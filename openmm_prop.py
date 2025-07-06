import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

from openmm.app import PDBFile, ForceField, DCDReporter, Simulation, NoCutoff, PME, HBonds, Topology, Modeller
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer, MonteCarloBarostat
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole, atmospheres
from threading import Lock
import pdbfixer

from datetime import datetime
import mdtraj
import numpy as np
import time
import os
import sys
import json
import math
import random

# Import TICA_PCoord from your existing file
from cg_prop import TICA_PCoord

class OpenMMPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)
        cgschnet_path = self.rc.config.require(['west', 'openmm', 'cgschnet_path']) 
        
        if cgschnet_path not in sys.path: 
            sys.path.append(cgschnet_path) 
          
        import simulate

        # Load parameters
        config = self.rc.config['west']['openmm']
        self.temperature = float(config.get('temperature', 300.0))
        self.timestep = float(config.get('timestep', 2.0))
        self.friction = float(config.get('friction', 1.0))
        self.pressure = float(config.get('pressure', 1.0))
        self.barostatInterval = int(config.get('barostatInterval', 25))
        self.constraintTolerance = float(config.get('constraintTolerance', 1e-6))
        self.hydrogenMass = float(config.get('hydrogenMass', 1.5))
        self.implicit_solvent = config.get('implicit_solvent', False)

        if self.implicit_solvent:
            self.saltCon = 0.15 # unit.molar
            self.solventDielectric = 78.5 # Default solvent dielectric: http://docs.openmm.org/latest/userguide/application/02_running_sims.html @ 2024.02.11
            self.implicitSolventKappa = 7.3*50.33355*math.sqrt(self.saltCon/self.solventDielectric/self.temperature)*(1/nanometer)


        self.steps = config['steps']
        self.save_steps = config['save_steps']


        try:
            platform = Platform.getPlatformByName('CUDA')
            self.num_gpus = int(config.get('num_gpus', 1))
            if self.num_gpus == -1:
                self.num_gpus = platform.getPropertyDefaultValue('CudaDeviceIndex').count(',') + 1 if ',' in platform.getPropertyDefaultValue('CudaDeviceIndex') else 1
        except Exception:
            self.num_gpus = 1

        self.gpu_precision = config.get('gpu_precision', 'single')


        # Topology and forcefield
        topology_path = os.path.expandvars(config['topology_path'])
        forcefield_files = config['forcefield']
        self.pdb = PDBFile(topology_path)
        self.forcefield = ForceField(*forcefield_files)

        fixer = pdbfixer.PDBFixer(topology_path)

        # find missing residues and atoms
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        print(f"Missing residues: {fixer.missingResidues}")
        print(f"Missing terminals: {fixer.missingTerminals}")
        print(f"Missing atoms: {fixer.missingAtoms}")

        # remove missing residues at the terminal
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            # terminal residues
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

        # check if the terminal residues are removed
        for key in list(keys):
            chain = chains[key[0]]
            assert key[1] != 0 or key[1] != len(list(chain.residues())), "Terminal residues are not removed."

        # find nonstandard residues
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        # add missing atoms, residues, and terminals
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # add missing hydrogens
        ph = 7.0
        fixer.addMissingHydrogens(ph)

        # make modeller
        modeller = Modeller(fixer.topology, fixer.positions)

        if self.implicit_solvent:
            modeller.deleteWater()
            ions_to_delete = [res for res in modeller.topology.residues() if res.name in ('NA', 'CL')]
            modeller.delete(ions_to_delete)

        self.pdb.topology = modeller.getTopology()
        self.pdb.positions = modeller.getPositions()

        # Create the system
        self.nonbondedMethod = NoCutoff if self.implicit_solvent else PME

        # pcoord calculator
        pcoord_config = dict(config['pcoord_calculator'])
        class_path = pcoord_config.pop("class")
        calculator_class = westpa.core.extloader.get_object(class_path)
        self.pcoord_calculator = calculator_class(**pcoord_config)

    def get_pcoord(self, state):

        if isinstance(state, BasisState):
            # Load initial structure (topology)
            ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'CA']
            positions = self.pdb.positions
            ca_positions = np.array([positions[i].value_in_unit(nanometer) for i in ca_indices])
            ca_positions = ca_positions[np.newaxis, :, :] * 10.0  # shape (1, N, 3) in Å
            state.pcoord = self.pcoord_calculator.calculate(ca_positions)
            return

        elif isinstance(state, InitialState):
            raise NotImplementedError

        raise NotImplementedError    


    def _get_next_gpu_index(self, segment_id):
        return segment_id % self.num_gpus

    def _create_simulation(self, seg_id):
        try:
            platform = Platform.getPlatformByName('CUDA')
            gpu_index = self._get_next_gpu_index(seg_id)
            print(f"Using gpu {gpu_index}")
            platform_properties = {
                'CudaDeviceIndex': str(gpu_index),
                'Precision': self.gpu_precision 
            }
        except Exception:
            print("CUDA not available, using CPU.")
            platform = Platform.getPlatformByName('CPU')
            platform_properties = {}

        if self.implicit_solvent:
            system = self.forcefield.createSystem(self.pdb.topology, nonbondedMethod=self.nonbondedMethod, constraints=HBonds, hydrogenMass=self.hydrogenMass, implicitSolventKappa=self.implicitSolventKappa)
        else:
            system = self.forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=self.nonbondedMethod,
                constraints=HBonds,
                rigidWater=True,
                hydrogenMass=self.hydrogenMass
            )

        if not self.implicit_solvent:
            system.addForce(MonteCarloBarostat(self.pressure * atmospheres,
                                                    self.temperature * kelvin,
                                                    self.barostatInterval))



        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond
        )
        integrator.setConstraintTolerance(self.constraintTolerance)
        
        seed = random.randint(1, 1000000)
        integrator.setRandomNumberSeed(seed)

        sim = Simulation(self.pdb.topology, system, integrator, platform, platform_properties)
        return sim

    def propagate(self, segments):
        starttime = time.time()
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']
        simulation = self._create_simulation(segments[0].seg_id)

        for segment in segments:
            segment_outdir = os.path.expandvars(segment_pattern.format(segment=segment))
            make_dir = False
            while not make_dir:
                try:
                    os.makedirs(segment_outdir, exist_ok=True)
                    make_dir = True
                except:
                    make_dir = False

            # Initialize state (positions/velocities)
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
                parent_outdir = os.path.expandvars(segment_pattern.format(segment=parent))
                state_file = os.path.join(parent_outdir, "seg.xml")

                print(f"Loading parent state from {state_file}")
                with open(state_file, 'r') as f:
                    xml = f.read()
                    simulation.context.setState(XmlSerializer.deserialize(xml))

                state = simulation.context.getState(getPositions=True)
                initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                initial_pos = np.array([initial_pos])
            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                print(f"Initializing new trajectory {segment.seg_id}")
                simulation.context.setPositions(self.pdb.positions)
                simulation.minimizeEnergy()
                simulation.context.setVelocitiesToTemperature(self.temperature)
                state = simulation.context.getState(getPositions=True)
                initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                initial_pos = np.array([initial_pos])
            else:
                raise ValueError(f"Unsupported segment initpoint type: {segment.initpoint_type}")

            
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            simulation.reporters.clear()
            simulation.reporters.append(DCDReporter(dcd_path, self.save_steps))

            print(f"Running {self.steps} steps for segment {segment.seg_id}")

            times = []
            forces = []
            energy_k = []
            energy_u = []

            assert self.steps % self.save_steps == 0, "total_steps must be divisible by report_steps"

            cur_steps = 0
            for i in range(self.steps//self.save_steps):
                simulation.step(self.save_steps)
                state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
                forces.append(state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer))
                times.append(state.getTime().value_in_unit(picosecond))
                energy_k.append(state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
                energy_u.append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))

            times = np.array(times)
            forces = np.array(forces)
            energy_k = np.array(energy_k)
            energy_u = np.array(energy_u)
            np.savez(os.path.join(segment_outdir, 'seg.npz'), times=times, forces=forces, energy_k=energy_k, energy_u=energy_u)


            # Save final state
            state = simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getForces=True,
                getEnergy=True,
                enforcePeriodicBox=True
            )
            with open(os.path.join(segment_outdir, "seg.xml"), 'w') as f:
                f.write(XmlSerializer.serialize(state))

            # Load trajectory
            md_top = mdtraj.Topology.from_openmm(self.pdb.topology)
            traj = mdtraj.load_dcd(dcd_path, top=md_top)
            all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])  # nm to Å

            # Select Cα atoms to match TICA training
            ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'CA']
            ca_positions = all_positions[:, ca_indices, :]
            segment.pcoord = self.pcoord_calculator.calculate(ca_positions)
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.time() - starttime

        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: Finished {len(segments)} segments in {time.time() - starttime:0.2f}s")
        return segments


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
# from cg_prop import TICA_PCoord

#We are using the RMSD Propagator from get_distance.py file calculations
class RMSDPropagator(WESTPropagator):
    def propagate(self, segment: Segment):
        seg_dir = segment.data_ref
        os.makedirs(seg_dir, exist_ok=True)
        os.chdir(seg_dir)

        # Link parent.xml based on init type
        if segment.initpoint_type == "SEG_INITPOINT_CONTINUES":
            os.symlink(os.path.join(segment.parent_data_ref, "seg.xml"), "parent.xml")
        else:
            os.symlink(segment.parent_data_ref, "parent.xml")

        # Step 1: Run dynamics with OpenMM
        subprocess.run([
            "python",
            os.path.join(self.sim_root, "common_files", "protein_prod.py")
        ], check=True)

        # Step 2: Compute RMSD using your get_distance.py
        subprocess.run([
            "python",
            os.path.join(self.sim_root, "common_files", "get_distance.py")
        ], check=True)

        # Step 3: Save RMSD to WESTPA
        with open("dist.dat") as f:
            rmsd_value = float(f.readline().strip())
        segment.pcoord = [rmsd_value]

class OpenMMPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)

        # No CGSchnet yet
        # cgschnet_path = self.rc.config.require(['west', 'openmm', 'cgschnet_path']) 
        
        # if cgschnet_path not in sys.path: 
        #     sys.path.append(cgschnet_path) 
        
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

        # pcoord calculator for TICA
        # pcoord_config = dict(config['pcoord_calculator']) #loading TICA knobs in the WESTPA config, under pcoord_calculator
        # class_path = pcoord_config.pop("class") #Grabbing whatever is stored in the class
        # calculator_class = westpa.core.extloader.get_object(class_path) 
        # self.pcoord_calculator = calculator_class(**pcoord_config)

        #RMSD configuration and reference prep
    #Config knobs: reference_pdb and rmsd_selection
        self.reference_pdb = os.path.expandvars(config.get('reference_pdb', config['topology_path']))
            #looks in west.cfg under [west][openmm] for a key called reference_pdb. If missing, it will fall back to config['topology_path'] the intial structure PDB
            #os.path.expandvars(...) lets you use the $WEST_SIM_ROOT or other environment variables in the path 

        #DNA safe default: phosphorus atoms; override via config if you like
        self.rmsd_selection = config.get('rmsd_selection', "name P")
            #this is safe for DNA backbone. 
            #User can override this in the config file, e.g. backbone or name CA

        #Build mdtraj topologies anad selectoin
        self.md_top = mdtraj.Topology.from_openmm(self.pdb.topology)
        #Converts the OpenMM topology you already loaded into a MDTraj Topology so we can  use MDTraj's RMSD functions
        #self.pdb was already created earlier in __init__ form your topology file 

        #Load reference: mdtraj auto-detects PDB/DCD/etc.
        self.ref_traj = mdtraj.load(self.reference_pdb) #reference structure as MDTraj object. If .pdb it loads a single frame, if .dcd it loads a trajectory
        self.sel_idx = self.md_top.select(self.rmsd_selection)
        #Creates an integer array of atoms indices from the simulation topology matching the selection string
            #Which atoms in the array are phosphorus P, may be [4, 10, 16, ...] 

        #Slice reference to selected atoms to match RMSD selection
        self.ref_traj = self.ref_traj.atom_slice(self.sel_idx)
        #Modifies the reference so it only contains the selected atoms
        #This ensures that when we compute RMSD later, the 
        #atom order and count match exactly between simulation frames and reference 

    def get_pcoord(self, state):

        # if isinstance(state, BasisState):
        #     # Load initial structure (topology)
        #     ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'CA']
        #     positions = self.pdb.positions
        #     ca_positions = np.array([positions[i].value_in_unit(nanometer) for i in ca_indices])
        #     ca_positions = ca_positions[np.newaxis, :, :] * 10.0  # shape (1, N, 3) in Å
        #     state.pcoord = self.pcoord_calculator.calculate(ca_positions)
        #     return

        if isinstance(state, BasisState):
            # Convert OpenMM units -> nm numpy array, shape (1, n_atoms, 3)
            pos_nm = np.array([[v.value_in_unit(nanometer) for v in self.pdb.position]], dtype=float)
            init_traj = mdtraj.Trajectory(xyz=pos_nm, topology=self.md_top)
            init_self = init_traj.atom_slice(self.sel_idx)
            ref_sel = self.ref_traj #already sliced 

            #RMSD vs reference frame 0; returns shape (1,)
            rmsd0 = mdtraj.rmsd(init_sel, ref_sel, frame=0)
            state.pcoord = rmsd0.reshape(-1, 1)
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
                if not os.path.isfile(state_file):
                    print(f"Missing segment state file: {state_file}")
                    continue

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
            # md_top = mdtraj.Topology.from_openmm(self.pdb.topology)
            # traj = mdtraj.load_dcd(dcd_path, top=md_top)
            # all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])  # nm to Å

            # # Select Cα atoms to match TICA training
            # ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'CA']
            # ca_positions = all_positions[:, ca_indices, :]
            # segment.pcoord = self.pcoord_calculator.calculate(ca_positions)
            # segment.status = Segment.SEG_STATUS_COMPLETE
            # segment.walltime = time.time() - starttime



            # #Use cached topology and selection
            # traj = mdtraj.load_dcd(dcd_path, top=self.md_top)
            # #Build a trajectory for the intiial frame and join so pcoord includes it
            # init_traj = mdtraj.Trajectory(xyz=initial_pos, topology=self.md_top) #initial_pos is nm, shape (1,n,3)
            # full_traj = init_traj.join(traj)

            # #Slice both to the selected atoms and compute RMSD vs reference (frame 0)
            # traj_sel = full_traj.atom_slice(self.sel_idx)
            # ref_sel = self.ref_traj #already sliced
            # rmsd_vals = mdtraj.rmsd(traj_sel, ref_sel, frame=0) #shape (n_frames,)

            # #WESTPA expects (n_frames, pcoord_dim)
            # segment.pcoord = rmsd_vals.reshape(-1, 1).astype(np.float32)
            # segment.status = Segment.SEG_STATUS_COMPLETE
            # segment.walltime = time.time() - starttime


            # Import RMSD helpers from get_distance.py
            from common_files.get_distance import build_distance_array_rmsd, traj_from_numpy

            # Prepare reference, parent, and segment trajectories
            base_traj = mdtraj.load(
                os.path.join(self.sim_root, "bstates", "bstate.xml"),
                top=os.path.join(self.sim_root, "bstates", "bstate.pdb")
            )
            parent_traj = mdtraj.load('parent.xml', top='bstate.pdb')
            traj = traj_from_numpy(parent_traj, os.path.join(segment_outdir, "seg.npz"))

            # Compute RMSD using helper
            d_arr = build_distance_array_rmsd(base_traj, parent_traj, traj)

            # WESTPA expects shape (n_frames, pcoord_dim)
            segment.pcoord = d_arr.reshape(-1, 1).astype(np.float32)
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.time() - starttime


        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"{current_time}: Finished {len(segments)} segments in {time.time() - starttime:0.2f}s")
        return segments

import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

from openmm.app import PDBFile, ForceField, DCDReporter, Simulation
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole

import mdtraj
import numpy as np
import time
import os
import sys
import json

# Import TICA_PCoord from your existing file
from cg_prop import TICA_PCoord

class OpenMMPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)


        cgschnet_path = self.rc.config.require(['west', 'openmm', 'cgschnet_path'])
        #FIXME: It would be better to do this without modifying sys.path
        if not cgschnet_path in sys.path:
            sys.path.append(cgschnet_path)
        import simulate  #pyright: ignore[reportMissingImports]


        # Load configuration
        topology_path = self.rc.config.require(['west', 'openmm', 'topology_path'])
        # Expand environment variables in topology path
        topology_path = os.path.expandvars(topology_path)
        forcefield_files = self.rc.config.require(['west', 'openmm', 'forcefield'])
        self.temperature = self.rc.config.get_typed(['west', 'openmm', 'temperature'], float, 300.0)
        self.timestep = self.rc.config.get_typed(['west', 'openmm', 'timestep'], float, 2.0)
        self.steps = self.rc.config.require(['west', 'openmm', 'steps'], int)
        self.save_steps = self.rc.config.require(['west', 'openmm', 'save_steps'], int)
        
        print(f"Initializing OpenMMPropagator with:")
        print(f"  topology_path: {topology_path}")
        print(f"  forcefield_files: {forcefield_files}")
        print(f"  temperature: {self.temperature}")
        print(f"  timestep: {self.timestep}")
        print(f"  steps: {self.steps}")
        print(f"  save_steps: {self.save_steps}")
        
        # Initialize OpenMM objects
        self.pdb = PDBFile(topology_path)
        self.forcefield = ForceField(*forcefield_files)
        
        # Create system with appropriate nonbonded method
        from openmm.app import NoCutoff, PME, HBonds
        self.system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=NoCutoff,  # or PME, CutoffNonPeriodic
            constraints=HBonds,
            rigidWater=True
        )
        
        # Create integrator
        self.integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            1.0 / picosecond,
            self.timestep * femtosecond
        )
        
        # Set up platform (CUDA/OpenCL for GPU)
        try:
            platform = Platform.getPlatformByName('CUDA')
            properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
        except Exception:
            print("CUDA platform not available, falling back to CPU")
            platform = Platform.getPlatformByName('CPU')
            properties = {}
        
        # Create simulation object
        self.simulation = Simulation(
            self.pdb.topology, 
            self.system, 
            self.integrator,
            platform, 
            properties
        )
        
        # Initialize pcoord calculator - using the same TICA_PCoord
        pcoord_config = dict(self.rc.config['west', 'openmm', 'pcoord_calculator'])
        pcoord_calculator_class_path = pcoord_config.pop("class")
        pcoord_calculator_class = westpa.core.extloader.get_object(pcoord_calculator_class_path)
        self.pcoord_calculator = pcoord_calculator_class(**pcoord_config)

        self.checkpoint_path = self.rc.config.require(['west', 'openmm', 'model_path'])
        checkpoint_path = self.checkpoint_path
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        assert os.path.exists(checkpoint_path)

        prior_path = os.path.join(checkpoint_dir, "priors.yaml")
        assert os.path.exists(prior_path)
        prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")

        with open(f"{prior_params_path}", 'r') as file:
            prior_params = json.load(file)
    
        self.mol, embeddings = simulate.load_molecule(prior_path, prior_params, topology_path, use_box=False, verbose=False)


    
    def get_pcoord(self, state):
        # Handle basis states
        if isinstance(state, BasisState):
            # print("state.auxref", state.auxref)
            # FIXME: Pay attention to what the intial state was
            bstate_mol = self.mol
            # print("bstate_mol.coords", bstate_mol.coords.shape)
            state.pcoord = self.pcoord_calculator.calculate(np.transpose(bstate_mol.coords, (2, 0, 1)))
            return
        elif isinstance(state, InitialState):
            raise NotImplementedError
            
        raise NotImplementedError
    
    def propagate(self, segments):
        starttime = time.time()
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']

        for segment in segments:
            segment_outdir = os.path.expandvars(segment_pattern.format(segment=segment))
            os.makedirs(segment_outdir, exist_ok=True)

            # NOTE: Removed basis state symlink
            # os.symlink(...)  ← This line is now gone

            # Set up initial positions and velocities
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
                parent_outdir = os.path.expandvars(segment_pattern.format(segment=parent))
                try:
                    with open(os.path.join(parent_outdir, "seg.xml"), 'r') as f:
                        xml = f.read()
                        self.simulation.context.setState(XmlSerializer.deserialize(xml))
                    state = self.simulation.context.getState(getPositions=True)
                    initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                    initial_pos = np.array([initial_pos])
                except Exception as e:
                    print(f"Error loading parent state: {e}")
                    raise

            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                # Start from topology PDB directly
                self.simulation.context.setPositions(self.pdb.positions)
                self.simulation.context.setVelocitiesToTemperature(self.temperature * kelvin)
                state = self.simulation.context.getState(getPositions=True)
                initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                initial_pos = np.array([initial_pos])

            # Use topology file directly for MDTraj
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            self.simulation.reporters.clear()
            self.simulation.reporters.append(DCDReporter(dcd_path, self.save_steps))

            print(f"Running {self.steps} steps for segment {segment.seg_id}")
            self.simulation.step(self.steps)

            # Save final state
            state = self.simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getForces=True,
                getEnergy=True,
                enforcePeriodicBox=True
            )

            with open(os.path.join(segment_outdir, "seg.xml"), 'w') as f:
                f.write(XmlSerializer.serialize(state))

            # Load trajectory using topology_path
            traj = mdtraj.load_dcd(dcd_path, top=self.pdb.topology)

            # Combine initial position with trajectory
            all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])  # Angstroms

            # Calculate and assign progress coordinate
            segment.pcoord = self.pcoord_calculator.calculate(all_positions)
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.time() - starttime

        print(f"Finished {len(segments)} segments in {time.time() - starttime:0.2f}s")
        return segments


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

# Import TICA_PCoord from your existing file
from cg_prop import TICA_PCoord

class OpenMMPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)
        
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
    
    def get_pcoord(self, state):
        # Handle basis states
        if isinstance(state, BasisState):
            pdb_path = state.auxref
            pdb = PDBFile(pdb_path)
            
            # Extract coordinates and reshape for TICA_PCoord
            positions = np.array([coord.value_in_unit(nanometer) for coord in pdb.positions])
            positions = positions.reshape(1, -1, 3)  
            
            # Convert nm to Angstroms if needed by TICA_PCoord
            positions *= 10  # Now in Angstroms for TICA_PCoord which will divide by 10
            
            state.pcoord = self.pcoord_calculator.calculate(positions)
            return
            
        elif isinstance(state, InitialState):
            raise NotImplementedError
            
        raise NotImplementedError
    
    def propagate(self, segments):
        starttime = time.time()
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']
        
        for segment in segments:
            # Create output directory
            segment_outdir = segment_pattern.format(segment=segment)
            segment_outdir = os.path.expandvars(segment_outdir)
            os.makedirs(segment_outdir, exist_ok=True)
            
            # Create symlink to basis state PDB for trajectory visualization
            bstate_path = os.path.join(self.rc.config.get(['west', 'data', 'data_refs', 'basis_state']).replace('{basis_state.auxref}', 'bstate.pdb'))
            os.symlink(bstate_path, os.path.join(segment_outdir, 'bstate.pdb'))
            
            # Set up initial positions and velocities
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                # Load from parent segment
                parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
                parent_outdir = os.path.expandvars(segment_pattern.format(segment=parent))
                
                # Load final state from parent
                try:
                    with open(os.path.join(parent_outdir, "seg.xml"), 'r') as f:
                        xml = f.read()
                        self.simulation.context.setState(XmlSerializer.deserialize(xml))
                        
                    # Get initial positions for pcoord calculation
                    state = self.simulation.context.getState(getPositions=True)
                    initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                    initial_pos = np.array([initial_pos])  # Add frame dimension
                except Exception as e:
                    print(f"Error loading parent state: {e}")
                    raise
                    
            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                # For a new trajectory, start from a basis state
                initial_state = self.initial_states[segment.initial_state_id]
                basis_state = self.basis_states[initial_state.basis_state_id]
                
                # Load positions from PDB
                pdb_path = basis_state.auxref
                pdb = PDBFile(pdb_path)
                self.simulation.context.setPositions(pdb.positions)
                
                # Set velocities according to temperature
                self.simulation.context.setVelocitiesToTemperature(self.temperature * kelvin)
                
                # Get initial positions for pcoord calculation
                state = self.simulation.context.getState(getPositions=True)
                initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                initial_pos = np.array([initial_pos])  # Add frame dimension
            
            # Setup DCD reporter for trajectory
            dcd_path = os.path.join(segment_outdir, 'seg.dcd')
            self.simulation.reporters.clear()
            self.simulation.reporters.append(DCDReporter(dcd_path, self.save_steps))
            
            # Run simulation
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
            
            xml_path = os.path.join(segment_outdir, "seg.xml")
            with open(xml_path, 'w') as f:
                f.write(XmlSerializer.serialize(state))
                
            # Load trajectory with MDTraj for pcoord calculation
            traj = mdtraj.load_dcd(dcd_path, top=os.path.join(segment_outdir, 'bstate.pdb'))
            
            # Convert to angstroms for TICA_PCoord
            all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])  # Convert nm to Angstroms
            
            # Calculate progress coordinate
            segment.pcoord = self.pcoord_calculator.calculate(all_positions)
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.time() - starttime
            
        print(f"Finished {len(segments)} segments in {time.time() - starttime:0.2f}s")
        return segments

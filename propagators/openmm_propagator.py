from propagators.base_propagator import BasePropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

from openmm.app import PDBFile, ForceField, Simulation, HBonds
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole

from file_system.md_store.save_npz import save_openmm_npz

import numpy as np
import time
import os
import sys
import random


class OpenMMPropagator(BasePropagator):
    
    def __init__(self, rc=None):
        super(OpenMMPropagator, self).__init__(rc)
    
    def _load_config(self):
        cgschnet_path = self.rc.config.require(['west', 'openmm', 'cgschnet_path'])
        if cgschnet_path not in sys.path:
            sys.path.append(cgschnet_path)
        import simulate
        
        config = self.rc.config['west']['openmm']
        self.temperature = float(config.get('temperature', 300.0))
        self.timestep = float(config.get('timestep', 2.0))
        self.friction = float(config.get('friction', 1.0))
        self.pressure = float(config.get('pressure', 1.0))
        self.barostatInterval = int(config.get('barostatInterval', 25))
        self.constraintTolerance = float(config.get('constraintTolerance', 1e-6))
        self.hydrogenMass = float(config.get('hydrogenMass', 1.5))
        self.implicit_solvent = config.get('implicit_solvent', False)
        self.steps = config['steps']
        self.save_steps = config['save_steps']
        self.save_format = self._get_save_format(['west', 'openmm'])
        
        try:
            platform = Platform.getPlatformByName('CUDA')
            self.num_gpus = int(config.get('num_gpus', 1))
            if self.num_gpus == -1:
                default_device = platform.getPropertyDefaultValue('CudaDeviceIndex')
                self.num_gpus = default_device.count(',') + 1 if ',' in default_device else 1
        except Exception:
            self.num_gpus = 1
        
        self.gpu_precision = config.get('gpu_precision', 'single')
        
        self.topology_path = os.path.expandvars(config['topology_path'])
        self.forcefield_files = config['forcefield']
        self.pdb = PDBFile(self.topology_path)
        self.forcefield = ForceField(*self.forcefield_files)
        self.nonbondedMethod = None
    
    def _get_pcoord_config(self):
        return self.rc.config['west']['openmm'].get('pcoord_calculator')
    
    def get_pcoord(self, state):
        if isinstance(state, BasisState):
            ca_indices = [atom.index for atom in self.pdb.topology.atoms() if atom.name == 'CA']
            positions = self.pdb.positions
            ca_positions = np.array([positions[i].value_in_unit(nanometer) for i in ca_indices])
            ca_positions = ca_positions[np.newaxis, :, :] * 10.0
            state.pcoord = self.pcoord_calculator.calculate(ca_positions)
            return
        elif isinstance(state, InitialState):
            raise NotImplementedError
        raise NotImplementedError
    
    def _get_next_gpu_index(self, segment_id):
        return segment_id % self.num_gpus
    
    def _get_platform(self, seg_id):
        try:
            platform = Platform.getPlatformByName('CUDA')
            gpu_index = self._get_next_gpu_index(seg_id)
            print(f"Using gpu {gpu_index}")
            properties = {'CudaDeviceIndex': str(gpu_index), 'Precision': self.gpu_precision}
        except Exception:
            print("CUDA not available, using CPU.")
            platform = Platform.getPlatformByName('CPU')
            properties = {}
        return platform, properties
    
    def _create_system(self):
        raise NotImplementedError
    
    def _create_simulation(self, seg_id):
        platform, properties = self._get_platform(seg_id)
        system = self._create_system()
        
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond
        )
        integrator.setConstraintTolerance(self.constraintTolerance)
        integrator.setRandomNumberSeed(random.randint(1, 1000000))
        
        return Simulation(self.pdb.topology, system, integrator, platform, properties)
    
    def _init_segment_state(self, simulation, segment):
        if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
            parent_outdir = self._get_parent_outdir(segment)
            state_file = os.path.join(parent_outdir, "seg.xml")
            
            print(f"Loading parent state from {state_file}")
            with open(state_file, 'r') as f:
                simulation.context.setState(XmlSerializer.deserialize(f.read()))
            
            state = simulation.context.getState(getPositions=True)
            initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            return np.array([initial_pos])
            
        elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
            print(f"Initializing new trajectory {segment.seg_id}")
            simulation.context.setPositions(self.pdb.positions)
            simulation.minimizeEnergy()
            simulation.context.setVelocitiesToTemperature(self.temperature)
            state = simulation.context.getState(getPositions=True)
            initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            return np.array([initial_pos])
        else:
            raise ValueError(f"Unsupported segment initpoint type: {segment.initpoint_type}")
    
    def _setup_reporters(self, simulation, segment_outdir):
        raise NotImplementedError
    
    def _run_simulation(self, simulation):
        print(f"Running {self.steps} steps")
        
        times, forces, energy_k, energy_u = [], [], [], []
        positions_list = []
        
        assert self.steps % self.save_steps == 0
        
        for i in range(self.steps // self.save_steps):
            simulation.step(self.save_steps)
            state = simulation.context.getState(getPositions=True, getForces=True, getEnergy=True)
            
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
            positions_list.append(pos_nm)
            
            forces.append(state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer))
            times.append(state.getTime().value_in_unit(picosecond))
            energy_k.append(state.getKineticEnergy().value_in_unit(kilojoule_per_mole))
            energy_u.append(state.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
        
        return times, forces, energy_k, energy_u, positions_list
    
    def _save_final_state(self, simulation, segment_outdir):
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True,
            getEnergy=True, enforcePeriodicBox=True
        )
        with open(os.path.join(segment_outdir, "seg.xml"), 'w') as f:
            f.write(XmlSerializer.serialize(state))
    
    def _calculate_pcoord(self, segment_outdir, initial_pos):
        raise NotImplementedError
    
    def propagate(self, segments):
        starttime = time.time()
        simulation = self._create_simulation(segments[0].seg_id)
        
        for segment in segments:
            segment_outdir = self._get_segment_outdir(segment)
            os.makedirs(segment_outdir, exist_ok=True)
            
            initial_pos = self._init_segment_state(simulation, segment)
            self._setup_reporters(simulation, segment_outdir)
            times, forces, energy_k, energy_u, positions_list = self._run_simulation(simulation)
            
            if self.save_format == 'npz':
                save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u, positions_list)
            else:
                save_openmm_npz(segment_outdir, times, forces, energy_k, energy_u)
            
            self._save_final_state(simulation, segment_outdir)
            segment.pcoord = self._calculate_pcoord(segment_outdir, initial_pos)
            self._finalize_segment(segment, starttime)
        
        self._print_completion(len(segments), time.time() - starttime)
        return segments

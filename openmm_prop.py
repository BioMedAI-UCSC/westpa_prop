import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

from openmm.app import PDBFile, ForceField, DCDReporter, Simulation
from openmm import Platform, LangevinMiddleIntegrator, XmlSerializer
from openmm.unit import kelvin, picosecond, femtosecond, nanometer, kilojoule_per_mole, atmospheres
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

        # Load paths
        cgschnet_path = self.rc.config.require(['west', 'openmm', 'cgschnet_path'])
        generate_path = self.rc.config.require(['west', 'openmm', 'openmm_generate_path'])
        if cgschnet_path not in sys.path:
            sys.path.append(cgschnet_path)
        if generate_path not in sys.path:
            sys.path.append(generate_path)

        import simulate
        from module import function, preprocess, simulation

        self.get_non_water_atom_indexes = function.get_non_water_atom_indexes
        self.prepare_protein = preprocess.prepare_protein
        self.run_openmm = simulation.run

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

        self.steps = config['steps']
        self.save_steps = config['save_steps']

        # Topology and forcefield
        topology_path = os.path.expandvars(config['topology_path'])
        forcefield_files = config['forcefield']
        self.pdb = PDBFile(topology_path)
        self.forcefield = ForceField(*forcefield_files)

        # Create the system
        from openmm.app import NoCutoff, PME, HBonds
        nonbondedMethod = NoCutoff if self.implicit_solvent else PME

        self.system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=nonbondedMethod,
            constraints=HBonds,
            rigidWater=True,
            hydrogenMass=self.hydrogenMass
        )

        if not self.implicit_solvent:
            from openmm import MonteCarloBarostat
            self.system.addForce(MonteCarloBarostat(self.pressure * atmospheres,
                                                    self.temperature * kelvin,
                                                    self.barostatInterval))

        # Integrator
        from openmm import LangevinMiddleIntegrator
        self.integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction / picosecond,
            self.timestep * femtosecond
        )
        self.integrator.setConstraintTolerance(self.constraintTolerance)

        # Platform
        try:
            platform = Platform.getPlatformByName('CUDA')
            platform_properties = {'Precision': 'mixed'}
        except Exception:
            print("CUDA not available, using CPU.")
            platform = Platform.getPlatformByName('CPU')
            platform_properties = {}

        # Simulation object
        from openmm.app import Simulation
        self.simulation = Simulation(
            self.pdb.topology, self.system, self.integrator,
            platform, platform_properties
        )

        # Load model
        self.checkpoint_path = os.path.expandvars(config['model_path'])
        checkpoint_file = self.checkpoint_path
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(checkpoint_file, "checkpoint-best.pth")
        assert os.path.exists(checkpoint_file)
        prior_params_path = os.path.join(os.path.dirname(checkpoint_file), "prior_params.json")
        prior_path = os.path.join(os.path.dirname(checkpoint_file), "priors.yaml")

        with open(prior_params_path, 'r') as f:
            prior_params = json.load(f)

        self.mol, embeddings = simulate.load_molecule(
            prior_path, prior_params, topology_path, use_box=False, verbose=False
        )

        # pcoord calculator
        pcoord_config = dict(config['pcoord_calculator'])
        class_path = pcoord_config.pop("class")
        calculator_class = westpa.core.extloader.get_object(class_path)
        self.pcoord_calculator = calculator_class(**pcoord_config)

    
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
        import os
        import time
        import numpy as np
        import mdtraj
        from openmm import XmlSerializer
        from westpa.core.segment import Segment

        starttime = time.time()
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']

        for segment in segments:
            segment_outdir = os.path.expandvars(segment_pattern.format(segment=segment))
            os.makedirs(segment_outdir, exist_ok=True)

            # Initialize state (positions/velocities)
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
                parent_outdir = os.path.expandvars(segment_pattern.format(segment=parent))
                state_file = os.path.join(parent_outdir, "seg.xml")

                print(f"Loading parent state from {state_file}")
                with open(state_file, 'r') as f:
                    xml = f.read()
                    self.simulation.context.setState(XmlSerializer.deserialize(xml))

                state = self.simulation.context.getState(getPositions=True)
                initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                initial_pos = np.array([initial_pos])

            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                print("Initializing from topology (new trajectory)")
                self.simulation.context.setPositions(self.pdb.positions)
                self.simulation.minimizeEnergy()
                self.simulation.context.setVelocitiesToTemperature(self.temperature)
                state = self.simulation.context.getState(getPositions=True)
                initial_pos = state.getPositions(asNumpy=True).value_in_unit(nanometer)
                initial_pos = np.array([initial_pos])

            else:
                raise ValueError(f"Unsupported segment initpoint type: {segment.initpoint_type}")

            # Set up reporter
            
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

            # Load trajectory
            import mdtraj as md
            md_top = md.Topology.from_openmm(self.pdb.topology)
            traj = mdtraj.load_dcd(dcd_path, top=md_top)
            all_positions = np.concatenate([initial_pos * 10, traj.xyz * 10])  # nm to Å

            # Calculate progress coordinate
            segment.pcoord = self.pcoord_calculator.calculate(all_positions)
            segment.status = Segment.SEG_STATUS_COMPLETE
            segment.walltime = time.time() - starttime

        print(f"Finished {len(segments)} segments in {time.time() - starttime:0.2f}s")
        return segments


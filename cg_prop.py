import torch
import westpa
from westpa.core.propagators import WESTPropagator
from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

import numpy as np
import time

import sys
import os
import json
import pickle

from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper

# https://westpa.readthedocs.io/en/stable/documentation/core/westpa.core.propagators.html
# https://github.com/westpa/westpa/blob/b3afe209fcffc6238c1d2ec700059c7e30f2adca/src/westpa/core/propagators/executable.py#L688


class TICA_PCoord():
    def __init__(self, model_path, components=[0]):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # assert isinstance(model, TicaModel)
            assert hasattr(model, "tica_model")
        self.tica_model = model.tica_model
        assert isinstance(components, list)
        self.components = components

    def calculate(self, data):
        assert len(data.shape) == 3, "Data must be (frames, atoms, xyz)"
        n_atoms = data.shape[1]
        pairs_a, pairs_b = np.triu_indices(n_atoms, k=1)
        distances = np.linalg.norm(data[:,pairs_a] - data[:, pairs_b], axis=2)
        distances /= 10 # The TICA model expects nm, the data is in Ang
        tica_comps = self.tica_model.transform(distances)
        return tica_comps[:, self.components]

class TestPropagator(WESTPropagator):
    def __init__(self, rc=None):
        super(TestPropagator, self).__init__(rc)

        device = "cuda"

        cgschnet_path = self.rc.config.require(['west', 'cg_prop', 'cgschnet_path'])
        #FIXME: It would be better to do this without modifying sys.path
        if not cgschnet_path in sys.path:
            sys.path.append(cgschnet_path)
        import simulate  #pyright: ignore[reportMissingImports]

        checkpoint_path = self.rc.config.require(['west', 'cg_prop', 'model_path'])
        topology_path = self.rc.config.require(['west', 'cg_prop', 'topology_path'])

        self.replicas = self.rc.config.get_typed(['west', 'propagation', 'block_size'], int, 1)
        use_box = self.rc.config.get_typed(['west', 'cg_prop', 'use_box'], bool, False)
        self.temperature = self.rc.config.get_typed(['west', 'cg_prop', 'temperature'], int, 300)

        self.steps = self.rc.config.require(['west', 'cg_prop', 'steps'], int)
        self.save_steps = self.rc.config.require(['west', 'cg_prop', 'save_steps'], int)
        self.timestep = self.rc.config.require(['west', 'cg_prop', 'timestep'], int)

        assert not use_box, "Box initialization not implemented yet"

        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        assert os.path.exists(checkpoint_path)

        prior_path = os.path.join(checkpoint_dir, "priors.yaml")
        assert os.path.exists(prior_path)
        prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")

        with open(f"{prior_params_path}", 'r') as file:
            prior_params = json.load(file)
        
        self.model = simulate.load_model(checkpoint_path, device, verbose=False)
        mol, embeddings = simulate.load_molecule(prior_path, prior_params, topology_path, use_box=use_box, verbose=False)

        calcs = []
        calcs.append(simulate.build_calc(self.model, mol, embeddings, use_box=use_box, replicas=self.replicas,
                                temperature=self.temperature, device=device))
        
        forceterms = prior_params["forceterms"]
        exclusions = prior_params["exclusions"]

        #FIXME: There should be as many mols as replicas
        system, forces = simulate.make_system([mol], prior_path, calcs, device, forceterms, exclusions, self.replicas, temperature=self.temperature, new_ff=True)
        self.md_system = system
        self.md_forces = forces

        langevin_gamma = 1

        self.integrator = Integrator(
            system,
            forces,
            self.timestep,
            device,
            gamma=langevin_gamma,
            T=self.temperature,
        )
        self.wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)
        self.mol = mol

        # Intialize pcoord calculator from the values in west.cfg
        pcoord_config = dict(self.rc.config['west', 'cg_prop', 'pcoord_calculator'])
        pcoord_calculator_class_path = pcoord_config.pop("class")
        pcoord_calculator_class = westpa.core.extloader.get_object(pcoord_calculator_class_path)
        self.pcoord_calculator = pcoord_calculator_class(**pcoord_config)

    def get_pcoord(self, state):
        # This function gets called to find the initial pcoord of a starting state
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
    
    def gen_istate(self, basis_state, initial_state):
        raise NotImplementedError
    
    def propagate(self, segments):
        starttime = time.time()
        segment_pattern = self.rc.config['west', 'data', 'data_refs', 'segment']

        assert len(segments) <= self.replicas

        # Initialize replicas for this iteration
        parent_pos = []
        for i, segment in enumerate(segments):
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                # Parent logic from the executable propagator
                parent = Segment(n_iter=segment.n_iter - 1, seg_id=segment.parent_id)
                parent_outdir = (segment_pattern.format(segment=parent))
                parent_outdir = os.path.expandvars(parent_outdir)

                parent_traj = np.load(os.path.join(parent_outdir, "seg.npz"))
                coords = torch.as_tensor(parent_traj["pos"][-1])
                velocities = torch.as_tensor(parent_traj["vel"][-1])

            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                # Could also be InitialState.ISTATE_TYPE_START but we don't support that mode
                initial_state = self.initial_states[segment.initial_state_id]
                assert initial_state.istate_type == InitialState.ISTATE_TYPE_BASIS
                # TODO: Pay attention to what the basis state was
                # basis_state = self.basis_states[initial_state.basis_state_id]

                bstate_mol = self.mol
                coords = torch.as_tensor(bstate_mol.coords.reshape(-1, 3))
                velocities = maxwell_boltzmann(self.md_forces.par.masses, self.temperature, 1)[0]

            self.md_system.pos[i][:] = coords
            self.md_system.vel[i][:] = velocities
            parent_pos.append(coords.numpy())

        #TODO: Set seed?

        # The integrator expects the forces to be valid before the first step() is called
        Epot = self.md_forces.compute(self.md_system.pos, self.md_system.box, self.md_system.forces)

        trajEpot = []
        trajEkin = []
        trajTemp = []
        trajTime = []
        trajPos  = []
        trajVel  = []
        for i in range(1, int(self.steps / self.save_steps) + 1):
            Ekin, Epot, T = self.integrator.step(niter=self.save_steps)
            self.wrapper.wrap(self.md_system.pos, self.md_system.box)
            currpos = self.md_system.pos.detach().cpu().numpy()
            currvel = self.md_system.vel.detach().cpu().numpy()

            trajEkin.append(Ekin)
            trajEpot.append(Epot)
            trajTemp.append(T)
            trajTime.append(np.repeat(i*self.timestep, self.replicas))
            trajPos.append(currpos)
            trajVel.append(currvel)

        # Save each replica to its corresponding segment
        for i, segment in enumerate(segments):
            segment_outdir = (segment_pattern.format(segment=segment))
            segment_outdir = os.path.expandvars(segment_outdir)
            os.makedirs(segment_outdir, exist_ok=True)

            np.savez(os.path.join(segment_outdir, "seg.npz"),
                epot = np.array([f[i] for f in trajEpot]),
                ekin = np.array([f[i] for f in trajEkin]),
                temp = np.array([f[i] for f in trajTemp]),
                time = np.array([f[i] for f in trajTime]),
                pos  = np.array([f[i] for f in trajPos]),
                vel  = np.array([f[i] for f in trajVel]),
            )

            pcoord_pos = np.array([parent_pos[i]] + [f[i] for f in trajPos])
            segment.pcoord = self.pcoord_calculator.calculate(pcoord_pos)
            # print("segment.pcoord[-1]", segment.pcoord[-1])

            segment.status = Segment.SEG_STATUS_COMPLETE

            segment.walltime = time.time() - starttime
            # segment.cputime = rusage.ru_utime
        print(f"Finished {len(segments)} segments in {time.time() - starttime:0.2f}s")
        
        return segments

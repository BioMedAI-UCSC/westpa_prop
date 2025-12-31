import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from westpa.core.states import BasisState, InitialState
from westpa.core.segment import Segment

from file_system.md_store.save_npz import save_cg_npz
from file_system.md_store.save_dcd import write_dcd_from_positions
from propagators.base_propagator import BasePropagator

import torch
import numpy as np
import time
import sys
import os
import json

from torchmd.integrator import maxwell_boltzmann, Integrator
from torchmd.wrapper import Wrapper

class CGMLPropagator(BasePropagator):
    
    def __init__(self, rc=None):
        super(CGMLPropagator, self).__init__(rc)
    
    def _load_config(self):
        device = "cuda"
        
        cgschnet_path = self.rc.config.require(['west', 'cg_prop', 'cgschnet_path'])
        if cgschnet_path not in sys.path:
            sys.path.append(cgschnet_path)
        import simulate
        
        checkpoint_path = self.rc.config.require(['west', 'cg_prop', 'model_path'])
        topology_path = self.rc.config.require(['west', 'cg_prop', 'topology_path'])
        
        self.replicas = self.rc.config.get_typed(['west', 'propagation', 'block_size'], int, 1)
        use_box = self.rc.config.get_typed(['west', 'cg_prop', 'use_box'], bool, False)
        self.temperature = self.rc.config.get_typed(['west', 'cg_prop', 'temperature'], int, 300)
        self.steps = self.rc.config.require(['west', 'cg_prop', 'steps'], int)
        self.save_steps = self.rc.config.require(['west', 'cg_prop', 'save_steps'], int)
        self.timestep = self.rc.config.require(['west', 'cg_prop', 'timestep'], int)
        self.save_format = self._get_save_format(['west', 'cg_prop'])
        
        assert not use_box
        
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        
        prior_path = os.path.join(checkpoint_dir, "priors.yaml")
        prior_params_path = os.path.join(checkpoint_dir, "prior_params.json")
        
        with open(prior_params_path, 'r') as f:
            prior_params = json.load(f)
        
        self.model = simulate.load_model(checkpoint_path, device, verbose=False)
        mol, embeddings = simulate.load_molecule(prior_path, prior_params, topology_path,
                                                 use_box=use_box, verbose=False)
        
        calcs = [simulate.build_calc(self.model, mol, embeddings, use_box=use_box,
                                     replicas=self.replicas, temperature=self.temperature,
                                     device=device)]
        
        forceterms = prior_params["forceterms"]
        exclusions = prior_params["exclusions"]
        
        system, forces = simulate.make_system([mol], prior_path, calcs, device, forceterms,
                                              exclusions, self.replicas,
                                              temperature=self.temperature, new_ff=True)
        self.md_system = system
        self.md_forces = forces
        
        
        self.integrator = Integrator(system, forces, self.timestep, device,
                                    gamma=1, T=self.temperature)
        self.wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)
        self.mol = mol
    
    def _get_pcoord_config(self):
        return self.rc.config['west', 'cg_prop'].get('pcoord_calculator')
    
    def get_pcoord(self, state):
        if isinstance(state, BasisState):
            state.pcoord = self.pcoord_calculator.calculate(
                np.transpose(self.mol.coords, (2, 0, 1))
            )
            return
        elif isinstance(state, InitialState):
            raise NotImplementedError
        raise NotImplementedError
    
    def propagate(self, segments):
        """Propagate a batch of WEST segments in parallel on one GPU.

        Each WEST segment is mapped to a replica in the underlying TorchMD
        System. All replicas are integrated together, just like in
        simulate.run_simulation / dynamics().
        """
        starttime = time.time()
        n_segments = len(segments)
        assert n_segments > 0, "No segments passed to CGMLPropagator.propagate"
        assert n_segments <= self.replicas, (
            f"Got {n_segments} segments but propagator was initialized with "
            f"only {self.replicas} replicas"
        )

        device = self.md_system.pos.device

        parent_pos = [None] * n_segments

        newtraj_indices = []
        cont_indices = []
        for i, segment in enumerate(segments):
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                cont_indices.append(i)
            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                newtraj_indices.append(i)
            else:
                raise ValueError(
                    f"Unsupported initpoint_type for segment {segment.seg_id}: "
                    f"{segment.initpoint_type}"
                )

        newtraj_velocities = None
        if newtraj_indices:
            newtraj_velocities = maxwell_boltzmann(
                self.md_forces.par.masses,
                self.temperature,
                len(newtraj_indices),
            )

        newtraj_vel_map = {}
        if newtraj_velocities is not None:
            for j, seg_idx in enumerate(newtraj_indices):
                newtraj_vel_map[seg_idx] = newtraj_velocities[j]

        for i, segment in enumerate(segments):
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                parent_outdir = self._get_parent_outdir(segment)
                parent_traj = np.load(os.path.join(parent_outdir, "seg.npz"))

                coords = torch.as_tensor(
                    parent_traj["pos"][-1], device=device
                )  # (n_atoms, 3)
                velocities = torch.as_tensor(
                    parent_traj["vel"][-1], device=device
                )  # (n_atoms, 3)

            elif segment.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ:
                # New trajectory: start from reference structure + fresh velocities
                initial_state = self.initial_states[segment.initial_state_id]
                assert (
                    initial_state.istate_type == InitialState.ISTATE_TYPE_BASIS
                ), "NEWTRAJ must be initialized from a BASIS state"

                coords = torch.as_tensor(
                    self.mol.coords.reshape(-1, 3),
                    device=device,
                )  # (n_atoms, 3)

                velocities = newtraj_vel_map[i].to(device)

            else:
                raise RuntimeError("Unexpected initpoint_type encountered")

            self.md_system.pos[i][:] = coords
            self.md_system.vel[i][:] = velocities

            parent_pos[i] = coords.detach().cpu().numpy()

        Epot = self.md_forces.compute(
            self.md_system.pos,
            self.md_system.box,
            self.md_system.forces,
        )

        trajEpot = []
        trajEkin = []
        trajTemp = []
        trajTime = []
        trajPos = []
        trajVel = []

        n_frames = int(self.steps / self.save_steps)

        for frame in range(1, n_frames + 1):
            Ekin, Epot, T = self.integrator.step(niter=self.save_steps)
            self.wrapper.wrap(self.md_system.pos, self.md_system.box)

            currpos = self.md_system.pos.detach().cpu().numpy()
            currvel = self.md_system.vel.detach().cpu().numpy()

            trajEkin.append(Ekin)
            trajEpot.append(Epot)
            trajTemp.append(T)
            trajTime.append(np.repeat(frame * self.timestep, self.replicas))
            trajPos.append(currpos)
            trajVel.append(currvel)

        for i, segment in enumerate(segments):
            segment_outdir = self._get_segment_outdir(segment)
            os.makedirs(segment_outdir, exist_ok=True)

            if self.save_format == "npz":
                save_cg_npz(
                    segment_outdir,
                    epot=[f[i] for f in trajEpot],
                    ekin=[f[i] for f in trajEkin],
                    temp=[f[i] for f in trajTemp],
                    time=[f[i] for f in trajTime],
                    pos=[f[i] for f in trajPos],
                    vel=[f[i] for f in trajVel],
                )
            else:
                dcd_path = os.path.join(segment_outdir, "seg.dcd")
                positions_nm = np.array([f[i] for f in trajPos]) / 10.0
                write_dcd_from_positions(dcd_path, positions_nm)

            pcoord_pos = np.array(
                [parent_pos[i]] + [f[i] for f in trajPos]
            )
            segment.pcoord = self.pcoord_calculator.calculate(pcoord_pos)

            self._finalize_segment(segment, starttime)

        self._print_completion(len(segments), time.time() - starttime)
        return segments


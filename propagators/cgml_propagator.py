import os
import sys
import json
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torchmd.integrator import maxwell_boltzmann, Integrator
from torchmd.wrapper import Wrapper
from westpa.core.states import BasisState
from westpa.core.segment import Segment

from file_system.md_store.save_npz import save_cg_npz
from file_system.md_store.save_dcd import write_dcd_from_positions
from propagators.base_propagator import BasePropagator


class CGMLPropagator(BasePropagator):

    def __init__(self, rc=None):
        super().__init__(rc)

    def _load_config(self):
        device = "cuda"

        cgschnet_path = self.rc.config.require(["west", "cg_prop", "cgschnet_path"])
        if cgschnet_path not in sys.path:
            sys.path.append(cgschnet_path)
        import simulate

        checkpoint_path  = self.rc.config.require(["west", "cg_prop", "model_path"])
        topology_path    = self.rc.config.require(["west", "cg_prop", "topology_path"])
        self.replicas    = self.rc.config.get_typed(["west", "propagation", "block_size"], int, 1)
        use_box          = self.rc.config.get_typed(["west", "cg_prop", "use_box"], bool, False)
        self.temperature = self.rc.config.get_typed(["west", "cg_prop", "temperature"], int, 300)
        self.steps       = self.rc.config.require(["west", "cg_prop", "steps"], int)
        self.save_steps  = self.rc.config.require(["west", "cg_prop", "save_steps"], int)
        self.timestep    = self.rc.config.require(["west", "cg_prop", "timestep"], int)
        self.save_format = self._get_save_format(["west", "cg_prop"])

        assert not use_box

        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint-best.pth")
        checkpoint_dir = os.path.dirname(checkpoint_path)

        with open(os.path.join(checkpoint_dir, "prior_params.json")) as f:
            prior_params = json.load(f)
        prior_path = os.path.join(checkpoint_dir, "priors.yaml")

        self.model      = simulate.load_model(checkpoint_path, device, verbose=False)
        mol, embeddings = simulate.load_molecule(prior_path, prior_params, topology_path, use_box=use_box, verbose=False)

        calcs = [simulate.build_calc(
            self.model, mol, embeddings,
            use_box=use_box, replicas=self.replicas,
            temperature=self.temperature, device=device,
        )]
        system, forces = simulate.make_system(
            [mol], prior_path, calcs, device,
            prior_params["forceterms"], prior_params["exclusions"],
            self.replicas, temperature=self.temperature, new_ff=True,
        )

        self.md_system  = system
        self.md_forces  = forces
        self.integrator = Integrator(system, forces, self.timestep, device, gamma=1, T=self.temperature)
        self.wrapper    = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)
        self.mol        = mol

    def _get_pcoord_config(self):
        return self.rc.config["west"]["cg_prop"].get("pcoord_calculator")

    def _get_recorded_configs(self):
        return self.rc.config["west"]["cg_prop"].get("recorded_calculators", [])

    def get_pcoord(self, state):
        if isinstance(state, BasisState):
            state.pcoord = self.pcoord_calculator.calculate(
                np.transpose(self.mol.coords, (2, 0, 1))
            )
            return
        raise NotImplementedError

    def propagate(self, segments):
        starttime  = time.time()
        n_segments = len(segments)
        assert 0 < n_segments <= self.replicas

        device     = self.md_system.pos.device
        parent_pos = [None] * n_segments

        newtraj_indices = [i for i, s in enumerate(segments) if s.initpoint_type == Segment.SEG_INITPOINT_NEWTRAJ]
        newtraj_vel_map = {}
        if newtraj_indices:
            vels = maxwell_boltzmann(self.md_forces.par.masses, self.temperature, len(newtraj_indices))
            for j, seg_idx in enumerate(newtraj_indices):
                newtraj_vel_map[seg_idx] = vels[j]

        for i, segment in enumerate(segments):
            if segment.initpoint_type == Segment.SEG_INITPOINT_CONTINUES:
                parent_traj = np.load(os.path.join(self._get_parent_outdir(segment), "seg.npz"))
                coords      = torch.as_tensor(parent_traj["pos"][-1], device=device)
                velocities  = torch.as_tensor(parent_traj["vel"][-1], device=device)
            else:
                coords     = torch.as_tensor(self.mol.coords.reshape(-1, 3), device=device)
                velocities = newtraj_vel_map[i].to(device)

            self.md_system.pos[i][:] = coords
            self.md_system.vel[i][:] = velocities
            parent_pos[i]            = coords.detach().cpu().numpy()

        self.md_forces.compute(self.md_system.pos, self.md_system.box, self.md_system.forces)

        traj_epot, traj_ekin, traj_temp, traj_time, traj_pos, traj_vel = [], [], [], [], [], []

        for frame in range(1, self.steps // self.save_steps + 1):
            ekin, epot, T = self.integrator.step(niter=self.save_steps)
            self.wrapper.wrap(self.md_system.pos, self.md_system.box)
            traj_pos.append(self.md_system.pos.detach().cpu().numpy())
            traj_vel.append(self.md_system.vel.detach().cpu().numpy())
            traj_ekin.append(ekin)
            traj_epot.append(epot)
            traj_temp.append(T)
            traj_time.append(np.repeat(frame * self.timestep, self.replicas))

        for i, segment in enumerate(segments):
            segment_outdir = self._get_segment_outdir(segment)
            os.makedirs(segment_outdir, exist_ok=True)

            pos_frames = [f[i] for f in traj_pos]
            vel_frames = [f[i] for f in traj_vel]

            if self.save_format == "npz":
                save_cg_npz(
                    segment_outdir,
                    epot=[f[i] for f in traj_epot],
                    ekin=[f[i] for f in traj_ekin],
                    temp=[f[i] for f in traj_temp],
                    time=[f[i] for f in traj_time],
                    pos=pos_frames,
                    vel=vel_frames,
                )
            else:
                write_dcd_from_positions(
                    os.path.join(segment_outdir, "seg.dcd"),
                    np.array(pos_frames) / 10.0,
                )

            pcoord_pos     = np.array([parent_pos[i]] + pos_frames)
            segment.pcoord = self.pcoord_calculator.calculate(pcoord_pos)

            self._run_recorded(pcoord_pos, segment_outdir, segment.n_iter, segment.seg_id)
            self._finalize_segment(segment, starttime)

        self._print_completion(len(segments), time.time() - starttime)
        return segments

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
# NOTE: save_dcd imports openmm.unit; the CG path uses npz (not dcd) and the
# wcmd-we image has no openmm. Import it lazily inside the dcd branch so the
# CG-npz path never pulls openmm. (save_format=dcd would need openmm in image.)
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
        self.friction    = self.rc.config.get_typed(["west", "cg_prop", "friction"], float, 1.0)
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

        # NOTE: this cgschnet `simulate.py` has no `build_calc`; the NN
        # calculator is built directly via `External(model, embeddings, device,
        # num_replicates, sequence=...)` (simulate.py:109), matching how
        # simulate.prepSim assembles `calcs`. Forces are in kcal/mol/Å.
        #
        # This checkpoint is sequence-conditioned (the "seq6" model:
        # representation_model.sequence_basis_radius != 0), so its forward
        # asserts the sequence tensor is present. Build it exactly as prepSim
        # does (dataset.build_sequence_for_mol = segid*20 + resid). For a pair
        # the two chains get distinct segids → distinct sequence blocks.
        sequence = None
        rep = getattr(self.model, "representation_model", None)
        if rep is not None and getattr(rep, "sequence_basis_radius", 0) != 0:
            from module import dataset as _cg_dataset
            sequence = _cg_dataset.build_sequence_for_mol(mol)
            print(f"[cgml] sequence-conditioned model: built sequence "
                  f"(len={len(sequence)})", flush=True)
        calcs = [simulate.External(self.model, embeddings, device, self.replicas,
                                   sequence=sequence)]
        system, forces = simulate.make_system(
            [mol], prior_path, calcs, device,
            prior_params["forceterms"], prior_params["exclusions"],
            self.replicas, temperature=self.temperature, new_ff=True,
        )

        self.md_system  = system
        self.md_forces  = forces
        self.integrator = Integrator(system, forces, self.timestep, device, gamma=self.friction, T=self.temperature)
        self.wrapper    = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)
        self.mol        = mol

        # Per-chain CA index groups + the docked-initial CA coords, for the
        # structure-integrity diagnostics (rmsd_from_init / rmsd_from_segstart).
        # Detects 3D collapse: WE pushes segments toward high interface-RMSD
        # (dissociation), and the single-chain CGSchNet can be unstable in those
        # high-energy states → intra-chain unfolding/collapse. Per-chain
        # superposed RMSD isolates that from (intended) inter-chain separation.
        import numpy as _np
        segid = _np.asarray(mol.segid)
        self._chain_ca_idx = [_np.where(segid == s)[0] for s in
                              sorted(set(segid.tolist()))]
        self._init_coords = _np.ascontiguousarray(
            mol.coords[:, :, 0].astype(_np.float64))   # (n_atoms, 3) Å, docked

    @staticmethod
    def _kabsch_rmsd(P, Q):
        """Superposed (rotation+translation-removed) RMSD between two (N,3)
        point sets, via the Kabsch algorithm. Returns Å (same units as input)."""
        import numpy as _np
        if P.shape[0] < 3:
            return float(_np.sqrt(_np.mean(_np.sum((P - Q) ** 2, axis=1))))
        Pc = P - P.mean(0); Qc = Q - Q.mean(0)
        H = Pc.T @ Qc
        V, S, Wt = _np.linalg.svd(H)
        d = _np.sign(_np.linalg.det(Wt.T @ V.T))
        D = _np.diag([1.0, 1.0, d])
        Pr = Pc @ (Wt.T @ D @ V.T).T
        return float(_np.sqrt(_np.mean(_np.sum((Pr - Qc) ** 2, axis=1))))

    def _structure_metrics(self, final_xyz, segstart_xyz):
        """rmsd_from_init = max over chains of per-chain superposed CA RMSD vs
        the docked pose (3D-integrity / collapse: ~native fold ⇒ small; unfold
        ⇒ large; INVARIANT to inter-chain dissociation). rmsd_from_segstart =
        whole-structure superposed RMSD vs this walker's segment start (per-
        segment motion; a divergence spike shows up here first)."""
        import numpy as _np
        rmsd_init = max(self._kabsch_rmsd(final_xyz[idx], self._init_coords[idx])
                        for idx in self._chain_ca_idx)
        rmsd_seg = self._kabsch_rmsd(final_xyz.astype(_np.float64),
                                     _np.asarray(segstart_xyz, dtype=_np.float64))
        return float(rmsd_init), float(rmsd_seg)

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

        traj_epot, traj_ekin, traj_temp, traj_time, traj_pos, traj_vel, traj_forces = \
            [], [], [], [], [], [], []

        for frame in range(1, self.steps // self.save_steps + 1):
            ekin, epot, T = self.integrator.step(niter=self.save_steps)
            self.wrapper.wrap(self.md_system.pos, self.md_system.box)
            traj_pos.append(self.md_system.pos.detach().cpu().numpy())
            traj_vel.append(self.md_system.vel.detach().cpu().numpy())
            # EXACT CG forces at the saved positions: torchmd velocity-Verlet's
            # last op each step is forces.compute(system.pos), so system.forces
            # ↔ system.pos with zero staleness / zero recomputation. Saved for
            # MS-CG force matching (V_θ supervision). (PBC wrap above is a no-op
            # since use_box=false, so it doesn't desync forces from pos.)
            traj_forces.append(self.md_system.forces.detach().cpu().numpy())
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
                    forces=[f[i] for f in traj_forces],
                )
            else:
                from file_system.md_store.save_dcd import write_dcd_from_positions
                write_dcd_from_positions(
                    os.path.join(segment_outdir, "seg.dcd"),
                    np.array(pos_frames) / 10.0,
                )

            pcoord_pos     = np.array([parent_pos[i]] + pos_frames)
            segment.pcoord = self.pcoord_calculator.calculate(pcoord_pos, None)

            # Structure-integrity diagnostics for this walker (logged to W&B by
            # run_pair_we to test the "3D collapse in high-energy states"
            # hypothesis vs pure dissociation).
            rmsd_init, rmsd_seg = self._structure_metrics(
                np.asarray(pos_frames[-1]), parent_pos[i])
            np.savez(os.path.join(segment_outdir, "struct_rmsd.npz"),
                     rmsd_from_init=np.float32(rmsd_init),
                     rmsd_from_segstart=np.float32(rmsd_seg))

            # BasePropagator._run_recorded signature is
            # (positions, energy_data, segment_outdir, n_iter, seg_id). The CG
            # path was calling it without energy_data → TypeError. Pass the
            # per-segment CG energies (epot/ekin) as energy_data; harmless when
            # no recorded_calculators are configured (it early-returns).
            energy_data = {"energy_u": [f[i] for f in traj_epot],
                           "energy_k": [f[i] for f in traj_ekin]}
            self._run_recorded(pcoord_pos, energy_data, segment_outdir,
                               segment.n_iter, segment.seg_id)
            self._finalize_segment(segment, starttime)

        self._print_completion(len(segments), time.time() - starttime)
        return segments

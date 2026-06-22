"""Interface-RMSD progress coordinate for protein-pair WE dissociation.

PHASE3_revised.md §2.2.1. A size-agnostic 1-D progress coordinate that is 0
at the docked pose and grows monotonically as the two chains separate:

  * Interface residues are identified ONCE from the reference docked pose —
    Cα atoms of chain A within `cutoff_A` of any Cα of chain B, and symmetric.
  * Per chain k: superpose each frame onto the reference using the *other*
    chain's Cα (a stable body frame), then RMSD over chain k's *interface* Cα.
    As the pair dissociates, the interface Cα drift from their docked
    positions → RMSD grows. (This is interface-restricted ligand-RMSD, done
    symmetrically for both chains.)
  * Combined to a scalar: sqrt(Σ_k RMSD_k²).

Why interface-restricted and Cα-only:
  * size-agnostic — depends on the interface separation, not chain length;
  * Cα-only so a Cα-resolution CG model (CGSchNet) can evaluate it directly;
  * the contact subset is what "binding" actually is, lower variance than
    whole-chain RMSD.

Subclasses MultiChainRMSDComputation's calculate() pattern but swaps the RMSD
atom set to the interface-Cα subset and the alignment set to the partner Cα.
"""
import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class InterfaceRMSDComputation(BaseComputation):

    def __init__(self, reference_pdb_path, reference_xml_path=None,
                 chainids=None, cutoff_A=8.0, sequence_separation_chains=True):
        # cutoff_A default 8.0: Cα-Cα ≤ 8 Å is the standard contact scale
        # (matches GAUSSIAN_CONTACT_SIGMA_A used by the TICA features) and
        # yields 40-100 interface Cα across 140-1800 aa pairs — robust,
        # low-variance, size-agnostic. 5 Å is too tight (≈10 Cα on some pairs).
        if reference_xml_path is not None:
            ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            ref = mdtraj.load(reference_pdb_path)
        self.reference_traj = ref[0]
        top = self.reference_traj.topology
        self.cutoff_nm = float(cutoff_A) / 10.0

        if chainids is None:
            chainids = [c.index for c in top.chains]
        if len(chainids) != 2:
            raise ValueError(f"InterfaceRMSD expects exactly 2 chains; got {chainids}")
        self.chainids = [int(c) for c in chainids]

        # Cα indices per chain.
        self.ca_by_chain = []
        for cid in self.chainids:
            idx = top.select(f"chainid {cid} and name CA")
            if len(idx) == 0:
                raise ValueError(f"No Cα atoms in chainid {cid}")
            self.ca_by_chain.append(np.asarray(idx, dtype=int))

        # Interface Cα: residues in chain k whose Cα is within cutoff of ANY
        # Cα in the other chain, computed on the reference docked pose.
        ref_xyz = self.reference_traj.xyz[0]   # (N,3) nm
        a_ca, b_ca = self.ca_by_chain
        # pairwise Cα-Cα distances between the two chains (nm)
        dmat = np.linalg.norm(
            ref_xyz[a_ca][:, None, :] - ref_xyz[b_ca][None, :, :], axis=-1)
        a_iface_local = np.where((dmat <= self.cutoff_nm).any(axis=1))[0]
        b_iface_local = np.where((dmat <= self.cutoff_nm).any(axis=0))[0]
        self.iface_ca = [a_ca[a_iface_local], b_ca[b_iface_local]]

        # Per-chain alignment set = the OTHER chain's full Cα (stable frame).
        self.align_ca = [self.ca_by_chain[1], self.ca_by_chain[0]]

        # Sanity: each chain must have >=3 interface Cα for a meaningful RMSD;
        # the caller (run_pair_we) falls back to COM-COM if this raises.
        for k, ic in enumerate(self.iface_ca):
            if len(ic) < 3:
                raise ValueError(
                    f"chain {self.chainids[k]} has only {len(ic)} interface Cα "
                    f"(<3) at cutoff {cutoff_A} Å — weak/glancing dock; "
                    f"use COM-COM progress coord for this pair instead.")
        self.n_chains = 2

    # ---- introspection (used by the inspection script + preflight) ----
    def interface_summary(self):
        top = self.reference_traj.topology
        out = []
        for k, cid in enumerate(self.chainids):
            n_res = len(self.ca_by_chain[k])
            n_iface = len(self.iface_ca[k])
            atom_names = sorted({top.atom(int(i)).name for i in self.iface_ca[k]})
            res = [(top.atom(int(i)).residue.name, top.atom(int(i)).residue.resSeq)
                   for i in self.iface_ca[k]]
            out.append(dict(chainid=cid, n_residues=n_res, n_interface=n_iface,
                            atom_names=atom_names, interface_residues=res))
        return out

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        traj = mdtraj.Trajectory(data / 10.0, self.reference_traj.topology)
        xyz0 = traj.xyz.copy()
        out_nm = np.zeros((traj.n_frames, self.n_chains), dtype=np.float32)
        for k in range(self.n_chains):
            traj.xyz = xyz0.copy()
            traj.superpose(self.reference_traj, atom_indices=self.align_ca[k])
            idx = self.iface_ca[k]
            X = traj.xyz[:, idx, :]
            Y = self.reference_traj.xyz[0, idx, :]
            diff = X - Y[None]
            out_nm[:, k] = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))
        result = np.sqrt(np.sum(out_nm ** 2, axis=1)) * 10.0   # Å, shape (n_frames,)
        if not np.all(np.isfinite(result)):
            bad = np.where(~np.isfinite(result))[0][:10]
            raise ValueError(f"Non-finite interface-RMSD at frames {bad}")
        # WESTPA expects pcoord shape (n_frames, pcoord_ndim); ndim=1 here, so
        # return (n_frames, 1) — the bare (n_frames,) triggers a broadcast error
        # ("(3,) into (3,1)") when the propagator assigns segment.pcoord.
        return result.reshape(-1, 1).astype(np.float32)

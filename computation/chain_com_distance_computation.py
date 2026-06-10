"""Inter-chain COM–COM distance progress coordinate for protein-pair
association / dissociation WE.

PHASE3_revised.md §2.2.1c. This mirrors the DISTANCE progress coordinate from
the prior working assembly runs (westpa_old_repos/twodimers_AAi/common_files/
dist.py: mean interface Cα–Cα distance), using the simplest robust analog: the
Euclidean distance between the two chains' Cα centres of mass.

  * monotonic + unbounded with separation (small when bound, large when apart),
    unlike interface-RMSD which saturates;
  * a clean target state for recycling (e.g. "bound" = distance ≤ d_bound + tol);
  * Cα-only so a Cα-resolution model could evaluate it, and so the atomistic
    propagator can pass the solute-Cα array directly;
  * NOT referenced to the docked pose, so it does not penalise non-native
    encounters the way interface-RMSD-from-docked does — any close approach
    counts as progress (what association WE needs).

The reference PDB is used ONLY to learn the per-chain Cα split (chain A then
chain B, in the order the propagator extracts solute Cα) and to report the
docked-pose `bound_distance` used to place the target state + bins.
"""
import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class ChainCOMDistanceComputation(BaseComputation):

    def __init__(self, reference_pdb_path, chainids=None):
        ref = mdtraj.load(reference_pdb_path)
        self.reference_traj = ref[0]
        top = self.reference_traj.topology

        if chainids is None:
            chainids = [c.index for c in top.chains]
        if len(chainids) != 2:
            raise ValueError(f"ChainCOMDistance expects exactly 2 chains; got {chainids}")
        self.chainids = [int(c) for c in chainids]

        # Cα indices per chain, as indices INTO the Cα array the propagator
        # passes (the reference here is itself Cα-only, so select gives those
        # indices directly; for a full-atom ref it would still be Cα indices
        # but the propagator passes Cα-only, so we require a Cα-only reference).
        self.ca_by_chain = []
        for cid in self.chainids:
            idx = top.select(f"chainid {cid} and name CA")
            if len(idx) == 0:
                raise ValueError(f"No Cα atoms in chainid {cid}")
            self.ca_by_chain.append(np.asarray(idx, dtype=int))

        n_ca = top.n_atoms
        if sum(len(c) for c in self.ca_by_chain) != n_ca:
            raise ValueError(
                "ChainCOMDistance requires a Cα-ONLY reference (every atom a Cα); "
                f"got {n_ca} atoms but {sum(len(c) for c in self.ca_by_chain)} Cα.")

        # docked-pose COM–COM distance (Å), for target/bin placement.
        self.bound_distance = float(self._com_distance(self.reference_traj.xyz[0] * 10.0))

    def _com_distance(self, ca_xyz_A):
        """ca_xyz_A: (n_ca, 3) in Å, chain-A Cα then chain-B Cα."""
        a, b = self.ca_by_chain
        com_a = ca_xyz_A[a].mean(axis=0)
        com_b = ca_xyz_A[b].mean(axis=0)
        return float(np.linalg.norm(com_a - com_b))

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        # data: (n_frames, n_ca, 3) in Å (solute Cα, chain A then chain B).
        self._validate_input(data)
        a, b = self.ca_by_chain
        com_a = data[:, a, :].mean(axis=1)
        com_b = data[:, b, :].mean(axis=1)
        d = np.linalg.norm(com_a - com_b, axis=1).astype(np.float32)
        if not np.all(np.isfinite(d)):
            bad = np.where(~np.isfinite(d))[0][:10]
            raise ValueError(f"Non-finite COM distance at frames {bad}")
        # WESTPA expects (n_frames, pcoord_ndim); ndim = 1.
        return d.reshape(-1, 1)

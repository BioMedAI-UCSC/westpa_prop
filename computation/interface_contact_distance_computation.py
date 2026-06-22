"""Interface-contact mean Cα–Cα distance progress coordinate (NO superposition).

PHASE3_revised.md §2.2.1c. Faithful port of the prior working assembly runs'
coordinate (westpa_old_repos/twodimers_AAi/common_files/dist.py):

    pcoord(frame) = mean over the native interface contact pairs (i,j) of
                    || Cα_i(chain A, frame) - Cα_j(chain B, frame) ||

  * The contact pairs (i,j) are identified ONCE, a priori, from the DOCKED
    reference pose: every chain-A Cα within `cutoff_A` of a chain-B Cα.
  * It is a DISTANCE between specific atoms → intrinsically invariant to global
    rotation/translation, so it needs NO superposition / alignment (this is the
    whole point: the user does not want any superpose step, whose effect is
    hard to reason about). Contrast InterfaceRMSDComputation, which superposes.
  * Small (~docked contact distance) when the native interface is formed, large
    when the chains are apart → monotonic association/dissociation coordinate;
    native-specific (only the native contacts count), so reaching the bound
    target means re-forming the NATIVE interface, not just any close approach.

The reference must be a Cα-only PDB with exactly 2 chains, atom order matching
what the propagator passes (chain A Cα then chain B Cα).
"""
import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class InterfaceContactDistanceComputation(BaseComputation):

    def __init__(self, reference_pdb_path, chainids=None, cutoff_A=8.0):
        ref = mdtraj.load(reference_pdb_path)
        self.reference_traj = ref[0]
        top = self.reference_traj.topology
        self.cutoff_A = float(cutoff_A)

        if chainids is None:
            chainids = [c.index for c in top.chains]
        if len(chainids) != 2:
            raise ValueError(f"InterfaceContactDistance expects exactly 2 chains; got {chainids}")
        self.chainids = [int(c) for c in chainids]

        ca_by_chain = []
        for cid in self.chainids:
            idx = top.select(f"chainid {cid} and name CA")
            if len(idx) == 0:
                raise ValueError(f"No Cα atoms in chainid {cid}")
            ca_by_chain.append(np.asarray(idx, dtype=int))
        a_ca, b_ca = ca_by_chain

        n_ca = top.n_atoms
        if len(a_ca) + len(b_ca) != n_ca:
            raise ValueError(
                "InterfaceContactDistance requires a Cα-ONLY reference (every atom "
                f"a Cα); got {n_ca} atoms but {len(a_ca) + len(b_ca)} Cα.")

        # Native interface contact pairs, from the DOCKED reference: all (i,j)
        # with i in chain A, j in chain B and Cα–Cα distance <= cutoff. Stored
        # as parallel index arrays into the full Cα array (one entry per pair).
        ref_xyz = self.reference_traj.xyz[0] * 10.0   # Å
        dmat = np.linalg.norm(
            ref_xyz[a_ca][:, None, :] - ref_xyz[b_ca][None, :, :], axis=-1)
        ii, jj = np.where(dmat <= self.cutoff_A)
        if len(ii) < 1:
            raise ValueError(
                f"no interface Cα–Cα contacts within {cutoff_A} Å in the docked "
                f"reference — weak/glancing dock; widen cutoff or use COM distance.")
        self.pair_a = a_ca[ii]            # chain-A Cα index per contact pair
        self.pair_b = b_ca[jj]            # chain-B Cα index per contact pair
        self.n_pairs = int(len(self.pair_a))
        self.bound_distance = float(dmat[ii, jj].mean())   # docked mean contact distance

    def n_contacts(self):
        return self.n_pairs

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        # data: (n_frames, n_ca, 3) in Å (solute Cα, chain A then chain B).
        # NO superposition: distances between specific atoms are already
        # invariant to global rotation/translation.
        self._validate_input(data)
        diff = data[:, self.pair_a, :] - data[:, self.pair_b, :]   # (n_frames, n_pairs, 3)
        dists = np.linalg.norm(diff, axis=2)                        # (n_frames, n_pairs)
        mean = dists.mean(axis=1).astype(np.float32)                # (n_frames,)
        if not np.all(np.isfinite(mean)):
            bad = np.where(~np.isfinite(mean))[0][:10]
            raise ValueError(f"Non-finite contact distance at frames {bad}")
        return mean.reshape(-1, 1)

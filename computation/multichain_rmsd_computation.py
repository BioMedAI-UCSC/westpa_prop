import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class MultiChainRMSDComputation(BaseComputation):
    """
    Per-chain RMSD where each chain is aligned using all other chains.

    For each chain k:
      1) Superpose onto reference using all atoms NOT in chain k.
      2) Compute RMSD over chain k atoms only.

    Returns (n_frames, n_chains) in Angstrom, then collapsed to
    sqrt(sum of squared per-chain RMSDs).
    """

    def __init__(self, reference_pdb_path, reference_xml_path=None, chain_selections=None, chainids=None, heavy_atoms_only=True):
        if reference_xml_path is not None:
            ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            ref = mdtraj.load(reference_pdb_path)

        self.reference_traj  = ref[0]
        self.heavy_atoms_only = bool(heavy_atoms_only)
        top = self.reference_traj.topology

        if chain_selections is None:
            chainids = chainids if chainids is not None else [c.index for c in top.chains]
            chain_selections = [f"chainid {int(cid)}" for cid in chainids]

        if len(chain_selections) < 2:
            raise ValueError("At least two chains required.")

        self.chain_atom_indices = []
        for sel in chain_selections:
            s = f"({sel})" + (" and not element H" if self.heavy_atoms_only else "")
            idx = top.select(s)
            if len(idx) == 0:
                raise ValueError(f"No atoms for selection: {s}")
            self.chain_atom_indices.append(np.array(idx, dtype=int))

        all_atoms = np.arange(self.reference_traj.n_atoms, dtype=int)
        self.align_atom_indices = []
        for chain_idx in self.chain_atom_indices:
            mask = np.ones(self.reference_traj.n_atoms, dtype=bool)
            mask[chain_idx] = False
            align_idx = all_atoms[mask]
            if align_idx.size == 0:
                raise ValueError("Alignment set is empty for a chain.")
            self.align_atom_indices.append(align_idx)

        self.n_chains = len(self.chain_atom_indices)

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        traj  = mdtraj.Trajectory(data / 10.0, self.reference_traj.topology)
        xyz0  = traj.xyz.copy()
        out_nm = np.zeros((traj.n_frames, self.n_chains), dtype=np.float32)

        for k in range(self.n_chains):
            traj.xyz = xyz0.copy()
            traj.superpose(self.reference_traj, atom_indices=self.align_atom_indices[k])
            idx  = self.chain_atom_indices[k]
            X    = traj.xyz[:, idx, :]
            Y    = self.reference_traj.xyz[0, idx, :]
            diff = X - Y[None]
            out_nm[:, k] = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))

        result = np.sqrt(np.sum(out_nm ** 2, axis=1)) * 10.0
        if not np.all(np.isfinite(result)):
            bad = np.where(~np.isfinite(result))[0][:10]
            raise ValueError(f"Non-finite output at frames {bad}")
        return result

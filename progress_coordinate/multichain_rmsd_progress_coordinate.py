import numpy as np
import mdtraj

from progress_coordinate.base_progress_coordinate import BaseProgressCoordinate


class MultiChainRMSDProgressCoordinate(BaseProgressCoordinate):
    """
    Per-chain RMSD progress coordinate (generalizes receptor/ligand to N chains).

    For each chain k:
      1) superpose each frame onto the reference using ALL OTHER chains (not k)
      2) compute RMSD over chain k atoms only (no further alignment)

    Returns array with shape (n_frames, n_chains) in Angstrom.

    Notes:
      - `data` is assumed Angstrom; mdtraj uses nm internally.
      - Trajectory atom ordering must match the reference topology.
    """

    def __init__(
        self,
        reference_pdb_path: str,
        reference_xml_path: str = None,
        chain_selections=None,          # e.g. ["chainid 0", "chainid 1", "chainid 2"]
        chainids=None,                  # e.g. [0, 1, 2] 
        heavy_atoms_only: bool = True
    ):
        super().__init__()

        if reference_pdb_path is None:
            raise ValueError("MultiChainRMSDProgressCoordinate requires reference_pdb_path.")

        self.reference_pdb_path = reference_pdb_path
        self.reference_xml_path = reference_xml_path
        self.heavy_atoms_only = bool(heavy_atoms_only)

        # Load reference
        if reference_xml_path is not None:
            full_ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            full_ref = mdtraj.load(reference_pdb_path)

        if full_ref.n_frames < 1:
            raise ValueError("Reference trajectory has no frames.")

        self.full_reference_traj = full_ref
        self.reference_traj = full_ref[0]
        top = self.reference_traj.topology

        # Build chain selection strings
        if chain_selections is None:
            if chainids is None:
                # default: all chains in topology
                chainids = [c.index for c in top.chains]
            chain_selections = [f"chainid {int(cid)}" for cid in chainids]

        if not isinstance(chain_selections, (list, tuple)) or len(chain_selections) < 2:
            raise ValueError("Provide at least two chains via chain_selections or chainids.")

        self.chain_selections = list(chain_selections)

        # Atom indices per chain (optionally heavy atoms only)
        self.chain_atom_indices = []
        for sel in self.chain_selections:
            s = f"({sel})"
            if self.heavy_atoms_only:
                s = f"{s} and not element H"
            idx = top.select(s)
            if len(idx) == 0:
                raise ValueError(f"No atoms found for chain selection: {s}")
            self.chain_atom_indices.append(np.array(idx, dtype=int))

        # Precompute alignment indices for each chain: all atoms NOT in that chain
        all_atoms = np.arange(self.reference_traj.n_atoms, dtype=int)
        self.align_atom_indices = []
        for chain_idx in self.chain_atom_indices:
            mask = np.ones(self.reference_traj.n_atoms, dtype=bool)
            mask[chain_idx] = False
            align_idx = all_atoms[mask]
            if align_idx.size == 0:
                raise ValueError(
                    "Alignment set is empty for a chain. "
                    "Need at least one other chain/atom to align on."
                )
            self.align_atom_indices.append(align_idx)

        self.n_chains = len(self.chain_atom_indices)

    def calculate(self, data):
        self._validate_data_shape(data, expected_ndim=3)

        # A -> nm
        data_nm = data / 10.0
        traj = mdtraj.Trajectory(data_nm, self.reference_traj.topology)

        # Save original coords (we'll re-superpose for each chain independently)
        xyz0 = traj.xyz.copy()

        out_nm = np.zeros((traj.n_frames, self.n_chains), dtype=np.float32)

        for k in range(self.n_chains):
            traj.xyz = xyz0.copy()

            # Align on all other chains
            traj.superpose(self.reference_traj, atom_indices=self.align_atom_indices[k])

            # RMSD over chain k atoms, no further alignment
            idx = self.chain_atom_indices[k]
            X = traj.xyz[:, idx, :]                  # (n_frames, n_atoms_k, 3)
            Y = self.reference_traj.xyz[0, idx, :]   # (n_atoms_k, 3)
            diff = X - Y[None, :, :]
            out_nm[:, k] = np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))

        final = np.sqrt(np.sum(out_nm ** 2, axis=1)) * 10.0
        if not np.all(np.isfinite(final)):
            bad = np.where(~np.isfinite(final))[0][:10]
            raise ValueError(f"Non-finite pcoord at frames {bad}: {final[bad]}")
        return final 


import numpy as np
import mdtraj

from progress_coordinate.base_progress_coordinate import BaseProgressCoordinate


class RefContactProgressCoordinate(BaseProgressCoordinate):
    """
    Reference-contact mean-distance progress coordinate for chains A–E.

    Chains (hard-coded):
      A -> chainid 0
      B -> chainid 1
      C -> chainid 2
      D -> chainid 3
      E -> chainid 4

    Pairs (hard-coded):
      (A,B), (B,C), (C,D), (D,E), (E,A)

    Algorithm:
      1) On the reference frame, find all atom pairs (i,j) within cutoff Å
         for each chain pair. These indices are frozen.
      2) For each frame, compute distances for those same (i,j) pairs
         and take the mean per pair.
      3) Sum the per-pair means and return sqrt(sum).

    Returns:
      pcoord shape (n_frames,), Angstrom units (after sqrt).
    """

    def __init__(
        self,
        reference_pdb_path: str,
        reference_xml_path: str = None,
        cutoff_angstrom: float = 3.0,
    ):
        super().__init__()

        self.cutoff_angstrom = float(cutoff_angstrom)

        # --- load reference ---
        if reference_xml_path is not None:
            full_ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            full_ref = mdtraj.load(reference_pdb_path)

        if full_ref.n_frames < 1:
            raise ValueError("Reference trajectory has no frames.")

        self.reference_traj = full_ref[0]
        top = self.reference_traj.topology

        # --- hard-coded chains A–E ---
        self.chain_labels = ["A", "B", "C", "D", "E"]
        self.chainids = [0, 1, 2, 3, 4]

        # atom indices per chain (ALL atoms)
        self.chain_atom_indices = []
        for cid in self.chainids:
            idx = top.select(f"chainid {cid}")
            if idx.size == 0:
                raise ValueError(f"No atoms found for chainid {cid}")
            self.chain_atom_indices.append(idx.astype(int))

        # --- hard-coded pairs ---
        # (A,B), (B,C), (C,D), (D,E), (E,A)
        self.pairs = [(0,1), (1,2), (2,3), (3,4), (4,0)]

        # --- precompute reference contact indices ---
        self.ref_contact_pairs = {}
        cutoff_nm = self.cutoff_angstrom / 10.0

        ref_xyz = self.reference_traj.xyz[0]  # nm

        for a, b in self.pairs:
            a_atoms = self.chain_atom_indices[a]
            b_atoms = self.chain_atom_indices[b]

            A = ref_xyz[a_atoms]  # (Na,3)
            B = ref_xyz[b_atoms]  # (Nb,3)

            # distance matrix (Na,Nb)
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
            ia, ib = np.where(d < cutoff_nm)

            if ia.size == 0:
                raise ValueError(
                    f"No reference contacts for pair {self.chain_labels[a]}{self.chain_labels[b]} "
                    f"at cutoff {self.cutoff_angstrom} Å"
                )

            self.ref_contact_pairs[(a, b)] = (
                a_atoms[ia].astype(int),
                b_atoms[ib].astype(int),
            )

    def calculate(self, data):
        """
        data: (n_frames, n_atoms, 3) in Angstrom
        returns: (n_frames,) float32
        """
        self._validate_data_shape(data, expected_ndim=3)

        # Angstrom -> nm
        traj = mdtraj.Trajectory(data / 10.0, self.reference_traj.topology)

        out = np.zeros(traj.n_frames, dtype=np.float32)

        for f in range(traj.n_frames):
            means_sum = 0.0
            xyz = traj.xyz[f]

            for a, b in self.pairs:
                ia, ib = self.ref_contact_pairs[(a, b)]
                A = xyz[ia]
                B = xyz[ib]
                dist_nm = np.sqrt(((A - B) ** 2).sum(axis=1))
                mean_angstrom = dist_nm.mean() * 10.0
                means_sum += mean_angstrom

            out[f] = np.sqrt(means_sum)

        if not np.all(np.isfinite(out)):
            bad = np.where(~np.isfinite(out))[0][:10]
            raise ValueError(f"Non-finite pcoord at frames {bad}: {out[bad]}")

        return out


import numpy as np
import mdtraj

from progress_coordinate.base_progress_coordinate import BaseProgressCoordinate


class RefContactProgressCoordinate(BaseProgressCoordinate):
    """
    Reference-contact mean-distance progress coordinate for specified selections.

    Algorithm:
      1) Define selection pairs to monitor
      2) On the reference frame, find all atom pairs (i,j) within cutoff Å
         for each selection pair. These indices are frozen.
      3) For each frame, compute distances for those same (i,j) pairs
         and take the mean per pair.
      4) Sum the per-pair means and return sqrt(sum).

    Returns:
      pcoord shape (n_frames,), Angstrom units (after sqrt).
    """

    def __init__(
        self,
        reference_pdb_path: str,
        selection_pairs: list,
        reference_xml_path: str = None,
        cutoff_angstrom: float = 3.0,
    ):
        """
        Args:
            reference_pdb_path: Path to reference PDB
            selection_pairs: List of tuples of MDTraj selection strings, e.g.,
                [("chainid 0", "chainid 1"), ("chainid 1", "chainid 2")]
                or [("resid 10 to 20", "resid 50 to 60")]
            reference_xml_path: Optional path to reference XML topology
            cutoff_angstrom: Distance cutoff for contacts
        """
        super().__init__()

        self.cutoff_angstrom = float(cutoff_angstrom)
        self.selection_pairs = selection_pairs

        # --- load reference ---
        if reference_xml_path is not None:
            full_ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            full_ref = mdtraj.load(reference_pdb_path)

        if full_ref.n_frames < 1:
            raise ValueError("Reference trajectory has no frames.")

        self.reference_traj = full_ref[0]
        top = self.reference_traj.topology

        # --- precompute reference contact indices ---
        self.ref_contact_pairs = {}
        cutoff_nm = self.cutoff_angstrom / 10.0

        ref_xyz = self.reference_traj.xyz[0]  # nm

        for pair_idx, (sel_a, sel_b) in enumerate(self.selection_pairs):
            # Get atom indices for each selection
            a_atoms = top.select(sel_a)
            b_atoms = top.select(sel_b)

            if a_atoms.size == 0:
                raise ValueError(f"No atoms found for selection '{sel_a}'")
            if b_atoms.size == 0:
                raise ValueError(f"No atoms found for selection '{sel_b}'")

            A = ref_xyz[a_atoms]  # (Na,3)
            B = ref_xyz[b_atoms]  # (Nb,3)

            # distance matrix (Na,Nb)
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
            ia, ib = np.where(d < cutoff_nm)

            if ia.size == 0:
                raise ValueError(
                    f"No reference contacts for pair ({sel_a}, {sel_b}) "
                    f"at cutoff {self.cutoff_angstrom} Å"
                )

            self.ref_contact_pairs[pair_idx] = (
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

            for pair_idx in self.ref_contact_pairs:
                ia, ib = self.ref_contact_pairs[pair_idx]
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

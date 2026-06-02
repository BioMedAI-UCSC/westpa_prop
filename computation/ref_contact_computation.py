import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class RefContactComputation(BaseComputation):
    """
    Mean distance over reference-frame contact pairs for each selection pair.

    For each (sel_a, sel_b) pair, finds all atom pairs within cutoff_angstrom
    in the reference structure, then tracks those same pairs in each frame.
    Returns sqrt of summed per-pair mean distances, shape (n_frames,).
    """

    def __init__(self, reference_pdb_path, selection_pairs, reference_xml_path=None, cutoff_angstrom=3.0):
        if reference_xml_path is not None:
            ref = mdtraj.load(reference_xml_path, top=reference_pdb_path)
        else:
            ref = mdtraj.load(reference_pdb_path)

        self.reference_traj  = ref[0]
        self.cutoff_angstrom = float(cutoff_angstrom)
        top      = self.reference_traj.topology
        ref_xyz  = self.reference_traj.xyz[0]  # nm
        cutoff_nm = self.cutoff_angstrom / 10.0

        self.contact_pairs = {}
        for idx, (sel_a, sel_b) in enumerate(selection_pairs):
            a_atoms = top.select(sel_a)
            b_atoms = top.select(sel_b)

            if a_atoms.size == 0:
                raise ValueError(f"No atoms for selection '{sel_a}'")
            if b_atoms.size == 0:
                raise ValueError(f"No atoms for selection '{sel_b}'")

            if self.cutoff_angstrom <= 0:
                ia, ib = np.meshgrid(np.arange(len(a_atoms)), np.arange(len(b_atoms)), indexing="ij")
                ia, ib = ia.ravel(), ib.ravel()
            else:
                A, B = ref_xyz[a_atoms], ref_xyz[b_atoms]
                d = np.sqrt(((A[:, None] - B[None]) ** 2).sum(axis=2))
                ia, ib = np.where(d < cutoff_nm)
                if ia.size == 0:
                    raise ValueError(
                        f"No contacts for pair ({sel_a}, {sel_b}) at {self.cutoff_angstrom} Å"
                    )

            self.contact_pairs[idx] = (a_atoms[ia].astype(int), b_atoms[ib].astype(int))

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        traj = mdtraj.Trajectory(data / 10.0, self.reference_traj.topology)
        out  = np.zeros(traj.n_frames, dtype=np.float32)

        for f in range(traj.n_frames):
            xyz = traj.xyz[f]
            means_sum = 0.0
            for ia, ib in self.contact_pairs.values():
                dist_nm = np.sqrt(((xyz[ia] - xyz[ib]) ** 2).sum(axis=1))
                means_sum += dist_nm.mean() * 10.0
            out[f] = np.sqrt(means_sum)

        if not np.all(np.isfinite(out)):
            bad = np.where(~np.isfinite(out))[0][:10]
            raise ValueError(f"Non-finite output at frames {bad}")

        return out

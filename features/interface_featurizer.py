import numpy as np

from computation.base_computation import BaseComputation
from computation.residue_utils import load_topology, residue_reps


class InterfaceFeaturizer(BaseComputation):
    """Reference-free interface descriptors.

    mode:
      "contact_map" -> (n_frames, na, nb) soft inter-chain residue contact map
                       (image input for the CVAE).
      "vector"      -> (n_frames, d) low-dim descriptor (com dist, k smallest
                       rep distances, total soft contacts, Rg terms); for PCA/
                       TICA/VAMPnet.
      "rigid"       -> (n_frames, 6) B center + orientation in A's body frame.
    """

    requires_positions = True
    requires_energy = False

    def __init__(self, topology_path, selection_a, selection_b,
                 mode="contact_map", scheme="cb", contact_cutoff_angstrom=8.0,
                 soft_n=6.0, k=10, exclude_hydrogens=True):
        if mode not in ("contact_map", "vector", "rigid"):
            raise ValueError(f"bad mode {mode!r}")
        self.mode = mode
        self.d0 = float(contact_cutoff_angstrom)
        self.soft_n = float(soft_n)
        self.k = int(k)
        top = load_topology(topology_path)
        self.res_a = residue_reps(top, selection_a, scheme, exclude_hydrogens)
        self.res_b = residue_reps(top, selection_b, scheme, exclude_hydrogens)
        self.rep_a = np.array([r["rep"] for r in self.res_a])
        self.rep_b = np.array([r["rep"] for r in self.res_b])
        self.heavy_a = np.concatenate([r["heavy"] for r in self.res_a])
        self.heavy_b = np.concatenate([r["heavy"] for r in self.res_b])

    @property
    def shape(self):
        na, nb = len(self.res_a), len(self.res_b)
        return {"contact_map": (na, nb), "vector": (self.k + 4,), "rigid": (6,)}[self.mode]

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        fn = {"contact_map": self._contact_map, "vector": self._vector,
              "rigid": self._rigid}[self.mode]
        return np.stack([fn(data[f]) for f in range(data.shape[0])]).astype(np.float32)

    def _rep_dists(self, xyz):
        A, B = xyz[self.rep_a], xyz[self.rep_b]
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))

    def _contact_map(self, xyz):
        d = self._rep_dists(xyz)
        return 1.0 / (1.0 + (d / self.d0) ** self.soft_n)

    def _vector(self, xyz):
        d = self._rep_dists(xyz).ravel()
        kmin = np.sort(np.partition(d, min(self.k, d.size) - 1)[:self.k])
        soft = float((1.0 / (1.0 + (d / self.d0) ** self.soft_n)).sum())
        ca, cb = xyz[self.heavy_a], xyz[self.heavy_b]
        com = float(np.linalg.norm(ca.mean(0) - cb.mean(0)))
        rg = lambda p: float(np.sqrt(((p - p.mean(0)) ** 2).sum(1).mean()))
        return np.concatenate([[com], kmin, [soft, rg(ca), rg(cb)]])

    def _rigid(self, xyz):
        A, B = xyz[self.heavy_a], xyz[self.heavy_b]
        ca = A.mean(0)
        R = self._principal_axes(A - ca)
        b_in_a = (B.mean(0) - ca) @ R
        Rb = self._principal_axes(B - B.mean(0))
        euler = self._matrix_to_euler(R.T @ Rb)
        return np.concatenate([b_in_a, euler])

    @staticmethod
    def _principal_axes(centered):
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        R = vt.T
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        return R

    @staticmethod
    def _matrix_to_euler(R):
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy > 1e-6:
            return np.array([np.arctan2(R[2, 1], R[2, 2]),
                             np.arctan2(-R[2, 0], sy),
                             np.arctan2(R[1, 0], R[0, 0])])
        return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])

import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


class InterfaceContactComputation(BaseComputation):
    """
    Reference-free protein-protein binding progress coordinate.

    Computes either:
      1. mean_k_min_distance:
           mean of the k smallest inter-chain heavy-atom distances, in Angstrom

      2. soft_contact_count:
           smooth count of inter-chain contacts

      3. neg_soft_contact_count:
           negative smooth contact count, useful with WESTPA direction [-1]

      4. com_distance:
           center-of-mass / center-of-geometry distance between selections, in Angstrom

      5. distance_and_contacts:
           two-dimensional pcoord:
             [mean_k_min_distance, -soft_contact_count]

    Input data convention:
        data shape = (n_frames, n_atoms, 3), Angstrom

    This does NOT require a known bound reference structure.
    It only needs a topology PDB so mdtraj can understand atom selections.
    """

    requires_positions = True
    requires_energy = False

    # Modes that aggregate over every pair of an N-chain `selections` list.
    # These are the "true N-body assembly" pcoords: every chain is randomized
    # independently and the pcoord summarises ALL C(N,2) inter-chain interfaces
    # at once. Existing 2-body modes are unchanged.
    ALL_PAIR_MODES = {
        "all_pair_worst_distance",                  # d0 = max pairwise mean-min-k dist
        "all_pair_mean_distance",                   # d0 = mean pairwise mean-min-k dist
        "all_pair_worst_distance_and_contacts",     # [d0=max dist, d1=-sum contacts]
        "all_pair_mean_distance_and_contacts",      # [d0=mean dist, d1=-sum contacts]
    }

    def __init__(
        self,
        topology_path: str,
        selection_a: str = None,
        selection_b: str = None,
        mode: str = "distance_and_contacts",
        k: int = 10,
        contact_cutoff_angstrom: float = 6.0,
        soft_n: float = 6.0,
        exclude_hydrogens: bool = True,
        selections: list = None,
    ):
        self.topology_path = topology_path
        self.selection_a = selection_a
        self.selection_b = selection_b
        self.mode = mode
        self.selections = selections
        self.k = int(k)
        self.contact_cutoff_angstrom = float(contact_cutoff_angstrom)
        self.soft_n = float(soft_n)
        self.exclude_hydrogens = bool(exclude_hydrogens)

        legacy_modes = {
            "mean_k_min_distance",
            "soft_contact_count",
            "neg_soft_contact_count",
            "com_distance",
            "distance_and_contacts",
        }
        valid_modes = legacy_modes | self.ALL_PAIR_MODES
        if self.mode not in valid_modes:
            raise ValueError(f"Unknown mode {self.mode!r}. Valid modes: {valid_modes}")

        if self.k <= 0:
            raise ValueError("k must be positive")

        if self.contact_cutoff_angstrom <= 0:
            raise ValueError("contact_cutoff_angstrom must be positive")

        if self.soft_n <= 0:
            raise ValueError("soft_n must be positive")

        ref = mdtraj.load(topology_path)
        self.topology = ref.topology

        if self.mode in self.ALL_PAIR_MODES:
            if not selections or len(selections) < 2:
                raise ValueError(
                    f"mode={self.mode!r} requires `selections`: a list of >=2 "
                    f"mdtraj selection strings, one per chain/group."
                )
            self._setup_all_pair(selections)
        else:
            if selection_a is None or selection_b is None:
                raise ValueError(
                    f"mode={self.mode!r} requires both selection_a and selection_b."
                )
            self._setup_two_body(selection_a, selection_b)

    def _setup_two_body(self, selection_a, selection_b):
        sel_a = selection_a
        sel_b = selection_b
        if self.exclude_hydrogens:
            sel_a = f"({selection_a}) and not element H"
            sel_b = f"({selection_b}) and not element H"

        self.atoms_a = self.topology.select(sel_a).astype(int)
        self.atoms_b = self.topology.select(sel_b).astype(int)

        if self.atoms_a.size == 0:
            raise ValueError(f"Selection A matched zero atoms: {sel_a!r}")
        if self.atoms_b.size == 0:
            raise ValueError(f"Selection B matched zero atoms: {sel_b!r}")

        ia, ib = np.meshgrid(self.atoms_a, self.atoms_b, indexing="ij")
        self.pair_i = ia.ravel().astype(int)
        self.pair_j = ib.ravel().astype(int)

        if self.pair_i.size == 0:
            raise ValueError("No inter-selection atom pairs were generated")

    def _setup_all_pair(self, selections):
        """Resolve each per-chain selection to atom indices and precompute
        the meshgrid of inter-chain atom-pair indices for every (i<j) pair.
        Stored as a list of (subset_i, subset_j, pair_i_idx, pair_j_idx).
        """
        self.subset_atoms = []
        for s in selections:
            sel = f"({s}) and not element H" if self.exclude_hydrogens else s
            atoms = self.topology.select(sel).astype(int)
            if atoms.size == 0:
                raise ValueError(f"all-pair selection matched zero atoms: {sel!r}")
            self.subset_atoms.append(atoms)

        self.subset_pairs = []
        n = len(self.subset_atoms)
        for i in range(n):
            for j in range(i + 1, n):
                ia, ib = np.meshgrid(
                    self.subset_atoms[i], self.subset_atoms[j], indexing="ij"
                )
                self.subset_pairs.append(
                    (i, j, ia.ravel().astype(int), ib.ravel().astype(int))
                )
        if not self.subset_pairs:
            raise ValueError("all-pair mode produced zero chain pairs")

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)

        n_frames = data.shape[0]
        is_2d_mode = (
            self.mode == "distance_and_contacts"
            or self.mode.endswith("distance_and_contacts")  # covers all_pair_*_distance_and_contacts
        )

        if is_2d_mode:
            out = np.zeros((n_frames, 2), dtype=np.float32)
        else:
            # Important: 1D pcoords should be shape (n_frames,), not (n_frames, 1)
            out = np.zeros(n_frames, dtype=np.float32)

        for f in range(n_frames):
            xyz = data[f]  # Angstrom

            if self.mode in self.ALL_PAIR_MODES:
                out[f] = self._calc_all_pair(xyz)
                continue

            A = xyz[self.atoms_a]
            B = xyz[self.atoms_b]

            pair_vecs = xyz[self.pair_i] - xyz[self.pair_j]
            dists = np.sqrt(np.sum(pair_vecs * pair_vecs, axis=1))  # Angstrom

            mean_k_dist = self._mean_k_smallest(dists)
            soft_contacts = self._soft_contact_count(dists) / len(dists)
            com_dist = self._center_distance(A, B)

            if self.mode == "mean_k_min_distance":
                out[f] = mean_k_dist

            elif self.mode == "soft_contact_count":
                out[f] = soft_contacts

            elif self.mode == "neg_soft_contact_count":
                out[f] = -soft_contacts

            elif self.mode == "com_distance":
                out[f] = com_dist

            elif self.mode == "distance_and_contacts":
                out[f, 0] = mean_k_dist
                out[f, 1] = -soft_contacts

        if not np.all(np.isfinite(out)):
            bad = np.argwhere(~np.isfinite(out))[:10]
            raise ValueError(f"Non-finite output at indices {bad}")

        # Critical WESTPA basis-state fix:
        # If this is a single basis-state frame and pcoord_ndim = 2,
        # return shape (2,), not shape (1, 2) or (2, 1).
        if n_frames == 1 and is_2d_mode:
            return out[0]

        return out

    def _calc_all_pair(self, xyz: np.ndarray):
        """Aggregate per-chain-pair (mean_k_min_distance, soft_contact_count)
        into a 1D or 2D pcoord per the active all_pair_* mode.

        NB: the all-pair modes use the RAW soft_contact_count (no /len(dists)
        normalisation) so the per-pair contribution is a real count, not a
        ratio. Aggregated total contacts across all chain pairs is meaningful;
        the legacy 2-body mode kept the normalisation for back-compat.
        """
        per_pair_dist = np.empty(len(self.subset_pairs), dtype=np.float64)
        per_pair_contacts = np.empty(len(self.subset_pairs), dtype=np.float64)
        for idx, (_, _, pi, pj) in enumerate(self.subset_pairs):
            vecs = xyz[pi] - xyz[pj]
            dists = np.sqrt(np.sum(vecs * vecs, axis=1))
            per_pair_dist[idx] = self._mean_k_smallest(dists)
            per_pair_contacts[idx] = self._soft_contact_count(dists)  # unnormalised

        is_worst = self.mode.startswith("all_pair_worst_")
        agg_d = float(per_pair_dist.max() if is_worst else per_pair_dist.mean())
        agg_c = float(per_pair_contacts.sum())  # sum across pairs in both modes

        if self.mode.endswith("distance_and_contacts"):
            return np.array([agg_d, -agg_c], dtype=np.float32)
        return np.float32(agg_d)

    def _mean_k_smallest(self, dists: np.ndarray) -> float:
        k = min(self.k, dists.size)
        smallest = np.partition(dists, k - 1)[:k]
        return float(np.mean(smallest))

    def _soft_contact_count(self, dists: np.ndarray) -> float:
        """
        Smooth contact function.

        For each inter-chain pair:
            s_ij = 1 / (1 + (d_ij / d0)^n)

        If d_ij << d0, contribution is close to 1.
        If d_ij >> d0, contribution is close to 0.
        """
        x = dists / self.contact_cutoff_angstrom
        return float(np.sum(1.0 / (1.0 + np.power(x, self.soft_n))))

    def _center_distance(self, A: np.ndarray, B: np.ndarray) -> float:
        center_a = np.mean(A, axis=0)
        center_b = np.mean(B, axis=0)
        return float(np.linalg.norm(center_a - center_b))

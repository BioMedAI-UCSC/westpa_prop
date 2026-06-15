import json

import numpy as np
import mdtraj

from computation.base_computation import BaseComputation


AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_INDEX = {a: i for i, a in enumerate(AA)}

KD = {"ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5, "GLN": -3.5,
      "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5, "LEU": 3.8, "LYS": -3.9,
      "MET": 1.9, "PHE": 2.8, "PRO": -1.6, "SER": -0.8, "THR": -0.7, "TRP": -0.9,
      "TYR": -1.3, "VAL": 4.2}

ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
           "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU",
           "LYN": "LYS", "MSE": "MET"}


class ContactEnergyComputation(BaseComputation):
    """Reference-free DoBi-style interface contact energy.

    Sums a residue-residue contact potential over inter-chain residue pairs in
    contact. Lower (more negative) = more favorable interface. Pluggable potential
    and contact scheme; no bound reference required.

    potential : "hydrophobic" (built-in Kyte-Doolittle desolvation proxy) or a
                path to .json/.npz with {'order': [3-letter...], 'matrix': NxN}.
    scheme    : "cb" (CB, CA for GLY), "ca", or "closest_heavy".
    mode      : "energy" or "energy_and_ncontacts".
    normalize : None, "n_contacts", or "sqrt_size".
    """

    requires_positions = True
    requires_energy = False

    def __init__(self, topology_path, selection_a, selection_b,
                 mode="energy", potential="hydrophobic", scheme="cb",
                 contact_cutoff_angstrom=8.0, exclude_hydrogens=True,
                 normalize=None):
        self.mode = mode
        self.scheme = scheme
        self.cutoff = float(contact_cutoff_angstrom)
        self.normalize = normalize
        if mode not in ("energy", "energy_and_ncontacts"):
            raise ValueError(f"bad mode {mode!r}")
        if scheme not in ("cb", "ca", "closest_heavy"):
            raise ValueError(f"bad scheme {scheme!r}")

        self.M = self._load_potential(potential)
        top = mdtraj.load(topology_path).topology
        self.res_a = self._residues(top, selection_a, exclude_hydrogens)
        self.res_b = self._residues(top, selection_b, exclude_hydrogens)
        if not self.res_a or not self.res_b:
            raise ValueError("A selection matched zero typed residues")
        self.types_a = np.array([r["type"] for r in self.res_a])
        self.types_b = np.array([r["type"] for r in self.res_b])
        self.pot_ab = self.M[np.ix_(self.types_a, self.types_b)]

    def _load_potential(self, potential):
        if potential == "hydrophobic":
            h = np.array([KD[a] for a in AA]) / 4.5
            return -np.outer(h, h)
        if potential.endswith(".npz"):
            d = np.load(potential, allow_pickle=True)
            order, mat = list(d["order"]), np.asarray(d["matrix"], float)
        else:
            with open(potential) as f:
                d = json.load(f)
            order, mat = d["order"], np.asarray(d["matrix"], float)
        idx = [AA_INDEX[ALIASES.get(o, o)] for o in order]
        M = np.zeros((20, 20))
        M[np.ix_(idx, idx)] = mat
        return M

    def _residues(self, top, selection, exclude_h):
        sel = f"({selection}) and not element H" if exclude_h else selection
        atom_set = set(top.select(sel).tolist())
        out = []
        for res in top.residues:
            name = ALIASES.get(res.name, res.name)
            if name not in AA_INDEX:
                continue
            atoms = [a for a in res.atoms if a.index in atom_set]
            if not atoms:
                continue
            entry = {"type": AA_INDEX[name],
                     "heavy": np.array([a.index for a in atoms], dtype=int)}
            if self.scheme in ("cb", "ca"):
                entry["rep"] = self._rep_atom(res, atom_set)
            out.append(entry)
        return out

    def _rep_atom(self, res, atom_set):
        want = "CA" if self.scheme == "ca" else "CB"
        names = {a.name: a.index for a in res.atoms if a.index in atom_set}
        for n in (want, "CA", "CB"):
            if n in names:
                return names[n]
        return next(iter(names.values()))

    def calculate(self, data: np.ndarray, energy: dict = None) -> np.ndarray:
        self._validate_input(data)
        n = data.shape[0]
        out = np.zeros((n, 2) if self.mode == "energy_and_ncontacts" else n,
                       dtype=np.float32)
        for f in range(n):
            mask = self._contact_mask(data[f])
            e = float(np.sum(self.pot_ab[mask]))
            nc = int(mask.sum())
            e = self._apply_norm(e, nc)
            if self.mode == "energy":
                out[f] = e
            else:
                out[f, 0], out[f, 1] = e, nc
        if not np.all(np.isfinite(out)):
            raise ValueError("non-finite contact energy")
        if n == 1 and self.mode == "energy_and_ncontacts":
            return out[0]
        return out

    def _contact_mask(self, xyz):
        if self.scheme == "closest_heavy":
            d = np.full((len(self.res_a), len(self.res_b)), np.inf)
            for i, ra in enumerate(self.res_a):
                A = xyz[ra["heavy"]]
                for j, rb in enumerate(self.res_b):
                    B = xyz[rb["heavy"]]
                    d[i, j] = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
            return d < self.cutoff
        A = xyz[[r["rep"] for r in self.res_a]]
        B = xyz[[r["rep"] for r in self.res_b]]
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        return d < self.cutoff

    def _apply_norm(self, e, nc):
        if self.normalize == "n_contacts":
            return e / max(nc, 1)
        if self.normalize == "sqrt_size":
            return e / np.sqrt(len(self.res_a) * len(self.res_b))
        return e

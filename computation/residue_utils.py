import mdtraj

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


def load_topology(topology_path):
    return mdtraj.load(topology_path).topology


def residue_reps(topology, selection, scheme="cb", exclude_h=True):
    """List of typed residues in `selection`.

    Each entry: {'type': int 0..19, 'rep': atom_index, 'heavy': int array}.
    scheme in {'cb','ca','closest_heavy'}; 'rep' falls back CB->CA->first heavy.
    """
    import numpy as np
    sel = f"({selection}) and not element H" if exclude_h else selection
    atom_set = set(topology.select(sel).tolist())
    out = []
    want = "CA" if scheme == "ca" else "CB"
    for res in topology.residues:
        name = ALIASES.get(res.name, res.name)
        if name not in AA_INDEX:
            continue
        atoms = [a for a in res.atoms if a.index in atom_set]
        if not atoms:
            continue
        names = {a.name: a.index for a in atoms}
        rep = next((names[n] for n in (want, "CA", "CB") if n in names),
                   next(iter(names.values())))
        out.append({"type": AA_INDEX[name], "rep": rep,
                    "heavy": np.array([a.index for a in atoms], dtype=int)})
    return out

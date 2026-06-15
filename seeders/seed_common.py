"""Shared rigid-body / clash / bstate-writing helpers for the seeders.

All geometry is done in Angstrom on (N, 3) float64 arrays. PDBs are read/written
through OpenMM (positions stored in nm), matching the rest of the framework
(see tools/randomize_chains.py).

Design notes
------------
* A "group" is one rigid body = one or more chains that move together
  (e.g. an antibody H+L pair). Default: every chain is its own group.
* DoBi (Guo et al. 2012) ideas applied here:
    - generate many candidate relative configurations (not one),
    - reject clashing poses (interior overlap),
    - keep poses whose surfaces are near contact (encounter complexes).
  We approximate DoBi's grid clash test with a heavy-atom minimum-distance
  test, which is cheaper and adequate for seeding.
"""

import json
import os
import sys
import time

import numpy as np

from openmm import Vec3, unit
from openmm.app import PDBFile


# --------------------------------------------------------------------------- #
# PDB / topology I/O
# --------------------------------------------------------------------------- #
def load_structure(path):
    """Return (topology, positions_A, chain_order) where positions are Angstrom."""
    if not os.path.isfile(path):
        sys.exit(f"Input not found: {path}")
    pdb = PDBFile(path)
    pos_nm = np.array([[v.x, v.y, v.z] for v in pdb.positions], dtype=np.float64)
    positions_A = pos_nm * 10.0
    chain_order = [c.id for c in pdb.topology.chains()]
    return pdb.topology, positions_A, chain_order


def index_chains(topology):
    """Map chain id -> {'all': [idx...], 'heavy': [idx...]} in topology order."""
    chains = {}
    order = []
    for atom in topology.atoms():
        cid = atom.residue.chain.id
        if cid not in chains:
            chains[cid] = {"all": [], "heavy": []}
            order.append(cid)
        chains[cid]["all"].append(atom.index)
        if atom.element is not None and atom.element.symbol != "H":
            chains[cid]["heavy"].append(atom.index)
    return chains, order


def parse_groups(group_tokens, chain_order):
    """['HL','Y'] or ['H,L','Y'] -> [['H','L'],['Y']], validated against chain_order."""
    groups = []
    for tok in group_tokens:
        chains = [c for c in tok.split(",") if c] if "," in tok else list(tok)
        groups.append(chains)
    flat = [c for g in groups for c in g]
    if len(flat) != len(set(flat)):
        sys.exit(f"--groups has duplicate chain IDs: {flat}")
    missing = [c for c in chain_order if c not in flat]
    if missing:
        sys.exit(f"--groups omits chain(s) present in the PDB: {missing}")
    unknown = [c for c in flat if c not in chain_order]
    if unknown:
        sys.exit(f"--groups references chain(s) not in the PDB: {unknown}")
    return groups


def group_atom_indices(groups, chains):
    """Return (all_idx_per_group, heavy_idx_per_group) as lists of int arrays."""
    all_idx = [
        np.array([i for c in g for i in chains[c]["all"]], dtype=int) for g in groups
    ]
    heavy_idx = [
        np.array([i for c in g for i in chains[c]["heavy"]], dtype=int) for g in groups
    ]
    for gi, h in enumerate(heavy_idx):
        if h.size == 0:
            sys.exit(f"Group {gi} ({groups[gi]}) has no heavy atoms.")
    return all_idx, heavy_idx


# --------------------------------------------------------------------------- #
# Rigid-body transforms
# --------------------------------------------------------------------------- #
def euler_matrix(rx, ry, rz):
    """Rz @ Ry @ Rx, angles in degrees (matches tools/randomize_chains.py)."""
    def rot(axis, deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        if axis == "x":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot("z", rz) @ rot("y", ry) @ rot("x", rx)


def random_rotation_matrix(rng):
    """Uniform random rotation on SO(3) via a random unit quaternion (Shoemake)."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, size=3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def apply_rigid(positions, idx, R, translation):
    """Rotate atoms `idx` about their centroid by R, then translate. In place-safe copy."""
    coords = positions[idx]
    centroid = coords.mean(axis=0)
    positions[idx] = (coords - centroid) @ R.T + centroid + translation
    return positions


# --------------------------------------------------------------------------- #
# Interface geometry checks
# --------------------------------------------------------------------------- #
def min_inter_distance(positions, idx_a, idx_b, chunk=2000):
    """Minimum pairwise distance (A) between two heavy-atom sets, memory-chunked."""
    pa, pb = positions[idx_a], positions[idx_b]
    best = np.inf
    for s in range(0, pa.shape[0], chunk):
        d = np.linalg.norm(pa[s:s + chunk, None, :] - pb[None, :, :], axis=2)
        best = min(best, float(d.min()))
    return best


def count_close_contacts(positions, idx_a, idx_b, cutoff=6.0, chunk=2000):
    """Number of heavy-atom pairs within `cutoff` A across the interface."""
    pa, pb = positions[idx_a], positions[idx_b]
    n = 0
    for s in range(0, pa.shape[0], chunk):
        d = np.linalg.norm(pa[s:s + chunk, None, :] - pb[None, :, :], axis=2)
        n += int(np.count_nonzero(d < cutoff))
    return n


# --------------------------------------------------------------------------- #
# bstate writing
# --------------------------------------------------------------------------- #
def write_pdb(topology, positions_A, out_path):
    """Write Angstrom positions back through OpenMM (expects nm)."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    quant = unit.Quantity(
        value=[Vec3(x / 10.0, y / 10.0, z / 10.0) for x, y, z in positions_A],
        unit=unit.nanometer,
    )
    with open(out_path, "w") as f:
        PDBFile.writeFile(topology, quant, f, keepIds=True)


def add_common_args(parser):
    parser.add_argument("input_pdb")
    parser.add_argument("--sim-root", default=".",
                        help="WEST_SIM_ROOT; bstates/ and bstates.txt are written here.")
    parser.add_argument("--groups", nargs="+", default=None,
                        help="Rigid-body grouping, e.g. 'AB' 'C' or 'A,B' 'C'. "
                             "Default: each chain is its own group.")
    parser.add_argument("--n-states", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clash-cutoff", type=float, default=2.0,
                        help="Reject pose if any inter-group heavy-atom pair is "
                             "closer than this (A).")
    parser.add_argument("--min-sep", type=float, default=4.0,
                        help="Required min heavy-atom separation, group0 vs group1 (A).")
    parser.add_argument("--max-sep", type=float, default=np.inf,
                        help="Max heavy-atom separation, group0 vs group1 (A).")
    parser.add_argument("--contact-cutoff", type=float, default=6.0)
    parser.add_argument("--max-tries", type=int, default=200,
                        help="Reseed attempts per requested state before giving up.")
    parser.add_argument("--label-prefix", default="seed")
    return parser


def prepare(args):
    topology, positions_A, chain_order = load_structure(args.input_pdb)
    chains, _ = index_chains(topology)
    groups = parse_groups(args.groups, chain_order) if args.groups else [[c] for c in chain_order]
    if len(groups) < 2:
        sys.exit(f"Need >=2 groups; got {len(groups)}")
    all_idx, heavy_idx = group_atom_indices(groups, chains)
    return dict(topology=topology, positions_A=positions_A, groups=groups,
                all_idx=all_idx, heavy_idx=heavy_idx)


def accept_pose(positions, heavy_idx, clash_cutoff, min_sep, max_sep):
    """All inter-group pairs clash-free; group0-group1 separation in [min_sep, max_sep]."""
    ng = len(heavy_idx)
    for i in range(ng):
        for j in range(i + 1, ng):
            if min_inter_distance(positions, heavy_idx[i], heavy_idx[j]) < clash_cutoff:
                return False, None
    sep01 = min_inter_distance(positions, heavy_idx[0], heavy_idx[1])
    return (min_sep <= sep01 <= max_sep), sep01


def write_bstates(sim_root, seeds, weights=None, label_prefix="seed"):
    """Write bstates/ pdbs + bstates.txt under sim_root.

    Parameters
    ----------
    sim_root : str           WEST_SIM_ROOT (bstates/ and bstates.txt go here).
    seeds : list of dict      each {'topology', 'positions_A', 'record'}.
    weights : list of float   defaults to uniform; normalized to sum 1.
    """
    bdir = os.path.join(sim_root, "bstates")
    os.makedirs(bdir, exist_ok=True)
    n = len(seeds)
    if n == 0:
        sys.exit("No seeds to write (all candidates were rejected).")
    if weights is None:
        weights = [1.0] * n
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    lines, records = [], []
    for i, seed in enumerate(seeds):
        label = f"{label_prefix}{i:03d}"
        fname = f"{label}.pdb"
        write_pdb(seed["topology"], seed["positions_A"], os.path.join(bdir, fname))
        lines.append(f"{label} {w[i]:.8f} {fname}")
        rec = dict(seed.get("record", {}))
        rec.update({"label": label, "weight": float(w[i]), "pdb": fname})
        records.append(rec)

    with open(os.path.join(sim_root, "bstates.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(sim_root, "bstates_manifest.json"), "w") as f:
        json.dump({"created_unix": time.time(), "n_states": n, "seeds": records},
                  f, indent=2)

    print(f"Wrote {n} basis states to {bdir}")
    print(f"Wrote {os.path.join(sim_root, 'bstates.txt')}")
    return os.path.join(sim_root, "bstates.txt")

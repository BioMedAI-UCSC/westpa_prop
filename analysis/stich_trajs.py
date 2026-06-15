#!/usr/bin/env python
"""
stitch_trajs.py – Stitch WESTPA segment trajectories into full HDF5 trajectories.

Reads defaults from west.cfg (YAML), with all options overridable via CLI.
"""

import argparse
import os
import sys
import yaml
import numpy as np
import mdtraj as md
from westpa.analysis import Run
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# west.cfg parsing
# ──────────────────────────────────────────────────────────────────────────────

class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader extended to silently ignore unknown YAML tags (e.g.
    ``!!python/name:numpy.float32``) rather than raising ConstructorError."""


def _ignore_unknown_tag(loader, tag_suffix, node):
    """Return the plain Python value for any unrecognised tag."""
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


_TolerantLoader.add_multi_constructor("", _ignore_unknown_tag)


def load_west_cfg(cfg_path: str) -> dict:
    """Load and return the parsed west.cfg YAML, or {} if not found.

    Uses a tolerant loader so that WESTPA-specific tags such as
    ``!!python/name:numpy.float32`` do not cause a ConstructorError.
    """
    if not os.path.exists(cfg_path):
        print(f"[warn] west.cfg not found at '{cfg_path}', using built-in defaults.")
        return {}
    with open(cfg_path) as fh:
        return yaml.load(fh, Loader=_TolerantLoader) or {}


def cfg_get(cfg: dict, *keys, default=None):
    """Safely walk nested dict keys, returning default if any key is missing."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def resolve_env(value: str) -> str:
    """Expand shell-style $VAR references in a string.

    $WEST_SIM_ROOT is treated as the current working directory when it is not
    already set in the environment.
    """
    if not isinstance(value, str):
        return value
    if "WEST_SIM_ROOT" not in os.environ:
        value = value.replace("$WEST_SIM_ROOT", os.getcwd())
    return os.path.expandvars(value)


def defaults_from_cfg(cfg: dict) -> dict:
    """
    Extract the fields we care about from a parsed west.cfg dict.

    Returned keys match the argparse dest names so they can be used as
    argument defaults directly.
    """
    west = cfg.get("west", {})

    # west.h5 path
    west_h5 = resolve_env(
        cfg_get(west, "data", "west_data_file", default="west.h5")
    )

    # traj_segs root from segment data_ref, e.g.
    #   $WEST_SIM_ROOT/traj_segs/{segment.n_iter:06d}/{segment.seg_id:06d}
    seg_ref = resolve_env(
        cfg_get(west, "data", "data_refs", "segment", default="")
    )
    sim_root = "."
    traj_segs_dir = "traj_segs"
    if seg_ref:
        # Split off the two format-field path components at the end
        parts = seg_ref.replace("\\", "/").split("/")
        # Drop trailing {segment.seg_id:…} and {segment.n_iter:…} parts
        path_parts = [p for p in parts if not p.startswith("{")]
        if path_parts:
            traj_segs_dir = path_parts[-1]
            sim_root = "/".join(path_parts[:-1]) or "."

    # Topology path (openmm block is project-specific but commonly present)
    topology_pdb = resolve_env(
        cfg_get(west, "openmm", "topology_path", default="")
    )

    return {
        "west_h5":      west_h5,
        "sim_root":     sim_root,
        "traj_segs_dir": traj_segs_dir,
        "topology_pdb": topology_pdb,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser(cfg_defaults: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stitch WESTPA segment DCDs into full HDF5 trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--west-cfg",
        default="west.cfg",
        metavar="FILE",
        help="Path to west.cfg (parsed for defaults before other flags).",
    )
    p.add_argument(
        "--west-h5",
        default=cfg_defaults.get("west_h5", "west.h5"),
        metavar="FILE",
        help="Path to the WESTPA HDF5 data file.",
    )
    p.add_argument(
        "--sim-root",
        default=cfg_defaults.get("sim_root", "."),
        metavar="DIR",
        help="Root directory of the simulation (where traj_segs/ lives).",
    )
    p.add_argument(
        "--traj-segs-dir",
        default=cfg_defaults.get("traj_segs_dir", "traj_segs"),
        metavar="DIR",
        help="Name of the trajectory segments subdirectory.",
    )
    p.add_argument(
        "--topology",
        default=cfg_defaults.get("topology_pdb", ""),
        metavar="PDB",
        dest="topology_pdb",
        help="Topology PDB for the system (required).",
    )
    p.add_argument(
        "--out-dir",
        default="full_cg_hdf5_trajs",
        metavar="DIR",
        help="Directory where stitched HDF5 trajectories are written.",
    )
    p.add_argument(
        "--iteration",
        type=int,
        default=None,
        metavar="N",
        help="Iteration to process (default: last iteration in west.h5).",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=100,
        metavar="N",
        help="Keep every Nth frame from the trace (1 = keep all).",
    )
    p.add_argument(
        "--walkers",
        type=int,
        nargs="+",
        default=None,
        metavar="IDX",
        help="Subset of walker indices to process (default: all walkers).",
    )
    p.add_argument(
        "--seg-filename",
        default="seg.dcd",
        metavar="NAME",
        help="Filename of the segment trajectory inside each seg directory.",
    )
    p.add_argument(
        "--nm",
        action="store_true",
        help="Coordinates in seg files are already in nm (skip ×10 conversion).",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Path helpers
# ──────────────────────────────────────────────────────────────────────────────

def segment_dcd_path(walker, sim_root: str, traj_segs_dir: str, seg_filename: str) -> str:
    """Return the path to the segment trajectory file for *walker*."""
    iter_dir = f"{walker.iteration.number:06d}"
    seg_dir  = f"{walker.index:06d}"
    return os.path.join(sim_root, traj_segs_dir, iter_dir, seg_dir, seg_filename)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_segment_positions(
    walker,
    topology,
    sim_root: str,
    traj_segs_dir: str,
    seg_filename: str,
    coords_in_nm: bool = False,
) -> np.ndarray:
    """
    Load coordinates from a segment trajectory file.

    Returns an array of shape (n_frames, n_atoms, 3) in Ångströms.
    """
    path = segment_dcd_path(walker, sim_root, traj_segs_dir, seg_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing segment trajectory for walker {walker} at '{path}'"
        )

    traj = md.load_dcd(path, top=topology)
    pos  = traj.xyz.copy()  # (n_frames, n_atoms, 3), nm by default from mdtraj

    if not coords_in_nm:
        pos = pos * 10.0  # nm → Å

    return pos


def save_hdf5_traj(
    path: str,
    coords: np.ndarray,
    topology,
    time: np.ndarray = None,
) -> None:
    """
    Save a trajectory as an mdtraj HDF5 file.

    coords : (n_frames, n_atoms, 3) in Å — converted to nm internally.
    """
    coords = np.asarray(coords, dtype=np.float32)

    kwargs = {"time": np.asarray(time, dtype=np.float32)} if time is not None else {}
    traj = md.Trajectory(coords, topology, **kwargs)
    traj.save_hdf5(path)


# ──────────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────────

def stitch_walker(
    walker,
    topology,
    last_iter: int,
    sim_root: str,
    traj_segs_dir: str,
    seg_filename: str,
    stride: int,
    coords_in_nm: bool,
) -> np.ndarray | None:
    """
    Trace *walker* back to its origin and concatenate segment coordinates.

    stride : keep every Nth segment in the trace (first and last are always kept).
    Returns a (n_frames, n_atoms, 3) array in Å, or None if the trace is empty.
    """
    trace = list(walker.trace())
    if not trace:
        return None

    seg_frames = []

    for i, seg_walker in enumerate(tqdm(trace, desc=f"  walker {walker.index}", leave=False)):
        # Always include the first and last segments; skip others per stride
        if i != 0 and i != len(trace) - 1 and i % stride != 0:
            continue

        pos = load_segment_positions(
            seg_walker, topology, sim_root, traj_segs_dir, seg_filename, coords_in_nm
        )

        if pos.shape[1] != topology.n_atoms:
            raise ValueError(
                f"Atom count mismatch: segment has {pos.shape[1]} atoms, "
                f"topology has {topology.n_atoms}. "
                f"Walker {seg_walker} at "
                f"'{segment_dcd_path(seg_walker, sim_root, traj_segs_dir, seg_filename)}'"
            )

        # Drop first frame of subsequent segments to avoid double-counting
        if seg_frames:
            pos = pos[1:]

        seg_frames.append(pos)

    if not seg_frames:
        return None

    return np.concatenate(seg_frames, axis=0)


def process_iteration(args: argparse.Namespace, run: Run) -> None:
    """Iterate over walkers in the chosen iteration and write HDF5 files."""
    target_iter = args.iteration if args.iteration is not None else run.num_iterations
    iteration   = run.iteration(target_iter)

    print(f"Processing iteration {target_iter} "
          f"({iteration.num_walkers} walkers total).")

    walkers = list(iteration.walkers)
    if args.walkers is not None:
        walkers = [w for w in walkers if w.index in set(args.walkers)]
        print(f"Restricting to {len(walkers)} requested walkers.")

    os.makedirs(args.out_dir, exist_ok=True)

    for walker in tqdm(walkers, desc="Walkers"):
        full_pos = stitch_walker(
            walker,
            topology       = args._topology_obj,
            last_iter      = target_iter,
            sim_root       = args.sim_root,
            traj_segs_dir  = args.traj_segs_dir,
            seg_filename   = args.seg_filename,
            stride         = args.stride,
            coords_in_nm   = args.nm,
        )

        if full_pos is None:
            print(f"  Walker {walker.index}: empty trace, skipping.")
            continue

        print(f"  Walker {walker.index}: {full_pos.shape[0]} frames, "
              f"{full_pos.shape[1]} atoms.")

        out_name = f"fulltraj_iter{target_iter:06d}_walker{walker.index:04d}.h5"
        out_path = os.path.join(args.out_dir, out_name)
        save_hdf5_traj(out_path, full_pos, args._topology_obj)
        print(f"  Saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    # ── Two-pass argument parsing ──────────────────────────────────────────
    # Pass 1: find --west-cfg so we can load defaults before building the
    #         full parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--west-cfg", default="west.cfg")
    pre_args, _ = pre.parse_known_args(argv)

    cfg          = load_west_cfg(pre_args.west_cfg)
    cfg_defaults = defaults_from_cfg(cfg)

    # Pass 2: full parser with cfg-derived defaults
    parser = build_parser(cfg_defaults)
    args   = parser.parse_args(argv)

    # ── Validate required inputs ───────────────────────────────────────────
    if not args.topology_pdb:
        parser.error(
            "Topology PDB is required. Supply --topology <file> or set "
            "west.openmm.topology_path in west.cfg."
        )
    if not os.path.exists(args.topology_pdb):
        parser.error(f"Topology PDB not found: '{args.topology_pdb}'")
    if not os.path.exists(args.west_h5):
        parser.error(f"WESTPA HDF5 file not found: '{args.west_h5}'")

    # ── Load topology ──────────────────────────────────────────────────────
    print(f"Loading topology from '{args.topology_pdb}' …")
    topo_traj          = md.load(args.topology_pdb)
    args._topology_obj = topo_traj.topology  # stash for reuse
    print(f"  {args._topology_obj.n_atoms} atoms in topology.")

    # ── Open WESTPA run ────────────────────────────────────────────────────
    print(f"Opening WESTPA run from '{args.west_h5}' …")
    run = Run.open(args.west_h5)

    try:
        process_iteration(args, run)
    finally:
        run.close()

    print("Done.")


if __name__ == "__main__":
    main()

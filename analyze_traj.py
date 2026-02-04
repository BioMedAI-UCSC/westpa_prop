#!/usr/bin/env python

import os
import numpy as np
import mdtraj as md
from westpa.analysis import Run
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configuration – adjust these to your layout
# ----------------------------------------------------------------------

# Path to your WESTPA HDF5 file
WEST_H5 = "west.h5"

# Root directory of the simulation (where traj_segs/ lives)
SIM_ROOT = "."

# Name of the trajectory segments directory
TRAJ_SEGS_DIR = "traj_segs"

# Topology PDB for the CG system used to define atoms/residues
TOPOLOGY_PDB = "/global/homes/a/awaghili/ctxB/structures/ctxBAFUnbindedCombinedFiltered.pdb"

# Where to write the stitched trajectories
OUT_DIR = "full_cg_hdf5_trajs"


# ----------------------------------------------------------------------
# Helpers for locating and loading segment DCD files
# ----------------------------------------------------------------------

def segment_dcd_path(walker):
    """
    Build the path to seg.dcd for a given Walker.

    Assumes layout:
        traj_segs/iter_000001/seg_000000/seg.dcd
    """
    iter_num = walker.iteration.number     # 1-based iteration number
    seg_idx = walker.index                 # 0-based walker index

    iter_dir = f"{iter_num:06d}"
    seg_dir = f"{seg_idx:06d}"

    return os.path.join(
        SIM_ROOT, TRAJ_SEGS_DIR, iter_dir, seg_dir, "seg.dcd"
    )


def load_segment_positions(walker, topology_all):
    """
    Load coordinates from seg.dcd using mdtraj.

    Returns array with shape (n_frames, n_atoms, 3) in Å as stored.
    """
    path = segment_dcd_path(walker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing seg.dcd for walker {walker} at {path}")

    # mdtraj.load_dcd requires topology loaded already
    traj = md.load_dcd(path, top=topology_all)
    pos = traj.xyz.copy()  # (n_frames, n_atoms, 3), nm by default

    # Convert nm → Å because original numpy npz code expected Å
    pos = pos * 10.0

    return pos


# ----------------------------------------------------------------------
# HDF5 saving helper
# ----------------------------------------------------------------------

def save_hdf5_traj(path, coords, topology, time=None):
    """
    Save a trajectory as an mdtraj HDF5 file.

    coords : np.ndarray with shape (n_frames, n_atoms, 3)
    topology : mdtraj.Topology
    time : optional 1D array of shape (n_frames,) in ps
    """
    coords = np.asarray(coords, dtype=np.float32)

    # Convert Å → nm (mdtraj storage convention)
    #coords = coords / 10.0

    if time is not None:
        time = np.asarray(time, dtype=np.float32)
        traj = md.Trajectory(coords, topology, time=time)
    else:
        traj = md.Trajectory(coords, topology)

    traj.save_hdf5(path)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load topology once
    print(f"Loading topology from {TOPOLOGY_PDB} ...")
    topo_traj = md.load(TOPOLOGY_PDB)
    topology_all = topo_traj.topology
    n_top_atoms = topology_all.n_atoms
    print(f"Full topology has {n_top_atoms} atoms")

    # Open WESTPA run
    print(f"Opening WESTPA run from {WEST_H5} ...")
    run = Run.open(WEST_H5)

    last_iter = run.num_iterations
    iteration = run.iteration(last_iter)

    print(f"Last iteration: {last_iter}")
    print(f"Number of walkers: {iteration.num_walkers}")

    first_full_pos_checked = False

    # Loop over walkers in the last iteration
    for walker in tqdm(iteration.walkers):
        trace = walker.trace()

        seg_frames = []

        for i, seg_walker in tqdm(enumerate(trace)):
            pos = load_segment_positions(seg_walker, topology_all)

            # Sanity check on atom count
            if pos.shape[1] != n_top_atoms:
                raise ValueError(
                    f"Atom count mismatch: seg.dcd has {pos.shape[1]} atoms, "
                    f"topology has {n_top_atoms}. "
                    f"Walker {seg_walker} at {segment_dcd_path(seg_walker)}"
                )

            # Drop 1st frame of subsequent segments to avoid double-counting
            if i > 0:
                pos = pos[1:]

            seg_frames.append(pos)

        if not seg_frames:
            print(f"Walker {walker.index} has empty trace, skipping.")
            continue

        full_pos = np.concatenate(seg_frames, axis=0)

        if not first_full_pos_checked:
            print(
                f"Example full trajectory for walker {walker.index}: "
                f"{full_pos.shape[0]} frames, {full_pos.shape[1]} atoms"
            )
            first_full_pos_checked = True

        out_name = f"fulltraj_iter{last_iter:06d}_walker{walker.index:04d}.h5"
        out_path = os.path.join(OUT_DIR, out_name)

        print(f"Saving {out_path} ...")
        save_hdf5_traj(out_path, full_pos, topology_all)

    run.close()
    print("Done building full CG HDF5 trajectories.")


if __name__ == "__main__":
    main()


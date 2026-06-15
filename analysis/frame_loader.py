import os
import glob

import numpy as np
import mdtraj


def load_segments(sim_root, topology_pdb, iter_stride=1, seg_stride=1,
                  max_iters=None, frames="all", min_iter=None, max_iter=None):
    """Load per-segment trajectories + RMSD labels from a WESTPA traj_segs tree.

    Returns (segments, rmsd, lengths):
        segments : list of (nf, n_atoms, 3) Angstrom arrays
        rmsd     : list of (nf,) arrays (from seg.npz['rmsd_ca'])
        lengths  : list of nf  (for lag-aware embeddings)
    frames="last" keeps only the final frame of each segment.
    """
    top = mdtraj.load(topology_pdb).topology
    iter_dirs = sorted(d for d in glob.glob(os.path.join(sim_root, "traj_segs", "[0-9]" * 6)))
    if min_iter is not None or max_iter is not None:
        lo, hi = min_iter or 0, max_iter or 10 ** 12
        iter_dirs = [d for d in iter_dirs if lo <= int(os.path.basename(d)) <= hi]
    iter_dirs = iter_dirs[::iter_stride]
    if max_iters:
        iter_dirs = iter_dirs[:max_iters]

    segs, rms, lens = [], [], []
    for idir in iter_dirs:
        seg_dirs = sorted(glob.glob(os.path.join(idir, "[0-9]" * 6)))[::seg_stride]
        for sdir in seg_dirs:
            dcd = os.path.join(sdir, "seg.dcd")
            npz = os.path.join(sdir, "seg.npz")
            if not os.path.isfile(dcd):
                continue
            try:
                xyz = mdtraj.load_dcd(dcd, top=top).xyz * 10.0
                # RMSD label is optional (only present when recorded calcs ran)
                r = (np.asarray(np.load(npz)["rmsd_ca"]).reshape(-1)
                     if os.path.isfile(npz) else np.full(len(xyz), np.nan, np.float32))
            except Exception:
                continue
            if frames == "last":
                xyz, r = xyz[-1:], r[-1:]
            n = min(len(xyz), len(r))
            if n == 0:
                continue
            segs.append(xyz[:n].astype(np.float32))
            rms.append(r[:n])
            lens.append(n)
    return segs, rms, lens


def featurize_segments(featurizer, segments):
    """Apply a featurizer to each segment; return stacked features (sum nf, ...)."""
    return np.concatenate([featurizer.calculate(s) for s in segments], axis=0)

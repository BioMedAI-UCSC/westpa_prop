#!/usr/bin/env python3
"""Post-hoc validation metrics for a WE run (computed from DCDs, MPI-safe).

Walks traj_segs, computes per-frame L-RMSD, i-RMSD, contact energy + contacts,
and global rmsd_ca, attaches WE weights from west.h5, and saves a tidy npz.

    python -m analysis.compute_metrics --sim-root runs/pilot_blind_tica \
        --topology /data/alex/pd1/3BIKFiltered.pdb --reference /data/alex/pd1/3BIKFiltered.pdb \
        --sel-a "chainid 0" --sel-b "chainid 1" --out analysis_out/pilot_tica_metrics.npz
"""
import argparse
import glob
import os

import numpy as np
import h5py

from computation.interface_rmsd_computation import InterfaceRMSDComputation
from computation.contact_energy_computation import ContactEnergyComputation
from computation.rmsd_computation import RMSDComputation
from analysis.frame_loader import load_segments


def seg_final_weights(sim_root):
    """Map (n_iter, seg_id) -> weight from west.h5."""
    w = {}
    with h5py.File(os.path.join(sim_root, "west.h5"), "r") as h:
        for k in h["iterations"]:
            n = int(k.split("_")[-1])
            for sid, row in enumerate(h["iterations"][k]["seg_index"]):
                w[(n, sid)] = float(row["weight"])
    return w


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-root", required=True)
    p.add_argument("--topology", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--sel-a", required=True)
    p.add_argument("--sel-b", required=True)
    p.add_argument("--iter-stride", type=int, default=1)
    p.add_argument("--seg-stride", type=int, default=1)
    p.add_argument("--frames", default="last", choices=["last", "all"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    calcs = {
        "lrmsd": InterfaceRMSDComputation(a.reference, a.sel_a, a.sel_b, mode="ligand"),
        "irmsd": InterfaceRMSDComputation(a.reference, a.sel_a, a.sel_b, mode="interface"),
        "rmsd_ca": RMSDComputation(a.reference, atom_selection="name CA"),
        "contact_energy": ContactEnergyComputation(a.topology, a.sel_a, a.sel_b,
                                                   mode="energy_and_ncontacts"),
    }
    # walk traj_segs preserving (iter, seg) identity via the dir structure
    iter_dirs = sorted(glob.glob(os.path.join(a.sim_root, "traj_segs", "[0-9]" * 6)))[::a.iter_stride]
    import mdtraj
    top = mdtraj.load(a.topology).topology
    weights = seg_final_weights(a.sim_root)

    rows = {k: [] for k in calcs}
    meta = {"n_iter": [], "seg_id": [], "weight": []}
    for idir in iter_dirs:
        n = int(os.path.basename(idir))
        for sdir in sorted(glob.glob(os.path.join(idir, "[0-9]" * 6)))[::a.seg_stride]:
            sid = int(os.path.basename(sdir))
            dcd = os.path.join(sdir, "seg.dcd")
            if not os.path.isfile(dcd):
                continue
            xyz = mdtraj.load_dcd(dcd, top=top).xyz * 10.0
            xyz = (xyz[-1:] if a.frames == "last" else xyz).astype(np.float32)
            for k, c in calcs.items():
                v = np.asarray(c.calculate(xyz))
                v = v.reshape(-1) if v.ndim == 1 else v[-1]  # last frame's vector
                rows[k].append(np.atleast_1d(v).astype(np.float32))
            meta["n_iter"].append(n)
            meta["seg_id"].append(sid)
            meta["weight"].append(weights.get((n, sid), np.nan))

    out = {k: np.array(v, dtype=np.float32) for k, v in rows.items()}
    out.update({k: np.array(v) for k, v in meta.items()})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, **out)
    lr = out["lrmsd"].reshape(-1)
    print(f"wrote {a.out}: {len(lr)} segments")
    print(f"  L-RMSD min/median/max = {lr.min():.1f}/{np.median(lr):.1f}/{lr.max():.1f} A")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize whether a WE run funnels toward the native interface.

Reads a compute_metrics .npz (per-segment lrmsd/irmsd/contact_energy + n_iter +
weight) and reports, per iteration: the best (min) i-RMSD/L-RMSD reached, the
weight carried by near-native walkers, and the running cumulative min. The
cumulative-min trend is the cleanest funnel signal: it should fall below the
seeded starting band over the run.

    python -m analysis.funnel_trend --metrics analysis_out/pilot_validation_tica_metrics.npz \
        --near 2.0 --col irmsd
"""
import argparse

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", required=True)
    p.add_argument("--col", default="irmsd", choices=["irmsd", "lrmsd"])
    p.add_argument("--near", type=float, default=2.0, help="near-native cutoff (A)")
    p.add_argument("--csv", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    d = np.load(a.metrics)
    val = d[a.col].reshape(-1)
    it = d["n_iter"].reshape(-1)
    w = d["weight"].reshape(-1).astype(np.float64)
    w = np.where(np.isfinite(w), w, 0.0)

    iters = np.unique(it)
    run_min = np.inf
    rows = []
    for n in iters:
        m = it == n
        v, wn = val[m], w[m]
        run_min = min(run_min, float(v.min()))
        near = v <= a.near
        rows.append((int(n), float(v.min()), run_min,
                     float(np.average(v, weights=wn) if wn.sum() else v.mean()),
                     float(wn[near].sum()), int(near.sum())))

    hdr = f"{'iter':>5} {'min':>7} {'cummin':>7} {'wmean':>7} {'w<=near':>9} {'n<=near':>7}"
    print(f"# {a.col}  near={a.near} A   ({len(val)} segs over {len(iters)} iters)")
    print(hdr)
    # print a sampled set of rows so it stays readable
    step = max(1, len(rows) // 25)
    for i, r in enumerate(rows):
        if i % step == 0 or i == len(rows) - 1:
            print(f"{r[0]:5d} {r[1]:7.2f} {r[2]:7.2f} {r[3]:7.2f} {r[4]:9.2e} {r[5]:7d}")

    first, last = rows[0], rows[-1]
    glob_min = min(r[1] for r in rows)
    peak_w = max(r[4] for r in rows)
    print(f"\nstart band min {a.col} = {first[1]:.2f} A (iter {first[0]})")
    print(f"global best  {a.col} = {glob_min:.2f} A")
    print(f"final cummin {a.col} = {last[2]:.2f} A (iter {last[0]})")
    print(f"max weight within {a.near} A in any iter = {peak_w:.2e}")
    verdict = "FUNNELS toward native" if glob_min < first[1] - 1.0 else "no clear funnel"
    print(f"=> {verdict}: improved {first[1] - glob_min:+.2f} A from start")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["iter", "min", "cummin", "wmean", "w_near", "n_near"])
            wcsv.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()

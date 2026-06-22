#!/usr/bin/env python3
"""VALIDATION seeder: perturb a known complex apart into N clash-free encounter poses.

Each group gets an independent random Euler rotation (+/- rmax) and translation
(+/- tmax) about its centroid, reproducing tools/randomize_chains.py but emitting
many basis states with separation/clash filters.

    python -m seeders.randomize_seeder complex.pdb --sim-root . \
        --groups AB C --n-states 16 --tmax 30 --rmax 60 --min-sep 6 --max-sep 25
"""
import argparse

import numpy as np

from seeders import seed_common as sc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(p)
    p.add_argument("--tmax", type=float, default=30.0, help="Max |translation| per axis (A).")
    p.add_argument("--rmax", type=float, default=60.0, help="Max |rotation| per axis (deg).")
    return p.parse_args()


def make_pose(base, ctx, rng, tmax, rmax):
    pos = base.copy()
    deltas = []
    for gi, idx in enumerate(ctx["all_idx"]):
        t = rng.uniform(-tmax, tmax, size=3)
        r = rng.uniform(-rmax, rmax, size=3)
        sc.apply_rigid(pos, idx, sc.euler_matrix(*r), t)
        deltas.append({"group_index": gi, "chain_ids": ctx["groups"][gi],
                       "translation_A": t.tolist(), "rotation_deg": r.tolist()})
    return pos, deltas


def main():
    args = parse_args()
    ctx = sc.prepare(args)
    rng = np.random.default_rng(args.seed)

    seeds, tries = [], 0
    while len(seeds) < args.n_states and tries < args.max_tries * args.n_states:
        tries += 1
        pos, deltas = make_pose(ctx["positions_A"], ctx, rng, args.tmax, args.rmax)
        ok, sep = sc.accept_pose(pos, ctx["heavy_idx"], args.clash_cutoff,
                                 args.min_sep, args.max_sep)
        if not ok:
            continue
        ncontacts = sc.count_close_contacts(pos, ctx["heavy_idx"][0],
                                            ctx["heavy_idx"][1], args.contact_cutoff)
        seeds.append({"topology": ctx["topology"], "positions_A": pos,
                      "record": {"mode": "validation", "sep01_A": sep,
                                 "n_contacts": ncontacts, "deltas": deltas}})
        print(f"  state {len(seeds):3d}/{args.n_states}: sep01={sep:.2f} A, "
              f"contacts={ncontacts} (try {tries})")

    sc.write_bstates(args.sim_root, seeds, label_prefix=args.label_prefix)


if __name__ == "__main__":
    main()

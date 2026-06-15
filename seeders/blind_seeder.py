#!/usr/bin/env python3
"""BLIND seeder: build N encounter poses with no memory of the native interface.

Group 0 is the anchor. Every other group is given a uniform-random SO(3)
orientation and approached from a random direction until its surface gap to the
already-placed assembly falls in [--gap-min, --gap-max]. Generalizes to multimers
(groups are placed one at a time around the growing assembly).

    python -m seeders.blind_seeder complex.pdb --sim-root . \
        --groups A B --n-states 32 --gap-min 5 --gap-max 12
"""
import argparse

import numpy as np

from seeders import seed_common as sc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sc.add_common_args(p)
    p.add_argument("--gap-min", type=float, default=5.0, help="Min target surface gap (A).")
    p.add_argument("--gap-max", type=float, default=12.0, help="Max target surface gap (A).")
    p.add_argument("--bisect-iters", type=int, default=30)
    p.add_argument("--span-A", type=float, default=400.0, help="Far bracket for approach (A).")
    return p.parse_args()


def rand_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _set_centroid(pos, idx, target_center):
    pos[idx] += target_center - pos[idx].mean(axis=0)


def place_at_gap(pos, placed_heavy, g_all, g_heavy, R, u, target_gap, args):
    """Rotate group by R, then bisect its offset along u from the assembly center
    until the inter-group surface gap matches target_gap. min_inter_distance is
    monotonic in offset, so bisection is well-posed."""
    coords = pos[g_all]
    c = coords.mean(axis=0)
    pos[g_all] = (coords - c) @ R.T + c
    center = pos[placed_heavy].mean(axis=0)

    lo, hi = 0.0, args.span_A
    for _ in range(args.bisect_iters):
        mid = 0.5 * (lo + hi)
        _set_centroid(pos, g_all, center + mid * u)
        if sc.min_inter_distance(pos, placed_heavy, g_heavy) > target_gap:
            hi = mid
        else:
            lo = mid
    _set_centroid(pos, g_all, center + hi * u)
    return sc.min_inter_distance(pos, placed_heavy, g_heavy)


def make_pose(ctx, rng, args):
    pos = ctx["positions_A"].copy()
    placed_heavy = ctx["heavy_idx"][0].copy()
    for gi in range(1, len(ctx["all_idx"])):
        target = rng.uniform(args.gap_min, args.gap_max)
        place_at_gap(pos, placed_heavy, ctx["all_idx"][gi], ctx["heavy_idx"][gi],
                     sc.random_rotation_matrix(rng), rand_unit(rng), target, args)
        placed_heavy = np.concatenate([placed_heavy, ctx["heavy_idx"][gi]])
    return pos


def main():
    args = parse_args()
    ctx = sc.prepare(args)
    rng = np.random.default_rng(args.seed)

    seeds, tries = [], 0
    while len(seeds) < args.n_states and tries < args.max_tries * args.n_states:
        tries += 1
        pos = make_pose(ctx, rng, args)
        ok, sep = sc.accept_pose(pos, ctx["heavy_idx"], args.clash_cutoff,
                                 args.min_sep, args.max_sep)
        if not ok:
            continue
        ncontacts = sc.count_close_contacts(pos, ctx["heavy_idx"][0],
                                            ctx["heavy_idx"][1], args.contact_cutoff)
        seeds.append({"topology": ctx["topology"], "positions_A": pos,
                      "record": {"mode": "blind", "sep01_A": sep,
                                 "n_contacts": ncontacts}})
        print(f"  state {len(seeds):3d}/{args.n_states}: sep01={sep:.2f} A, "
              f"contacts={ncontacts} (try {tries})")

    sc.write_bstates(args.sim_root, seeds, label_prefix=args.label_prefix)


if __name__ == "__main__":
    main()

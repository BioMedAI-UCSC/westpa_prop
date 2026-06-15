#!/usr/bin/env python3
"""Offline validation of learned-PC embeddings on existing WE data.

Loads frames + RMSD labels from a run, featurizes, fits each requested
embedding, and scores how well the latent coordinate tracks proximity to the
native state (|Spearman| of best latent dim vs RMSD; silhouette of near vs far).
No simulations are launched.

    python -m analysis.embedding_offline \
        --sim-root /data/alex/pd1/contact_interface \
        --topology /data/alex/pd1/3BIK_random1Filtered.pdb \
        --sel-a "chainid 0" --sel-b "chainid 1" \
        --methods pca tica cvae vampnet --iter-stride 5 --seg-stride 3
"""
import argparse
import json
import os

import numpy as np

from features.interface_featurizer import InterfaceFeaturizer
from embeddings import build
from analysis.frame_loader import load_segments, featurize_segments


def score_latent(Z, rmsd, near_quantile=0.25):
    from scipy.stats import spearmanr
    rho = [abs(spearmanr(Z[:, j], rmsd).correlation) for j in range(Z.shape[1])]
    best = int(np.nanargmax(rho))
    try:
        from sklearn.metrics import silhouette_score
        thr = np.quantile(rmsd, near_quantile)
        labels = (rmsd <= thr).astype(int)
        sil = float(silhouette_score(Z, labels)) if len(set(labels)) > 1 else float("nan")
    except Exception:
        sil = float("nan")
    return {"best_dim": best, "spearman_abs": float(np.nanmax(rho)),
            "spearman_per_dim": [float(x) for x in rho], "silhouette_near_far": sil}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-root", required=True)
    p.add_argument("--topology", required=True)
    p.add_argument("--sel-a", required=True)
    p.add_argument("--sel-b", required=True)
    p.add_argument("--methods", nargs="+", default=["pca", "tica", "cvae", "vampnet"])
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--iter-stride", type=int, default=5)
    p.add_argument("--seg-stride", type=int, default=3)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--out", default="analysis_out/embedding_offline.json")
    return p.parse_args()


def main():
    args = parse_args()
    map_feat = InterfaceFeaturizer(args.topology, args.sel_a, args.sel_b, mode="contact_map")
    vec_feat = InterfaceFeaturizer(args.topology, args.sel_a, args.sel_b, mode="vector")

    print("loading frames ...")
    segs, rms, lens = load_segments(args.sim_root, args.topology,
                                    iter_stride=args.iter_stride,
                                    seg_stride=args.seg_stride, max_iters=args.max_iters)
    n = sum(lens)
    print(f"  {len(segs)} segments, {n} frames")
    rmsd = np.concatenate(rms)

    feats = {}
    results = {"n_frames": int(n), "n_segments": len(segs),
               "rmsd_range": [float(rmsd.min()), float(rmsd.max())], "methods": {}}

    for m in args.methods:
        kw = dict(n_components=args.n_components)
        if m in ("tica", "vampnet"):
            kw["lag"] = args.lag
        if m in ("cvae", "vampnet"):
            kw["epochs"] = args.epochs
        feat_kind = "map" if m == "cvae" else "vector"
        if feat_kind not in feats:
            fz = map_feat if feat_kind == "map" else vec_feat
            feats[feat_kind] = featurize_segments(fz, segs)
        X = feats[feat_kind]
        try:
            emb = build(m, **kw)
            Z = emb.fit_transform(X, lengths=lens)
            sc = score_latent(Z, rmsd)
            results["methods"][m] = sc
            print(f"  [{m}] |spearman|={sc['spearman_abs']:.3f} "
                  f"silhouette={sc['silhouette_near_far']:.3f}")
        except Exception as e:
            results["methods"][m] = {"error": repr(e)}
            print(f"  [{m}] FAILED: {e}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

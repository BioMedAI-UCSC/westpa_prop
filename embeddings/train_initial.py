#!/usr/bin/env python3
"""Train an initial learned-PC embedding before w_run.

Featurizes a set of structures (bstates, a PDB glob, or a prior run's traj_segs)
and fits the chosen embedding, writing model_path (+ .meta) for LearnedPCoord /
LearnedPCPlugin to consume.

    # from basis-state ensemble (blind: no native info)
    python -m embeddings.train_initial --topology complex.pdb \
        --sel-a "chainid 0" --sel-b "chainid 1" \
        --pdb-glob "bstates/*.pdb" --method tica --model-path models/pc.model

    # from a prior run
    python -m embeddings.train_initial ... --sim-root /path/to/run
"""
import argparse
import glob

import numpy as np
import mdtraj

from features.interface_featurizer import InterfaceFeaturizer
from embeddings import build, save_model, method_kwargs, default_feature_mode
from analysis.frame_loader import load_segments, featurize_segments


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--topology", required=True)
    p.add_argument("--sel-a", required=True)
    p.add_argument("--sel-b", required=True)
    p.add_argument("--method", default="tica")
    p.add_argument("--model-path", required=True)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--feature-mode", default=None)
    p.add_argument("--lag", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-glob", help="Glob of PDBs (e.g. bstates/*.pdb).")
    src.add_argument("--sim-root", help="Prior WESTPA run root (uses traj_segs).")
    p.add_argument("--iter-stride", type=int, default=5)
    p.add_argument("--seg-stride", type=int, default=3)
    return p.parse_args()


def main():
    a = parse_args()
    mode = a.feature_mode or default_feature_mode(a.method)
    fz = InterfaceFeaturizer(a.topology, a.sel_a, a.sel_b, mode=mode)

    if a.pdb_glob:
        segs = [(mdtraj.load(f).xyz * 10.0).astype(np.float32)
                for f in sorted(glob.glob(a.pdb_glob))]
        lens = [len(s) for s in segs]
    else:
        segs, _, lens = load_segments(a.sim_root, a.topology,
                                      iter_stride=a.iter_stride, seg_stride=a.seg_stride)
    if not segs:
        raise SystemExit("No structures found for training.")
    X = featurize_segments(fz, segs)
    emb = build(a.method, **method_kwargs(a.method, a.n_components, a.lag, a.epochs))
    emb.fit(X, lengths=lens)
    save_model(emb, a.method, a.model_path)
    print(f"trained {a.method} on {X.shape[0]} frames -> {a.model_path}")


if __name__ == "__main__":
    main()

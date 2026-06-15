#!/bin/bash
# Phase-3 integration smoke test: run the learned-PC WE loop for each PC method
# for a few iterations on 1 GPU. Validates seeder -> w_init -> w_run with
# LearnedPCoord + retrain plugin + outlier driver.
set -u
ROOT=/data/alex/pd1/learned_pc
TOPO=/data/alex/pd1/3BIKFiltered.pdb
TEMPLATE=$ROOT/configs/west_learned_pc.cfg
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ITERS=${ITERS:-3}
NSTATES=${NSTATES:-8}

for M in pca tica cvae; do
  if [ "$M" = "cvae" ]; then FEAT=contact_map; else FEAT=vector; fi
  RUN=$ROOT/runs/smoke_$M
  echo "############################## $M ($FEAT) -> $RUN"
  rm -rf "$RUN"; mkdir -p "$RUN/models"

  python -m seeders.blind_seeder "$TOPO" --sim-root "$RUN" \
      --groups A B --n-states "$NSTATES" --gap-min 5 --gap-max 12 --seed 1 \
      >"$RUN/seed.log" 2>&1 || { echo "[$M] SEED FAILED"; continue; }

  python - "$TEMPLATE" "$RUN/west.cfg" "$M" "$FEAT" "$ITERS" <<'PY'
import sys
tmpl, out, method, feat, iters = sys.argv[1:6]
s = open(tmpl).read()
s = s.replace("/data/alex/pd1/learned_pc/models/pc.model", "$WEST_SIM_ROOT/models/pc.model")
s = s.replace("/data/alex/pd1/3BIK_random1Filtered.pdb", "/data/alex/pd1/3BIKFiltered.pdb")
s = s.replace("method:       tica", f"method:       {method}")
s = s.replace("feature_mode: vector", f"feature_mode: {feat}")
s = s.replace("retrain_period: 5", "retrain_period: 2")
s = s.replace("warmup:       5", "warmup:       2")
s = s.replace("epochs:       30", "epochs:       8")
s = s.replace("max_total_iterations: 1000", f"max_total_iterations: {iters}")
s = s.replace("num_gpus: 4", "num_gpus: 1")
open(out, "w").write(s)
PY

  python -m embeddings.train_initial --topology "$TOPO" \
      --sel-a "chainid 0" --sel-b "chainid 1" --pdb-glob "$RUN/bstates/*.pdb" \
      --method pca --feature-mode "$FEAT" --model-path "$RUN/models/pc.model" \
      >"$RUN/train_init.log" 2>&1 || { echo "[$M] TRAIN_INIT FAILED"; continue; }

  ( cd "$RUN" && export WEST_SIM_ROOT="$RUN" && \
    w_init --bstates-from bstates.txt --segs-per-state 1 >w_init.log 2>&1 && \
    w_run >w_run.log 2>&1 ) || { echo "[$M] W_RUN FAILED (see $RUN/w_run.log)"; tail -15 "$RUN/w_run.log"; continue; }

  python - "$RUN" "$M" <<'PY'
import sys, h5py, numpy as np, os, glob
run, method = sys.argv[1:3]
h = h5py.File(os.path.join(run, "west.h5"), "r")
its = sorted(k for k in h["iterations"])
last = h["iterations"][its[-1]]
pc = np.array(last["pcoord"])
seg = last["seg_index"]
w = np.array([s["weight"] for s in seg])
print(f"[{method}] iters={len(its)} last_nsegs={pc.shape[0]} pcoord_dim={pc.shape[-1]} "
      f"weight_sum={w.sum():.4f} pcoord0={np.round(pc[0,-1],3)}")
meta = glob.glob(os.path.join(run, "models", "pc.model.meta"))
if meta:
    import json; print(f"[{method}] current model method:", json.load(open(meta[0]))["method"])
PY
  echo "[$M] OK"
done
echo "ALL SMOKE DONE"

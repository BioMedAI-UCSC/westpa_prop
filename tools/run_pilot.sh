#!/bin/bash
# Phase-4 pilot: one blind-mode discovery run to test whether Stage-1 drives
# walkers toward native-like interfaces (i-RMSD down / contact_energy down).
set -u
ROOT=/data/alex/pd1/learned_pc
TOPO=/data/alex/pd1/3BIKFiltered.pdb
TEMPLATE=$ROOT/configs/west_learned_pc.cfg
export PYTHONPATH=$ROOT:${PYTHONPATH:-}
export HDF5_USE_FILE_LOCKING=0
METHOD=${METHOD:-tica}
ITERS=${ITERS:-50}
NSTATES=${NSTATES:-24}
NTRAIN=${NTRAIN:-128}
NWORKERS=${NWORKERS:-4}
FEAT=vector; [ "$METHOD" = "cvae" ] && FEAT=contact_map
RUN=${RUN:-$ROOT/runs/pilot_${MODE:-blind}_$METHOD}
echo "pilot: $METHOD ($FEAT) iters=$ITERS states=$NSTATES -> $RUN"
rm -rf "$RUN"; mkdir -p "$RUN/models" "$RUN/seedpool"

# run bstates + a larger pool for a non-degenerate initial model
MODE=${MODE:-blind}            # blind | validation
TMAX=${TMAX:-8}; RMAX=${RMAX:-15}; MINSEP=${MINSEP:-2}; MAXSEP=${MAXSEP:-20}
seed_into() {  # $1=dir $2=n_states $3=seed
  if [ "$MODE" = "validation" ]; then
    python -m seeders.randomize_seeder "$TOPO" --sim-root "$1" --groups A B \
      --n-states "$2" --tmax "$TMAX" --rmax "$RMAX" --min-sep "$MINSEP" \
      --max-sep "$MAXSEP" --seed "$3"
  else
    python -m seeders.blind_seeder "$TOPO" --sim-root "$1" --groups A B \
      --n-states "$2" --gap-min 5 --gap-max 12 --seed "$3"
  fi
}
seed_into "$RUN" "$NSTATES" 1 >"$RUN/seed.log" 2>&1
seed_into "$RUN/seedpool" "$NTRAIN" 100 >"$RUN/seedpool.log" 2>&1

python - "$TEMPLATE" "$RUN/west.cfg" "$METHOD" "$FEAT" "$ITERS" <<'PY'
import sys
tmpl, out, method, feat, iters = sys.argv[1:6]
s = open(tmpl).read()
s = s.replace("/data/alex/pd1/learned_pc/models/pc.model", "$WEST_SIM_ROOT/models/pc.model")
s = s.replace("/data/alex/pd1/3BIK_random1Filtered.pdb", "/data/alex/pd1/3BIKFiltered.pdb")
s = s.replace("method:       tica", f"method:       {method}")
s = s.replace("feature_mode: vector", f"feature_mode: {feat}")
s = s.replace("iter_stride:  1", "iter_stride:  2")
s = s.replace("seg_stride:   1", "seg_stride:   4")
s = s.replace("window:       20", "window:       10")
s = s.replace("bin_target_counts: 6", "bin_target_counts: 4")
s = s.replace("max_total_iterations: 1000", f"max_total_iterations: {iters}")
# Drop in-run recorded calculators: under MPI, parallel workers writing aux to the
# shared west.h5 contend. Validation metrics are computed post-hoc from DCDs.
s = s.split("\n    recorded_calculators:")[0].rstrip() + "\n"
open(out, "w").write(s)
PY

python -m embeddings.train_initial --topology "$TOPO" --sel-a "chainid 0" --sel-b "chainid 1" \
    --pdb-glob "$RUN/seedpool/bstates/*.pdb" --method pca --feature-mode "$FEAT" \
    --model-path "$RUN/models/pc.model" >"$RUN/train_init.log" 2>&1

cd "$RUN" && export WEST_SIM_ROOT="$RUN"
w_init --bstates-from bstates.txt --segs-per-state 1 >w_init.log 2>&1
echo "w_init done; starting w_run ($NWORKERS workers)"
# MPI (separate processes) avoids the CUDA-after-fork failure that breaks the
# processes/zmq work managers once libcuda is loaded in the master.
if [ "${WM:-mpi}" = "mpi" ]; then
  mpirun -np $((NWORKERS + 1)) w_run --work-manager mpi >w_run.log 2>&1
else
  w_run --work-manager "${WM}" --n-workers "$NWORKERS" >w_run.log 2>&1
fi
echo "PILOT $METHOD DONE (exit $?)"

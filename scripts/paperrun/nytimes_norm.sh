#!/bin/bash
# nytimes rerun on the angular-ready copy (zero vectors dropped, unit length;
# see tools/normalize_fvecs.py). Same six-experiment layout as paper_run_ads.sh,
# once per backend; tags nyn_<backend>_..., faiss baseline first so the
# groundtruth is written by a single process before the MPI runs read it.
cd ~/harmony-o || exit 1
DS=nytimes_norm
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
LOG=~/paperrun/logs
mkdir -p "$LOG"

for np in 16 32 64 128; do
  tag=nyn_faiss_np$np
  echo "### $tag"
  timeout 3600 mpirun -n 1 release/bin/query --benchmarks_path ./benchmarks \
      --dataset $DS --k 100 --cache --nprobes $np --run_faiss > "$LOG/$tag.log" 2>&1
  echo "exit=$?"; grep -E "Faiss nprobe" "$LOG/$tag.log" | tail -1
done

run () {
  tag=$1; shift
  echo "### $tag  $*"
  timeout 3600 $M release/bin/query --benchmarks_path ./benchmarks \
      --dataset $DS --k 100 --cache "$@" > "$LOG/$tag.log" 2>&1
  echo "exit=$?"
  grep -E "^HSTATS|^HPRUNE" "$LOG/$tag.log"
}

for be in harmony adsampling; do
  p=nyn_${be}; B="--backend $be"
  for np in 16 32 64 128; do run ${p}_1x4_np$np $B --team 1 --group 2 --block 2 --nprobes $np; done
  for np in 16 32 64 128; do run ${p}_2x2_np$np $B --team 2 --group 2 --block 2 --nprobes $np; done
  for np in 16 32 64 128; do run ${p}_4x1_np$np $B --team 4 --group 4 --block 2 --nprobes $np; done
  run ${p}_abl_blocksend $B --team 1 --group 2 --block 2 --nprobes 32 --blockSend
  run ${p}_abl_noprune   $B --team 1 --group 2 --block 2 --nprobes 32 --disablePruning
  run ${p}_abl_noorder   $B --team 1 --group 2 --block 2 --nprobes 32 --disableOrderOpt
  for r in 0.0 0.4 0.8; do
    run ${p}_skew_w$r $B --team 2 --group 2 --block 2 --nprobes 32 \
        --HardInBalance --HardInBalanceTeam 2 --HardInBalanceRatio $r
  done
done
rm -rf benchmarks/$DS/index
echo "ALL DONE"; df -h ~ | tail -1

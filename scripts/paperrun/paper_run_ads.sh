#!/bin/bash
# The same six paper experiments, with the ADSampling backend instead of the
# exact one. Tags are prefixed ads_ so they do not collide with the baseline
# logs, and the rows land in the same measurements/*.csv with backend
# "adsampling" in their own column.
cd ~/harmony-o || exit 1

M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
LOG=~/paperrun/logs
mkdir -p "$LOG"
B="--backend adsampling"   # --adeps 2.1 and --addelta 32 are the authors' defaults

run () {
  tag=$1; shift
  echo "### $tag  $*"
  timeout 7200 $M release/bin/query --benchmarks_path ./benchmarks \
      --k 100 --cache $B "$@" > "$LOG/$tag.log" 2>&1
  echo "exit=$?"
  grep -E "^HSTATS" "$LOG/$tag.log"
  grep -E "^HPRUNE" "$LOG/$tag.log"
}

dataset_runs () {
  ds=$1
  echo "======================================== $ds"
  for np in 16 32 64 128; do run ads_${ds}_1x4_np$np --dataset $ds --team 1 --group 2 --block 2 --nprobes $np; done
  for np in 16 32 64 128; do run ads_${ds}_2x2_np$np --dataset $ds --team 2 --group 2 --block 2 --nprobes $np; done
  for np in 16 32 64 128; do run ads_${ds}_4x1_np$np --dataset $ds --team 4 --group 4 --block 2 --nprobes $np; done

  run ads_${ds}_abl_blocksend --dataset $ds --team 1 --group 2 --block 2 --nprobes 32 --blockSend
  run ads_${ds}_abl_noprune   --dataset $ds --team 1 --group 2 --block 2 --nprobes 32 --disablePruning
  run ads_${ds}_abl_noorder   --dataset $ds --team 1 --group 2 --block 2 --nprobes 32 --disableOrderOpt

  for r in 0.0 0.4 0.8; do
    run ads_${ds}_skew_w$r --dataset $ds --team 2 --group 2 --block 2 --nprobes 32 \
        --HardInBalance --HardInBalanceTeam 2 --HardInBalanceRatio $r
  done

  echo "### $ds done, freeing the index"
  rm -rf benchmarks/$ds/index
  df -h ~ | tail -1
}

for ds in "$@"; do
  dataset_runs $ds
done
echo "ALL DONE"

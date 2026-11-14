#!/bin/bash
cd ~/harmony-o || exit 1
mkdir -p results logs
Q="./release/bin/query --benchmarks_path ./benchmarks --dataset gist --cache"
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
H="--team 2 --group 2 --block 2"   # Harmony            2x2
D="--team 1 --group 2 --block 2"   # Harmony-dimension  1x4
V="--team 4 --group 4 --block 2"   # Harmony-vector     4x1

run() {
  tag=$1; shift
  printf "### %-22s %s  " "$tag" "$(date +%H:%M:%S)"
  timeout 420 $M $Q "$@" --csv ./results/$tag.csv > logs/$tag.log 2>&1
  rc=$?
  [ $rc -ne 0 ] && printf "FAILED(rc=%s) " $rc
  grep -hoE "query_time: [0-9.]+|recall: [0-9.]+" logs/$tag.log | tr '\n' ' '
  echo
}

echo "===== 6.2.1 QPS-recall ====="
for np in 1 4 16 32 64 128; do
  run 621_harmony_$np   $H --nprobes $np
  run 621_dimension_$np $D --nprobes $np
  run 621_vector_$np    $V --nprobes $np
done

echo "===== 6.3.2 optimization ladder ====="
run 632_L0_nonopt    $V --nprobes 32 --blockSend --disablePruning
run 632_L1_pipeline  $V --nprobes 32 --disablePruning
run 632_L2_balanced  $H --nprobes 32 --disablePruning
run 632_L3_pruning   $H --nprobes 32

echo "===== 6.2.2 skew ====="
for r in 0.0 0.33 0.66 1.0; do
  run 622_harmony_$r   $H --nprobes 32 --HardInBalance --HardInBalanceTeam 2 --HardInBalanceRatio $r
  run 622_dimension_$r $D --nprobes 32 --HardInBalance --HardInBalanceTeam 2 --HardInBalanceRatio $r
  run 622_vector_$r    $V --nprobes 32 --HardInBalance --HardInBalanceTeam 2 --HardInBalanceRatio $r
done
echo "===== DONE ====="

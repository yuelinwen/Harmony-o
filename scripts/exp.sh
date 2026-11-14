#!/bin/bash
# harmony-o on gist: reproduce paper sections 6.2.1, 6.2.2, 6.3.2
cd ~/harmony-o || exit 1
mkdir -p results
Q="./release/bin/query --benchmarks_path ./benchmarks --dataset gist --cache"
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
NP="1 2 4 8 16 32 64 128"

# Paper's three variants:
#   Harmony           = hybrid    2x2  --team 2
#   Harmony-dimension = dim only  1x4  --team 1
#   Harmony-vector    = vec only  4x1  --team 4
H="--team 2 --group 2 --block 2"
D="--team 1 --group 2 --block 2"
V="--team 4 --group 4 --block 2"

echo "########## 6.2.1 QPS-recall trade-off ##########"
for cfg in "harmony:$H" "dimension:$D" "vector:$V"; do
  name="${cfg%%:*}"; args="${cfg#*:}"
  echo "===== 6.2.1 $name ====="
  $M $Q $args --nprobes $NP --csv ./results/621_$name.csv 2>&1 \
    | grep -E "nprobe:|recall:|Error|error"
done

echo "########## 6.3.2 Contribution of optimizations (nprobe 32) ##########"
echo "===== L0 non-optimize (vector + blocking send + no pruning) ====="
$M $Q $V --nprobes 32 --blockSend --disablePruning --csv ./results/632_L0.csv 2>&1 | grep -E "nprobe:|recall:|Error"
echo "===== L1 +pipeline/async ====="
$M $Q $V --nprobes 32 --disablePruning --csv ./results/632_L1.csv 2>&1 | grep -E "nprobe:|recall:|Error"
echo "===== L2 +balanced load (hybrid partition) ====="
$M $Q $H --nprobes 32 --disablePruning --csv ./results/632_L2.csv 2>&1 | grep -E "nprobe:|recall:|Error"
echo "===== L3 +pruning ====="
$M $Q $H --nprobes 32 --csv ./results/632_L3.csv 2>&1 | grep -E "nprobe:|recall:|Error"

echo "########## 6.2.2 Skewed workloads (nprobe 32) ##########"
for r in 0.0 0.25 0.5 0.75 1.0; do
  for cfg in "harmony:$H" "dimension:$D" "vector:$V"; do
    name="${cfg%%:*}"; args="${cfg#*:}"
    echo "===== 6.2.2 $name ratio=$r ====="
    $M $Q $args --nprobes 32 --HardInBalance --HardInBalanceTeam 2 \
      --HardInBalanceRatio $r --csv ./results/622_$name.csv 2>&1 \
      | grep -E "nprobe:|recall:|Error|error"
  done
done

echo "########## DONE ##########"
df -h ~ | tail -1

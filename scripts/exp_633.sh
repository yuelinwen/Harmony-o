#!/bin/bash
cd ~/harmony-o
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
B="release/bin/query --benchmarks_path ./benchmarks --dataset gist --group 2 --block 2 --k 100 --cache"

run () {  # name, extra args...
  n=$1; shift
  echo "=== $n ==="
  timeout 1800 $M $B "$@" > /tmp/p633_$n.log 2>&1
  echo "exit=$?"
  grep -E "HSTATS|HPRUNE" /tmp/p633_$n.log
}

for np in 16 32 64 128; do
  run t1_np$np --team 1 --nprobes $np
done
run t2_np32 --team 2 --nprobes 32
run t4_np32 --team 4 --group 4 --nprobes 32

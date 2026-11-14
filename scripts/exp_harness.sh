#!/bin/bash
cd ~/harmony-o
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
B="release/bin/query --benchmarks_path ./benchmarks --dataset gist --k 100 --cache"
run () { n=$1; shift
  echo "=== $n ==="
  timeout 1800 $M $B "$@" > /tmp/h_$n.log 2>&1
  echo "exit=$?"
  grep -E "HSTATS|HPRUNE" /tmp/h_$n.log
}
for np in 16 32 64 128; do run 1x4_np$np --team 1 --group 2 --block 2 --nprobes $np; done
for np in 16 32 64 128; do run 2x2_np$np --team 2 --group 2 --block 2 --nprobes $np; done
for np in 16 32 64 128; do run 4x1_np$np --team 4 --group 4 --block 2 --nprobes $np; done

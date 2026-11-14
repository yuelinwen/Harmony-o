#!/bin/bash
cd ~/harmony-o
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
Q="release/bin/query --benchmarks_path ./benchmarks --dataset gist --team 1 --group 2 --block 2 --nprobes 32 --k 100 --cache"

run () {  # name, extra args...
  n=$1; shift
  echo "=== $n ==="
  timeout 1800 $M $Q "$@" --dumpLabels /tmp/lab_$n.bin > /tmp/ads_$n.log 2>&1
  echo "exit=$?"
  grep -E "HSTATS|rotated" /tmp/ads_$n.log
}

run exact
run rot1 --rotate 1
run eps15 --rotate 1 --adsampling --adeps 1.5
run eps21 --rotate 1 --adsampling --adeps 2.1
run eps30 --rotate 1 --adsampling --adeps 3.0

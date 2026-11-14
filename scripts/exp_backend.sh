#!/bin/bash
cd ~/harmony-o
M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"
B="release/bin/query --benchmarks_path ./benchmarks --dataset gist --group 2 --block 2 --k 100 --cache --team 1 --nprobes 32"
echo "=== bad name ==="; $M $B --backend bogus 2>&1 | grep -i "unknown --backend" | head -1
for b in harmony rotation adsampling; do
  echo "=== $b ==="
  timeout 1800 $M $B --backend $b > /tmp/bk_$b.log 2>&1; echo "exit=$?"
  grep -E "^node .* backend" /tmp/bk_$b.log | sort -u | head -4
  grep -E "HSTATS|HPRUNE" /tmp/bk_$b.log
done

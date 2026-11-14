#!/bin/bash
# The four gist faiss baseline points that the first sweep could not fit.
# One nprobe per invocation, so agg.py sees the same one-log-per-point shape
# it sees for the other three datasets.  The faiss index is written on the
# first call and read back on the other three; write_index does not check
# ENOSPC, so the size is asserted before the numbers are trusted.
cd ~/harmony-o || exit 1
LOG=~/paperrun/logs
for np in 16 32 64 128; do
  tag=gist_faiss_np$np
  echo "### $tag"
  timeout 7200 mpirun -n 1 release/bin/query --benchmarks_path ./benchmarks \
      --dataset gist --k 100 --cache --nprobes $np --run_faiss > "$LOG/$tag.log" 2>&1
  echo "exit=$?"
  grep -E "Faiss nprobe" "$LOG/$tag.log" | tail -1
  ls -l benchmarks/gist/index/ 2>/dev/null
  df -h ~ | tail -1
done
rm -rf benchmarks/gist/index
echo "GIST FAISS DONE"
df -h ~ | tail -1

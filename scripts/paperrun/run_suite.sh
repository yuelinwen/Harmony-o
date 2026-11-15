#!/bin/bash
# Full paper-run suite for one backend on the four datasets, against the
# ann-benchmarks groundtruth (tools/official_gt.py). Everything a run produces
# goes to ~/expirementresult/<YYYYMMDD_HHMM local>_<label>/.
# usage: run_suite.sh BACKEND LABEL      e.g. run_suite.sh harmony baseline
BE=$1; LABEL=$2
cd ~/harmony-o || exit 1
STAMP=$(TZ=America/Toronto date +%Y%m%d_%H%M)
OUT=~/expirementresult/${STAMP}_${LABEL}
LOG=$OUT/logs
mkdir -p "$LOG"
cp "$0" "$OUT/run_suite.sh"
echo "$OUT"

M="mpirun -n 5 -bind-to none --host localhost,vdb2,vdb3,vdb4,vdb5"

# all five nodes must run the same binary
{
  echo "binary md5 per node:"
  echo "vdb1 $(md5sum release/bin/query | cut -c1-32)"
  for h in vdb2 vdb3 vdb4 vdb5; do echo "$h $(ssh $h md5sum harmony-o/release/bin/query | cut -c1-32)"; done
  echo "groundtruth md5:"
  md5sum benchmarks/{mnist,nytimes_norm,sift,gist}/result/groundtruth_100.bin
} > "$OUT/env.txt"
if [ "$(grep -cE '^vdb' "$OUT/env.txt")" != 5 ] || [ "$(awk '/^vdb/{print $2}' "$OUT/env.txt" | sort -u | wc -l)" != 1 ]; then
  echo "BINARY MISMATCH, see $OUT/env.txt"; exit 1
fi

# measurements/*.csv are appended to by every run; remember where we start
for f in measurements/*.csv; do wc -l < "$f" > "$OUT/.start_$(basename "$f")"; done

run () {
  tag=$1; shift
  echo "### $tag  $*"
  timeout 7200 $M release/bin/query --benchmarks_path ./benchmarks \
      --k 100 --cache --backend $BE "$@" > "$LOG/$tag.log" 2>&1
  echo "exit=$?"
  grep -E "^HSTATS|^HPRUNE" "$LOG/$tag.log"
}

for ds in mnist nytimes_norm sift gist; do
  echo "======================================== $ds $(date +%H:%M:%S)"
  rm -rf benchmarks/$ds/index
  for np in 16 32 64 128; do
    tag=${ds}_faiss_np$np
    echo "### $tag"
    timeout 7200 mpirun -n 1 release/bin/query --benchmarks_path ./benchmarks \
        --dataset $ds --k 100 --cache --nprobes $np --run_faiss > "$LOG/$tag.log" 2>&1
    echo "exit=$?"; grep -E "Faiss nprobe" "$LOG/$tag.log" | tail -1
  done
  rm -rf benchmarks/$ds/index   # gist cannot hold the faiss and harmony index at once
  for np in 16 32 64 128; do run ${ds}_1x4_np$np --dataset $ds --team 1 --group 2 --block 2 --nprobes $np; done
  for np in 16 32 64 128; do run ${ds}_2x2_np$np --dataset $ds --team 2 --group 2 --block 2 --nprobes $np; done
  for np in 16 32 64 128; do run ${ds}_4x1_np$np --dataset $ds --team 4 --group 4 --block 2 --nprobes $np; done
  run ${ds}_abl_blocksend --dataset $ds --team 1 --group 2 --block 2 --nprobes 32 --blockSend
  run ${ds}_abl_noprune   --dataset $ds --team 1 --group 2 --block 2 --nprobes 32 --disablePruning
  run ${ds}_abl_noorder   --dataset $ds --team 1 --group 2 --block 2 --nprobes 32 --disableOrderOpt
  for r in 0.0 0.4 0.8; do
    run ${ds}_skew_w$r --dataset $ds --team 2 --group 2 --block 2 --nprobes 32 \
        --HardInBalance --HardInBalanceTeam 2 --HardInBalanceRatio $r
  done
  rm -rf benchmarks/$ds/index
  df -h ~ | tail -1
done

# keep this suite's rows of every measurements csv (header + new rows)
mkdir -p "$OUT/measurements"
for f in measurements/*.csv; do
  n=$(cat "$OUT/.start_$(basename "$f")")
  { head -1 "$f"; tail -n +$((n + 1)) "$f"; } > "$OUT/measurements/$(basename "$f")"
done
rm -f "$OUT"/.start_*
echo "ALL DONE $(date +%H:%M:%S)"

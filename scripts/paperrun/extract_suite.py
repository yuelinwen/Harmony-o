"""Collect a run_suite.sh result folder into one JSON (runs on vdb1).
usage: extract_suite.py RESULT_DIR > summary.json
Output: {dataset: {"sweep": {layout: {nprobe: stats}}, "abl": {...},
                   "skew": {...}, "faiss": {nprobe: {recall, qps, time}}, "nq": n}}
stats are the HSTATS fields plus "hprune" (cumulative pruned share per chain
position, from the raw skip/compare counts) and "busy_skew" (from runs.csv,
which is what the result pages plot)."""
import csv
import os
import re
import sys

OUT = sys.argv[1]
LOG = os.path.join(OUT, "logs")
runs = list(csv.DictReader(open(os.path.join(OUT, "measurements", "runs.csv"))))
DATASETS = ("mnist", "nytimes_norm", "sift", "gist")
NPS = (16, 32, 64, 128)


def parse(path):
    txt = open(path).read()
    m = re.search(r"^HSTATS (.*)$", txt, re.M)
    if not m:
        raise SystemExit("no HSTATS in " + path)
    h = {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}
    h["hprune"] = [int(sk) / int(cp) if int(cp) else 0.0
                   for sk, cp in re.findall(r"^HPRUNE .* skip=(\d+) compare=(\d+) ", txt, re.M)]
    rows = [r for r in runs if r["qps"] == "%.3f" % h["qps"] and r["recall"] == "%.6f" % h["recall"]]
    if len(rows) != 1:
        raise SystemExit("%s: %d runs.csv rows match" % (path, len(rows)))
    h["busy_skew"] = float(rows[0]["busy_skew"])
    h["nq"] = int(rows[0]["nq"])
    return h


out = {}
for ds in DATASETS:
    d = {"sweep": {}, "abl": {}, "skew": {}, "faiss": {}}
    for lay in ("1x4", "2x2", "4x1"):
        d["sweep"][lay] = {str(np): parse("%s/%s_%s_np%d.log" % (LOG, ds, lay, np)) for np in NPS}
    for a in ("blocksend", "noprune", "noorder"):
        d["abl"][a] = parse("%s/%s_abl_%s.log" % (LOG, ds, a))
    for r in ("0.0", "0.4", "0.8"):
        d["skew"][r] = parse("%s/%s_skew_w%s.log" % (LOG, ds, r))
    d["nq"] = d["sweep"]["2x2"]["32"]["nq"]
    for np in NPS:
        txt = open("%s/%s_faiss_np%d.log" % (LOG, ds, np)).read()
        m = re.findall(r"Faiss nprobe: (\d+) time: ([\d.]+) recall: ([\d.]+)", txt)
        if not m:
            raise SystemExit("no faiss line for %s np%d" % (ds, np))
        _, t, r = m[-1]
        d["faiss"][str(np)] = {"time": float(t), "recall": float(r), "qps": d["nq"] / float(t)}
    out[ds] = d

import json
json.dump(out, sys.stdout, indent=1)

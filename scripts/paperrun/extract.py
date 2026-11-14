"""Collect one dataset's paper-run numbers into JSON (runs on vdb1).
usage: extract.py BACKEND=LOGPREFIX [...] --faiss-csv PATH --nq N
e.g.   extract.py adsampling=ads_nytimes_ --faiss-csv .../log_faiss.csv --nq 1024"""
import csv, glob, json, os, re, sys

LOG = os.path.expanduser("~/paperrun/logs")
RUNS = os.path.expanduser("~/harmony-o/measurements/runs.csv")
args = sys.argv[1:]
fcsv = args[args.index("--faiss-csv") + 1]
nq = int(args[args.index("--nq") + 1])
pairs = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]

runs = list(csv.DictReader(open(RUNS)))


def parse(path):
    txt = open(path).read()
    m = re.search(r"^HSTATS (.*)$", txt, re.M)
    if not m:
        raise SystemExit("no HSTATS in " + path)
    h = {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}
    h["hprune"] = [int(sk) / int(cp) if int(cp) else 0.0
                   for sk, cp in re.findall(r"^HPRUNE .* skip=(\d+) compare=(\d+) ", txt, re.M)]
    # busy_skew in the pages comes from runs.csv; match the row by its qps string
    q = "%.3f" % h["qps"]
    rows = [r for r in runs if r["qps"] == q and r["recall"] == "%.6f" % h["recall"]]
    if len(rows) != 1:
        raise SystemExit("%s: %d runs.csv rows match qps %s" % (path, len(rows), q))
    h["busy_skew"] = float(rows[0]["busy_skew"])
    return h


out = {}
for be, pre in pairs:
    d = {"sweep": {}, "abl": {}, "skew": {}}
    for lay in ("1x4", "2x2", "4x1"):
        d["sweep"][lay] = {str(np): parse("%s/%s%s_np%d.log" % (LOG, pre, lay, np)) for np in (16, 32, 64, 128)}
    for a in ("blocksend", "noprune", "noorder"):
        d["abl"][a] = parse("%s/%sabl_%s.log" % (LOG, pre, a))
    for r in ("0.0", "0.4", "0.8"):
        d["skew"][r] = parse("%s/%sskew_w%s.log" % (LOG, pre, r))
    out[be] = d
out["faiss"] = {}
for r in csv.DictReader(open(fcsv)):
    out["faiss"][r["nprobe"]] = {"recall": float(r["recall"]), "qps": nq / float(r["time"])}
json.dump(out, sys.stdout, indent=1)

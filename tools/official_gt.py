"""Replace a dataset's self-computed groundtruth with the ann-benchmarks one.

usage: official_gt.py H5 DATASET [--norm-of RAWDATASET]
Checks first that our fvecs are the h5's train/test (bit-exact, or for a
normalized copy, equal after the same normalization), then writes
benchmarks/DATASET/result/groundtruth_100.bin in harmony's layout (int64 ids,
then float32 distances, both nq x 100) with distances as squared L2:
  euclidean h5: official distance is plain L2  -> squared
  angular h5:   official distance is 1 - cos   -> on unit vectors L2^2 = 2(1 - cos)
The previous file is kept as groundtruth_100.self_l2.bin.
"""
import os
import shutil
import sys

import h5py
import numpy as np

K = 100
h5p, ds = sys.argv[1], sys.argv[2]
raw = sys.argv[sys.argv.index("--norm-of") + 1] if "--norm-of" in sys.argv else None
B = os.path.expanduser("~/harmony-o/benchmarks")


def fvecs(p):
    a = np.fromfile(p, dtype=np.float32)
    d = a[:1].view(np.int32)[0]
    return a.reshape(-1, d + 1)[:, 1:]


base = fvecs("%s/%s/origin/%s_base.fvecs" % (B, ds, ds))
qry = fvecs("%s/%s/origin/%s_query.fvecs" % (B, ds, ds))
h = h5py.File(h5p, "r")
train, test = h["train"], h["test"]
nbr, dst = h["neighbors"], h["distances"]
metric = h.attrs.get("distance", "?")
if isinstance(metric, bytes):
    metric = metric.decode()
print("%s: h5 train %s test %s neighbors %s metric %s | ours base %s query %s"
      % (ds, train.shape, test.shape, nbr.shape, metric, base.shape, qry.shape))

rng = np.random.default_rng(0)
zero = None
if raw is None:
    nq = len(qry)
    assert np.array_equal(np.asarray(test[:nq], dtype=np.float32), qry), "query mismatch"
    idx = np.sort(rng.choice(len(base), 2000, replace=False))
    assert len(base) == train.shape[0], "base size mismatch"
    assert np.array_equal(np.asarray(train[idx.tolist()], dtype=np.float32), base[idx]), "base mismatch"
    qsel = np.arange(nq)
    idmap = None
else:
    T = np.asarray(train, dtype=np.float32)
    zero = np.linalg.norm(T, axis=1) == 0
    keep = ~zero
    nb = T[keep] / np.linalg.norm(T[keep], axis=1)[:, None]
    assert nb.shape == base.shape and np.allclose(nb, base, atol=1e-6), "normalized base mismatch"
    rawq = fvecs("%s/%s/origin/%s_query.fvecs" % (B, raw, raw))
    assert np.array_equal(np.asarray(test[:len(rawq)], dtype=np.float32), rawq), "raw query mismatch"
    qn = np.linalg.norm(rawq, axis=1)
    qsel = np.nonzero(qn != 0)[0][:len(qry)]
    assert np.allclose(rawq[qsel] / qn[qsel, None], qry, atol=1e-6), "normalized query mismatch"
    idmap = np.cumsum(keep) - 1  # old id -> new id (valid where keep)
    print("  zero base vectors dropped: %d; h5 query rows used %d..%d; zero query rows skipped: %s"
          % (zero.sum(), qsel[0], qsel[-1], np.nonzero(qn[:qsel[-1] + 1] == 0)[0].tolist()))

I = np.asarray(nbr[qsel.tolist()], dtype=np.int64)[:, :K]
Dh = np.asarray(dst[qsel.tolist()], dtype=np.float64)[:, :K]
if idmap is not None:
    bad = zero[I]
    print("  official neighbours that are zero vectors: %d" % bad.sum())
    assert not bad.any(), "official GT contains zero vectors"
    I = idmap[I]
D = 2.0 * Dh if metric == "angular" else Dh ** 2

# recompute a sample of distances ourselves to confirm the conversion
s = rng.choice(len(qsel), 50, replace=False)
calc = ((base[I[s, :]].astype(np.float64) - qry[s][:, None, :]) ** 2).sum(-1)
err = np.max(np.abs(calc - D[s]) / np.maximum(D[s], 1e-6))
print("  distance check: max rel err %.2e over 50x100 pairs" % err)
assert err < 1e-3, "distance conversion mismatch"

out = "%s/%s/result/groundtruth_%d.bin" % (B, ds, K)
old = out.replace(".bin", ".self_l2.bin")
if os.path.exists(out) and not os.path.exists(old):
    shutil.copy2(out, old)
if os.path.exists(old):
    raw_old = open(old, "rb").read()
    oi = np.frombuffer(raw_old[:len(qsel) * K * 8], dtype=np.int64).reshape(-1, K)
    ov = np.mean([len(set(a.tolist()) & set(b.tolist())) / K for a, b in zip(oi, I)])
    print("  overlap of self-computed L2 groundtruth with official: %.4f" % ov)
with open(out, "wb") as f:
    f.write(I.astype(np.int64).tobytes())
    f.write(D.astype(np.float32).tobytes())
print("  wrote %s  (%d x %d)" % (out, len(qsel), K))

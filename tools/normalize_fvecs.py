"""Make an angular-ready copy of an fvecs dataset: drop all-zero base and query vectors
(cosine is undefined for them), trim queries to a multiple of 8, and scale every remaining base and query vector
to unit length. On unit vectors L2 order equals cosine order, so harmony can
keep its L2 metric and its monotone partial-sum pruning.
usage: normalize_fvecs.py SRC_DIR SRC_NAME DST_DIR DST_NAME
"""
import sys, numpy as np
src, sname, dst, dname = sys.argv[1:5]
def load(p):
    a = np.fromfile(p, dtype=np.float32)
    d = a[:1].view(np.int32)[0]
    return a.reshape(-1, d + 1)[:, 1:].copy(), d
def save(p, x):
    d = x.shape[1]
    out = np.empty((x.shape[0], d + 1), dtype=np.float32)
    out[:, 0] = np.array([d], dtype=np.int32).view(np.float32)[0]
    out[:, 1:] = x
    out.tofile(p)
MULT = 8  # harmony needs nq % (group*block) == 0; 8 covers group<=4, block 2
for part in ("base", "query"):
    x, d = load("%s/%s_%s.fvecs" % (src, sname, part))
    n = np.linalg.norm(x, axis=1)
    zero = n == 0
    x = x[~zero] / n[~zero, None]
    if part == "query":
        x = x[: len(x) // MULT * MULT]  # zero queries removed above; trim to a multiple of MULT
    save("%s/%s_%s.fvecs" % (dst, dname, part), x)
    nn = np.linalg.norm(x, axis=1)
    print("%s: %d -> %d vectors (dropped %d zero), d=%d, norm %.6f..%.6f" % (part, len(n), len(x), zero.sum(), d, nn.min(), nn.max()))

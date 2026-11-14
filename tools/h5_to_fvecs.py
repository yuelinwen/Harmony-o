"""ann-benchmarks hdf5 -> fvecs, the only format harmony-o reads.

fvecs is as simple as it sounds: per vector, a 4-byte int32 saying how many
dimensions follow, then that many float32. No file header, no count -- the
reader divides the file size by (4 + 4*d) to learn n, which is why the size
assertion below is worth having.

Read in chunks rather than whole: gist's train is 3.84 GB and the machine
has to hold the index later. The header is written by reinterpreting the
int32's bits as a float32 so the whole record is one contiguous row and
goes out in a single tofile.
"""
import os
import sys

import h5py
import numpy as np

src, outdir, name = sys.argv[1], sys.argv[2], sys.argv[3]
CHUNK = 50000

os.makedirs(outdir, exist_ok=True)

with h5py.File(src, "r") as h:
    for key, suffix in (("train", "base"), ("test", "query")):
        ds = h[key]
        n, d = ds.shape
        header = np.array([d], dtype=np.int32).view(np.float32)[0]
        path = os.path.join(outdir, "%s_%s.fvecs" % (name, suffix))
        with open(path, "wb") as f:
            for start in range(0, n, CHUNK):
                block = np.asarray(ds[start:start + CHUNK], dtype=np.float32)
                m = block.shape[0]
                out = np.empty((m, d + 1), dtype=np.float32)
                out[:, 0] = header
                out[:, 1:] = block
                out.tofile(f)

        size = os.path.getsize(path)
        expect = n * (4 + 4 * d)
        assert size == expect, "size %d, expected %d" % (size, expect)
        # Read the first record's header back: proves the bit reinterpretation
        # round-trips, which a size check alone would not catch.
        got = int(np.fromfile(path, dtype=np.int32, count=1)[0])
        assert got == d, "header %d, expected %d" % (got, d)
        print("  %s  %d x %d  %.2f GB  ok" % (path, n, d, size / 1e9))

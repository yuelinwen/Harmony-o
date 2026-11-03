#ifndef ROTATION_H
#define ROTATION_H

#include <mkl.h>

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace harmony {

// A random orthogonal d x d basis, built by modified Gram-Schmidt on a
// Gaussian matrix. The accumulation is in double because the classical
// version loses orthogonality at d ~ 1000, and an R that is not orthogonal
// changes distances, which would make the control useless.
inline std::vector<float> makeOrthogonal(size_t d, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> m(d * d);
    for (size_t i = 0; i < d * d; i++) {
        m[i] = gauss(gen);
    }

    for (size_t i = 0; i < d; i++) {
        double* row = m.data() + i * d;
        for (size_t j = 0; j < i; j++) {
            const double* prev = m.data() + j * d;
            double dot = 0;
            for (size_t c = 0; c < d; c++) {
                dot += row[c] * prev[c];
            }
            for (size_t c = 0; c < d; c++) {
                row[c] -= dot * prev[c];
            }
        }
        double norm = 0;
        for (size_t c = 0; c < d; c++) {
            norm += row[c] * row[c];
        }
        norm = std::sqrt(norm);
        for (size_t c = 0; c < d; c++) {
            row[c] /= norm;
        }
    }

    std::vector<float> r(d * d);
    for (size_t i = 0; i < d * d; i++) {
        r[i] = static_cast<float>(m[i]);
    }
    return r;
}

// Rewrites n vectors of d dimensions as v' = R v, in place.
//
// One gemm per chunk rather than a loop of dot products: same arithmetic,
// but the shape MKL is built for. Chunked because gemm cannot write into
// its own input, and MKL here is the sequential build, so the parallelism
// has to come from the chunk loop.
inline void rotateInPlace(float* data, size_t n, size_t d, const std::vector<float>& r) {
    const size_t kChunk = 16384;
    size_t chunks = (n + kChunk - 1) / kChunk;

#pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < chunks; c++) {
        size_t start = c * kChunk;
        size_t m = (start + kChunk <= n) ? kChunk : (n - start);
        std::vector<float> out(m * d);
        // out = data * R^T, i.e. out[i][j] = sum_k data[i][k] * R[j][k]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (MKL_INT)m, (MKL_INT)d, (MKL_INT)d, 1.0f,
                    data + start * d, (MKL_INT)d,
                    r.data(), (MKL_INT)d,
                    0.0f, out.data(), (MKL_INT)d);
        std::copy(out.begin(), out.end(), data + start * d);
    }
}

}  // namespace harmony

#endif

#include "backend.h"

#include <cmath>

#include "Index.h"
// Byte-for-byte the authors' header (ADSampling, Gao and Long, SIGMOD 2023),
// so the test this prunes on is theirs and not a transcription of it. It has
// file-scope globals and non-inline definitions, so it is included here and
// nowhere else; backend.h stays clean of it and the rest of the program never
// sees adsampling::.
#include "vendor/adsampling.h"

namespace harmony {

namespace {

// The order the dimensions already arrive in, with the exact test on top of
// them. This is the paper's Harmony, and both halves do nothing, so it is
// also the proof that the interface costs nothing: a run with it has to
// match a run from before the interface existed, to the digit.
class HarmonyBackend : public PruningBackend {
   public:
    const char* name() const override { return "harmony"; }
};

// ADSampling (Gao and Long, SIGMOD 2023), which is the first thing here
// allowed to be wrong.
//
// The exact test waits for the partial sum S over d' dimensions to pass the
// full threshold T. That is proof, and it is late. A random projection is
// what turns S into an estimator of the whole: with the variance spread
// evenly across the axes, the full squared distance is around S * D / d', so
// a candidate can go as soon as that estimate is confidently above T rather
// than once it is certainly so. The paper's test, for squared distances:
//
//     S / d'  >  (T / D) * (1 + eps / sqrt(d'))^2
//
// which is S > T * m with m = (d'/D) * (1 + eps/sqrt(d'))^2 -- and that m is
// all this has to return. eps is the confidence knob: larger is more
// cautious and prunes later.
//
// Granularity: deltaD() carries the reference's delta_d, so the test runs
// every 32 dimensions as dist_comp does, not only where one worker hands the
// sum to the next. At 4x1 the first and only hop already holds the whole
// distance; the slice is then the whole vector and this is exactly the
// reference's own single-node setting.
//
// One thing the chain cannot copy: ivf.h tightens distK inside the scan
// ("if(KNNs.size() == k && KNNs.top().first < distK) distK = ..."). Here the
// threshold is fixed for a whole query group, because only the chain tail
// holds a complete distance and only the master owns the heap. The test is
// the same; the threshold it is given is looser.
class AdSamplingBackend : public PruningBackend {
   public:
    AdSamplingBackend(unsigned seed, float eps, size_t deltaD)
        : seed_(seed), deltaD_(deltaD) {
        // The knobs the vendored header reads are its own globals, set the
        // way search_ivf.cpp sets them from argv. One backend exists per
        // process and it is built before the first search, so this is the
        // same single assignment their main() makes.
        adsampling::epsilon0 = eps;
        adsampling::delta_d = (unsigned)deltaD;
    }

    // The rotation is part of the method, not a separate option. An
    // orthogonal map preserves every distance, so on its own it changes
    // nothing; what it buys is that the dimensions one hop has seen become a
    // random projection of the vector, which is the only condition under
    // which the partial sum estimates the whole. Running the test on the raw
    // axes is not a stricter variant of it, it is a different and
    // unjustified rule.
    void apply(Index& index, float* queries, size_t nq) override {
        index.rotate(seed_, queries, nq);
    }

    const char* name() const override { return "adsampling"; }

    size_t deltaD() const override { return deltaD_; }

    // Their ratio(), called rather than reimplemented: i is dimsSoFar, D is
    // dimsTotal, epsilon0 is the global set above. The i == 0 case is the one
    // thing they do not handle, because dist_comp only tests once i has
    // advanced -- and 0 * (1 + eps/sqrt(0))^2 is a NaN, not a 0, which would
    // make every comparison false and prune nothing.
    float thresholdScale(size_t dimsSoFar, size_t dimsTotal) const override {
        if (dimsSoFar == 0) {
            return 1.0f;  // nothing summed yet: no guessing
        }
        if (dimsSoFar > dimsTotal) {
            dimsSoFar = dimsTotal;  // ratio is only defined up to D
        }
        return adsampling::ratio((int)dimsTotal, (int)dimsSoFar);
    }

   private:
    unsigned seed_;
    size_t deltaD_;
};

}  // namespace

std::unique_ptr<PruningBackend> makeBackend(const std::string& name, unsigned seed, float eps,
                                            size_t deltaD) {
    if (name == "harmony") {
        return std::unique_ptr<PruningBackend>(new HarmonyBackend());
    }
    if (name == "adsampling") {
        return std::unique_ptr<PruningBackend>(new AdSamplingBackend(seed, eps, deltaD));
    }
    return nullptr;
}

std::string backendNames() {
    return "harmony, adsampling";
}

}  // namespace harmony

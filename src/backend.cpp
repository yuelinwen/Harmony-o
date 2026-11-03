#include "backend.h"

#include <cmath>

#include "Index.h"

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

// A random orthogonal basis, and nothing else.
//
// An orthogonal map preserves every distance, so this changes which
// dimensions land in which slice but not a single comparison's outcome:
// recall has to come out unchanged, and a recall that moved means the
// transform was applied incompletely somewhere, not that the method is
// approximate. That is what makes it the control.
//
// It is also the floor adsampling is read against. Adsampling rotates too,
// so a difference between the two is the estimate talking, with the basis
// held fixed.
class RotationBackend : public PruningBackend {
   public:
    explicit RotationBackend(unsigned seed) : seed_(seed) {}

    void apply(Index& index, float* queries, size_t nq) override {
        index.rotate(seed_, queries, nq);
    }

    const char* name() const override { return "rotation"; }

   private:
    unsigned seed_;
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
// It inherits the rotation rather than merely requiring it. The estimator is
// only valid if the dimensions a hop has seen are a random projection of the
// vector, and on the raw axes they are not; running this without rotating is
// not a stricter variant of it, it is a different and unjustified rule.
//
// What the chain costs it: the paper re-tests every 32 dimensions, this can
// only test where one worker hands the sum to the next, so teamSize times at
// multiples of block_dim. On gist at 1x4 that is 240, 480, 720 against the
// paper's 32; at 4x1 the first and only sum is already the whole distance
// and this degenerates into the exact test. Its headroom here is bounded by
// block_dim, which is worth knowing before reading the numbers.
class AdSamplingBackend : public RotationBackend {
   public:
    AdSamplingBackend(unsigned seed, float eps) : RotationBackend(seed), eps_(eps) {}

    const char* name() const override { return "adsampling"; }

    float thresholdScale(size_t dimsSoFar, size_t dimsTotal) const override {
        if (dimsSoFar == 0 || dimsSoFar >= dimsTotal) {
            return 1.0f;  // nothing summed yet, or all of it: no guessing
        }
        float f = 1.0f + eps_ / std::sqrt((float)dimsSoFar);
        return ((float)dimsSoFar / (float)dimsTotal) * f * f;
    }

   private:
    float eps_;
};

}  // namespace

std::unique_ptr<PruningBackend> makeBackend(const std::string& name, unsigned seed, float eps) {
    if (name == "harmony") {
        return std::unique_ptr<PruningBackend>(new HarmonyBackend());
    }
    if (name == "rotation") {
        return std::unique_ptr<PruningBackend>(new RotationBackend(seed));
    }
    if (name == "adsampling") {
        return std::unique_ptr<PruningBackend>(new AdSamplingBackend(seed, eps));
    }
    return nullptr;
}

std::string backendNames() {
    return "harmony, rotation, adsampling";
}

}  // namespace harmony

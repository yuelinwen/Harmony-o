#ifndef BACKEND_H
#define BACKEND_H

#include <cstddef>
#include <memory>
#include <string>

namespace harmony {

class Index;

// How the pruning decision gets its dimensions, and what it is allowed to
// conclude from them.
//
// A relay chain prunes a candidate as soon as its running sum passes the
// threshold, so two things decide how early that happens: which dimensions
// are summed first, and how strict the test on the partial sum is. The two
// are a pair -- a method that reorders the axes usually also wants to read
// the partial sum differently -- so one interface covers both:
//
//   apply            rewrite base, query and centroids on the master
//   thresholdScale   how much to scale the threshold at a hop
//
// Nothing else moves. The grid, the block order, the chain, the inner test
// are untouched, and the transform is applied before a byte is sent, so a
// worker never learns which backend is running -- it only sees the scales.
//
// Forward-declared Index on purpose: node.h has to include this header to
// hold a backend, and Index.h already includes node.h. The concrete
// backends therefore live in backend.cpp, where Index is complete.
class PruningBackend {
   public:
    virtual ~PruningBackend() {}

    // Rewrite all three in place, once, before the first search.
    //
    // All three is the part that is easy to miss: a query reaches a cluster
    // by its distance to the centroid, so rewriting the vectors and not the
    // centroids sends every query to the wrong clusters and still returns
    // results that look plausible.
    virtual void apply(Index& index, float* queries, size_t nq) {
        (void)index;
        (void)queries;
        (void)nq;
    }

    // How much to scale the pruning threshold at a hop that has summed
    // dimsSoFar of dimsTotal dimensions.
    //
    // 1.0 is the exact test, and what makes it safe is that a partial sum
    // only grows: a candidate already past the threshold can never come back
    // under it. A method willing to guess from the partial sum -- that the
    // remaining dimensions will not rescue this candidate -- says so by
    // returning less than 1.0, which prunes earlier at the cost of being
    // wrong sometimes.
    //
    // Asked once per block, never per candidate: the result is folded into
    // the threshold before the loop, so the inner test keeps its shape and a
    // backend costs nothing in there.
    virtual float thresholdScale(size_t dimsSoFar, size_t dimsTotal) const {
        (void)dimsSoFar;
        (void)dimsTotal;
        return 1.0f;
    }

    // As given to --backend, and as recorded in the backend column.
    virtual const char* name() const = 0;
};

// Returns nullptr for a name that is not a backend, so the caller can print
// the list rather than run something the user did not ask for.
std::unique_ptr<PruningBackend> makeBackend(const std::string& name, unsigned seed, float eps);

// Every name makeBackend accepts, for the error message and for --help.
std::string backendNames();

}  // namespace harmony

#endif

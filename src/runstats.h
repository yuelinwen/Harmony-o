#ifndef RUNSTATS_H
#define RUNSTATS_H

#include <mpi.h>
#include <sys/resource.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace harmony {

// Longest relay chain we split the pruning counters over, i.e. the largest
// teamSize these runs use. A chain longer than this folds into the last
// bucket rather than writing out of bounds.
static const int kMaxChainPos = 8;

// What one worker reports once it has finished every group.
//
// The counters it carries already existed -- waitTime, searchTime, totalSkip
// and totalCompare are maintained by GroupWorker and were only ever read by
// a print that utils.h disables. Gathering them to the master instead keeps
// that print switched off and still gets the numbers out, in a form that can
// be parsed rather than eyeballed.
struct WorkerStat {
    double waitTime = 0;
    double searchTime = 0;
    double totalTime = 0;
    long long totalSkip = 0;
    long long totalCompare = 0;
    long long bytesFwd = 0;
    long long maxRssKb = 0;
    long long teamId = 0;
    long long rankInsideTeam = 0;
    long long blockDim = 0;
    // Per chain position, so the master can rebuild prune rate as a function
    // of dimensions seen. Summed across workers, not averaged: a position is
    // occupied by every worker for some blocks.
    long long skipByPos[kMaxChainPos] = {0};
    long long compareByPos[kMaxChainPos] = {0};
    long long prunedByPos[kMaxChainPos] = {0};
    // Dimensions the worker really summed. A backend that tests inside a
    // slice abandons candidates part-way through one, so candidates * blockDim
    // counts arithmetic that never happened.
    long long dimsEvalByPos[kMaxChainPos] = {0};
    long long bytesByPos[kMaxChainPos] = {0};
    long long heapBytesByPos[kMaxChainPos] = {0};
};

inline long long maxRssKb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (long long)ru.ru_maxrss;  // Linux reports kilobytes
}

// Both sides of one MPI_Gather. Every rank calls this; rank 0 ends up with
// workerCount + 1 entries and ignores its own.
inline std::vector<WorkerStat> gatherWorkerStats(const WorkerStat& mine, int rank, int worldSize) {
    std::vector<WorkerStat> all(worldSize);
    MPI_Gather(&mine, sizeof(WorkerStat), MPI_BYTE,
               all.data(), sizeof(WorkerStat), MPI_BYTE,
               0, MPI_COMM_WORLD);
    if (rank != 0) {
        all.clear();
    }
    return all;
}

// One line, parseable, printed by the master. Everything here is derived
// from the gathered counters except qps/recall, which the caller passes in.
inline void printRunStats(const std::vector<WorkerStat>& all, size_t nq, double queryTime, float recall) {
    if (all.size() < 2) {
        return;
    }
    size_t workers = all.size() - 1;

    double waitSum = 0, searchSum = 0, otherSum = 0, searchMax = 0;
    long long dimOps = 0, candOffered = 0, bytesFwd = 0, rssKb = 0;
    long long skipSum = 0, compareSum = 0;

    for (size_t r = 1; r < all.size(); r++) {
        const WorkerStat& s = all[r];
        waitSum += s.waitTime;
        searchSum += s.searchTime;
        otherSum += s.totalTime - s.waitTime - s.searchTime;
        searchMax = std::max(searchMax, s.searchTime);
        skipSum += s.totalSkip;
        compareSum += s.totalCompare;
        for (int p = 0; p < kMaxChainPos; p++) {
            dimOps += s.dimsEvalByPos[p];
        }
        bytesFwd += s.bytesFwd;
        rssKb = std::max(rssKb, s.maxRssKb);
        // Every worker in a team walks the same candidate array, so counting
        // the first of each team once is what "candidates offered" means.
        if (s.rankInsideTeam == 1) {
            candOffered += s.totalCompare;
        }
    }

    double searchMean = searchSum / (double)workers;
    double busySkew = (searchMean > 0) ? (searchMax / searchMean) : 0;
    double pruneRate = (compareSum > 0) ? ((double)skipSum / (double)compareSum) : 0;

    printf("HSTATS qps=%.3f latency_ms=%.4f recall=%.6f dim_ops=%lld bytes_fwd=%lld "
           "cand_offered=%lld busy_skew=%.4f comm_time=%.4f compute_time=%.4f "
           "other_time=%.4f mem_mb=%lld prune_rate=%.4f\n",
           (double)nq / queryTime,
           queryTime / (double)nq * 1000.0,
           (double)recall,
           dimOps,
           bytesFwd,
           candOffered,
           busySkew,
           waitSum / (double)workers,
           searchMean,
           otherSum / (double)workers,
           rssKb / 1024,
           pruneRate);
    // One line per chain position: the 6.3.3 breakdown. blockDim is the same
    // on every worker of a run, so the first worker's is the slice width.
    //
    // skip counts candidates that arrive already INFINITY, so it reports what
    // the PREDECESSORS pruned: at position p the partial sum has been through
    // p slices, i.e. D' = p * block_dim dimensions. Position 0 therefore always
    // reads 0.0000, and that is the check that the bucketing is right.
    long long blockDim = all[1].blockDim;
    for (int pos = 0; pos < kMaxChainPos; pos++) {
        long long skip = 0, compare = 0;
        for (size_t r = 1; r < all.size(); r++) {
            skip += all[r].skipByPos[pos];
            compare += all[r].compareByPos[pos];
        }
        if (compare == 0) {
            continue;  // no worker ever sat at this position
        }
        printf("HPRUNE pos=%d dims=%lld skip=%lld compare=%lld rate=%.4f\n",
               pos, (long long)pos * blockDim, skip, compare,
               (double)skip / (double)compare);
    }
    fflush(stdout);
}

}  // namespace harmony

#endif

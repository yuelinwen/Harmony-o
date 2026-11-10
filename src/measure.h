#ifndef MEASURE_H
#define MEASURE_H

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <ctime>
#include <string>
#include <vector>

#include "runstats.h"

// The five measurements the reference implementation records, written in the
// same shape so a harmony-o row and a harmony row can be read side by side:
// dimensions evaluated, candidates forwarded, bytes, busy-time skew and
// threshold staleness -- each at the grain its own quantity has, joined by
// run_id.
//
// Two of them say something different here than they do there, and the
// difference is the point rather than a defect:
//
//   bytes_per_hop  harmony-o forwards one float per (query, candidate) pair
//                  whether or not the candidate is still alive, so its bytes
//                  answer only to the candidate count. A row where bytes
//                  track the pruning rate is a row from an implementation
//                  that compacts.
//
//   threshold_staleness  the stage here is a vector partition's turn at a
//                  query group (--group), not a dimension block, because that
//                  is the only point at which this code revisits a threshold.
//
// No flag turns this on: a run nobody recorded cannot be compared later.

namespace harmony {

inline void writeMeasurements(const std::vector<WorkerStat>& all,
                              const std::vector<double>& stalenessSum,
                              const std::vector<double>& stalenessMax,
                              const std::vector<long long>& stalenessCount,
                              const std::vector<long long>& noBoundCount,
                              const std::string& backend, size_t teamCount, size_t teamSize,
                              size_t nprobe, size_t k, size_t nq, size_t d,
                              double queryTime, float recall) {
    if (all.size() < 2) {
        return;  // not the master, or a mode that gathers nothing
    }
    mkdir("measurements", 0755);  // already there is not an error worth reporting

    // Unique per run: two runs can start in the same second, but not from the
    // same process.
    char runId[64];
    time_t now = time(nullptr);
    struct tm tmv;
    localtime_r(&now, &tmv);
    char stamp[32];
    strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H:%M:%S", &tmv);
    snprintf(runId, sizeof(runId), "%s-%d", stamp, (int)getpid());

    // Repeated into every file so one can be read without a join.
    char id[256];
    snprintf(id, sizeof(id), "%s,%s,%d,%d,%d,%d", runId, backend.c_str(), (int)teamCount,
             (int)teamSize, (int)nprobe, (int)k);

    struct Open {
        static FILE* at(const char* path, const char* header) {
            bool isNew = true;
            FILE* probe = fopen(path, "rb");
            if (probe != nullptr) {
                fseek(probe, 0, SEEK_END);
                isNew = (ftell(probe) == 0);
                fclose(probe);
            }
            FILE* f = fopen(path, "ab");
            if (f != nullptr && isNew) {
                fprintf(f, "%s\n", header);
            }
            return f;
        }
    };

    // Summed over workers: every worker occupies every chain position for
    // some blocks, so a position is a property of the run, not of a machine.
    long long skip[kMaxChainPos] = {0};
    long long pruned[kMaxChainPos] = {0};
    long long compare[kMaxChainPos] = {0};
    long long bytes[kMaxChainPos] = {0};
    long long heapBytes[kMaxChainPos] = {0};
    long long dimsEval[kMaxChainPos] = {0};
    for (size_t r = 1; r < all.size(); r++) {
        for (int p = 0; p < kMaxChainPos; p++) {
            skip[p] += all[r].skipByPos[p];
            pruned[p] += all[r].prunedByPos[p];
            compare[p] += all[r].compareByPos[p];
            bytes[p] += all[r].bytesByPos[p];
            heapBytes[p] += all[r].heapBytesByPos[p];
            dimsEval[p] += all[r].dimsEvalByPos[p];
        }
    }
    long long blockDim = all[1].blockDim;
    int hops = 0;
    for (int p = 0; p < kMaxChainPos; p++) {
        if (compare[p] > 0) {
            hops = p + 1;
        }
    }

    // ---- 1. dimensions evaluated: per hop, against doing all of it ----
    FILE* f = Open::at("measurements/dims_evaluated.csv",
                       "run_id,backend,bvec,bdim,nprobe,k,hop_id,dims_evaluated,"
                       "dims_no_pruning,pct_of_no_pruning");
    if (f != nullptr) {
        for (int p = 0; p < hops; p++) {
            // The workers' own count. It equals (compare - skip) * blockDim
            // whenever nothing abandons a slice part-way, and is smaller than
            // it when a backend tests inside one.
            long long ops = dimsEval[p];
            long long bare = compare[p] * blockDim;
            fprintf(f, "%s,%d,%lld,%lld,%.4f\n", id, p + 1, ops, bare,
                    (bare > 0) ? (100.0 * (double)ops / (double)bare) : 0.0);
        }
        fclose(f);
    }

    // ---- 2. candidates per hop ----
    // forwarded is the column that separates this from an implementation
    // that compacts: here it equals reached, alive or not.
    f = Open::at("measurements/candidates_per_hop.csv",
                 "run_id,backend,bvec,bdim,nprobe,k,hop_id,reached,alive,skipped_pct,forwarded");
    if (f != nullptr) {
        long long offered = compare[0];
        for (int p = 0; p < hops; p++) {
            long long reached = compare[p] - skip[p];
            long long alive = compare[p] - skip[p] - pruned[p];
            long long fwd = (p + 1 < hops) ? compare[p] : 0;  // the tail sends a heap
            fprintf(f, "%s,%d,%lld,%lld,%.4f,%lld\n", id, p + 1, reached, alive,
                    (offered > 0) ? (100.0 * (1.0 - (double)reached / (double)offered)) : 0.0, fwd);
        }
        fclose(f);
    }

    // ---- 3. bytes per hop ----
    f = Open::at("measurements/bytes_per_hop.csv",
                 "run_id,backend,bvec,bdim,nprobe,k,hop_id,bytes_fwd,bytes_per_query,heap_bytes");
    if (f != nullptr) {
        for (int p = 0; p < hops; p++) {
            fprintf(f, "%s,%d,%lld,%.2f,%lld\n", id, p + 1, bytes[p],
                    (nq > 0) ? ((double)bytes[p] / (double)nq) : 0.0, heapBytes[p]);
        }
        fclose(f);
    }

    // ---- 4. node busy-time skew: the per-worker seconds it comes from ----
    double busyLo = -1.0, busyHi = 0.0;
    for (size_t r = 1; r < all.size(); r++) {
        double c = all[r].searchTime;
        if (busyLo < 0.0 || c < busyLo) {
            busyLo = c;
        }
        if (c > busyHi) {
            busyHi = c;
        }
    }
    double runSkew = (busyLo > 0.0) ? (busyHi / busyLo) : 0.0;
    f = Open::at("measurements/busy_skew.csv",
                 "run_id,backend,bvec,bdim,nprobe,k,worker,row,col,"
                 "busy_s,total_s,idle_pct,other_pct,run_skew");
    if (f != nullptr) {
        for (size_t r = 1; r < all.size(); r++) {
            const WorkerStat& s = all[r];
            double other = s.totalTime - s.waitTime - s.searchTime;
            fprintf(f, "%s,%d,%lld,%lld,%.6f,%.6f,%.2f,%.2f,%.4f\n", id, (int)r, s.teamId,
                    s.rankInsideTeam, s.searchTime, s.totalTime,
                    (s.totalTime > 0) ? (100.0 * s.waitTime / s.totalTime) : 0.0,
                    (s.totalTime > 0) ? (100.0 * other / s.totalTime) : 0.0, runSkew);
        }
        fclose(f);
    }

    // ---- 5. threshold staleness: per vector-level stage ----
    f = Open::at("measurements/threshold_staleness.csv",
                 "run_id,backend,bvec,bdim,nprobe,k,round,mean_pct,worst_pct,queries,no_bound");
    if (f != nullptr) {
        for (size_t r = 0; r < stalenessSum.size(); r++) {
            long long cnt = (r < stalenessCount.size()) ? stalenessCount[r] : 0;
            double mean = (cnt > 0) ? (stalenessSum[r] / (double)cnt) : 0.0;
            fprintf(f, "%s,%d,%.4f,%.4f,%lld,%lld\n", id, (int)r + 1, 100.0 * mean,
                    100.0 * ((r < stalenessMax.size()) ? stalenessMax[r] : 0.0), cnt,
                    (r < noBoundCount.size()) ? noBoundCount[r] : 0);
        }
        fclose(f);
    }

    // ---- 6. one row per run, to join the five against ----
    f = Open::at("measurements/runs.csv",
                 "run_id,backend,bvec,bdim,nprobe,k,nq,d,workers,qps,latency_ms,recall,"
                 "dim_ops,bytes_fwd,cand_offered,busy_skew,stale_mean_pct,mem_mb");
    if (f != nullptr) {
        long long dimOps = 0, bytesFwd = 0, rssKb = 0;
        for (int p = 0; p < hops; p++) {
            dimOps += dimsEval[p];
            bytesFwd += bytes[p];
        }
        for (size_t r = 1; r < all.size(); r++) {
            if (all[r].maxRssKb > rssKb) {
                rssKb = all[r].maxRssKb;
            }
        }
        double staleSum = 0;
        long long staleN = 0;
        for (size_t r = 0; r < stalenessSum.size(); r++) {
            staleSum += stalenessSum[r];
            staleN += (r < stalenessCount.size()) ? stalenessCount[r] : 0;
        }
        fprintf(f, "%s,%d,%d,%d,%.3f,%.4f,%.6f,%lld,%lld,%lld,%.4f,%.4f,%lld\n", id, (int)nq, (int)d,
                (int)(all.size() - 1), (double)nq / queryTime, queryTime / (double)nq * 1000.0,
                (double)recall, dimOps, bytesFwd, compare[0], runSkew,
                (staleN > 0) ? (100.0 * staleSum / (double)staleN) : 0.0, rssKb / 1024);
        fclose(f);
    }
}

}  // namespace harmony

#endif

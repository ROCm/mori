// Pipeline AllReduce ? compare serial vs Pipeline (default chunk).
// Minimal output: PE0 only prints a summary table. Verbose: MORI_SDMA_VERBOSE=1
// Chunk sweep: MORI_PIPELINE_CHUNK_SWEEP=1
//
// Verification matches sdma_allreduce_sync: copy_output_to_user=true, D2H from
// hipMalloc devOut; avoids D2H from symmetric transit on multi-GPU.

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <string>
#include <cstdarg>
#include <functional>

#include "mori/application/utils/check.hpp"
#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "mori/shmem/shmem.hpp"

using namespace mori::core;
using namespace mori::application;
using namespace mori::shmem;
using namespace mori::collective;

#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error("HIP failed"); \
        } \
    } while (0)

static uint32_t computeExpected(int npes) {
    uint32_t sum = 0;
    for (int pe = 0; pe < npes; pe++) sum += (pe + 1) * 1000;
    return sum;
}

static bool verifyResult(const uint32_t* data, size_t elems, uint32_t expected, int myPe) {
    for (size_t i = 0; i < elems; i++) {
        if (data[i] != expected) {
            fprintf(stderr, "PE %d VERIFY [%zu] exp %u got %u\n", myPe, i, expected, data[i]);
            return false;
        }
    }
    return true;
}

using BenchFn = std::function<bool(uint32_t* in, uint32_t* out, size_t elems, hipStream_t s)>;

static bool runBenchMs(BenchFn fn, uint32_t* inBuf, uint32_t* devOut,
                       const std::vector<uint32_t>& hostData,
                       size_t elemsPerPe, size_t bytesPerPe, int npes, int myPe,
                       hipStream_t stream, int warmup, int iterations,
                       double* outMs, double* outAlgoGb) {
    for (int w = 0; w < 3; w++) {
        CHECK_HIP(hipMemcpyAsync(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice,
                                 stream));
        CHECK_HIP(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        fn(inBuf, devOut, elemsPerPe, stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myPe == 0) fprintf(stderr, "  warmup done\n");

    CHECK_HIP(hipMemcpyAsync(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice, stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = fn(inBuf, devOut, elemsPerPe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    if (ok) {
        std::vector<uint32_t> result(elemsPerPe);
        CHECK_HIP(hipMemcpy(result.data(), devOut, bytesPerPe, hipMemcpyDeviceToHost));
        ok = verifyResult(result.data(), elemsPerPe, computeExpected(npes), myPe);
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (gok != npes) {
        if (myPe == 0) fprintf(stderr, "  VERIFY FAILED (%d/%d)\n", gok, npes);
        return false;
    }
    if (myPe == 0) fprintf(stderr, "  verify OK\n");

    std::vector<double> times;
    for (int i = 0; i < warmup + iterations; i++) {
        CHECK_HIP(hipMemcpyAsync(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice,
                                 stream));
        CHECK_HIP(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        fn(inBuf, devOut, elemsPerPe, stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        double t1 = MPI_Wtime();
        if (i >= warmup) times.push_back(t1 - t0);
    }
    if (myPe == 0) fprintf(stderr, "  bench done\n");
    MPI_Barrier(MPI_COMM_WORLD);

    double avg = 0;
    for (double t : times) avg += t;
    avg /= times.size();

    double g_sum = 0;
    MPI_Reduce(&avg, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myPe == 0) {
        double g_avg = g_sum / npes;
        *outMs = g_avg * 1000.0;
        *outAlgoGb = bytesPerPe / g_avg / (1024.0 * 1024.0 * 1024.0);
    }
    return true;
}

static bool wantChunkSweep() {
    const char* e = std::getenv("MORI_PIPELINE_CHUNK_SWEEP");
    return e && e[0] == '1' && e[1] == '\0';
}

static void sweepPrintf(std::vector<std::string>* lines, const char* fmt, ...) {
    if (!lines) return;
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    lines->emplace_back(buf);
}

void testPipelinedAllreduce() {
    MPI_Init(NULL, NULL);
    int status = ShmemMpiInit(MPI_COMM_WORLD);
    assert(!status);

    int myPe = ShmemMyPe();
    int npes = ShmemNPes();

    const std::vector<size_t> dataSizesMB = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
    const std::vector<size_t> chunkSizesKB = {
        128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    const int warmup = 3;
    const int iterations = 5;
    const bool chunkSweep = wantChunkSweep();
    std::vector<std::string> sweep_lines;

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    std::vector<double> serialMs, sdmaMs, p2pMs;
    std::vector<double> serialGb, sdmaGb, p2pGb;
    std::vector<bool> okSerial, okSdma, okP2p;

    for (size_t dataMB : dataSizesMB) {
        size_t bytesPerPe = dataMB * 1024 * 1024;
        size_t elemsPerPe = bytesPerPe / sizeof(uint32_t);
        uint32_t fillValue = static_cast<uint32_t>((myPe + 1) * 1000);

        uint32_t* inBuf = nullptr;
        uint32_t* devOut = nullptr;
        CHECK_HIP(hipMalloc(&inBuf, bytesPerPe));
        CHECK_HIP(hipMalloc(&devOut, bytesPerPe));
        std::vector<uint32_t> hostData(elemsPerPe, fillValue);

        size_t outputBufSize = static_cast<size_t>(npes) * (elemsPerPe / npes + 64) * sizeof(uint32_t);
        auto ar = std::make_unique<AllreduceSdma<uint32_t>>(
            myPe, npes, bytesPerPe, outputBufSize, true);

        double ms = 0, gb = 0;
        bool ok = false;

        // Serial skipped ? output zeros to avoid polluting pipeline signal space
        ok = true;
        if (myPe == 0) {
            serialMs.push_back(ms);
            serialGb.push_back(gb);
            okSerial.push_back(ok);
        }

        if (myPe == 0) fprintf(stderr, "--- %zuMB SDMA pipe ---\n", dataMB);

        // Pipeline SDMA default
        ms = 0;
        gb = 0;
        ok = runBenchMs(
            [&](uint32_t* in, uint32_t* out, size_t n, hipStream_t s) {
                return ar->pipelined(in, out, n, 0, 0, s);
            },
            inBuf, devOut, hostData, elemsPerPe, bytesPerPe, npes, myPe, stream, warmup,
            iterations, &ms, &gb);
        if (myPe == 0) {
            sdmaMs.push_back(ok ? ms : -1.0);
            sdmaGb.push_back(ok ? gb : 0.0);
            okSdma.push_back(ok);
        }

        if (chunkSweep && myPe == 0) {
            sweepPrintf(&sweep_lines, "  [%zu MB/PE] SDMA chunk sweep", dataMB);
        }
        if (chunkSweep) {
            for (size_t chunkKB : chunkSizesKB) {
                size_t chunkBytes = chunkKB * 1024;
                size_t chunkElems = chunkBytes / sizeof(uint32_t);
                if (chunkBytes > bytesPerPe) continue;
                double sm = 0, sg = 0;
                bool sk = runBenchMs(
                    [&](uint32_t* in, uint32_t* out, size_t n, hipStream_t s) {
                        return ar->pipelined(in, out, n, chunkElems, 0, s);
                    },
                    inBuf, devOut, hostData, elemsPerPe, bytesPerPe, npes, myPe, stream, warmup,
                    iterations, &sm, &sg);
                if (myPe == 0 && sk) {
                    sweepPrintf(&sweep_lines, "    SDMA %5zuKB  %7.3f ms  %6.1f GB/s",
                                chunkKB, sm, sg);
                }
            }
        }

        if (myPe == 0) fprintf(stderr, "--- %zuMB P2P pipe ---\n", dataMB);

        // P2P default
        ms = 0;
        gb = 0;
        ok = runBenchMs(
            [&](uint32_t* in, uint32_t* out, size_t n, hipStream_t s) {
                return ar->pipelined(in, out, n, 0, 1, s);
            },
            inBuf, devOut, hostData, elemsPerPe, bytesPerPe, npes, myPe, stream, warmup,
            iterations, &ms, &gb);
        if (myPe == 0) {
            p2pMs.push_back(ok ? ms : -1.0);
            p2pGb.push_back(ok ? gb : 0.0);
            okP2p.push_back(ok);
        }

        if (chunkSweep && myPe == 0) {
            sweepPrintf(&sweep_lines, "  [%zu MB/PE] P2P chunk sweep", dataMB);
        }
        if (chunkSweep) {
            for (size_t chunkKB : chunkSizesKB) {
                size_t chunkBytes = chunkKB * 1024;
                size_t chunkElems = chunkBytes / sizeof(uint32_t);
                if (chunkBytes > bytesPerPe) continue;
                double sm = 0, sg = 0;
                bool sk = runBenchMs(
                    [&](uint32_t* in, uint32_t* out, size_t n, hipStream_t s) {
                        return ar->pipelined(in, out, n, chunkElems, 1, s);
                    },
                    inBuf, devOut, hostData, elemsPerPe, bytesPerPe, npes, myPe, stream, warmup,
                    iterations, &sm, &sg);
                if (myPe == 0 && sk) {
                    sweepPrintf(&sweep_lines, "    P2P  %5zuKB  %7.3f ms  %6.1f GB/s",
                                chunkKB, sm, sg);
                }
            }
        }

        ar.reset();
        CHECK_HIP(hipFree(devOut));
        CHECK_HIP(hipFree(inBuf));
    }

    if (myPe == 0) {
        const char* sep =
            "-------------------------------------------------------------------------------------";
        printf("\n%s\n", sep);
        printf("  AllReduce Pipeline Benchmark\n");
        printf("  npes=%-3d  dtype=uint32  warmup=%d  iters=%d  chunk_sweep=%s\n",
               npes, warmup, iterations, chunkSweep ? "on" : "off");
        printf("  Serial = ReduceScatter + AllGather  (single-shot, no pipeline)\n");
        printf("  SDMA   = Pipeline SDMA scatter/reduce/AG  (auto 2-chunk, 3-stage)\n");
        printf("  P2P    = Pipeline with P2P scatter mode\n");
        printf("%s\n\n", sep);

        printf("%-7s | --- Latency (ms) --- | --- Bandwidth (GB/s) --- | -- Speedup -- | Winner\n",
               "MB/PE");
        printf("%-7s | %7s %7s %7s | %8s %8s %8s | %6s %6s |\n",
               "", "Serial", "SDMA", "P2P",
               "Serial", "SDMA", "P2P",
               "SDMA", "P2P");
        printf("%s\n", sep);

        int wins = 0, ncmp = 0;
        for (size_t i = 0; i < dataSizesMB.size(); i++) {
            double s  = (i < serialMs.size()) ? serialMs[i] : -1.0;
            double d  = (i < sdmaMs.size())   ? sdmaMs[i]   : -1.0;
            double p  = (i < p2pMs.size())    ? p2pMs[i]    : -1.0;
            double sg = (i < serialGb.size()) ? serialGb[i] : 0.0;
            double dg = (i < sdmaGb.size())   ? sdmaGb[i]   : 0.0;
            double pg = (i < p2pGb.size())    ? p2pGb[i]    : 0.0;
            bool os = (i < okSerial.size()) && okSerial[i];
            bool od = (i < okSdma.size())   && okSdma[i];
            bool op = (i < okP2p.size())    && okP2p[i];

            printf("%-7zu |", dataSizesMB[i]);

            auto fmtMs = [](bool ok, double v) {
                if (!ok)       return printf(" %7s", "FAIL");
                if (v <= 0)    return printf(" %7s", "-");
                return                printf(" %7.3f", v);
            };
            auto fmtGb = [](bool ok, double v) {
                if (!ok)       return printf(" %8s", "FAIL");
                if (v <= 0)    return printf(" %8s", "-");
                return                printf(" %8.1f", v);
            };

            fmtMs(os, s); fmtMs(od, d); fmtMs(op, p);
            printf(" |");
            fmtGb(os, sg); fmtGb(od, dg); fmtGb(op, pg);
            printf(" |");

            if (os && od && s > 0 && d > 0) {
                printf(" %5.2fx", s / d);
                ncmp++;
                if (d < s) wins++;
            } else {
                printf(" %6s", "-");
            }

            if (os && op && s > 0 && p > 0)
                printf(" %5.2fx", s / p);
            else
                printf(" %6s", "-");

            printf(" |");

            if (os && od && s > 0 && d > 0) {
                if (d < s)      printf(" SDMA\n");
                else if (d > s) printf(" Serial\n");
                else            printf(" Tie\n");
            } else {
                printf(" -\n");
            }
        }
        printf("%s\n", sep);
        if (ncmp > 0)
            printf("SDMA Pipeline vs Serial: %d/%d faster\n", wins, ncmp);

        if (chunkSweep && !sweep_lines.empty()) {
            printf("\n--- chunk sweep (MORI_PIPELINE_CHUNK_SWEEP=1) ---\n");
            for (const auto& s : sweep_lines) printf("%s\n", s.c_str());
        }
    }

    CHECK_HIP(hipStreamDestroy(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    ShmemFinalize();
}

int main(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    testPipelinedAllreduce();
    return 0;
}

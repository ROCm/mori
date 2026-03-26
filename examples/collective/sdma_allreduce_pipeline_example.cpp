// Pipeline AllReduce — 对比串行 vs Pipeline（整块默认 chunk）。
// 少打印：默认只输出 PE0 一张汇总表。详细库日志: MORI_SDMA_VERBOSE=1
// chunk 扫描: MORI_PIPELINE_CHUNK_SWEEP=1（表后附加简短行）
//
// 校验与 sdma_allreduce_sync 一致：copy_output_to_user=true，从 hipMalloc 的 devOut
// D2H；避免从对称 transit 直接 D2H 在多卡/部分驱动路径上的异常。collective 前
// 输入用 hipMemcpyAsync(..., stream)+sync，与 kernel 同流有序。

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

// PE0: sets *outMs / *outGb; others: unchanged. Returns global pass.
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
        if (myPe == 0) {
            fprintf(stderr, "VERIFY FAILED（若日志中有 RDMA/SDMA device None，需正确设备与 rank-GPU 绑定）\n");
        }
        return false;
    }

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

    int nGpu = 0;
    CHECK_HIP(hipGetDeviceCount(&nGpu));
    if (nGpu > 0) {
        MPI_Comm localComm;
        int localRank = 0;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localComm);
        MPI_Comm_rank(localComm, &localRank);
        CHECK_HIP(hipSetDevice(localRank % nGpu));
        MPI_Comm_free(&localComm);
    }

    int status = ShmemMpiInit(MPI_COMM_WORLD);
    assert(!status);

    int myPe = ShmemMyPe();
    int npes = ShmemNPes();

    const std::vector<size_t> dataSizesMB = {32, 64, 128, 256};
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

        // 串行（结果 D2D 到 devOut，与 sync example out-of-place 一致）
        {
            for (int i = 0; i < 10; i++) {
                CHECK_HIP(hipMemcpyAsync(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice,
                                         stream));
                CHECK_HIP(hipMemsetAsync(devOut, 0, bytesPerPe, stream));
                CHECK_HIP(hipStreamSynchronize(stream));
                MPI_Barrier(MPI_COMM_WORLD);
                (*ar)(inBuf, devOut, elemsPerPe, stream);
                CHECK_HIP(hipStreamSynchronize(stream));
                MPI_Barrier(MPI_COMM_WORLD);
            }
            std::vector<uint32_t> result(elemsPerPe);
            CHECK_HIP(hipMemcpyAsync(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice,
                                     stream));
            CHECK_HIP(hipStreamSynchronize(stream));
            (*ar)(inBuf, devOut, elemsPerPe, stream);
            CHECK_HIP(hipStreamSynchronize(stream));
            CHECK_HIP(hipMemcpy(result.data(), devOut, bytesPerPe, hipMemcpyDeviceToHost));
            int v = verifyResult(result.data(), elemsPerPe, computeExpected(npes), myPe) ? 1 : 0;
            int gv = 0;
            MPI_Allreduce(&v, &gv, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            ok = (gv == npes);
            if (ok) {
                std::vector<double> times;
                for (int i = 0; i < warmup + iterations; i++) {
                    CHECK_HIP(hipMemcpyAsync(inBuf, hostData.data(), bytesPerPe,
                                             hipMemcpyHostToDevice, stream));
                    CHECK_HIP(hipStreamSynchronize(stream));
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t0 = MPI_Wtime();
                    (*ar)(inBuf, devOut, elemsPerPe, stream);
                    CHECK_HIP(hipStreamSynchronize(stream));
                    double t1 = MPI_Wtime();
                    if (i >= warmup) times.push_back(t1 - t0);
                }
                MPI_Barrier(MPI_COMM_WORLD);
                double avg = 0;
                for (double t : times) avg += t;
                avg /= times.size();
                double g_sum = 0;
                MPI_Reduce(&avg, &g_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                if (myPe == 0) {
                    double g_avg = g_sum / npes;
                    ms = g_avg * 1000.0;
                    gb = bytesPerPe / g_avg / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
        if (myPe == 0) {
            serialMs.push_back(ms);
            serialGb.push_back(gb);
            okSerial.push_back(ok);
        }

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
        printf("pipeline_bench  npes=%d  sweep=%s\n\n", npes, chunkSweep ? "on" : "off");
        printf("%-7s %8s %8s %8s %7s %7s %5s %5s\n",
               "MB/PE", "串行ms", "SDMAms", "P2Pms", "SDMA×", "P2P×", "S算法", "SDMA胜");
        printf("%s\n", std::string(65, '-').c_str());

        int wins = 0, ncmp = 0;
        for (size_t i = 0; i < dataSizesMB.size(); i++) {
            double s = (i < serialMs.size()) ? serialMs[i] : -1.0;
            double d = (i < sdmaMs.size()) ? sdmaMs[i] : -1.0;
            double p = (i < p2pMs.size()) ? p2pMs[i] : -1.0;
            bool os = (i < okSerial.size()) && okSerial[i];
            bool od = (i < okSdma.size()) && okSdma[i];
            bool op = (i < okP2p.size()) && okP2p[i];

            if (!os) {
                printf("%-7zu %8s %8s %8s %7s %7s %5s %5s\n",
                       dataSizesMB[i], "FAIL", "-", "-", "-", "-", "-", "-");
                continue;
            }
            printf("%-7zu %8.3f ", dataSizesMB[i], s);
            if (od && d > 0)
                printf("%8.3f ", d);
            else
                printf("%8s ", "-");
            if (op && p > 0)
                printf("%8.3f ", p);
            else
                printf("%8s ", "-");

            if (od && d > 0) {
                double r = s / d;
                printf("%7.2f ", r);
                ncmp++;
                if (d < s) wins++;
            } else {
                printf("%7s ", "-");
            }

            if (op && p > 0)
                printf("%7.2f ", s / p);
            else
                printf("%7s ", "-");

            if (i < serialGb.size())
                printf("%5.0f ", serialGb[i]);
            else
                printf("%5s ", "-");

            if (od && d > 0)
                printf("%5s\n", (d < s) ? "是" : "否");
            else
                printf("%5s\n", "-");
        }
        printf("%s\n", std::string(65, '-').c_str());
        if (ncmp > 0)
            printf("SDMA整块 vs 串行: %d/%d 更快\n", wins, ncmp);

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

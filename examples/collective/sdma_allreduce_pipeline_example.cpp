// Pipeline AllReduce — 示例只做一件事：对照串行 operator，看 Pipeline 能否更快。
// 只跑库默认整块 chunk（chunk_elems=0），不扫小 chunk（小 chunk 必然拖慢，与「赢串行」无关）。
// 扩展扫 chunk：设置环境变量 MORI_PIPELINE_CHUNK_SWEEP=1

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
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
            fprintf(stderr, "HIP Error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            throw std::runtime_error("HIP call failed"); \
        } \
    } while(0)

static uint32_t computeExpected(int npes) {
    uint32_t sum = 0;
    for (int pe = 0; pe < npes; pe++) sum += (pe + 1) * 1000;
    return sum;
}

static bool verifyResult(const uint32_t* data, size_t elems, uint32_t expected, int myPe) {
    for (size_t i = 0; i < elems; i++) {
        if (data[i] != expected) {
            printf("PE %d: FAILED at [%zu]: expected %u, got %u\n", myPe, i, expected, data[i]);
            return false;
        }
    }
    return true;
}

struct BenchResult {
    std::string label;
    size_t dataBytes;
    size_t chunkBytes;
    int scatterMode;
    double avgMs;
    double algoBw;
    double busBw;
    bool passed;
};

using BenchFn = std::function<bool(uint32_t* in, uint32_t* out, int elems, hipStream_t s)>;

static BenchResult runBench(const char* label, BenchFn fn,
                            uint32_t* inBuf, void* verifyBuf,
                            const std::vector<uint32_t>& hostData,
                            int elemsPerPe, size_t bytesPerPe, int npes, int myPe,
                            hipStream_t stream, int warmup, int iterations,
                            size_t chunkBytes, int scatterMode) {
    BenchResult res;
    res.label = label;
    res.dataBytes = bytesPerPe;
    res.chunkBytes = chunkBytes;
    res.scatterMode = scatterMode;
    res.passed = false;
    res.avgMs = res.algoBw = res.busBw = 0.0;

    uint32_t* outBuf = reinterpret_cast<uint32_t*>(verifyBuf);

    for (int w = 0; w < 3; w++) {
        CHECK_HIP(hipMemcpy(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        fn(inBuf, outBuf, elemsPerPe, stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    CHECK_HIP(hipMemcpy(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
    CHECK_HIP(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    bool ok = fn(inBuf, outBuf, elemsPerPe, stream);
    CHECK_HIP(hipStreamSynchronize(stream));

    if (ok) {
        std::vector<uint32_t> result(elemsPerPe);
        CHECK_HIP(hipMemcpy(result.data(), verifyBuf, bytesPerPe, hipMemcpyDeviceToHost));
        ok = verifyResult(result.data(), elemsPerPe, computeExpected(npes), myPe);
    }

    int lok = ok ? 1 : 0, gok = 0;
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (gok != npes) {
        if (myPe == 0) printf("  %-40s VERIFY FAILED\n", label);
        return res;
    }

    std::vector<double> times;
    for (int i = 0; i < warmup + iterations; i++) {
        CHECK_HIP(hipMemcpy(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        fn(inBuf, outBuf, elemsPerPe, stream);
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
        res.avgMs = g_avg * 1000.0;
        res.algoBw = bytesPerPe / g_avg / (1024.0 * 1024.0 * 1024.0);
        res.busBw = res.algoBw * 2.0 * (npes - 1) / npes;
        res.passed = true;
    }
    return res;
}

static bool wantChunkSweep() {
    const char* e = std::getenv("MORI_PIPELINE_CHUNK_SWEEP");
    return e && e[0] == '1' && e[1] == '\0';
}

void testPipelinedAllreduce() {
    MPI_Init(NULL, NULL);
    int status = ShmemMpiInit(MPI_COMM_WORLD);
    assert(!status);

    int myPe = ShmemMyPe();
    int npes = ShmemNPes();

    std::vector<size_t> dataSizesMB = {32, 64, 128, 256};
    const std::vector<size_t> chunkSizesKB = {
        128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

    const int warmup = 3;
    const int iterations = 5;
    const bool chunkSweep = wantChunkSweep();

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    if (myPe == 0) {
        printf("\n======================================================================\n");
        printf("Pipeline AllReduce — 目标: Pipeline SDMA(整块) 快于 串行 operator()\n");
        printf("  PE=%d  仅默认 chunk；小 chunk 扫描: %s\n", npes,
               chunkSweep ? "开启(MORI_PIPELINE_CHUNK_SWEEP=1)" : "关闭");
        printf("======================================================================\n");
    }

    std::vector<BenchResult> allResults;
    std::vector<double> serialMsBySize;
    std::vector<double> sdmaDefaultMsBySize;

    for (size_t dataMB : dataSizesMB) {
        size_t bytesPerPe = dataMB * 1024 * 1024;
        int elemsPerPe = bytesPerPe / sizeof(uint32_t);
        uint32_t fillValue = static_cast<uint32_t>((myPe + 1) * 1000);

        uint32_t* inBuf = nullptr;
        CHECK_HIP(hipMalloc(&inBuf, bytesPerPe));
        std::vector<uint32_t> hostData(elemsPerPe, fillValue);

        size_t outputBufSize = static_cast<size_t>(npes) * (elemsPerPe / npes + 64) * sizeof(uint32_t);
        auto ar = std::make_unique<AllreduceSdma<uint32_t>>(
            myPe, npes, bytesPerPe, outputBufSize, false);

        double serialMs = -1.0;

        if (myPe == 0) {
            printf("\n--- 数据大小: %zu MB/PE ---\n", dataMB);
            printf("%-42s %10s %12s %12s %8s\n",
                   "配置", "时间(ms)", "算法BW(GB/s)", "总线BW(GB/s)", "状态");
            printf("%s\n", std::string(90, '-').c_str());
        }

        // 1) 串行基准
        {
            uint32_t* outBuf = nullptr;
            CHECK_HIP(hipMalloc(&outBuf, bytesPerPe));

            for (int i = 0; i < 10; i++) {
                CHECK_HIP(hipMemcpy(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
                CHECK_HIP(hipMemset(outBuf, 0, bytesPerPe));
                MPI_Barrier(MPI_COMM_WORLD);
                (*ar)(inBuf, outBuf, elemsPerPe, stream);
                CHECK_HIP(hipStreamSynchronize(stream));
                MPI_Barrier(MPI_COMM_WORLD);
            }

            void* transitBuf = ar->getOutputTransitBuffer();
            std::vector<uint32_t> result(elemsPerPe);
            CHECK_HIP(hipMemcpy(result.data(), transitBuf, bytesPerPe, hipMemcpyDeviceToHost));

            bool ok = verifyResult(result.data(), elemsPerPe, computeExpected(npes), myPe);
            int lok = ok ? 1 : 0, gok = 0;
            MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            BenchResult res;
            res.label = "串行 operator()";
            res.dataBytes = bytesPerPe;
            res.chunkBytes = 0;
            res.scatterMode = -1;
            res.passed = (gok == npes);

            if (res.passed) {
                std::vector<double> times;
                for (int i = 0; i < warmup + iterations; i++) {
                    CHECK_HIP(hipMemcpy(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
                    MPI_Barrier(MPI_COMM_WORLD);
                    double t0 = MPI_Wtime();
                    (*ar)(inBuf, outBuf, elemsPerPe, stream);
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
                    res.avgMs = g_avg * 1000.0;
                    res.algoBw = bytesPerPe / g_avg / (1024.0 * 1024.0 * 1024.0);
                    res.busBw = res.algoBw * 2.0 * (npes - 1) / npes;
                    serialMs = res.avgMs;
                    printf("%-42s %10.3f %12.2f %12.2f %8s\n",
                           res.label.c_str(), res.avgMs, res.algoBw, res.busBw, "基准");
                }
            } else {
                if (myPe == 0) printf("  %-40s VERIFY FAILED\n", res.label.c_str());
            }
            allResults.push_back(res);
            CHECK_HIP(hipFree(outBuf));
        }

        if (myPe == 0) {
            serialMsBySize.push_back(serialMs);
        }

        // 2) Pipeline SDMA 整块 — 主赛道
        {
            const char* label = "Pipeline SDMA chunk=default(0)";
            auto res = runBench(label,
                [&](uint32_t* in, uint32_t* out, int n, hipStream_t s) {
                    return ar->pipelined(in, out, n, 0, 0, s);
                },
                inBuf, ar->getOutputTransitBuffer(), hostData, elemsPerPe, bytesPerPe, npes, myPe,
                stream, warmup, iterations, 0, 0);
            if (myPe == 0 && res.passed) {
                printf("%-42s %10.3f %12.2f %12.2f %8s\n",
                       label, res.avgMs, res.algoBw, res.busBw, "SDMA");
                sdmaDefaultMsBySize.push_back(res.avgMs);

                if (serialMs > 0.0) {
                    double fasterPct = (serialMs - res.avgMs) / serialMs * 100.0;
                    printf("\n  >>> 对决: 串行 %.3f ms  vs  Pipeline SDMA %.3f ms", serialMs, res.avgMs);
                    if (fasterPct > 0.0)
                        printf("  |  Pipeline 快 %.1f%%  [%s]\n", fasterPct, "达成目标");
                    else
                        printf("  |  Pipeline 慢 %.1f%%  [%s]\n", -fasterPct, "未达成，继续优化 kernel");
                }
            } else {
                if (myPe == 0) sdmaDefaultMsBySize.push_back(-1.0);
            }
            allResults.push_back(res);
        }

        if (chunkSweep) {
            for (size_t chunkKB : chunkSizesKB) {
                size_t chunkBytes = chunkKB * 1024;
                size_t chunkElems = chunkBytes / sizeof(uint32_t);
                if (chunkBytes > bytesPerPe) continue;

                char label[64];
                snprintf(label, sizeof(label), "Pipeline SDMA chunk=%zuKB", chunkKB);
                auto res = runBench(label,
                    [&](uint32_t* in, uint32_t* out, int n, hipStream_t s) {
                        return ar->pipelined(in, out, n, chunkElems, 0, s);
                    },
                    inBuf, ar->getOutputTransitBuffer(), hostData, elemsPerPe, bytesPerPe, npes, myPe,
                    stream, warmup, iterations, chunkBytes, 0);
                if (myPe == 0 && res.passed) {
                    printf("%-42s %10.3f %12.2f %12.2f %8s\n",
                           label, res.avgMs, res.algoBw, res.busBw, "SDMA");
                }
                allResults.push_back(res);
            }
        }

        // 3) Pipeline P2P 整块（参考，主目标仍是 SDMA）
        {
            const char* label = "Pipeline P2P  chunk=default(0)";
            auto res = runBench(label,
                [&](uint32_t* in, uint32_t* out, int n, hipStream_t s) {
                    return ar->pipelined(in, out, n, 0, 1, s);
                },
                inBuf, ar->getOutputTransitBuffer(), hostData, elemsPerPe, bytesPerPe, npes, myPe,
                stream, warmup, iterations, 0, 1);
            if (myPe == 0 && res.passed) {
                printf("%-42s %10.3f %12.2f %12.2f %8s\n",
                       label, res.avgMs, res.algoBw, res.busBw, "P2P");
            }
            allResults.push_back(res);
        }

        if (chunkSweep) {
            for (size_t chunkKB : chunkSizesKB) {
                size_t chunkBytes = chunkKB * 1024;
                size_t chunkElems = chunkBytes / sizeof(uint32_t);
                if (chunkBytes > bytesPerPe) continue;

                char label[64];
                snprintf(label, sizeof(label), "Pipeline P2P  chunk=%zuKB", chunkKB);
                auto res = runBench(label,
                    [&](uint32_t* in, uint32_t* out, int n, hipStream_t s) {
                        return ar->pipelined(in, out, n, chunkElems, 1, s);
                    },
                    inBuf, ar->getOutputTransitBuffer(), hostData, elemsPerPe, bytesPerPe, npes, myPe,
                    stream, warmup, iterations, chunkBytes, 1);
                if (myPe == 0 && res.passed) {
                    printf("%-42s %10.3f %12.2f %12.2f %8s\n",
                           label, res.avgMs, res.algoBw, res.busBw, "P2P");
                }
                allResults.push_back(res);
            }
        }

        ar.reset();
        CHECK_HIP(hipFree(inBuf));
    }

    if (myPe == 0) {
        printf("\n======================================================================\n");
        printf("汇总: 串行 vs Pipeline SDMA(整块) — 加速比 (>1 为更快)\n");
        printf("======================================================================\n");
        printf("%-10s %12s %18s %12s\n", "MB/PE", "串行(ms)", "Pipeline默认(ms)", "加速比");
        printf("%s\n", std::string(56, '-').c_str());

        for (size_t i = 0; i < dataSizesMB.size(); i++) {
            double s = (i < serialMsBySize.size()) ? serialMsBySize[i] : -1.0;
            double p = (i < sdmaDefaultMsBySize.size()) ? sdmaDefaultMsBySize[i] : -1.0;
            if (s > 0 && p > 0) {
                double speedup = s / p;
                printf("%-10zu %12.3f %18.3f %12.2fx\n",
                       dataSizesMB[i], s, p, speedup);
            } else {
                printf("%-10zu %12s %18s %12s\n",
                       dataSizesMB[i], "—", "—", "—");
            }
        }

        int wins = 0, total = 0;
        for (size_t i = 0; i < dataSizesMB.size(); i++) {
            double s = (i < serialMsBySize.size()) ? serialMsBySize[i] : -1.0;
            double p = (i < sdmaDefaultMsBySize.size()) ? sdmaDefaultMsBySize[i] : -1.0;
            if (s > 0 && p > 0) {
                total++;
                if (p < s) wins++;
            }
        }
        printf("\n  Pipeline SDMA(整块) 在 %d / %d 个数据量上快于串行\n", wins, total);

        printf("\n======================================================================\n");
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

// Pipeline AllReduce SDMA 基准测试
// 遍历不同数据大小、不同 chunk 大小、SDMA/P2P 两种模式
// 输出对比表格，并根据结果判断最优配置

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <cassert>
#include <cstdio>
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

    // Warmup: 第一次调用可能因 L2 缓存残留导致 transit buffer 数据不正确
    // 多次执行让 SDMA 写入的数据填充 L2 缓存
    for (int w = 0; w < 3; w++) {
        CHECK_HIP(hipMemcpy(inBuf, hostData.data(), bytesPerPe, hipMemcpyHostToDevice));
        CHECK_HIP(hipDeviceSynchronize());
        MPI_Barrier(MPI_COMM_WORLD);
        fn(inBuf, outBuf, elemsPerPe, stream);
        CHECK_HIP(hipStreamSynchronize(stream));
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 验证
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
    MPI_Allreduce(&lok, &gok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
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

void testPipelinedAllreduce() {
    MPI_Init(NULL, NULL);
    int status = ShmemMpiInit(MPI_COMM_WORLD);
    assert(!status);

    int myPe = ShmemMyPe();
    int npes = ShmemNPes();

    // 测试数据大小：4MB, 16MB, 32MB, 64MB, 128MB, 256MB (per PE)
    // 从 32MB 开始（小数据量可能遇到 L2 缓存残留问题）
    std::vector<size_t> dataSizesMB = {32, 64, 128, 256};

    // 测试 chunk 大小：128KB, 512KB, 1MB, 2MB, 4MB, 8MB
    std::vector<size_t> chunkSizesKB = {128, 512, 1024, 2048, 4096, 8192};

    const int warmup = 3;
    const int iterations = 5;

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));

    if (myPe == 0) {
        printf("\n======================================================================\n");
        printf("Pipeline AllReduce SDMA 综合基准测试\n");
        printf("  节点数: %d GPU\n", npes);
        printf("  预热次数: %d, 测量次数: %d\n", warmup, iterations);
        printf("======================================================================\n");
    }

    // 存储所有结果用于最终汇总
    std::vector<BenchResult> allResults;

    for (size_t dataMB : dataSizesMB) {
        size_t bytesPerPe = dataMB * 1024 * 1024;
        int elemsPerPe = bytesPerPe / sizeof(uint32_t);
        size_t totalBytes = bytesPerPe * npes;
        uint32_t fillValue = static_cast<uint32_t>((myPe + 1) * 1000);

        uint32_t* inBuf = nullptr;
        CHECK_HIP(hipMalloc(&inBuf, bytesPerPe));
        std::vector<uint32_t> hostData(elemsPerPe, fillValue);

        // copy_output_to_user=false，与 allreduce_sdma_sync 一致
        size_t outputBufSize = static_cast<size_t>(npes) * (elemsPerPe / npes + 64) * sizeof(uint32_t);
        auto ar = std::make_unique<AllreduceSdma<uint32_t>>(
            myPe, npes, bytesPerPe, outputBufSize, false);

        if (myPe == 0) {
            printf("\n--- 数据大小: %zu MB/PE ---\n", dataMB);
            printf("%-42s %10s %12s %12s %8s\n",
                   "配置", "时间(ms)", "算法BW(GB/s)", "总线BW(GB/s)", "状态");
            printf("%s\n", std::string(90, '-').c_str());
        }

        // 1) 基准：串行模式 (当前 operator())
        {
            char label[64];
            snprintf(label, sizeof(label), "串行 operator()");
            auto res = runBench(label,
                [&](uint32_t* in, uint32_t* out, int n, hipStream_t s) {
                    return (*ar)(in, out, n, s);
                },
                inBuf, ar->getOutputTransitBuffer(), hostData, elemsPerPe, bytesPerPe, npes, myPe,
                stream, warmup, iterations, 0, -1);
            if (myPe == 0 && res.passed) {
                printf("%-42s %10.3f %12.2f %12.2f %8s\n",
                       label, res.avgMs, res.algoBw, res.busBw, "基准");
            }
            allResults.push_back(res);
        }

        // 2) Pipeline SDMA scatter 模式，遍历不同 chunk 大小
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

        // 3) Pipeline P2P read 模式，遍历不同 chunk 大小
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

        ar.reset();
        CHECK_HIP(hipFree(inBuf));
    }

    // ================================================================
    // 汇总：每种数据大小下的最优配置
    // ================================================================
    if (myPe == 0) {
        printf("\n======================================================================\n");
        printf("最优配置汇总\n");
        printf("======================================================================\n");
        printf("%-12s %-42s %10s %12s %12s\n",
               "数据大小", "最优配置", "时间(ms)", "算法BW", "总线BW");
        printf("%s\n", std::string(90, '-').c_str());

        for (size_t dataMB : dataSizesMB) {
            size_t targetBytes = dataMB * 1024 * 1024;
            double bestBw = 0;
            const BenchResult* bestRes = nullptr;

            for (const auto& r : allResults) {
                if (r.dataBytes == targetBytes && r.passed && r.algoBw > bestBw) {
                    bestBw = r.algoBw;
                    bestRes = &r;
                }
            }

            if (bestRes) {
                printf("%-12s %-42s %10.3f %12.2f %12.2f\n",
                       (std::to_string(dataMB) + " MB").c_str(),
                       bestRes->label.c_str(),
                       bestRes->avgMs, bestRes->algoBw, bestRes->busBw);
            }
        }

        // 模式对比总结
        printf("\n--- 模式对比（所有数据大小平均） ---\n");
        double serialAvg = 0, sdmaAvg = 0, p2pAvg = 0;
        int serialCnt = 0, sdmaCnt = 0, p2pCnt = 0;
        for (const auto& r : allResults) {
            if (!r.passed) continue;
            if (r.scatterMode == -1) { serialAvg += r.algoBw; serialCnt++; }
            else if (r.scatterMode == 0) { sdmaAvg += r.algoBw; sdmaCnt++; }
            else if (r.scatterMode == 1) { p2pAvg += r.algoBw; p2pCnt++; }
        }
        if (serialCnt > 0) printf("  串行模式平均算法BW:     %.2f GB/s (%d 个测试)\n", serialAvg / serialCnt, serialCnt);
        if (sdmaCnt > 0)   printf("  Pipeline SDMA 平均算法BW: %.2f GB/s (%d 个测试)\n", sdmaAvg / sdmaCnt, sdmaCnt);
        if (p2pCnt > 0)    printf("  Pipeline P2P  平均算法BW: %.2f GB/s (%d 个测试)\n", p2pAvg / p2pCnt, p2pCnt);

        printf("\n======================================================================\n");
    }

    CHECK_HIP(hipStreamDestroy(stream));
    MPI_Barrier(MPI_COMM_WORLD);
    ShmemFinalize();
}

int main(int argc, char* argv[]) {
    testPipelinedAllreduce();
    return 0;
}

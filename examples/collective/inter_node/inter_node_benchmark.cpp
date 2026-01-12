// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <getopt.h>
#include <hip/hip_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mori/application/utils/check.hpp"
#include "mori/collective/core/allreduce_config.hpp"
#include "mori/collective/core/allreduce_executor.hpp"
#include "mori/collective/core/topology_detector.hpp"
#include "mori/collective/inter_node/executors/one_shot.hpp"
#include "mori/collective/inter_node/executors/ring_1d.hpp"
#include "mori/shmem/shmem.hpp"

namespace {
inline void hipCheck(hipError_t err, const char* expr, const char* file, int line) {
  if (err != hipSuccess) {
    std::ostringstream oss;
    oss << "HIP error: " << hipGetErrorString(err) << " (" << static_cast<int>(err) << ") at "
        << file << ":" << line << " in " << expr;
    throw std::runtime_error(oss.str());
  }
}
}  // namespace

#define HIP_CHECK(expr) hipCheck((expr), #expr, __FILE__, __LINE__)

// Algorithm selection
enum class BenchAlgorithm { ONE_SHOT, RING_1D };

// Default arguments
struct Args {
  size_t min_bytes = 1024;            // 1KB
  size_t max_bytes = (128ull << 20);  // 128MB
  size_t step_factor = 2;
  int warmup_iters = 5;
  int iters = 20;
  bool check = false;
  BenchAlgorithm algorithm = BenchAlgorithm::ONE_SHOT;
  int max_blocks = 80;
  int threads_per_block = 512;
};

void printUsage(const char* prog) {
  std::cerr << "Usage: " << prog << " [options]\n"
            << "Options:\n"
            << "  -b <size>    Minimum buffer size (e.g., 1K, 1M, 1G). Default: 1K\n"
            << "  -e <size>    Maximum buffer size. Default: 128M\n"
            << "  -f <factor>  Step factor for size progression. Default: 2\n"
            << "  -w <iters>   Warmup iterations. Default: 5\n"
            << "  -n <iters>   Benchmark iterations. Default: 20\n"
            << "  -c           Enable correctness check\n"
            << "  -a <algo>    Algorithm: 'oneshot' or 'ring'. Default: oneshot\n"
            << "  -B <blocks>  Max blocks for kernel launch. Default: 80\n"
            << "  -T <threads> Threads per block. Default: 512\n"
            << "  -h           Show this help message\n";
}

size_t parseSize(const std::string& s) {
  std::string str = s;
  size_t multiplier = 1;
  if (!str.empty()) {
    char suffix = str.back();
    if (suffix == 'G' || suffix == 'g') {
      multiplier = 1ull << 30;
      str.pop_back();
    } else if (suffix == 'M' || suffix == 'm') {
      multiplier = 1ull << 20;
      str.pop_back();
    } else if (suffix == 'K' || suffix == 'k') {
      multiplier = 1ull << 10;
      str.pop_back();
    }
  }
  return std::stoull(str) * multiplier;
}

void parseArgs(int argc, char** argv, Args& args) {
  int opt;
  while ((opt = getopt(argc, argv, "b:e:f:w:n:ca:B:T:h")) != -1) {
    switch (opt) {
      case 'b':
        args.min_bytes = parseSize(optarg);
        break;
      case 'e':
        args.max_bytes = parseSize(optarg);
        break;
      case 'f':
        args.step_factor = std::stoull(optarg);
        if (args.step_factor < 2) args.step_factor = 2;
        break;
      case 'w':
        args.warmup_iters = std::stoi(optarg);
        break;
      case 'n':
        args.iters = std::stoi(optarg);
        break;
      case 'c':
        args.check = true;
        break;
      case 'a':
        if (std::string(optarg) == "ring" || std::string(optarg) == "ring1d") {
          args.algorithm = BenchAlgorithm::RING_1D;
        } else {
          args.algorithm = BenchAlgorithm::ONE_SHOT;
        }
        break;
      case 'B':
        args.max_blocks = std::stoi(optarg);
        break;
      case 'T':
        args.threads_per_block = std::stoi(optarg);
        break;
      case 'h':
        printUsage(argv[0]);
        MPI_Finalize();
        exit(0);
      default:
        break;
    }
  }
}

std::string formatBytes(size_t bytes) {
  char buf[64];
  if (bytes < 1024)
    snprintf(buf, sizeof(buf), "%4zu  B", bytes);
  else if (bytes < 1024 * 1024)
    snprintf(buf, sizeof(buf), "%4.0f KB", bytes / 1024.0);
  else if (bytes < 1024 * 1024 * 1024)
    snprintf(buf, sizeof(buf), "%4.0f MB", bytes / (1024.0 * 1024.0));
  else
    snprintf(buf, sizeof(buf), "%4.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
  return std::string(buf);
}

/**
 * Calculate bus bandwidth factor for AllReduce algorithms
 */
inline double calculateBusFactor(BenchAlgorithm algo, int num_ranks) {
  if (num_ranks <= 1) return 1.0;

  switch (algo) {
    case BenchAlgorithm::ONE_SHOT:
      return static_cast<double>(num_ranks - 1);

    case BenchAlgorithm::RING_1D:
      return 2.0 * static_cast<double>(num_ranks - 1) / static_cast<double>(num_ranks);

    default:
      return 1.0;
  }
}

std::string algorithmName(BenchAlgorithm algo) {
  switch (algo) {
    case BenchAlgorithm::ONE_SHOT:
      return "OneShot";
    case BenchAlgorithm::RING_1D:
      return "Ring1D";
    default:
      return "Unknown";
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  Args args;
  parseArgs(argc, argv, args);

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    if (rank == 0) std::cerr << "No HIP devices found." << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int device_id = rank % device_count;
  HIP_CHECK(hipSetDevice(device_id));

  int status = mori::shmem::ShmemMpiInit(MPI_COMM_WORLD);
  if (status != 0) {
    if (rank == 0) std::cerr << "Failed to initialize SHMEM" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int myPe = mori::shmem::ShmemMyPe();
  int npes = mori::shmem::ShmemNPes();

  if (rank == 0) {
    std::cout << "# Mori Inter-Node AllReduce Benchmark" << std::endl;
    std::cout << "# Ranks: " << num_ranks << ", Algorithm: " << algorithmName(args.algorithm)
              << std::endl;
    std::cout << "# MaxBlocks: " << args.max_blocks
              << ", ThreadsPerBlock: " << args.threads_per_block << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(15) << "size" << std::setw(15) << "count" << std::setw(10) << "type"
              << std::setw(10) << "redop" << std::setw(15) << "time(us)" << std::setw(15)
              << "algbw(GB/s)" << std::setw(15) << "busbw(GB/s)" << std::setw(10) << "status"
              << std::endl;
  }

  mori::collective::AllReduceConfig config;
  config.maxBlocks = args.max_blocks;
  config.threadsPerBlock = args.threads_per_block;

  using T = float;
  std::unique_ptr<mori::collective::AllReduceExecutor<T>> executor;

  try {
    if (args.algorithm == BenchAlgorithm::ONE_SHOT) {
      executor =
          std::make_unique<mori::collective::OneShotAllReduceExecutor<T>>(npes, myPe, config);
    } else {
      executor = std::make_unique<mori::collective::Ring1DAllReduceExecutor<T>>(npes, myPe, config);
    }
  } catch (const std::exception& e) {
    if (rank == 0) std::cerr << "Failed to initialize executor: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  T* d_in = nullptr;
  T* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, args.max_bytes));
  HIP_CHECK(hipMalloc(&d_out, args.max_bytes));

  std::vector<T> h_input;
  std::vector<T> h_output;

  if (args.check) {
    h_input.resize(args.max_bytes / sizeof(T));
    h_output.resize(args.max_bytes / sizeof(T));
  }

  if (args.algorithm == BenchAlgorithm::ONE_SHOT) {
    auto* oneShotExec =
        dynamic_cast<mori::collective::OneShotAllReduceExecutor<T>*>(executor.get());
    if (oneShotExec) {
      int rc = oneShotExec->RegisterBuffers(d_in, d_out, args.max_bytes / sizeof(T));
      if (rc != 0 && rank == 0) {
        std::cerr << "Warning: Failed to pre-register buffers, using on-the-fly registration"
                  << std::endl;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Benchmark loop
  for (size_t bytes = args.min_bytes; bytes <= args.max_bytes; bytes *= args.step_factor) {
    size_t count = bytes / sizeof(T);
    if (count == 0) continue;

    if (args.check) {
      for (size_t i = 0; i < count; ++i) h_input[i] = 1.0f;
      HIP_CHECK(hipMemcpy(d_in, h_input.data(), count * sizeof(T), hipMemcpyHostToDevice));
    } else {
      HIP_CHECK(hipMemsetAsync(d_in, 0, bytes, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
    }

    HIP_CHECK(hipMemsetAsync(d_out, 0, bytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    bool warmup_ok = true;
    for (int i = 0; i < args.warmup_iters; ++i) {
      int rc = executor->Execute(d_in, d_out, count, stream);
      if (rc != 0) {
        std::cerr << "Rank " << rank << " Execute() failed during warmup, rc=" << rc << std::endl;
        warmup_ok = false;
        break;
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    int warmup_success = warmup_ok ? 1 : 0;
    int all_warmup_success = 0;
    MPI_Allreduce(&warmup_success, &all_warmup_success, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

    if (!all_warmup_success) {
      if (rank == 0) {
        std::cout << std::setw(15) << formatBytes(bytes) << std::setw(15) << count << std::setw(10)
                  << "float" << std::setw(10) << "sum" << std::setw(15) << "N/A" << std::setw(15)
                  << "N/A" << std::setw(15) << "N/A" << std::setw(10) << "FAIL" << std::endl;
      }
      continue;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (args.check) {
      HIP_CHECK(hipMemcpy(d_in, h_input.data(), count * sizeof(T), hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipMemsetAsync(d_out, 0, bytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) {
      int rc = executor->Execute(d_in, d_out, count, stream);
      if (rc != 0) {
        std::cerr << "Rank " << rank << " Execute() failed during measure, rc=" << rc << std::endl;
        break;
      }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    MPI_Barrier(MPI_COMM_WORLD);

    std::string verify_status = "OK";
    if (args.check) {
      HIP_CHECK(hipMemcpy(h_output.data(), d_out, count * sizeof(T), hipMemcpyDeviceToHost));
      bool correct = true;
      T expected = static_cast<T>(num_ranks);
      for (size_t i = 0; i < count; ++i) {
        if (std::abs(h_output[i] - expected) > 1e-3f) {
          correct = false;
          if (rank == 0) {
            std::cerr << "Verification failed at index " << i << ": expected " << expected
                      << ", got " << h_output[i] << std::endl;
          }
          break;
        }
      }
      int ok = correct ? 1 : 0;
      int all_ok = 0;
      MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
      verify_status = all_ok ? "PASS" : "FAIL";
    }

    double elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    double avg_time_us_local = elapsed_us / args.iters;
    double avg_time_us = 0.0;
    MPI_Reduce(&avg_time_us_local, &avg_time_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
      double alg_bw = (double)bytes / (avg_time_us * 1e-6) / 1e9;
      double bus_factor = calculateBusFactor(args.algorithm, num_ranks);
      double bus_bw = alg_bw * bus_factor;

      std::cout << std::setw(15) << formatBytes(bytes) << std::setw(15) << count << std::setw(10)
                << "float" << std::setw(10) << "sum" << std::setw(15) << std::fixed
                << std::setprecision(2) << avg_time_us << std::setw(15) << std::fixed
                << std::setprecision(2) << alg_bw << std::setw(15) << std::fixed
                << std::setprecision(2) << bus_bw << std::setw(10) << verify_status << std::endl;
    }
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipStreamDestroy(stream));

  executor.reset();

  mori::shmem::ShmemFinalize();

  return 0;
}

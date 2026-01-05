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
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mori/collective/intra_node/executor.hpp"

// Default arguments
struct Args {
  size_t min_bytes = 1024;           // 1KB
  size_t max_bytes = 128 * 1 << 20;  // 128MB
  size_t step_factor = 2;
  int warmup_iters = 5;
  int iters = 20;
  bool check = false;
};

void parseArgs(int argc, char** argv, Args& args) {
  int opt;
  while ((opt = getopt(argc, argv, "b:e:f:w:n:c")) != -1) {
    switch (opt) {
      case 'b': {
        std::string s(optarg);
        size_t multiplier = 1;
        if (s.back() == 'G')
          multiplier = 1 << 30;
        else if (s.back() == 'M')
          multiplier = 1 << 20;
        else if (s.back() == 'K')
          multiplier = 1 << 10;
        if (multiplier > 1) s.pop_back();
        args.min_bytes = std::stoull(s) * multiplier;
        break;
      }
      case 'e': {
        std::string s(optarg);
        size_t multiplier = 1;
        if (s.back() == 'G')
          multiplier = 1 << 30;
        else if (s.back() == 'M')
          multiplier = 1 << 20;
        else if (s.back() == 'K')
          multiplier = 1 << 10;
        if (multiplier > 1) s.pop_back();
        args.max_bytes = std::stoull(s) * multiplier;
        break;
      }
      case 'f':
        args.step_factor = std::stoull(optarg);
        if (args.step_factor < 2) args.step_factor = 2;  // prevent infinite loops
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
      default:
        break;
    }
  }
}

std::string formatBytes(size_t bytes) {
  char buf[64];
  if (bytes < 1024)
    snprintf(buf, sizeof(buf), "%4lu  B", bytes);
  else if (bytes < 1024 * 1024)
    snprintf(buf, sizeof(buf), "%4.0f KB", bytes / 1024.0);
  else if (bytes < 1024 * 1024 * 1024)
    snprintf(buf, sizeof(buf), "%4.0f MB", bytes / (1024.0 * 1024.0));
  else
    snprintf(buf, sizeof(buf), "%4.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
  return std::string(buf);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, num_ranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

  Args args;
  parseArgs(argc, argv, args);

  if (rank == 0) {
    std::cout << "# Mori Intra-Node AllReduce Benchmark" << std::endl;
    std::cout << "# Dist: " << num_ranks << " ranks" << std::endl;
    std::cout << std::endl;
    std::cout << std::setw(15) << "size" << std::setw(15) << "count" << std::setw(10) << "type"
              << std::setw(10) << "redop" << std::setw(15) << "time(us)" << std::setw(15)
              << "algbw(GB/s)" << std::setw(15) << "busbw(GB/s)" << std::endl;
  }

  hipSetDevice(rank);

  using T = float;
  size_t executor_max_size = std::max((size_t)8 * 1024 * 1024, args.max_bytes);

  std::unique_ptr<mori::collective::AllReduceExecutor<T>> executor;
  try {
    executor = std::make_unique<mori::collective::IntraNodeAllReduceExecutor<T>>(
        num_ranks, rank, MPI_COMM_WORLD, executor_max_size);
  } catch (const std::exception& e) {
    if (rank == 0) std::cerr << "Failed to initialize executor: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  hipStream_t stream;
  hipStreamCreate(&stream);

  T* d_data = nullptr;
  hipMalloc(&d_data, args.max_bytes);

  std::vector<T> h_input;
  std::vector<T> h_output;

  if (args.check) {
    h_input.resize(args.max_bytes / sizeof(T));
    h_output.resize(args.max_bytes / sizeof(T));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (size_t bytes = args.min_bytes; bytes <= args.max_bytes; bytes *= args.step_factor) {
    size_t count = bytes / sizeof(T);
    if (count == 0) continue;

    if (args.check) {
      for (size_t i = 0; i < count; ++i) h_input[i] = 1.0f;
      hipMemcpy(d_data, h_input.data(), count * sizeof(T), hipMemcpyHostToDevice);
    }

    // Warmup
    for (int i = 0; i < args.warmup_iters; ++i) {
      executor->Execute(d_data, d_data, count, stream);
    }
    hipStreamSynchronize(stream);
    MPI_Barrier(MPI_COMM_WORLD);

    // Measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < args.iters; ++i) {
      executor->Execute(d_data, d_data, count, stream);
    }
    hipStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    if (args.check) {
      hipMemcpy(h_output.data(), d_data, count * sizeof(T), hipMemcpyDeviceToHost);
      bool correct = true;
      for (size_t i = 0; i < count; ++i) {
        if (std::abs(h_output[i] - (float)num_ranks) > 1e-5) {
          correct = false;
          break;
        }
      }
      if (!correct) {
        std::cerr << "Rank " << rank << " Verification FAILED at size " << formatBytes(bytes)
                  << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      double elapsed_us =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
      double avg_time_us = elapsed_us / args.iters;

      double alg_bw = (double)bytes / (avg_time_us * 1e-6) / 1e9;  // GB/s
      double bus_bw = alg_bw * (2.0 * (num_ranks - 1) / num_ranks);

      std::cout << std::setw(15) << formatBytes(bytes) << std::setw(15) << count << std::setw(10)
                << "float" << std::setw(10) << "sum" << std::setw(15) << std::fixed
                << std::setprecision(2) << avg_time_us << std::setw(15) << std::fixed
                << std::setprecision(2) << alg_bw << std::setw(15) << std::fixed
                << std::setprecision(2) << bus_bw << std::endl;
    }
  }

  hipFree(d_data);
  hipStreamDestroy(stream);

  MPI_Finalize();
  return 0;
}

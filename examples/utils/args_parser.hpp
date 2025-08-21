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
#pragma once

#include <getopt.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "common_utils.hpp"
#include "mori/core/transport/rdma/primitives.hpp"

using namespace mori::core;

// Enums and types would typically be defined in a header file
enum MORIDataType {
  MORI_INT,
  MORI_LONG,
  MORI_LONGLONG,
  MORI_ULONGLONG,
  MORI_SIZE,
  MORI_PTRDIFF,
  MORI_FLOAT,
  MORI_DOUBLE,
  MORI_UINT,
  MORI_INT32,
  MORI_INT64,
  MORI_UINT32,
  MORI_UINT64,
  MORI_FP16,
  MORI_BF16
};

enum MORIReduceOp { MORI_MIN, MORI_MAX, MORI_SUM, MORI_PROD, MORI_AND, MORI_OR, MORI_XOR };

enum MORIScope { MORI_THREAD, MORI_WARP, MORI_BLOCK, MORI_ALL_SCOPES };

enum Direction { READ, WRITE };

enum PutGetIssue { ON_STREAM, HOST };

struct Datatype {
  MORIDataType type;
  size_t size;
  std::string name;
};

struct ReduceOp {
  MORIReduceOp type;
  std::string name;
};

struct ThreadgroupScope {
  MORIScope type;
  std::string name;
};

struct AMO {
  atomicType type;
  std::string name;
};

struct DirectionConfig {
  Direction type;
  std::string name;
};

struct PutGetIssueConfig {
  PutGetIssue type;
  std::string name;
};

class BenchmarkConfig {
 public:
  // 解析命令行
  void readArgs(int argc, char** argv);

  // 简单 getter
  size_t getMinSize() const { return min_size; }
  size_t getMaxSize() const { return max_size; }
  size_t getNumBlocks() const { return num_blocks; }
  size_t getThreadsPerBlock() const { return threads_per_block; }
  size_t getIters() const { return iters; }
  size_t getWarmupIters() const { return warmup_iters; }
  size_t getStepFactor() const { return step_factor; }
  size_t getMaxSizeLog() const { return max_size_log; }
  size_t getStride() const { return stride; }

  bool isBidirectional() const { return bidirectional; }
  bool isReportMsgrate() const { return report_msgrate; }

  Datatype getDatatype() const { return datatype; }
  ReduceOp getReduceOp() const { return reduce_op; }
  ThreadgroupScope getThreadgroupScope() const { return threadgroup_scope; }
  AMO getTestAMO() const { return test_amo; }
  PutGetIssueConfig getPutGetIssue() const { return putget_issue; }
  DirectionConfig getDirection() const { return dir; }

 private:
  size_t min_size = 4;
  size_t max_size = min_size * 1024 * 1024;
  size_t num_blocks = 1;
  size_t threads_per_block = 1;
  size_t iters = 10;
  size_t warmup_iters = 5;
  size_t step_factor = 2;
  size_t max_size_log = 1;
  size_t stride = 1;

  bool bidirectional = false;
  bool report_msgrate = false;

  Datatype datatype = {MORI_UINT64, sizeof(uint64_t), "uint64_t"};
  ReduceOp reduce_op = {MORI_SUM, "sum"};
  ThreadgroupScope threadgroup_scope = {MORI_ALL_SCOPES, "all_scopes"};
  AMO test_amo = {AMO_INC, "inc"};
  PutGetIssueConfig putget_issue = {ON_STREAM, "on_stream"};
  DirectionConfig dir = {WRITE, "write"};

  void datatypeParse(const char* optarg);
  void reduceOpParse(const char* str);
  void atomicOpParse(const char* str);
  int atolScaled(const char* str, size_t* out);
};

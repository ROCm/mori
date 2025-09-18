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
#include "args_parser.hpp"

#include <cctype>

void BenchmarkConfig::datatypeParse(const char* optarg) {
  if (!strcmp(optarg, "int")) {
    datatype = {MORI_INT, sizeof(int), "int"};
  } else if (!strcmp(optarg, "long")) {
    datatype = {MORI_LONG, sizeof(long), "long"};
  } else if (!strcmp(optarg, "longlong")) {
    datatype = {MORI_LONGLONG, sizeof(long long), "longlong"};
  } else if (!strcmp(optarg, "ulonglong")) {
    datatype = {MORI_ULONGLONG, sizeof(unsigned long long), "ulonglong"};
  } else if (!strcmp(optarg, "size")) {
    datatype = {MORI_SIZE, sizeof(size_t), "size"};
  } else if (!strcmp(optarg, "ptrdiff")) {
    datatype = {MORI_PTRDIFF, sizeof(std::ptrdiff_t), "ptrdiff"};
  } else if (!strcmp(optarg, "float")) {
    datatype = {MORI_FLOAT, sizeof(float), "float"};
  } else if (!strcmp(optarg, "double")) {
    datatype = {MORI_DOUBLE, sizeof(double), "double"};
  } else if (!strcmp(optarg, "uint")) {
    datatype = {MORI_UINT, sizeof(unsigned int), "uint"};
  } else if (!strcmp(optarg, "int32")) {
    datatype = {MORI_INT32, sizeof(int32_t), "int32"};
  } else if (!strcmp(optarg, "int64")) {
    datatype = {MORI_INT64, sizeof(int64_t), "int64"};
  } else if (!strcmp(optarg, "uint32")) {
    datatype = {MORI_UINT32, sizeof(uint32_t), "uint32"};
  } else if (!strcmp(optarg, "uint64")) {
    datatype = {MORI_UINT64, sizeof(uint64_t), "uint64"};
  } else if (!strcmp(optarg, "fp16")) {
    datatype = {MORI_FP16, sizeof(__fp16), "fp16"};
  } else if (!strcmp(optarg, "bf16")) {
    datatype = {MORI_BF16, sizeof(__bf16), "bf16"};
  }
}

void BenchmarkConfig::reduceOpParse(const char* str) {
  if (!strcmp(str, "min")) {
    reduce_op = {MORI_MIN, "min"};
  } else if (!strcmp(str, "max")) {
    reduce_op = {MORI_MAX, "max"};
  } else if (!strcmp(str, "sum")) {
    reduce_op = {MORI_SUM, "sum"};
  } else if (!strcmp(str, "prod")) {
    reduce_op = {MORI_PROD, "prod"};
  } else if (!strcmp(str, "and")) {
    reduce_op = {MORI_AND, "and"};
  } else if (!strcmp(str, "or")) {
    reduce_op = {MORI_OR, "or"};
  } else if (!strcmp(str, "xor")) {
    reduce_op = {MORI_XOR, "xor"};
  }
}

void BenchmarkConfig::atomicOpParse(const char* str) {
  size_t string_length = strnlen(str, 20);
  if (strncmp(str, "inc", string_length) == 0) {
    test_amo = {AMO_INC, "inc"};
  } else if (strncmp(str, "fetch_inc", string_length) == 0) {
    test_amo = {AMO_FETCH_INC, "fetch_inc"};
  } else if (strncmp(str, "set", string_length) == 0) {
    test_amo = {AMO_SET, "set"};
  } else if (strncmp(str, "add", string_length) == 0) {
    test_amo = {AMO_ADD, "add"};
  } else if (strncmp(str, "fetch_add", string_length) == 0) {
    test_amo = {AMO_FETCH_ADD, "fetch_add"};
  } else if (strncmp(str, "and", string_length) == 0) {
    test_amo = {AMO_AND, "and"};
  } else if (strncmp(str, "fetch_and", string_length) == 0) {
    test_amo = {AMO_FETCH_AND, "fetch_and"};
  } else if (strncmp(str, "or", string_length) == 0) {
    test_amo = {AMO_OR, "or"};
  } else if (strncmp(str, "fetch_or", string_length) == 0) {
    test_amo = {AMO_FETCH_OR, "fetch_or"};
  } else if (strncmp(str, "xor", string_length) == 0) {
    test_amo = {AMO_XOR, "xor"};
  } else if (strncmp(str, "fetch_xor", string_length) == 0) {
    test_amo = {AMO_FETCH_XOR, "fetch_xor"};
  } else if (strncmp(str, "swap", string_length) == 0) {
    test_amo = {AMO_SWAP, "swap"};
  } else if (strncmp(str, "compare_swap", string_length) == 0) {
    test_amo = {AMO_COMPARE_SWAP, "compare_swap"};
  } else {
    test_amo = {AMO_ACK, "ack"};
  }
}

int BenchmarkConfig::atolScaled(const char* str, size_t* out) {
  int scale = 0, n;
  double p = -1.0;
  char f = 0;

  n = sscanf(str, "%lf%c", &p, &f);
  if (n == 2) {
    switch (f) {
      case 'k':
      case 'K':
        scale = 10;
        break;
      case 'm':
      case 'M':
        scale = 20;
        break;
      case 'g':
      case 'G':
        scale = 30;
        break;
      case 't':
      case 'T':
        scale = 40;
        break;
      default:
        return 1;
    }
  } else if (p < 0) {
    return 1;
  }
  *out = static_cast<size_t>(std::ceil(p * (1lu << scale)));
  return 0;
}

void BenchmarkConfig::readArgs(int argc, char** argv) {
  int c;
  static struct option long_opts[] = {{"bidir", no_argument, 0, 0},
                                      {"report_msgrate", no_argument, 0, 0},
                                      {"dir", required_argument, 0, 0},
                                      {"issue", required_argument, 0, 0},
                                      {"help", no_argument, 0, 'h'},
                                      {"min_size", required_argument, 0, 'b'},
                                      {"max_size", required_argument, 0, 'e'},
                                      {"step", required_argument, 0, 'f'},
                                      {"iters", required_argument, 0, 'n'},
                                      {"warmup_iters", required_argument, 0, 'w'},
                                      {"ctas", required_argument, 0, 'c'},
                                      {"threads_per_cta", required_argument, 0, 't'},
                                      {"datatype", required_argument, 0, 'd'},
                                      {"reduce_op", required_argument, 0, 'o'},
                                      {"scope", required_argument, 0, 's'},
                                      {"atomic_op", required_argument, 0, 'a'},
                                      {"stride", required_argument, 0, 'i'},
                                      {"num_qp", required_argument, 0, 'q'},
                                      {0, 0, 0, 0}};

  int opt_idx = 0;
  while ((c = getopt_long(argc, argv, "hb:e:f:n:w:c:t:d:o:s:a:i:q:", long_opts, &opt_idx)) != -1) {
    switch (c) {
      case 'h':
        printf(
            "Accepted arguments: \n"
            "-b, --min_size <minbytes> \n"
            "-e, --max_size <maxbytes> \n"
            "-f, --step <step factor for message sizes> \n"
            "-n, --iters <number of iterations> \n"
            "-w, --warmup_iters <number of warmup iterations> \n"
            "-c, --ctas <number of CTAs to launch> \n"
            "-t, --threads_per_cta <number of threads per block> \n"
            "-d, --datatype: "
            "<int, int32_t, uint32_t, int64_t, uint64_t, long, longlong, ulonglong, size, "
            "ptrdiff, float, double, fp16, bf16> \n"
            "-o, --reduce_op <min, max, sum, prod, and, or, xor> \n"
            "-s, --scope <thread, warp, block, all> \n"
            "-i, --stride stride between elements \n"
            "-a, --atomic_op <inc, add, and, or, xor, set, swap, fetch_<inc, add, and, or, "
            "xor>, compare_swap> \n"
            "--bidir: run bidirectional test \n"
            "--msgrate: report message rate (MMPs)\n"
            "--dir: <read, write> (whether to run put or get operations) \n"
            "--issue: <on_stream, host> (applicable in some host pt-to-pt tests) \n");
        exit(0);
      case 0: {
        auto name = long_opts[opt_idx].name;
        if (!strcmp(name, "bidir"))
          bidirectional = true;
        else if (!strcmp(name, "report_msgrate"))
          report_msgrate = true;
        else if (!strcmp(name, "dir"))
          dir = (!strcmp(optarg, "read") ? DirectionConfig{READ, "read"}
                                         : DirectionConfig{WRITE, "write"});
        else if (!strcmp(name, "issue"))
          putget_issue = (!strcmp(optarg, "on_stream") ? PutGetIssueConfig{ON_STREAM, "on_stream"}
                                                       : PutGetIssueConfig{HOST, "host"});
        break;
      }
      case 'b':
        atolScaled(optarg, &min_size);
        break;
      case 'e':
        atolScaled(optarg, &max_size);
        break;
      case 'f':
        atolScaled(optarg, &step_factor);
        break;
      case 'n':
        atolScaled(optarg, &iters);
        break;
      case 'w':
        atolScaled(optarg, &warmup_iters);
        break;
      case 'c':
        atolScaled(optarg, &num_blocks);
        break;
      case 't':
        atolScaled(optarg, &threads_per_block);
        break;
      case 'i':
        atolScaled(optarg, &stride);
        break;
      case 'd':
        datatypeParse(optarg);
        break;
      case 'o':
        reduceOpParse(optarg);
        break;
      case 's':
        if (!strcmp(optarg, "thread"))
          threadgroup_scope = {MORI_THREAD, "thread"};
        else if (!strcmp(optarg, "warp"))
          threadgroup_scope = {MORI_WARP, "warp"};
        else if (!strcmp(optarg, "block"))
          threadgroup_scope = {MORI_BLOCK, "block"};
        break;
      case 'a':
        atomicOpParse(optarg);
        break;
      case 'q':
        atolScaled(optarg, &num_qp);
        break;
      case '?':
        if (optopt == 'c') {
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        } else if (isprint(optopt)) {
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        return;
      default:
        abort();
    }
  }

  max_size_log = 1;
  size_t tmp = max_size;
  while (tmp) {
    max_size_log++;
    tmp >>= 1;
  }

  assert(min_size <= max_size);

  printf("Runtime options after parsing command line arguments \n");
  printf(
      "min_size: %zu, max_size: %zu, num_qp: %zu, step_factor: %zu, iterations: %zu, "
      "warmup iterations: %zu, number of ctas: %zu, threads per cta: %zu "
      "stride: %zu, datatype: %s, reduce_op: %s, threadgroup_scope: %s, "
      "atomic_op: %s, dir: %s, report_msgrate: %d, bidirectional: %d, "
      "putget_issue: %s\n",
      min_size, max_size, num_qp, step_factor, iters, warmup_iters, num_blocks, threads_per_block,
      stride, datatype.name.c_str(), reduce_op.name.c_str(), threadgroup_scope.name.c_str(),
      test_amo.name.c_str(), dir.name.c_str(), report_msgrate, bidirectional,
      putget_issue.name.c_str());
  printf(
      "Note: Above is full list of options, any given test will use only a "
      "subset of these variables.\n");
}

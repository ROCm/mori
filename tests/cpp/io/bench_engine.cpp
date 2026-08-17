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
// bench_engine.cpp
//
// A C++ benchmark for MORI-IO whose measurement methodology exactly matches
// nixlbench (NVIDIA NIXL's xferbench). The point is an apples-to-apples RDMA
// throughput/latency comparison between MORI-IO and NIXL on the same fabric,
// eliminating the gaps that exist between MORI's Python benchmark and nixlbench:
//
//   1. WHOLE-LOOP TIMER (not per-iteration sum).
//      nixlbench wraps ONE timer around the entire num_iter loop
//      (nixl_worker.cpp: total_timer -> total_duration = total_timer.lap()),
//      then throughput = total_bytes / total_duration. MORI's Python bench
//      instead times each iteration (launch+transfer) and SUMS them, which
//      excludes the inter-iteration gap. Here we use a single whole-loop timer.
//
//   2. INLINE SPIN-POLL COMPLETION in the benchmark thread.
//      nixlbench polls agent->getXferStatus() in a tight spin (NIXL_IN_PROG ->
//      continue) directly in the bench thread. We mirror that exactly: the timed
//      loop spins on the transfer status flag (stored by MORI's CQ worker
//      thread). Completion is ALWAYS spin here -- never a cv-block Wait() -- so
//      the measured latency matches nixl (a cv wakeup would add ~5-10us and make
//      numbers non-comparable). MORI's blocking Wait() path is exercised only by
//      the Python benchmark, not this nixl-parity tool.
//
//   3. WARMUP excluded from timing (nixl: --warmup_iter, default 100).
//
//   4. throughput_gb = total_bytes / 1e9 / (total_duration_us / 1e6), GB=10^9,
//      identical unit to nixl. total_bytes = msg * batch * num_iter, and
//      avg_latency = total_duration / (num_iter * batch) -- i.e. PER SINGLE
//      TRANSFER, matching nixl's total_duration/(per_thread_iter*batch_size)
//      (nixl counts block_size*batch_size descriptors per request).
//
// Rendezvous: 2 processes (one per node) exchange EngineDesc and MemoryDesc over
// mori::application::SocketBootstrapNetwork (msgpack, same as MORI's pybind
// pack()/unpack()).
// Rank 0 = initiator, rank 1 = target. Initiator drives all transfers (RDMA
// one-sided WRITE/READ); target only registers memory and waits.
//
// Build: the `bench_engine` target in tests/cpp/CMakeLists.txt (needs
// -DBUILD_IO=ON). Run `bench_engine --help` for the flag list, and see
// docs/MORI-IO-BENCHMARK.md for the two-node walkthrough.

#include <hip/hip_runtime.h>
#include <signal.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <msgpack.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mori/application/bootstrap/socket_bootstrap.hpp"
#include "mori/io/io.hpp"

using namespace mori::io;
using Clock = std::chrono::steady_clock;

namespace {

// A local HIP_CHECK, as in the other standalone tests and benchmarks. It throws
// rather than terminating, like the one in examples/collective/intra_node, so a
// failure unwinds through main() and releases the memory registration before the
// pages it covers; mori::application::HIP_RUNTIME_CHECK cannot be used here
// because it exits the process and would leave the MR behind.
[[noreturn]] void ThrowHipError(hipError_t e, const char* file, int line) {
  throw std::runtime_error(std::string("HIP error ") + hipGetErrorString(e) + " at " + file + ":" +
                           std::to_string(line));
}

}  // namespace

#define HIP_CHECK(expr)                                          \
  do {                                                           \
    hipError_t _e = (expr);                                      \
    if (_e != hipSuccess) ThrowHipError(_e, __FILE__, __LINE__); \
  } while (0)

namespace {

// --------------------------- buffer ownership -------------------------------
// Owns the benchmark buffer together with its NIC registration. The backend
// holds an MR over these pages, so the registration has to be dropped before
// they are released -- on every exit path, including one taken by an exception
// from the engine or the bootstrap. That ordering requirement is why this is a
// scope guard instead of two calls at the end of the run.
class BenchBuffer {
 public:
  BenchBuffer(IOEngine& engine, size_t bytes, int device, MemoryLocationType loc)
      : engine(engine), cpu(loc == MemoryLocationType::CPU), device(device) {
    if (cpu)
      HIP_CHECK(hipHostMalloc(&ptr, bytes, 0));
    else
      HIP_CHECK(hipMalloc(&ptr, bytes));
    try {
      desc = engine.RegisterMemory(ptr, bytes, device, loc);
    } catch (...) {
      Free();
      throw;
    }
    registered = true;
  }

  ~BenchBuffer() {
    // Teardown of a failing run must not mask the original error, so both steps
    // report instead of throwing.
    if (registered) {
      try {
        engine.DeregisterMemory(desc);
      } catch (const std::exception& e) {
        std::cerr << "DeregisterMemory failed: " << e.what() << std::endl;
      }
    }
    Free();
  }

  BenchBuffer(const BenchBuffer&) = delete;
  BenchBuffer& operator=(const BenchBuffer&) = delete;

  void* Data() const { return ptr; }
  const MemoryDesc& Desc() const { return desc; }

 private:
  void Free() {
    if (!ptr) return;
    // Restore the owning device first. Allocation sets it explicitly, but the
    // destructor runs at scope exit under whatever device is current then --
    // with two buffers on two GPUs (--xgmi-single-process) that is the wrong one
    // for at least one of them.
    int prev = -1;
    const bool switched = !cpu && hipGetDevice(&prev) == hipSuccess && prev != device &&
                          hipSetDevice(device) == hipSuccess;
    hipError_t e = cpu ? hipHostFree(ptr) : hipFree(ptr);
    if (e != hipSuccess) {
      std::cerr << "buffer free failed: " << hipGetErrorString(e) << std::endl;
    }
    if (switched) (void)hipSetDevice(prev);
    ptr = nullptr;
  }

  IOEngine& engine;
  const bool cpu;
  const int device;
  void* ptr{nullptr};
  bool registered{false};
  MemoryDesc desc;
};

// ------------------------------ validation ---------------------------------
// The Python benchmark checks correctness as part of the run (benchmark.py
// _validate_rdma): it performs a transfer, ships the target's buffer back over
// the control plane and byte-compares it against the initiator's source. We do
// the same check here, but exchange a per-slot checksum instead of the payload,
// so validating a multi-hundred-MiB sweep point stays O(batch) on the wire while
// still covering every transferred byte.
//
// Both buffers are seeded with a rank-dependent, offset-dependent pattern, so an
// un-issued transfer, a short transfer, or one landing at the wrong offset all
// leave the two sides disagreeing.

uint64_t Fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

// Byte at absolute offset p is a function of (p, rank), so the two ranks start
// out differing everywhere and every byte position is distinguishable.
void FillPattern(void* buf, size_t bytes, int rank) {
  constexpr size_t kTile = 1u << 20;
  std::vector<uint8_t> tile(std::min(bytes, kTile));
  for (size_t base = 0; base < bytes; base += tile.size()) {
    const size_t n = std::min(tile.size(), bytes - base);
    for (size_t i = 0; i < n; ++i) {
      const uint64_t p = base + i;
      tile[i] = static_cast<uint8_t>((p * 2654435761ull + rank * 0x9E3779B9ull) >> 13);
    }
    HIP_CHECK(hipMemcpy(static_cast<char*>(buf) + base, tile.data(), n, hipMemcpyHostToDevice));
  }
}

// One checksum per transferred slot, so a mismatch names the slot that differs
// rather than just failing the whole point.
std::vector<uint64_t> SlotChecksums(const void* buf, const SizeVec& offsets, size_t msg) {
  std::vector<uint64_t> sums(offsets.size());
  std::vector<uint8_t> host(msg);
  for (size_t i = 0; i < offsets.size(); ++i) {
    HIP_CHECK(hipMemcpy(host.data(), static_cast<const char*>(buf) + offsets[i], msg,
                        hipMemcpyDeviceToHost));
    sums[i] = Fnv1a(host.data(), msg);
  }
  return sums;
}

// ------------------------- control-plane rendezvous -------------------------
// Adapter over MORI's SocketBootstrapNetwork -- the same bootstrap layer
// src/cco/cco_init.cpp and src/shmem/init.cpp use to bring up multi-node jobs --
// so the benchmark inherits the library's rendezvous policy instead of carrying a
// second one: connect and accept retried over the MORI_BOOTSTRAP_TIMEOUT budget
// (300s default), a Barrier that blocks for as long as the initiator's sweep
// takes, failures raised as exceptions, and Finalize() from the destructor.
//
// Both ranks derive the same UniqueId from the master endpoint, so nothing has to
// carry it between them. The bootstrap picks its own local interface; set
// MORI_SOCKET_IFNAME when that choice has no route to the peer.
//
// World size is a runtime value, not a constant: with --num-initiator-dev N /
// --num-target-dev N there are N processes per node, one per GPU, and the world
// is numInitiatorDev + numTargetDev ranks. Global ranks are laid out initiators
// first, so initiator local i is global i and target local i is global
// numInitiatorDev + i -- the same numbering the Python bench gives torch.dist
// (benchmark.py _setup_rdma), which keeps the two directly comparable.
class Rendezvous {
 public:
  Rendezvous(int globalRank, int worldSize, int peerRank, const std::string& masterIp,
             uint16_t port)
      : boot(mori::application::SocketBootstrapNetwork::GenerateUniqueId(masterIp, port),
             globalRank, worldSize),
        myRank(globalRank),
        worldSize(worldSize),
        peerRank(peerRank) {
    boot.Initialize();
  }

  // Contribute this rank's blob, return every rank's, indexed by global rank.
  // Allgather is fixed-size while packed descriptors are not, so sizes go first
  // and the payload round pads every rank up to the largest of them.
  std::vector<std::string> AllgatherBlobs(const std::string& mine) {
    std::vector<uint64_t> sizes(static_cast<size_t>(worldSize), 0);
    uint64_t mySize = mine.size();
    boot.Allgather(&mySize, sizes.data(), sizeof(mySize));

    const size_t stride = *std::max_element(sizes.begin(), sizes.end());
    std::vector<std::string> out(static_cast<size_t>(worldSize));
    if (stride == 0) return out;

    std::vector<char> sendBuf(stride, 0), recvBuf(stride * static_cast<size_t>(worldSize), 0);
    std::memcpy(sendBuf.data(), mine.data(), mine.size());
    boot.Allgather(sendBuf.data(), recvBuf.data(), stride);

    for (int r = 0; r < worldSize; ++r) {
      out[static_cast<size_t>(r)] = std::string(recvBuf.data() + r * stride, sizes[r]);
    }
    return out;
  }

  // Contribute this rank's blob, return the paired rank's. Every rank still
  // takes part in the underlying allgather -- it is a collective, so a rank that
  // sat one out would leave the rest blocked until the bootstrap timeout.
  std::string ExchangeBlob(const std::string& mine) {
    return AllgatherBlobs(mine)[static_cast<size_t>(peerRank)];
  }

  void Barrier() { boot.Barrier(); }

  int WorldSize() const { return worldSize; }

 private:
  mori::application::SocketBootstrapNetwork boot;
  const int myRank;
  const int worldSize;
  const int peerRank;
};

template <typename T>
std::string Pack(const T& v) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, v);
  return std::string(buf.data(), buf.size());
}
template <typename T>
T Unpack(const std::string& b) {
  auto oh = msgpack::unpack(b.data(), b.size());
  return oh.get().as<T>();
}

// Compare the transferred slots of both ranks' buffers after the sweep. Both
// sides derive the geometry from the same args, so the only thing that crosses
// the wire is one checksum per slot. Only the initiator reports.
//
// BOTH ranks must call this unconditionally, even when --skip-validate is set:
// the exchange is a collective, so a rank that returned early would leave its
// peer blocked in Allgather until the bootstrap timeout. Opting out therefore
// contributes an empty checksum vector rather than skipping the call, which also
// makes the flag safe to pass to only one side -- either empty vector downgrades
// the run to "skipped" instead of hanging or reporting a bogus failure.
// Checksums of the slots the sweep actually touched, or an empty vector when the
// caller opted out with --skip-validate.
std::vector<uint64_t> TransferChecksums(const void* buf, size_t msg, int batch, bool batched,
                                        bool batchContiguous, bool wanted) {
  if (!wanted) return {};
  // Mirror the offsets the sweep actually used: the batched path strides by
  // msg+1 unless --batch-contiguous, while the singles path is always
  // contiguous (same asymmetry as the Python bench's run_single_once).
  const size_t stride = batched ? (batchContiguous ? msg : msg + 1) : msg;
  SizeVec offsets(static_cast<size_t>(batch));
  for (int i = 0; i < batch; ++i) offsets[i] = static_cast<size_t>(i) * stride;
  return SlotChecksums(buf, offsets, msg);
}

// Compare two checksum vectors and report. `label` prefixes every line so the
// output stays readable when N initiator ranks report concurrently.
bool CompareChecksums(const std::vector<uint64_t>& mine, const std::vector<uint64_t>& peer,
                      size_t msg, int batch, const std::string& label) {
  if (mine.empty() || peer.empty()) {
    std::cout << label << "validation: skipped"
              << (mine.empty() ? "" : " (peer opted out with --skip-validate)") << std::endl;
    return true;
  }
  if (peer.size() != mine.size()) {
    std::cerr << label << "VALIDATION FAILED: peer returned " << peer.size()
              << " checksums, expected " << mine.size() << std::endl;
    return false;
  }
  for (size_t i = 0; i < mine.size(); ++i) {
    if (mine[i] != peer[i]) {
      std::cerr << label << "VALIDATION FAILED: slot " << i << " of " << mine.size()
                << " differs at msg=" << msg << " batch=" << batch << " (initiator " << std::hex
                << mine[i] << " vs target " << peer[i] << std::dec << ")" << std::endl;
      return false;
    }
  }
  std::cout << label << "validation: OK (" << batch << " slot(s) x " << msg << " B byte-identical)"
            << std::endl;
  return true;
}

bool ValidateTransfer(Rendezvous& rdv, const void* buf, size_t msg, int batch, bool batched,
                      bool batchContiguous, bool isInitiator, bool wanted,
                      const std::string& label) {
  const auto mine = TransferChecksums(buf, msg, batch, batched, batchContiguous, wanted);
  const auto peer = Unpack<std::vector<uint64_t>>(rdv.ExchangeBlob(Pack(mine)));
  if (!isInitiator) return true;
  return CompareChecksums(mine, peer, msg, batch, label);
}

// ------------------------------- config ------------------------------------
// Flags mirror MORI's Python benchmark (tests/python/io/benchmark.py) so runs
// are directly comparable. Names use the Python spelling where they exist.
struct Args {
  int rank = 0;           // 0=initiator, 1=target
  std::string master_ip;  // bootstrap root (rank 0's data IP); both ranks pass the same
  std::string self_ip;    // this rank's own reachable IP (advertised in EngineDesc)
  uint16_t port = 18515;
  int gpu = 0;                // --gpu / --src-gpu
  int dst_gpu = -1;           // --dst-gpu (-1 = derive from --target-dev-offset)
  int target_dev_offset = 0;  // --target-dev-offset (target GPU = (gpu+offset)%ndev)

  // Multi-device fan-out (Python --num-initiator-dev / --num-target-dev): each
  // side forks one process per GPU, initiator local i pairing with target local
  // i. Must be equal, as in the Python bench.
  int num_initiator_dev = 1;
  int num_target_dev = 1;

  std::string op = "write";      // write|read
  std::string backend = "rdma";  // --backend rdma|xgmi|fabric
  // --xgmi-single-process: run BOTH sides in one process over two local GPUs, no
  // rendezvous (the Python bench's default XGMI mode). The normal two-rank path
  // is already what Python calls --xgmi-multiprocess.
  bool xgmi_single_process = false;
  bool xgmi_multiprocess = false;  // accepted for parity; already the default

  // Sweep control (Python parity). --all sweeps message size; --all-batch sweeps
  // batch size. Neither set => single run at (buffer_size, batch).
  bool sweep_all = false;       // --all
  bool sweep_batch = false;     // --all-batch
  size_t buffer_size = 32768;   // --buffer-size (single message size when not sweeping)
  size_t sweep_start = 8;       // --sweep-start-size (Python default 8)
  size_t sweep_max = 1u << 20;  // --sweep-max-size (Python default 2^20)
  size_t sweep_step = 0;        // --sweep-step (0 = geometric x2; >0 = linear +step)
  int iters = 500;
  int warmup = 50;  // --warmup-iters

  // RdmaBackendConfig knobs
  int qp_per_transfer = 4;               // --num-qp-per-transfer
  int worker_threads = 1;                // --num-worker-threads
  int post_batch_size = -1;              // --post-batch-size
  std::string poll_cq_mode = "polling";  // polling|event
  bool disable_chunking = false;         // --disable-chunking (default: chunking ON, like Python)
  size_t chunk_bytes = 65536;            // --chunk-bytes
  int max_chunks = 64;                   // --max-chunks
  int max_send_wr = 0;                   // --max-send-wr (0 = leave default)
  int max_cqe_num = 0;                   // --max-cqe-num
  int max_msg_sge = 0;                   // --max-msg-sge

  // XgmiBackendConfig knobs (--backend xgmi)
  int num_streams = 64;  // --num-streams
  int num_events = 64;   // --num-events

  // batch / session
  int batch = 1;                      // --transfer-batch-size (transfers per request)
  bool enable_batch_transfer = true;  // --enable-batch-transfer / --disable-batch-transfer.
                                      // ON (default): batch>1 => ONE N-descriptor batch request
                                      // (nixl-equivalent). OFF: batch>1 => N individual single
                                      // transfers per iteration (Python run_single_once path).
  bool batch_contiguous = false;      // --batch-contiguous (adjacent offsets → merged WR);
                                      // default strided (each transfer a separate WR)
  bool enable_sess = false;           // --enable-sess (session fast-path); Python default: off

  std::string mem_type = "gpu";  // --mem-type gpu|cpu
  std::string init_mem_type;     // --initiator-mem-type (empty => mem_type)
  std::string target_mem_type;   // --target-mem-type   (empty => mem_type)

  std::string log_level = "info";  // --log-level trace|debug|info|warning|error|critical

  // Correctness check on the last sweep point, on by default to match the Python
  // benchmark (which always validates). --skip-validate opts out.
  bool skip_validate = false;  // --skip-validate

  bool help = false;  // -h / --help
};

void PrintUsage(const char* argv0) {
  std::printf(
      "Usage: %s --rank <0|1> --master-ip <IP> [OPTIONS]\n"
      "\n"
      "nixlbench-matching MORI-IO transfer benchmark. Start rank 1 (the target)\n"
      "first, then rank 0 (the initiator), which drives every transfer and prints\n"
      "the results. Both ranks must be given the same workload arguments. See\n"
      "docs/MORI-IO-BENCHMARK.md for the full walkthrough.\n"
      "\n"
      "Rendezvous:\n"
      "  --rank N                 0 = initiator, 1 = target\n"
      "  --master-ip IP           rank 0's IP; pass the same value on both ranks\n"
      "  --self-ip IP             this rank's peer-reachable IP (default: --master-ip)\n"
      "  --port N                 bootstrap port (default: 18515)\n"
      "\n"
      "Workload:\n"
      "  --op <write|read>        transfer direction (default: write)\n"
      "  --backend <rdma|xgmi|fabric>  (default: rdma; fabric = UALink scale-up vPOD)\n"
      "  --buffer-size N          message size in bytes when not sweeping (default: 32768)\n"
      "  --transfer-batch-size N  transfers per iteration (default: 1)\n"
      "  --enable-batch-transfer  one N-descriptor batch request per iteration (default)\n"
      "  --disable-batch-transfer N individual transfers per iteration instead\n"
      "  --batch-contiguous       adjacent slot offsets (default: strided by msg+1)\n"
      "  --iters N                timed iterations (default: 500)\n"
      "  --warmup-iters N         untimed warmup iterations (default: 50)\n"
      "  --enable-sess            use the session fast path (default: off)\n"
      "  --disable-sess           call the engine APIs directly\n"
      "\n"
      "Sweeps (default: a single point at --buffer-size):\n"
      "  --all                    sweep message size --sweep-start..--sweep-max\n"
      "  --all-batch              sweep batch 1..32768 at --buffer-size\n"
      "  --sweep-start N          (default: 8)\n"
      "  --sweep-max N            (default: 1048576)\n"
      "  --sweep-step N           0 = geometric x2, >0 = linear +N (default: 0)\n"
      "\n"
      "Memory:\n"
      "  --mem-type <gpu|cpu>     (default: gpu)\n"
      "  --initiator-mem-type T   overrides --mem-type on rank 0\n"
      "  --target-mem-type T      overrides --mem-type on rank 1\n"
      "  --gpu N                  local device ordinal (default: 0); alias --src-gpu\n"
      "  --dst-gpu N              peer device ordinal; sets --target-dev-offset (dst - gpu),\n"
      "                           and is the destination GPU under --xgmi-single-process\n"
      "  --target-dev-offset N    target GPU = (gpu + offset) %% device count\n"
      "\n"
      "Multi-device (one process per GPU, initiator i <-> target i):\n"
      "  --num-initiator-dev N    GPUs/processes on rank 0 (default: 1)\n"
      "  --num-target-dev N       GPUs/processes on rank 1 (default: 1; must equal the above)\n"
      "                           Rank r's process i drives GPU (gpu + i) and reports its own\n"
      "                           row; rank 0 also prints an AGGREGATE line summing all pairs.\n"
      "\n"
      "RDMA tuning:\n"
      "  --num-qp-per-transfer N  (default: 4)\n"
      "  --num-worker-threads N   (default: 1)\n"
      "  --post-batch-size N      (default: -1, backend default)\n"
      "  --poll_cq_mode <polling|event>   (default: polling)\n"
      "  --disable-chunking       chunking is on by default\n"
      "  --chunk-bytes N          (default: 65536)\n"
      "  --max-chunks N           (default: 64)\n"
      "  --max-send-wr N          0 = backend default\n"
      "  --max-cqe-num N          0 = backend default\n"
      "  --max-msg-sge N          0 = backend default\n"
      "\n"
      "XGMI / FABRIC tuning:\n"
      "  --num-streams N          (default: 64)\n"
      "  --num-events N           (default: 64)\n"
      "  --xgmi-single-process    run both sides in ONE process over --src-gpu/--dst-gpu with\n"
      "                           no rendezvous (Python's default XGMI mode). Without it the\n"
      "                           usual two-rank path runs, which is already Python's\n"
      "                           --xgmi-multiprocess (distinct engine keys => HIP IPC).\n"
      "  --xgmi-multiprocess      accepted for Python parity; this is the default, so it only\n"
      "                           documents intent (rejected with --xgmi-single-process)\n"
      "\n"
      "Other:\n"
      "  --skip-validate          skip the post-sweep checksum comparison\n"
      "  --log-level LEVEL        trace|debug|info|warning|error|critical (default: info)\n"
      "  -h, --help               show this message\n",
      argv0);
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    // Checked rather than argv[++i] directly: a value-taking flag in last
    // position would otherwise construct a std::string from argv[argc], which
    // is null.
    auto next = [&]() {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + k);
      return std::string(argv[++i]);
    };
    if (k == "-h" || k == "--help")
      a.help = true;
    else if (k == "--rank")
      a.rank = std::stoi(next());
    else if (k == "--master-ip" || k == "--host")  // --host: Python spelling
      a.master_ip = next();
    else if (k == "--self-ip")
      a.self_ip = next();
    else if (k == "--port")
      a.port = static_cast<uint16_t>(std::stoi(next()));
    else if (k == "--gpu" || k == "--src-gpu")
      a.gpu = std::stoi(next());
    else if (k == "--dst-gpu")
      a.dst_gpu = std::stoi(next());
    else if (k == "--target-dev-offset")
      a.target_dev_offset = std::stoi(next());
    else if (k == "--num-initiator-dev")
      a.num_initiator_dev = std::stoi(next());
    else if (k == "--num-target-dev")
      a.num_target_dev = std::stoi(next());
    else if (k == "--xgmi-single-process")
      a.xgmi_single_process = true;
    else if (k == "--xgmi-multiprocess")
      a.xgmi_multiprocess = true;
    else if (k == "--op" || k == "--op-type")
      a.op = next();
    else if (k == "--backend")
      a.backend = next();
    else if (k == "--all")
      a.sweep_all = true;
    else if (k == "--all-batch")
      a.sweep_batch = true;
    else if (k == "--buffer-size")
      a.buffer_size = std::stoull(next());
    else if (k == "--sweep-start" || k == "--sweep-start-size")
      a.sweep_start = std::stoull(next());
    else if (k == "--sweep-max" || k == "--sweep-max-size")
      a.sweep_max = std::stoull(next());
    else if (k == "--sweep-step")
      a.sweep_step = std::stoull(next());
    else if (k == "--iters")
      a.iters = std::stoi(next());
    else if (k == "--warmup" || k == "--warmup-iters")
      a.warmup = std::stoi(next());
    else if (k == "--qp-per-transfer" || k == "--num-qp-per-transfer")
      a.qp_per_transfer = std::stoi(next());
    else if (k == "--worker-threads" || k == "--num-worker-threads")
      a.worker_threads = std::stoi(next());
    else if (k == "--post-batch-size")
      a.post_batch_size = std::stoi(next());
    else if (k == "--poll_cq_mode" || k == "--poll-cq-mode")
      a.poll_cq_mode = next();
    else if (k == "--disable-chunking")
      a.disable_chunking = true;
    else if (k == "--chunk-bytes")
      a.chunk_bytes = std::stoull(next());
    else if (k == "--max-chunks")
      a.max_chunks = std::stoi(next());
    else if (k == "--max-send-wr")
      a.max_send_wr = std::stoi(next());
    else if (k == "--max-cqe-num")
      a.max_cqe_num = std::stoi(next());
    else if (k == "--max-msg-sge")
      a.max_msg_sge = std::stoi(next());
    else if (k == "--num-streams")
      a.num_streams = std::stoi(next());
    else if (k == "--num-events")
      a.num_events = std::stoi(next());
    else if (k == "--batch" || k == "--transfer-batch-size")
      a.batch = std::stoi(next());
    else if (k == "--enable-batch-transfer")
      a.enable_batch_transfer = true;
    else if (k == "--disable-batch-transfer")
      a.enable_batch_transfer = false;
    else if (k == "--batch-contiguous")
      a.batch_contiguous = true;
    else if (k == "--enable-sess")
      a.enable_sess = true;
    else if (k == "--disable-sess")
      a.enable_sess = false;
    else if (k == "--mem-type")
      a.mem_type = next();
    else if (k == "--initiator-mem-type")
      a.init_mem_type = next();
    else if (k == "--target-mem-type")
      a.target_mem_type = next();
    else if (k == "--skip-validate")
      a.skip_validate = true;
    else if (k == "--log-level")
      a.log_level = next();
    else
      throw std::invalid_argument("unknown arg " + k + " (try --help)");
  }
  return a;
}

// Rejected up front because each of these either corrupts the run or reports a
// number for something the caller did not ask for: rank feeds the bootstrap's
// peer indexing, iters and batch are divisors of the reported latency, a zero
// message size makes the geometric sweep loop forever, and an unrecognised op or
// memory type would otherwise fall back silently to write/GPU.
void ValidateArgs(Args& a) {
  if (a.rank != 0 && a.rank != 1) {
    throw std::invalid_argument("--rank must be 0 (initiator) or 1 (target)");
  }
  if (a.backend != "rdma" && a.backend != "xgmi" && a.backend != "fabric") {
    throw std::invalid_argument("--backend must be rdma, xgmi or fabric (got " + a.backend + ")");
  }
  if (a.op != "read" && a.op != "write") {
    throw std::invalid_argument("--op must be read or write (got " + a.op + ")");
  }
  if (a.iters <= 0) throw std::invalid_argument("--iters must be > 0");
  if (a.batch <= 0) throw std::invalid_argument("--transfer-batch-size must be > 0");
  if (a.warmup < 0) throw std::invalid_argument("--warmup-iters must be >= 0");
  if (a.sweep_all ? a.sweep_start == 0 : a.buffer_size == 0) {
    throw std::invalid_argument("message size must be > 0");
  }
  for (const std::string& m : {a.mem_type, a.init_mem_type, a.target_mem_type}) {
    if (!m.empty() && m != "gpu" && m != "cpu") {
      throw std::invalid_argument("memory type must be gpu or cpu (got " + m + ")");
    }
  }
  // FABRIC is a GPU scale-up transport (UALink vPOD); it has no host-memory path,
  // and the Python bench rejects the combination the same way.
  if (a.backend == "fabric") {
    for (const std::string& m : {a.mem_type, a.init_mem_type, a.target_mem_type}) {
      if (m == "cpu") throw std::invalid_argument("--backend fabric supports GPU memory only");
    }
  }
  if (a.num_initiator_dev < 1 || a.num_target_dev < 1) {
    throw std::invalid_argument("--num-initiator-dev / --num-target-dev must be >= 1");
  }
  // Same restriction the Python bench asserts: the two sides pair up one-to-one,
  // so an unequal split would leave processes without a partner, blocked in the
  // bootstrap until it times out.
  if (a.num_initiator_dev != a.num_target_dev) {
    throw std::invalid_argument("--num-initiator-dev must equal --num-target-dev (got " +
                                std::to_string(a.num_initiator_dev) + " vs " +
                                std::to_string(a.num_target_dev) + ")");
  }
  if (a.xgmi_single_process && a.xgmi_multiprocess) {
    throw std::invalid_argument("--xgmi-single-process and --xgmi-multiprocess are exclusive");
  }
  if (a.xgmi_single_process && a.backend != "xgmi") {
    throw std::invalid_argument("--xgmi-single-process requires --backend xgmi");
  }
  if (a.xgmi_single_process && a.num_initiator_dev != 1) {
    throw std::invalid_argument("--xgmi-single-process runs one process; drop --num-*-dev");
  }
  // Rejected rather than ignored: a two-node launcher that keeps --rank 1 on the
  // second node would otherwise run a full, independent GPU0->GPU1 benchmark on
  // BOTH hosts, print a plausible-looking table on each, and transfer nothing
  // between them -- a wrong answer that reads as a successful run.
  if (a.xgmi_single_process && a.rank != 0) {
    throw std::invalid_argument(
        "--xgmi-single-process is a single-node, single-process mode; --rank 1 is meaningless "
        "(drop --rank, or drop --xgmi-single-process for the two-rank XGMI path)");
  }
  // --dst-gpu is the Python spelling of a destination ordinal; the two-rank path
  // expresses the same thing as an offset, so fold one into the other rather than
  // carrying two sources of truth into the run.
  if (a.dst_gpu >= 0 && !a.xgmi_single_process) {
    if (a.target_dev_offset != 0 && a.target_dev_offset != a.dst_gpu - a.gpu) {
      throw std::invalid_argument("--dst-gpu and --target-dev-offset disagree");
    }
    a.target_dev_offset = a.dst_gpu - a.gpu;
  }
}

// Build the run plan: a list of (msgSize, batch) points.
std::vector<std::pair<size_t, int>> BuildPlan(const Args& a) {
  //   --all       => sweep message size (geometric x2, or linear when --sweep-step>0),
  //                  batch fixed at --transfer-batch-size.
  //   --all-batch => msg fixed at --buffer-size, batch = 1,2,4,...,32768.
  //   neither     => single point (--buffer-size, --transfer-batch-size).
  std::vector<std::pair<size_t, int>> plan;
  if (a.sweep_all) {
    for (size_t msg = a.sweep_start; msg <= a.sweep_max;
         msg = (a.sweep_step > 0) ? msg + a.sweep_step : msg * 2) {
      plan.emplace_back(msg, a.batch);
    }
  } else if (a.sweep_batch) {
    for (int b = 1; b <= 32768; b *= 2) plan.emplace_back(a.buffer_size, b);
  } else {
    plan.emplace_back(a.buffer_size, a.batch);
  }
  // The last point is validated after the sweep, so an empty plan would read off
  // the end of it.
  if (plan.empty()) {
    throw std::invalid_argument("empty sweep: --sweep-start exceeds --sweep-max");
  }
  return plan;
}

// Buffer must hold the largest single REQUEST across the whole plan. Strided
// batch (default) needs (msg+1)*batch to keep slots non-adjacent; contiguous
// needs msg*batch.
size_t PlanBufferBytes(const Args& a, const std::vector<std::pair<size_t, int>>& plan) {
  size_t bufBytes = 0;
  for (auto& planEntry : plan) {
    const size_t msg = planEntry.first;
    const int b = planEntry.second;
    const bool pBatched = b > 1;
    const size_t slotStride = a.batch_contiguous ? msg : (msg + 1);
    const size_t need = pBatched ? slotStride * static_cast<size_t>(b) : msg;
    bufBytes = std::max(bufBytes, need);
  }
  return bufBytes;
}

// XGMI moves data over Infinity Fabric between two GPUs on the SAME host and
// FABRIC over UALink between hosts in one vPOD, so both take stream/event depth
// rather than QPs and chunking. The RDMA-only flags are simply unused there.
void CreateBackendFor(IOEngine& engine, const Args& a) {
  if (a.backend == "xgmi" || a.backend == "fabric") {
    if (a.backend == "xgmi") {
      XgmiBackendConfig xgmiCfg{};
      xgmiCfg.numStreams = a.num_streams;
      xgmiCfg.numEvents = a.num_events;
      engine.CreateBackend(BackendType::XGMI, xgmiCfg);
    } else {
      FabricBackendConfig fabricCfg{};
      fabricCfg.numStreams = a.num_streams;
      fabricCfg.numEvents = a.num_events;
      engine.CreateBackend(BackendType::FABRIC, fabricCfg);
    }
    return;
  }
  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.qpPerTransfer = a.qp_per_transfer;
  rdmaCfg.postBatchSize = a.post_batch_size;
  rdmaCfg.numWorkerThreads = a.worker_threads;
  rdmaCfg.pollCqMode = (a.poll_cq_mode == "event") ? PollCqMode::EVENT : PollCqMode::POLLING;
  rdmaCfg.enableNotification = false;                    // match MORI Python bench RDMA path
  rdmaCfg.enableTransferChunking = !a.disable_chunking;  // chunking ON by default (Python parity)
  rdmaCfg.chunkBytes = a.chunk_bytes;
  rdmaCfg.maxChunksPerTransfer = a.max_chunks;
  if (a.max_send_wr > 0) rdmaCfg.maxSendWr = a.max_send_wr;
  if (a.max_cqe_num > 0) rdmaCfg.maxCqeNum = a.max_cqe_num;
  if (a.max_msg_sge > 0) rdmaCfg.maxMsgSge = a.max_msg_sge;
  engine.CreateBackend(BackendType::RDMA, rdmaCfg);
}

// One row of the report. Kept per sweep point so the multi-device path can sum
// bandwidth across GPU pairs after the fact.
struct SweepResult {
  size_t msg;
  int batch;
  double bw;   // GB/s, GB=10^9
  double lat;  // us per single transfer
  double dur;  // us, whole timed loop
  // Python-benchmark-format metrics for the perf report (tests/python/perf_report.py).
  // Python times each iteration separately and reports bandwidth/latency PER
  // ITERATION (whole batch), deriving max_bw/min_lat from the FASTEST iteration.
  // The nixl fields above (bw/lat) are per-single-transfer over one whole-loop
  // timer, so they are NOT interchangeable -- these mirror what benchmark.py emits.
  double avg_bw_gbps;  // total_mem_mb/1e3 / avg_iter_duration_s
  double max_bw_gbps;  // total_mem_mb/1e3 / min_iter_duration_s
  double avg_lat_us;   // mean per-iteration duration
  double min_lat_us;   // fastest per-iteration duration
  double total_mb;     // msg * batch / 1e6
  MSGPACK_DEFINE(msg, batch, bw, lat, dur, avg_bw_gbps, max_bw_gbps, avg_lat_us, min_lat_us,
                 total_mb);
};

// Minimal JSON string escaper for the perf record. Only the characters that can
// appear in our values (backslash, quote, control chars) need handling.
std::string JsonEscape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string EnvOr(const char* name, const std::string& dflt = "") {
  const char* v = std::getenv(name);
  return v ? std::string(v) : dflt;
}

// Append perf records for the whole sweep to $MORI_PERF_OUT as JSON lines,
// matching tests/python/perf_report.py::record_perf exactly so the C++ bench
// and the Python bench feed one report pipeline (tools/perf/build_report.py).
// No-op unless MORI_PERF_OUT is set. Call from the initiator root only, so a
// record is not written once per rank. Any error is swallowed -- perf reporting
// must never fail a benchmark run.
void WritePerfRecords(const Args& a, const std::vector<SweepResult>& rows) {
  const std::string out = EnvOr("MORI_PERF_OUT");
  if (out.empty() || rows.empty()) return;

  const std::string run_id = EnvOr("MORI_PERF_RUN_ID");
  const std::string commit = EnvOr("MORI_PERF_COMMIT");
  const std::string platform = EnvOr("MORI_PERF_PLATFORM");
  const std::string python = EnvOr("MORI_PERF_PYTHON");
  const std::string date = EnvOr("MORI_PERF_DATE");
  const double ts = static_cast<double>(std::time(nullptr));

  // Build the whole payload first, then take the lock once and write it, so a
  // concurrent writer never sees a half-written record. Keys are emitted in
  // sorted order to match Python's json.dumps(sort_keys=True).
  std::ostringstream payload;
  for (const auto& r : rows) {
    std::ostringstream line;
    line.setf(std::ios::fixed);
    line.precision(6);
    line << '{' << "\"category\": \"io\", "
         << "\"commit\": \"" << JsonEscape(commit) << "\", "
         << "\"date\": \"" << JsonEscape(date) << "\", "
         << "\"metrics\": {"
         << "\"avg_bw_gbps\": " << r.avg_bw_gbps << ", "
         << "\"avg_lat_us\": " << r.avg_lat_us << ", "
         << "\"max_bw_gbps\": " << r.max_bw_gbps << ", "
         << "\"min_lat_us\": " << r.min_lat_us << ", "
         << "\"total_mb\": " << r.total_mb << "}, "
         << "\"params\": {"
         << "\"backend\": \"" << JsonEscape(a.backend) << "\", "
         << "\"batch_size\": " << r.batch << ", "
         << "\"enable_batch_transfer\": " << (a.enable_batch_transfer ? "true" : "false") << ", "
         << "\"enable_sess\": " << (a.enable_sess ? "true" : "false") << ", "
         << "\"msg_size\": " << r.msg << ", "
         << "\"num_initiator_dev\": " << a.num_initiator_dev << ", "
         << "\"num_qp_per_transfer\": " << a.qp_per_transfer << ", "
         << "\"num_target_dev\": " << a.num_target_dev << ", "
         << "\"num_worker_threads\": " << a.worker_threads << ", "
         << "\"op_type\": \"" << JsonEscape(a.op) << "\", "
         << "\"transfer_batch_size\": " << a.batch << "}, "
         << "\"platform\": \"" << JsonEscape(platform) << "\", "
         << "\"python\": \"" << JsonEscape(python) << "\", "
         << "\"run_id\": \"" << JsonEscape(run_id) << "\", "
         << "\"ts\": " << ts << "}\n";
    payload << line.str();
  }

  // Create the parent directory if it does not exist, matching Python's
  // record_perf (os.makedirs(parent, exist_ok=True)). On a clean nightly
  // workspace MORI_PERF_OUT points at $WORKSPACE/_perf/perf.jsonl and no earlier
  // step creates _perf, so without this fopen fails and every IO row is silently
  // dropped from the uploaded perf data. mkdir each path component in turn.
  const auto slash = out.find_last_of('/');
  if (slash != std::string::npos && slash > 0) {
    const std::string dir = out.substr(0, slash);
    std::string partial;
    for (size_t i = 0; i < dir.size(); ++i) {
      partial += dir[i];
      if (dir[i] == '/' || i + 1 == dir.size()) {
        if (partial != "/" && ::mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) {
          std::fprintf(stderr, "[perf_report] failed to create %s: %s\n", partial.c_str(),
                       std::strerror(errno));
          break;
        }
      }
    }
  }

  std::FILE* fh = std::fopen(out.c_str(), "a");
  if (fh == nullptr) {
    std::fprintf(stderr, "[perf_report] failed to open %s: %s\n", out.c_str(),
                 std::strerror(errno));
    return;
  }
  int fd = fileno(fh);
  if (fd >= 0) flock(fd, LOCK_EX);
  std::fputs(payload.str().c_str(), fh);
  if (fd >= 0) flock(fd, LOCK_UN);
  std::fclose(fh);
}

// Drive every point in `plan` from localMem into peerMem and print one row each.
// `label` prefixes every row, so N initiator processes writing to the same
// terminal stay attributable.
std::vector<SweepResult> RunSweep(const Args& a, IOEngine& engine, const MemoryDesc& localMem,
                                  const MemoryDesc& peerMem,
                                  const std::vector<std::pair<size_t, int>>& plan,
                                  const std::string& label) {
  // Session fast-path (matches MORI --enable-sess). When --disable-sess, we call
  // the engine batch/single APIs directly with explicit MemoryDesc + uid vecs.
  const bool useSess = a.enable_sess;
  IOEngineSession* sessPtr = nullptr;
  std::optional<IOEngineSession> sessOpt;
  if (useSess) {
    sessOpt = engine.CreateSession(localMem, peerMem);
    if (!sessOpt) throw std::runtime_error("CreateSession failed");
    sessPtr = &*sessOpt;
  }
  const bool isRead = (a.op == "read");

  std::vector<SweepResult> results;
  results.reserve(plan.size());

  for (auto& planEntry : plan) {
    const size_t msg = planEntry.first;
    const int curBatch = planEntry.second;
    // Two ways to move curBatch transfers per iteration, mirroring the Python bench:
    //   --enable-batch-transfer (default, batched): ONE N-descriptor batch request
    //     (BatchWrite/BatchRead) -- the nixl-equivalent (nixl always batches).
    //   --disable-batch-transfer (singles): curBatch INDIVIDUAL single-transfer
    //     submissions per iteration (Python run_single_once), each its own status.
    // batch==1 is a single transfer either way.
    const bool batched = a.enable_batch_transfer && curBatch > 1;
    const int perSlot = batched ? 1 : curBatch;  // statuses per request
    // Strict stop-and-wait: one request outstanding at a time (nixl
    // --pipeline_depth 1). [perSlot] status array (TransferStatus is
    // non-copyable, so a flat vector rather than nested).
    std::vector<TransferStatus> st(static_cast<size_t>(perSlot));

    // Batch layout for the batched path. Strided (default): transfer i at
    // (msg+1)*i so the N transfers stay SEPARATE WRs (stresses SQ, real batching)
    // -- matches Python default. Contiguous (--batch-contiguous): transfer i at
    // msg*i, adjacent, so MORI may merge them into one big WR (fast but not really
    // batching, and hits the 1 GiB max_msg_sz without chunking).
    const size_t stride = a.batch_contiguous ? msg : (msg + 1);
    SizeVec offsets(curBatch), sizes(curBatch);
    for (int i = 0; i < curBatch; ++i) {
      offsets[i] = static_cast<size_t>(i) * stride;
      sizes[i] = msg;
    }
    // Engine (non-session) batch path needs vec-of-vec + desc vectors. These and
    // the status/id vectors below are built once per sweep point and reused every
    // iteration: allocating them inside post() would charge two heap
    // allocation/free pairs per iteration to the reported latency.
    MemDescVec locVec{localMem}, remVec{peerMem};
    BatchSizeVec offVec{offsets}, sizeVec{sizes};
    TransferStatusPtrVec statusPtrs{&st[0]};
    TransferUniqueIdVec batchIds(1);

    auto post = [&]() {
      TransferStatus* base = &st[0];
      for (int i = 0; i < perSlot; ++i) base[i].SetCode(StatusCode::INIT);
      if (batched) {
        // ONE N-descriptor batch request (nixl / Python --enable-batch-transfer).
        TransferUniqueId id =
            useSess ? sessPtr->AllocateTransferUniqueId() : engine.AllocateTransferUniqueId();
        if (useSess) {
          if (isRead)
            sessPtr->BatchRead(offsets, offsets, sizes, &base[0], id);
          else
            sessPtr->BatchWrite(offsets, offsets, sizes, &base[0], id);
        } else {
          statusPtrs[0] = &base[0];
          batchIds[0] = id;
          if (isRead)
            engine.BatchRead(locVec, offVec, remVec, offVec, sizeVec, statusPtrs, batchIds);
          else
            engine.BatchWrite(locVec, offVec, remVec, offVec, sizeVec, statusPtrs, batchIds);
        }
      } else {
        // curBatch individual single-transfer submissions (Python run_single_once);
        // contiguous offsets i*msg, matching Python's single path. curBatch==1 is a
        // single transfer at offset 0.
        for (int i = 0; i < curBatch; ++i) {
          const size_t off = static_cast<size_t>(i) * msg;
          TransferUniqueId id =
              useSess ? sessPtr->AllocateTransferUniqueId() : engine.AllocateTransferUniqueId();
          if (useSess) {
            if (isRead)
              sessPtr->Read(off, off, msg, &base[i], id);
            else
              sessPtr->Write(off, off, msg, &base[i], id);
          } else {
            if (isRead)
              engine.Read(localMem, off, peerMem, off, msg, &base[i], id);
            else
              engine.Write(localMem, off, peerMem, off, msg, &base[i], id);
          }
        }
      }
    };

    // The request is complete only when ALL perSlot sub-transfers have left
    // INIT/IN_PROGRESS (mirrors Python waiting on the whole status_list). Spin,
    // like the timed loop. reqFailed returns the first failed status, or nullptr.
    auto reqDone = [&]() {
      TransferStatus* base = &st[0];
      for (int i = 0; i < perSlot; ++i)
        if (base[i].InProgress() || base[i].Init()) return false;
      return true;
    };
    auto reqFailed = [&]() -> TransferStatus* {
      TransferStatus* base = &st[0];
      for (int i = 0; i < perSlot; ++i)
        if (base[i].Failed()) return &base[i];
      return nullptr;
    };

    // ---- warmup (excluded from timing) ----
    for (int w = 0; w < a.warmup; ++w) {
      post();
      while (!reqDone()) { /* spin: CQ worker thread stores status */
      }
      if (TransferStatus* f = reqFailed()) {
        throw std::runtime_error("warmup transfer failed: " + f->Message());
      }
    }

    // ---- timed region: ONE whole-loop timer, nixl-style ----
    // Strict stop-and-wait (nixl --pipeline_depth 1): post one request, spin
    // until it completes, then post the next.
    // A per-iteration clock read is also taken so we can emit the Python
    // benchmark's per-iteration metrics (min/avg) to the perf report without
    // disturbing the whole-loop nixl figure (t1-t0 stays authoritative).
    double min_iter_us = std::numeric_limits<double>::max();
    auto t0 = Clock::now();

    for (int completed = 0; completed < a.iters; ++completed) {
      auto it0 = Clock::now();
      post();
      // spin-poll, mirroring nixl's "check status, if IN_PROG continue" scan
      // (status flags stored by MORI's CQ worker thread)
      while (!reqDone()) { /* spin */
      }
      if (TransferStatus* f = reqFailed()) {
        throw std::runtime_error("transfer failed: " + f->Message());
      }
      double iter_us =
          std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(Clock::now() - it0)
              .count();
      if (iter_us < min_iter_us) min_iter_us = iter_us;
    }
    auto t1 = Clock::now();

    double total_us =
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(t1 - t0).count();
    // nixl parity (nixl_worker/utils.cpp): count block_size*batch_size*num_iter
    // bytes over ONE whole-loop timer, and report latency PER SINGLE TRANSFER.
    // Both batched and singles modes move curBatch transfers of `msg` per iter:
    //   total_bytes  = msg * curBatch * iters
    //   avg_bw       = total_bytes/1e9 / (total_us/1e6)                 [GB/s, GB=10^9]
    //   avg_latency  = total_us / (iters * curBatch)                    [us per transfer]
    // matching nixl's avg_latency = total_duration/(per_thread_iter*batch_size).
    const double numXfers = static_cast<double>(a.iters) * curBatch;
    double total_bytes = static_cast<double>(msg) * numXfers;
    double avg_bw = (total_bytes / 1e9) / (total_us / 1e6);  // GB/s, GB=10^9
    double avg_lat = total_us / numXfers;                    // us per single transfer

    std::printf("%s%-11zu %-6d %-6d %-12.2f %-11.2f %-.1f\n", label.c_str(), msg, curBatch, a.iters,
                avg_bw, avg_lat, total_us);
    std::fflush(stdout);

    // Python-benchmark-format metrics (tests/python/io/benchmark.py _run_and_compute):
    // per-iteration timing, bandwidth/latency PER ITERATION (whole batch), with
    // max_bw/min_lat taken from the fastest iteration. avg iteration duration is
    // the whole-loop time / iters (identical to averaging the per-iter samples,
    // minus the negligible gap between the loop timer and the first/last sample).
    const double total_mb = static_cast<double>(msg) * curBatch / 1e6;
    const double avg_iter_us = total_us / a.iters;
    const double avg_bw_gbps = (total_mb / 1e3) / (avg_iter_us / 1e6);
    const double max_bw_gbps = (total_mb / 1e3) / (min_iter_us / 1e6);
    results.push_back(SweepResult{msg, curBatch, avg_bw, avg_lat, total_us, avg_bw_gbps,
                                  max_bw_gbps, avg_iter_us, min_iter_us, total_mb});
  }

  return results;
}

// ---------------------- driver: two-node, N GPUs a side ---------------------
// One process per GPU. Global ranks are initiators first, so initiator local i
// is global i and target local i is global numInitiatorDev + i, and the pair
// (i, numInitiatorDev + i) shares a session. With the defaults this is exactly
// the original two-rank run.
int RunDistributed(const Args& a, int localRank) {
  const bool isInitiator = (a.rank == 0);
  const int worldSize = a.num_initiator_dev + a.num_target_dev;
  const int globalRank = isInitiator ? localRank : a.num_initiator_dev + localRank;
  const int peerRank = isInitiator ? a.num_initiator_dev + localRank : localRank;
  const bool multiDev = worldSize > 2;

  // Per-role memory type: initiator (rank 0) / target (rank 1) may override the
  // shared --mem-type, enabling mixed CPU<->GPU transfers. Each process only
  // allocates its own side, so no cross-node coupling is needed.
  const std::string myMem = isInitiator
                                ? (!a.init_mem_type.empty() ? a.init_mem_type : a.mem_type)
                                : (!a.target_mem_type.empty() ? a.target_mem_type : a.mem_type);
  const bool cpuMem = (myMem == "cpu");
  const MemoryLocationType memLoc = cpuMem ? MemoryLocationType::CPU : MemoryLocationType::GPU;

  // Process i of a side drives GPU (--gpu + i); the target additionally shifts by
  // --target-dev-offset for cross-rail pairing (GPU memory only, matches Python).
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  const int numDev = isInitiator ? a.num_initiator_dev : a.num_target_dev;
  if (ndev > 0 && a.gpu + numDev > ndev) {
    throw std::invalid_argument("--gpu " + std::to_string(a.gpu) + " + " + std::to_string(numDev) +
                                " devices exceeds the " + std::to_string(ndev) + " visible GPUs");
  }
  int gpu = a.gpu + localRank;
  if (!isInitiator && !cpuMem && a.target_dev_offset != 0 && ndev > 0) {
    // Floored modulo: C++ % truncates toward zero, so a negative offset (legal,
    // and what --dst-gpu produces when it names a lower ordinal) would otherwise
    // yield a negative device and fail in hipSetDevice -- after the peer has
    // already entered the bootstrap, leaving it to block until the timeout.
    const int shifted = a.gpu + localRank + a.target_dev_offset;
    gpu = ((shifted % ndev) + ndev) % ndev;
  }
  HIP_CHECK(hipSetDevice(gpu));  // valid device context needed even for host mem

  const std::string label = multiDev ? ("[gpu " + std::to_string(gpu) + "] ") : std::string();

  const auto plan = BuildPlan(a);
  const size_t bufBytes = PlanBufferBytes(a, plan);

  // Out-of-band control endpoint for the MORI engine. host MUST be an IP the
  // peer can reach (advertised via EngineDesc for RDMA QP setup). Defaults to
  // master_ip: correct for rank 0 (it IS the master); rank 1 should pass
  // --self-ip when its reachable IP differs from the master's.
  //
  // Both the port and the engine key have to vary per process: with N processes
  // per node, a per-side constant would have every local process binding the
  // same port and advertising the same key, and peers would resolve to whichever
  // registered last. The port uses the global rank so it is unique across both
  // nodes; the key uses the local rank because the role prefix already separates
  // the sides, which also matches the Python bench's f"{role.name}-{role_rank}".
  IOEngineConfig cfg;
  cfg.host = !a.self_ip.empty() ? a.self_ip : a.master_ip;
  cfg.port = static_cast<uint16_t>(a.port + 1 + globalRank);
  const std::string key = (isInitiator ? "initiator-" : "target-") + std::to_string(localRank);
  IOEngine engine(key, cfg);

  CreateBackendFor(engine, a);

  // Declared after the engine so it is destroyed first, i.e. the MR is dropped
  // while the engine that owns it is still alive.
  BenchBuffer buffer(engine, bufBytes, gpu, memLoc);
  void* buf = buffer.Data();
  const MemoryDesc& localMem = buffer.Desc();

  // Rank-dependent seed pattern rather than zeros, so the post-sweep validation
  // can tell "the transfer moved my bytes" from "both buffers happened to match".
  if (a.skip_validate) {
    HIP_CHECK(hipMemset(buf, 0, bufBytes));
  } else {
    FillPattern(buf, bufBytes, a.rank);
  }

  // --- rendezvous over MORI's socket bootstrap ------------------------------
  Rendezvous rdv(globalRank, worldSize, peerRank, a.master_ip, a.port);

  // Exchange EngineDesc then register the remote engine.
  EngineDesc myEng = engine.GetEngineDesc();
  EngineDesc peerEng = Unpack<EngineDesc>(rdv.ExchangeBlob(Pack(myEng)));
  engine.RegisterRemoteEngine(peerEng);

  // Exchange MemoryDesc.
  MemoryDesc peerMem = Unpack<MemoryDesc>(rdv.ExchangeBlob(Pack(localMem)));

  rdv.Barrier();

  // Every collective below runs on BOTH sides in the same order: the target has
  // no transfers to drive, but skipping a barrier or an allgather would leave the
  // initiators blocked until the bootstrap timeout.
  std::vector<SweepResult> mine;
  if (isInitiator) {
    if (globalRank == 0) {
      std::cout << "MsgSize(B)  Batch  Iters  AvgBW(GB/s)  AvgLat(us)  TotalDur(us)" << std::endl;
    }
    mine = RunSweep(a, engine, localMem, peerMem, plan, label);
  }

  rdv.Barrier();  // initiators done sweeping

  // Aggregate: sum bandwidth over the initiator ranks, point by point. A single
  // pair already reports its own number, so only fan-out runs print this.
  const auto allBlobs = rdv.AllgatherBlobs(Pack(mine));
  if (multiDev && globalRank == 0) {
    std::vector<std::vector<SweepResult>> perRank;
    for (int r = 0; r < a.num_initiator_dev; ++r) {
      perRank.push_back(Unpack<std::vector<SweepResult>>(allBlobs[static_cast<size_t>(r)]));
    }
    std::cout << "--- AGGREGATE over " << a.num_initiator_dev << " GPU pairs ---" << std::endl;
    for (size_t p = 0; p < plan.size(); ++p) {
      double sumBw = 0, sumLat = 0;
      int n = 0;
      for (const auto& rows : perRank) {
        if (p >= rows.size()) continue;
        sumBw += rows[p].bw;
        sumLat += rows[p].lat;
        ++n;
      }
      if (n == 0) continue;
      std::printf("[AGGREGATE] %-11zu %-6d %-6d %-12.2f %-11.2f (mean lat over %d pairs)\n",
                  plan[p].first, plan[p].second, a.iters, sumBw, sumLat / n, n);
    }
    std::fflush(stdout);
  }

  // Perf report: each initiator appends its own rows (matches the Python bench's
  // per-rank _emit_io_perf; build_report.py dedups by op/backend/msg/batch).
  // No-op unless MORI_PERF_OUT is set.
  if (isInitiator) WritePerfRecords(a, mine);

  // Correctness check on the last sweep point, matching the Python benchmark's
  // always-on validation. Runs after the timed region so it cannot perturb the
  // reported numbers.
  const auto& last = plan.back();
  const bool valid = ValidateTransfer(rdv, buf, last.first, last.second,
                                      a.enable_batch_transfer && last.second > 1,
                                      a.batch_contiguous, isInitiator, !a.skip_validate, label);
  return valid ? 0 : 1;
}

// ------------------ driver: XGMI, both sides in one process -----------------
// The Python bench's DEFAULT xgmi mode: one process owns both GPUs, so there is
// no rendezvous, no remote engine, and no IPC -- source and destination are two
// registrations against the same engine. (The two-rank path above is already
// what Python calls --xgmi-multiprocess: distinct engine keys make the backend
// take its hipIpc* route.)
int RunXgmiSingleProcess(const Args& a) {
  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  const int src = a.gpu;
  const int dst = (a.dst_gpu >= 0) ? a.dst_gpu : a.gpu + 1;
  if (src == dst) throw std::invalid_argument("--src-gpu and --dst-gpu must differ");
  if (src < 0 || dst < 0 || src >= ndev || dst >= ndev) {
    throw std::invalid_argument("--src-gpu/--dst-gpu out of range (" + std::to_string(ndev) +
                                " visible GPUs)");
  }

  const auto plan = BuildPlan(a);
  const size_t bufBytes = PlanBufferBytes(a, plan);

  HIP_CHECK(hipSetDevice(src));

  // No peer ever dials in, but the engine still wants an endpoint; the XGMI
  // backend never binds it.
  IOEngineConfig cfg;
  cfg.host = !a.self_ip.empty() ? a.self_ip : (!a.master_ip.empty() ? a.master_ip : "127.0.0.1");
  cfg.port = static_cast<uint16_t>(a.port + 1);
  IOEngine engine("xgmi-benchmark", cfg);
  CreateBackendFor(engine, a);

  // hipMalloc lands on the CURRENT device, so the device has to be switched
  // around each allocation rather than set once up front.
  HIP_CHECK(hipSetDevice(src));
  BenchBuffer srcBuf(engine, bufBytes, src, MemoryLocationType::GPU);
  HIP_CHECK(hipSetDevice(dst));
  BenchBuffer dstBuf(engine, bufBytes, dst, MemoryLocationType::GPU);
  HIP_CHECK(hipSetDevice(src));

  // Same two-sided seeding as the distributed path, so a transfer that never
  // happened cannot pass validation.
  if (a.skip_validate) {
    HIP_CHECK(hipMemset(srcBuf.Data(), 0, bufBytes));
    HIP_CHECK(hipMemset(dstBuf.Data(), 0, bufBytes));
  } else {
    FillPattern(srcBuf.Data(), bufBytes, 0);
    FillPattern(dstBuf.Data(), bufBytes, 1);
  }

  std::cout << "XGMI single-process: GPU" << src << " -> GPU" << dst << std::endl;
  std::cout << "MsgSize(B)  Batch  Iters  AvgBW(GB/s)  AvgLat(us)  TotalDur(us)" << std::endl;
  const auto results = RunSweep(a, engine, srcBuf.Desc(), dstBuf.Desc(), plan, "");
  // Emit perf records here too: this path returns before the distributed writer,
  // and the equivalent Python XGMI mode records perf. No-op unless MORI_PERF_OUT
  // is set.
  WritePerfRecords(a, results);

  // Both buffers are local, so the checksums compare directly -- no exchange.
  const auto& last = plan.back();
  const bool batched = a.enable_batch_transfer && last.second > 1;
  const bool wanted = !a.skip_validate;
  const auto mine = TransferChecksums(srcBuf.Data(), last.first, last.second, batched,
                                      a.batch_contiguous, wanted);
  const auto peer = TransferChecksums(dstBuf.Data(), last.first, last.second, batched,
                                      a.batch_contiguous, wanted);
  return CompareChecksums(mine, peer, last.first, last.second, "") ? 0 : 1;
}

// ------------------------------ main ---------------------------------------
int RunBenchmark(int argc, char** argv) {
  Args a = ParseArgs(argc, argv);
  if (a.help) {
    PrintUsage(argv[0]);
    return 0;
  }
  SetLogLevel(a.log_level);
  ValidateArgs(a);

  if (a.xgmi_single_process) return RunXgmiSingleProcess(a);

  const int numDev = (a.rank == 0) ? a.num_initiator_dev : a.num_target_dev;
  if (numDev == 1) return RunDistributed(a, 0);

  // One process per GPU, forked BEFORE any HIP call so each child builds its own
  // device context and its own engine: the RDMA backend's queues, worker threads
  // and MRs are per-process state that does not survive being shared, which is
  // why this fans out with processes rather than threads (and why the Python
  // bench uses mp.spawn rather than a thread pool).
  std::vector<pid_t> kids;
  kids.reserve(static_cast<size_t>(numDev - 1));
  for (int i = 1; i < numDev; ++i) {
    const pid_t pid = fork();
    if (pid < 0) {
      // Reap after signalling, for the same reason the loop below does: an
      // unreaped child keeps its bootstrap port and fails the next launch with
      // "Address already in use". The peer node still has to wait out its own
      // bootstrap timeout -- nothing here can tell it the ring will never close.
      const int err = errno;
      for (pid_t k : kids) kill(k, SIGTERM);
      for (pid_t k : kids) {
        int status = 0;
        waitpid(k, &status, 0);
      }
      throw std::runtime_error(std::string("fork failed: ") + std::strerror(err));
    }
    if (pid == 0) {
      // Child. _exit rather than return: RunDistributed has already run its
      // destructors, and unwinding back through main would flush the parent's
      // inherited stdio buffers a second time.
      int rc = 1;
      try {
        rc = RunDistributed(a, i);
      } catch (const std::exception& e) {
        // Labelled by local index, not GPU ordinal: the ordinal is only valid
        // once the device-count check inside RunDistributed has passed, and
        // naming a GPU that does not exist is exactly the confusing case.
        std::cerr << "bench_engine[dev " << i << "]: " << e.what() << std::endl;
      }
      _exit(rc);
    }
    kids.push_back(pid);
  }

  int rc = 1;
  try {
    rc = RunDistributed(a, 0);
  } catch (const std::exception& e) {
    std::cerr << "bench_engine[dev 0]: " << e.what() << std::endl;
  }

  // Reap every child even after a local failure, so a partial run does not leave
  // processes holding their bootstrap ports against the next launch.
  for (pid_t k : kids) {
    int status = 0;
    if (waitpid(k, &status, 0) < 0) {
      rc = 1;
      continue;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) rc = 1;
  }
  return rc;
}

}  // namespace

int main(int argc, char** argv) {
  // Both the bootstrap layer and the IO engine report failures by throwing, so
  // unwinding here runs the destructors instead of leaving teardown to an exit
  // path.
  try {
    return RunBenchmark(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "bench_engine: " << e.what() << std::endl;
    return 1;
  }
}

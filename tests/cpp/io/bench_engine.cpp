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
// Rendezvous: 2 processes (one per node), TCP socket exchange of EngineDesc and
// MemoryDesc (msgpack, same as MORI's pybind pack()/unpack()). Rank 0 =
// initiator, rank 1 = target. Initiator drives all transfers (RDMA one-sided
// WRITE/READ); target only registers memory and waits.
//
// Build: see build.sh (links libmori_io + libmori_application from the editable
// install, includes 3rdparty/msgpack-c + HIP).

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>
#include <msgpack.hpp>

#include "mori/io/io.hpp"

using namespace mori::io;
using Clock = std::chrono::steady_clock;

#define HIP_CHECK(expr)                                                                   \
  do {                                                                                    \
    hipError_t _e = (expr);                                                               \
    if (_e != hipSuccess) {                                                               \
      std::cerr << "HIP error " << hipGetErrorString(_e) << " at " << __FILE__ << ":"     \
                << __LINE__ << std::endl;                                                 \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// ------------------------- tiny TCP rendezvous -----------------------------
// Rank 0 listens, rank 1 connects. Then symmetric length-prefixed blob swap.

static int TcpListenAccept(uint16_t port) {
  int lfd = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    perror("bind");
    std::exit(1);
  }
  listen(lfd, 1);
  int fd = accept(lfd, nullptr, nullptr);
  close(lfd);
  return fd;
}

static int TcpConnect(const std::string& host, uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  for (int retry = 0; retry < 100; ++retry) {
    if (connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) return fd;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  std::cerr << "connect failed to " << host << ":" << port << std::endl;
  std::exit(1);
}

static void SendAll(int fd, const void* buf, size_t n) {
  const char* p = static_cast<const char*>(buf);
  while (n) {
    ssize_t k = send(fd, p, n, 0);
    if (k <= 0) { perror("send"); std::exit(1); }
    p += k; n -= k;
  }
}
static void RecvAll(int fd, void* buf, size_t n) {
  char* p = static_cast<char*>(buf);
  while (n) {
    ssize_t k = recv(fd, p, n, 0);
    if (k <= 0) { perror("recv"); std::exit(1); }
    p += k; n -= k;
  }
}
static void SendBlob(int fd, const std::string& b) {
  uint64_t len = b.size();
  SendAll(fd, &len, sizeof(len));
  SendAll(fd, b.data(), len);
}
static std::string RecvBlob(int fd) {
  uint64_t len = 0;
  RecvAll(fd, &len, sizeof(len));
  std::string b(len, '\0');
  RecvAll(fd, b.data(), len);
  return b;
}
static void Barrier(int fd) {  // symmetric 1-byte ping-pong
  char c = 'x';
  SendAll(fd, &c, 1);
  RecvAll(fd, &c, 1);
}

template <typename T>
static std::string Pack(const T& v) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, v);
  return std::string(buf.data(), buf.size());
}
template <typename T>
static T Unpack(const std::string& b) {
  auto oh = msgpack::unpack(b.data(), b.size());
  return oh.get().as<T>();
}

// ------------------------------- config ------------------------------------
// Flags mirror MORI's Python benchmark (tests/python/io/benchmark.py) so runs
// are directly comparable. Names use the Python spelling where they exist.
struct Args {
  int rank = 0;                  // 0=initiator, 1=target
  std::string master_ip;         // rank 1 connects here (rank 0's data IP) for TCP rendezvous
  std::string self_ip;           // this rank's own reachable IP (advertised in EngineDesc)
  uint16_t port = 18515;
  int gpu = 0;
  int target_dev_offset = 0;     // --target-dev-offset (target GPU = (gpu+offset)%ndev)
  std::string op = "write";      // write|read

  // Sweep control (Python parity). --all sweeps message size; --all-batch sweeps
  // batch size. Neither set => single run at (buffer_size, batch).
  bool sweep_all = false;        // --all
  bool sweep_batch = false;      // --all-batch
  size_t buffer_size = 32768;    // --buffer-size (single message size when not sweeping)
  size_t sweep_start = 8;        // --sweep-start-size (Python default 8)
  size_t sweep_max = 1u << 20;   // --sweep-max-size (Python default 2^20)
  size_t sweep_step = 0;         // --sweep-step (0 = geometric x2; >0 = linear +step)
  int iters = 500;
  int warmup = 50;               // --warmup-iters

  // RdmaBackendConfig knobs
  int qp_per_transfer = 4;       // --num-qp-per-transfer
  int worker_threads = 1;        // --num-worker-threads
  int post_batch_size = -1;      // --post-batch-size
  std::string poll_cq_mode = "polling";  // polling|event
  bool disable_chunking = false; // --disable-chunking (default: chunking ON, like Python)
  size_t chunk_bytes = 65536;    // --chunk-bytes
  int max_chunks = 64;           // --max-chunks
  int max_send_wr = 0;           // --max-send-wr (0 = leave default)
  int max_cqe_num = 0;           // --max-cqe-num
  int max_msg_sge = 0;           // --max-msg-sge

  // batch / session
  int batch = 1;                 // --transfer-batch-size (transfers per request)
  bool enable_batch_transfer = true;  // --enable-batch-transfer / --disable-batch-transfer.
                                 // ON (default): batch>1 => ONE N-descriptor batch request
                                 // (nixl-equivalent). OFF: batch>1 => N individual single
                                 // transfers per iteration (Python run_single_once path).
  bool batch_contiguous = false; // --batch-contiguous (adjacent offsets → merged WR);
                                 // default strided (each transfer a separate WR)
  bool enable_sess = false;      // --enable-sess (session fast-path); Python default: off

  std::string mem_type = "gpu";  // --mem-type gpu|cpu
  std::string init_mem_type;     // --initiator-mem-type (empty => mem_type)
  std::string target_mem_type;   // --target-mem-type   (empty => mem_type)

  std::string log_level = "info";  // --log-level trace|debug|info|warning|error|critical
};

static Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (k == "--rank") a.rank = std::stoi(next());
    else if (k == "--master-ip") a.master_ip = next();
    else if (k == "--self-ip") a.self_ip = next();
    else if (k == "--port") a.port = static_cast<uint16_t>(std::stoi(next()));
    else if (k == "--gpu") a.gpu = std::stoi(next());
    else if (k == "--target-dev-offset") a.target_dev_offset = std::stoi(next());
    else if (k == "--op" || k == "--op-type") a.op = next();
    else if (k == "--all") a.sweep_all = true;
    else if (k == "--all-batch") a.sweep_batch = true;
    else if (k == "--buffer-size") a.buffer_size = std::stoull(next());
    else if (k == "--sweep-start" || k == "--sweep-start-size") a.sweep_start = std::stoull(next());
    else if (k == "--sweep-max" || k == "--sweep-max-size") a.sweep_max = std::stoull(next());
    else if (k == "--sweep-step") a.sweep_step = std::stoull(next());
    else if (k == "--iters") a.iters = std::stoi(next());
    else if (k == "--warmup" || k == "--warmup-iters") a.warmup = std::stoi(next());
    else if (k == "--qp-per-transfer" || k == "--num-qp-per-transfer") a.qp_per_transfer = std::stoi(next());
    else if (k == "--worker-threads" || k == "--num-worker-threads") a.worker_threads = std::stoi(next());
    else if (k == "--post-batch-size") a.post_batch_size = std::stoi(next());
    else if (k == "--poll_cq_mode" || k == "--poll-cq-mode") a.poll_cq_mode = next();
    else if (k == "--disable-chunking") a.disable_chunking = true;
    else if (k == "--chunk-bytes") a.chunk_bytes = std::stoull(next());
    else if (k == "--max-chunks") a.max_chunks = std::stoi(next());
    else if (k == "--max-send-wr") a.max_send_wr = std::stoi(next());
    else if (k == "--max-cqe-num") a.max_cqe_num = std::stoi(next());
    else if (k == "--max-msg-sge") a.max_msg_sge = std::stoi(next());
    else if (k == "--batch" || k == "--transfer-batch-size") a.batch = std::stoi(next());
    else if (k == "--enable-batch-transfer") a.enable_batch_transfer = true;
    else if (k == "--disable-batch-transfer") a.enable_batch_transfer = false;
    else if (k == "--batch-contiguous") a.batch_contiguous = true;
    else if (k == "--enable-sess") a.enable_sess = true;
    else if (k == "--disable-sess") a.enable_sess = false;
    else if (k == "--mem-type") a.mem_type = next();
    else if (k == "--initiator-mem-type") a.init_mem_type = next();
    else if (k == "--target-mem-type") a.target_mem_type = next();
    else if (k == "--log-level") a.log_level = next();
    else { std::cerr << "unknown arg " << k << std::endl; std::exit(1); }
  }
  return a;
}

// ------------------------------ main ---------------------------------------
int main(int argc, char** argv) {
  Args a = ParseArgs(argc, argv);
  SetLogLevel(a.log_level);

  // Per-role memory type: initiator (rank 0) / target (rank 1) may override the
  // shared --mem-type, enabling mixed CPU<->GPU transfers. Each process only
  // allocates its own side, so no cross-node coupling is needed.
  const std::string myMem = (a.rank == 0)
      ? (!a.init_mem_type.empty() ? a.init_mem_type : a.mem_type)
      : (!a.target_mem_type.empty() ? a.target_mem_type : a.mem_type);
  const bool cpuMem = (myMem == "cpu");
  const MemoryLocationType memLoc = cpuMem ? MemoryLocationType::CPU : MemoryLocationType::GPU;

  // Target GPU shift for cross-rail pairing (GPU memory only, matches Python).
  int gpu = a.gpu;
  if (a.rank == 1 && !cpuMem && a.target_dev_offset != 0) {
    int ndev = 0;
    HIP_CHECK(hipGetDeviceCount(&ndev));
    if (ndev > 0) gpu = (a.gpu + a.target_dev_offset) % ndev;
  }
  HIP_CHECK(hipSetDevice(gpu));  // valid device context needed even for host mem

  // Build the run plan: a list of (msgSize, batch) points.
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

  // Buffer must hold the largest single REQUEST across the whole plan. Strided
  // batch (default) needs (msg+1)*batch to keep slots non-adjacent; contiguous
  // needs msg*batch.
  size_t bufBytes = 0;
  for (auto& planEntry : plan) {
    const size_t msg = planEntry.first;
    const int b = planEntry.second;
    const bool pBatched = b > 1;
    const size_t slotStride = a.batch_contiguous ? msg : (msg + 1);
    const size_t need = pBatched ? slotStride * static_cast<size_t>(b) : msg;
    bufBytes = std::max(bufBytes, need);
  }

  void* buf = nullptr;
  if (cpuMem) {
    HIP_CHECK(hipHostMalloc(&buf, bufBytes, 0));
  } else {
    HIP_CHECK(hipMalloc(&buf, bufBytes));
  }
  HIP_CHECK(hipMemset(buf, 0, bufBytes));

  // Out-of-band control endpoint for the MORI engine. host MUST be an IP the
  // peer can reach (advertised via EngineDesc for RDMA QP setup). Defaults to
  // master_ip: correct for rank 0 (it IS the master); rank 1 should pass
  // --self-ip when its reachable IP differs from the master's.
  IOEngineConfig cfg;
  cfg.host = !a.self_ip.empty() ? a.self_ip : a.master_ip;
  cfg.port = static_cast<uint16_t>(a.port + 1 + a.rank);  // distinct per rank
  std::string key = a.rank == 0 ? "initiator" : "target";
  IOEngine engine(key, cfg);

  RdmaBackendConfig rdmaCfg{};
  rdmaCfg.qpPerTransfer = a.qp_per_transfer;
  rdmaCfg.postBatchSize = a.post_batch_size;
  rdmaCfg.numWorkerThreads = a.worker_threads;
  rdmaCfg.pollCqMode = (a.poll_cq_mode == "event") ? PollCqMode::EVENT : PollCqMode::POLLING;
  rdmaCfg.enableNotification = false;         // match MORI Python bench RDMA path
  rdmaCfg.enableTransferChunking = !a.disable_chunking;  // chunking ON by default (Python parity)
  rdmaCfg.chunkBytes = a.chunk_bytes;
  rdmaCfg.maxChunksPerTransfer = a.max_chunks;
  if (a.max_send_wr > 0) rdmaCfg.maxSendWr = a.max_send_wr;
  if (a.max_cqe_num > 0) rdmaCfg.maxCqeNum = a.max_cqe_num;
  if (a.max_msg_sge > 0) rdmaCfg.maxMsgSge = a.max_msg_sge;
  engine.CreateBackend(BackendType::RDMA, rdmaCfg);

  MemoryDesc localMem = engine.RegisterMemory(buf, bufBytes, gpu, memLoc);

  // --- rendezvous over a side TCP socket -----------------------------------
  int sock = (a.rank == 0) ? TcpListenAccept(a.port) : TcpConnect(a.master_ip, a.port);
  int one = 1;
  setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  // Exchange EngineDesc then register the remote engine.
  EngineDesc myEng = engine.GetEngineDesc();
  SendBlob(sock, Pack(myEng));
  EngineDesc peerEng = Unpack<EngineDesc>(RecvBlob(sock));
  engine.RegisterRemoteEngine(peerEng);

  // Exchange MemoryDesc.
  SendBlob(sock, Pack(localMem));
  MemoryDesc peerMem = Unpack<MemoryDesc>(RecvBlob(sock));

  Barrier(sock);

  if (a.rank == 1) {
    // Target: nothing to drive. RDMA one-sided ops complete without target CPU.
    // Just hold memory registered until the initiator says it's done.
    Barrier(sock);   // wait for initiator to finish the whole sweep
    if (cpuMem) HIP_CHECK(hipHostFree(buf)); else HIP_CHECK(hipFree(buf));
    close(sock);
    return 0;
  }

  // -------------------- initiator: run the sweep ---------------------------
  // Session fast-path (matches MORI --enable-sess). When --disable-sess, we call
  // the engine batch/single APIs directly with explicit MemoryDesc + uid vecs.
  const bool useSess = a.enable_sess;
  IOEngineSession* sessPtr = nullptr;
  std::optional<IOEngineSession> sessOpt;
  if (useSess) {
    sessOpt = engine.CreateSession(localMem, peerMem);
    if (!sessOpt) { std::cerr << "CreateSession failed" << std::endl; std::exit(1); }
    sessPtr = &*sessOpt;
  }
  const bool isRead = (a.op == "read");

  std::cout << "MsgSize(B)  Batch  Iters  AvgBW(GB/s)  AvgLat(us)  TotalDur(us)" << std::endl;

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
    const int perSlot = batched ? 1 : curBatch;    // statuses per request
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
    // Engine (non-session) batch path needs vec-of-vec + desc vectors.
    MemDescVec locVec{localMem}, remVec{peerMem};
    BatchSizeVec offVec{offsets}, sizeVec{sizes};

    auto post = [&]() {
      TransferStatus* base = &st[0];
      for (int i = 0; i < perSlot; ++i) base[i].SetCode(StatusCode::INIT);
      if (batched) {
        // ONE N-descriptor batch request (nixl / Python --enable-batch-transfer).
        TransferUniqueId id =
            useSess ? sessPtr->AllocateTransferUniqueId() : engine.AllocateTransferUniqueId();
        if (useSess) {
          if (isRead) sessPtr->BatchRead(offsets, offsets, sizes, &base[0], id);
          else        sessPtr->BatchWrite(offsets, offsets, sizes, &base[0], id);
        } else {
          TransferStatusPtrVec sp{&base[0]};
          TransferUniqueIdVec ids{id};
          if (isRead) engine.BatchRead(locVec, offVec, remVec, offVec, sizeVec, sp, ids);
          else        engine.BatchWrite(locVec, offVec, remVec, offVec, sizeVec, sp, ids);
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
            if (isRead) sessPtr->Read(off, off, msg, &base[i], id);
            else        sessPtr->Write(off, off, msg, &base[i], id);
          } else {
            if (isRead) engine.Read(localMem, off, peerMem, off, msg, &base[i], id);
            else        engine.Write(localMem, off, peerMem, off, msg, &base[i], id);
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
      while (!reqDone()) { /* spin: CQ worker thread stores status */ }
      if (TransferStatus* f = reqFailed()) {
        std::cerr << "warmup transfer failed: " << f->Message() << std::endl;
        std::exit(1);
      }
    }

    // ---- timed region: ONE whole-loop timer, nixl-style ----
    // Strict stop-and-wait (nixl --pipeline_depth 1): post one request, spin
    // until it completes, then post the next.
    auto t0 = Clock::now();

    for (int completed = 0; completed < a.iters; ++completed) {
      post();
      // spin-poll, mirroring nixl's "check status, if IN_PROG continue" scan
      // (status flags stored by MORI's CQ worker thread)
      while (!reqDone()) { /* spin */ }
      if (TransferStatus* f = reqFailed()) {
        std::cerr << "transfer failed: " << f->Message() << std::endl;
        std::exit(1);
      }
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
    const int effBatch = curBatch;
    const double numXfers = static_cast<double>(a.iters) * effBatch;
    double total_bytes = static_cast<double>(msg) * numXfers;
    double avg_bw = (total_bytes / 1e9) / (total_us / 1e6);   // GB/s, GB=10^9
    double avg_lat = total_us / numXfers;                      // us per single transfer

    std::printf("%-11zu %-6d %-6d %-12.2f %-11.2f %-.1f\n", msg, effBatch, a.iters, avg_bw, avg_lat,
                total_us);
    std::fflush(stdout);
  }

  Barrier(sock);   // tell target we're done
  if (cpuMem) HIP_CHECK(hipHostFree(buf)); else HIP_CHECK(hipFree(buf));
  close(sock);
  return 0;
}

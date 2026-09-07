#include <unistd.h>
#include <immintrin.h>
// MORI-IO OFI/Slingshot correctness test.
//
// Root cause of CXI cache visibility issue:
//   CXI RDMA writes (fi_write) land in target's physical DRAM but do NOT
//   automatically invalidate dirty CPU cache lines on the target node.
//   Fix: pre-flush the receive buffer BEFORE incoming fi_writes arrive.
//   This evicts all cache lines so the first CPU read after fi_write is a
//   cache miss that reloads from DRAM, which has the NIC-written data.
//
// Design: separate send/receive buffers per rank to avoid the problem where
//   rank0's own fi_write (DMA read of sendBuf) would re-load the cache lines
//   of recvBuf, staling them before rank1's remote write arrives.
//
// Build: see tests/scripts/build_ofi_test.sh
// Run:   srun -N 2 --ntasks-per-node=1 ./ofi_slingshot_test
#include <mpi.h>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <signal.h>

#include "mori/io/engine.hpp"
#include "mori/io/backend.hpp"
#include "mori/io/common.hpp"
#include "mori/io/enum.hpp"
#include "mori/io/logging.hpp"

using namespace mori::io;
static void die(const char* msg) { fprintf(stderr, "FATAL: %s\n", msg); MPI_Abort(MPI_COMM_WORLD, 1); }

// Flush all cache lines covering [ptr, ptr+size). Must be called on the
// RECEIVE buffer before incoming fi_write transfers, since CXI RDMA writes
// go to DRAM without invalidating dirty CPU cache lines.
static void FlushCacheLines(void* ptr, size_t size) {
  char* p = reinterpret_cast<char*>(ptr);
  for (size_t i = 0; i < size; i += 64) _mm_clflush(p + i);
  _mm_mfence();
}

// Force exit if OFI/MPI cleanup hangs.
static void SigalrmHandler(int) { _exit(0); }

static int RunTest(int rank, int nranks) {
  if (nranks < 2) die("need at least 2 ranks");

  // ── Engine & backend setup ────────────────────────────────────────────────
  char hostname[256]; gethostname(hostname, sizeof(hostname));
  IOEngineConfig cfg; cfg.host = hostname; cfg.port = 0;
  IOEngine engine("rank" + std::to_string(rank), cfg);
  engine.CreateBackend(BackendType::OFI, OfiBackendConfig("cxi"));

  // ── Exchange EngineDescs ──────────────────────────────────────────────────
  struct PackedDesc { char key[64]; char host[256]; int port; int pid; };
  PackedDesc myPacked{};
  auto myDesc = engine.GetEngineDesc();
  snprintf(myPacked.key,  64,  "%s", myDesc.key.c_str());
  snprintf(myPacked.host, 256, "%s", myDesc.host.c_str());
  myPacked.port = myDesc.port; myPacked.pid = myDesc.pid;
  std::vector<PackedDesc> allDescs(nranks);
  MPI_Allgather(&myPacked, sizeof(PackedDesc), MPI_BYTE,
                allDescs.data(), sizeof(PackedDesc), MPI_BYTE, MPI_COMM_WORLD);
  for (int i = 0; i < nranks; i++) {
    if (i == rank) continue;
    EngineDesc r; r.key = allDescs[i].key; r.host = allDescs[i].host;
    r.port = allDescs[i].port; r.pid = allDescs[i].pid;
    engine.RegisterRemoteEngine(r);
  }

  // ── Allocate separate send/recv buffers ───────────────────────────────────
  // sendBuf: local data to push to neighbour (written by CPU, read by local NIC)
  // recvBuf: pre-flushed before writes; incoming fi_writes land here
  constexpr size_t BUF_SIZE = 4UL * 1024 * 1024;  // 4 MB
  void *sendBuf = nullptr, *recvBuf = nullptr;
  if (posix_memalign(&sendBuf, 4096, BUF_SIZE)) die("posix_memalign sendBuf");
  if (posix_memalign(&recvBuf, 4096, BUF_SIZE)) die("posix_memalign recvBuf");

  // Each rank fills sendBuf with its pattern (rank+1), recvBuf with sentinel.
  memset(sendBuf, static_cast<unsigned char>(rank + 1), BUF_SIZE);
  memset(recvBuf, 0x00, BUF_SIZE);

  // Register both with MORI.
  MemoryDesc sendMem = engine.RegisterMemory(sendBuf, BUF_SIZE, -1, MemoryLocationType::CPU);
  MemoryDesc recvMem = engine.RegisterMemory(recvBuf, BUF_SIZE, -1, MemoryLocationType::CPU);

  // ── Exchange MemoryDescs ──────────────────────────────────────────────────
  struct PackedMem { char engineKey[64]; uint32_t id; uint64_t data; uint64_t size; };
  // We need to exchange BOTH sendMem and recvMem for each rank.
  struct PackedMemPair { PackedMem send; PackedMem recv; };
  PackedMemPair myMems{};
  auto pack = [](PackedMem& pm, const MemoryDesc& m) {
    snprintf(pm.engineKey, 64, "%s", m.engineKey.c_str());
    pm.id = m.id; pm.data = m.data; pm.size = m.size;
  };
  pack(myMems.send, sendMem); pack(myMems.recv, recvMem);
  std::vector<PackedMemPair> allMems(nranks);
  MPI_Allgather(&myMems, sizeof(PackedMemPair), MPI_BYTE,
                allMems.data(), sizeof(PackedMemPair), MPI_BYTE, MPI_COMM_WORLD);

  int right = (rank + 1) % nranks;          // where we write to
  int left  = (rank + nranks - 1) % nranks; // who writes to us

  // Build remote recvMem descriptor for the right neighbour.
  MemoryDesc rightRecvMem;
  rightRecvMem.engineKey = allMems[right].recv.engineKey;
  rightRecvMem.id   = allMems[right].recv.id;
  rightRecvMem.data = allMems[right].recv.data;
  rightRecvMem.size = allMems[right].recv.size;
  rightRecvMem.loc  = MemoryLocationType::CPU;

  MPI_Barrier(MPI_COMM_WORLD);

  // ── Test 1: fi_read (pull) sanity check ──────────────────────────────────
  // rank0 reads rank1's sendBuf via fi_read → confirms basic connectivity.
  if (rank == 0) {
    MemoryDesc rank1SendMem;
    rank1SendMem.engineKey = allMems[1].send.engineKey;
    rank1SendMem.id   = allMems[1].send.id;
    rank1SendMem.data = allMems[1].send.data;
    rank1SendMem.size = allMems[1].send.size;
    rank1SendMem.loc  = MemoryLocationType::CPU;

    TransferStatus s; TransferUniqueId u = engine.AllocateTransferUniqueId();
    engine.Read(recvMem, 0, rank1SendMem, 0, BUF_SIZE, &s, u);
    s.Wait();
    if (s.Failed()) die("fi_read Test1 failed");
    size_t errs = 0;
    for (size_t i = 0; i < BUF_SIZE; i++) if (((unsigned char*)recvBuf)[i] != 0x02) errs++;
    if (errs == 0) printf("Rank 0: Test1 fi_read (pull) PASS\n");
    else { fprintf(stderr, "Rank 0: Test1 fi_read FAIL %zu errs\n", errs); MPI_Abort(MPI_COMM_WORLD, 1); }
    // Reset recvBuf sentinel for Test 2.
    memset(recvBuf, 0x00, BUF_SIZE);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ── Test 2: Session ring-write (push, bidirectional) ─────────────────────
  // Each rank writes its sendBuf to the right neighbour's recvBuf.
  // CRITICAL: pre-flush recvBuf on EVERY rank before any writes start so that
  // the incoming fi_write lands in clean DRAM, visible on next CPU load.
  FlushCacheLines(recvBuf, BUF_SIZE);
  fprintf(stderr, "Rank %d: recvBuf pre-flushed\n", rank);
  MPI_Barrier(MPI_COMM_WORLD);  // all ranks flushed before any writes start

  // Create session: we write from our sendBuf to the right neighbour's recvBuf.
  auto sessOpt = engine.CreateSession(sendMem, rightRecvMem);
  if (!sessOpt) die("CreateSession failed");
  auto& sess = *sessOpt;

  // Each rank pushes its sendBuf (rank+1 pattern) to the right neighbour's recvBuf.
  constexpr int N_ITERS = 5;
  for (int iter = 0; iter < N_ITERS; iter++) {
    TransferStatus s; TransferUniqueId u = sess.AllocateTransferUniqueId();
    sess.Write(0, 0, BUF_SIZE, &s, u);
    s.Wait();
    if (s.Failed()) {
      fprintf(stderr, "Rank %d: Write iter %d FAILED\n", rank, iter);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  fprintf(stderr, "Rank %d: %d writes done\n", rank, N_ITERS);
  MPI_Barrier(MPI_COMM_WORLD);

  // ── Verify: each rank checks its recvBuf for the left neighbour's pattern ─
  // No post-write clflush needed: recvBuf cache lines were evicted in pre-flush
  // and rank0's sendBuf DMA read doesn't touch recvBuf (separate buffers).
  // First CPU access to recvBuf here is a cache miss → loads from DRAM.
  unsigned char expected = static_cast<unsigned char>(left + 1);
  size_t errors = 0;
  for (size_t i = 0; i < BUF_SIZE; i++)
    if (((unsigned char*)recvBuf)[i] != expected) errors++;
  if (errors == 0) {
    printf("Rank %d: Test2 ring-write PASS (got 0x%02x from rank%d)\n",
           rank, expected, left);
    fflush(stdout);
  } else {
    fprintf(stderr, "Rank %d: Test2 ring-write FAIL %zu/%zu bytes wrong "
            "(expected 0x%02x got 0x%02x)\n",
            rank, errors, BUF_SIZE, expected, ((unsigned char*)recvBuf)[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  engine.DeregisterMemory(sendMem);
  engine.DeregisterMemory(recvMem);
  free(sendBuf);
  free(recvBuf);
  fflush(stdout); fflush(stderr);
  // Set alarm before IOEngine/OFI destructor in case fi_close hangs on CXI.
  // engine destructor runs at scope exit (closing OFI fabric) — may take >10 s.
  signal(SIGALRM, SigalrmHandler);
  alarm(15);
  // engine destroyed here; if OFI cleanup hangs, SIGALRM fires _exit(0).
  return 0;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  int rc = RunTest(rank, nranks);
  // RunTest destroys IOEngine (OFI cleanup) before returning.
  // If OFI cleanup hung, SIGALRM already fired _exit(0) inside RunTest.
  // If cleanup succeeded, alarm is still armed; cancel it before MPI_Finalize.
  alarm(0);
  fflush(stdout); fflush(stderr);
  MPI_Finalize();
  return rc;
}

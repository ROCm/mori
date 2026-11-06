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

// test_fence_xnode.cpp
//
// Two-process, two-node version of test_fence, shaped like the KV push it protects:
//
//   receiver (decode)   owns a registered buffer of 64 slots and fences the producer
//   producer (prefill)  streams one write per slot, 10 ms apart, without waiting
//
// The receiver fences the producer while the stream is still in flight, the way a
// decode-side timeout fires while prefill is mid-way through its per-layer writes.
// Slots the producer wrote before the fence legitimately landed; what must hold is
// that NOTHING lands once FenceRemoteEngine has returned. A third step asks what a
// producer that is still alive can do afterwards, which the single-process test
// cannot answer: a producer that is still running must be able to write again, which
// means it has to notice its connection died and rebuild it.
//
// Run one process per node (the receiver first), for example:
//   test_fence_xnode --role receiver --ip <recv-ip> --loc gpu
//   test_fence_xnode --role producer --ip <prod-ip> --peer <recv-ip> --loc gpu
// The receiver's exit status is the verdict.

#include <arpa/inet.h>
#include <hip/hip_runtime_api.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <msgpack.hpp>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "mori/io/io.hpp"

using namespace mori::io;

namespace {

constexpr size_t kSlotBytes = 4096;
constexpr int kSlots = 64;
constexpr size_t kBufBytes = kSlotBytes * kSlots;
constexpr int kStreamGapMs = 10;
constexpr int kFenceAfterSlots = 5;
constexpr int kSetupMs = 60000;
constexpr int kWaitMs = 120000;
constexpr uint8_t kStreamByte = 0xA1;
constexpr uint8_t kRetryByte = 0xCC;
// The producer's rebuilds while fenced were refused, and failed QP replacement backs
// off, so after unfencing it may take a few attempts to get back in.
constexpr int kRetryBudgetMs = 20000;
const char* const kReceiverKey = "xn_receiver";
const char* const kProducerKey = "xn_producer";

struct Args {
  std::string role;
  std::string ip{"127.0.0.1"};
  std::string peer{"127.0.0.1"};
  int enginePort{31400};
  int rvPort{31500};
  bool gpu{true};
};

[[noreturn]] void Die(const std::string& msg) {
  std::fprintf(stderr, "FATAL: %s\n", msg.c_str());
  std::exit(2);
}

Args Parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string k = argv[i], v = argv[i + 1];
    if (k == "--role")
      a.role = v;
    else if (k == "--ip")
      a.ip = v;
    else if (k == "--peer")
      a.peer = v;
    else if (k == "--port")
      a.enginePort = std::atoi(v.c_str());
    else if (k == "--rv")
      a.rvPort = std::atoi(v.c_str());
    else if (k == "--loc")
      a.gpu = (v == "gpu");
    else
      Die("unknown argument " + k);
  }
  if (a.role != "receiver" && a.role != "producer") Die("--role receiver|producer required");
  return a;
}

/* ------------------------------ rendezvous over plain TCP ------------------------------ */

std::string Hex(const std::string& s) {
  static const char* d = "0123456789abcdef";
  std::string o;
  for (unsigned char c : s) {
    o += d[c >> 4];
    o += d[c & 15];
  }
  return o;
}

std::string Unhex(const std::string& h) {
  std::string o;
  for (size_t i = 0; i + 1 < h.size(); i += 2) {
    o += static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16));
  }
  return o;
}

template <typename T>
std::string PackHex(const T& v) {
  msgpack::sbuffer b;
  msgpack::pack(b, v);
  return Hex(std::string(b.data(), b.size()));
}

template <typename T>
T UnpackHex(const std::string& h) {
  std::string raw = Unhex(h);
  return msgpack::unpack(raw.data(), raw.size()).get().as<T>();
}

class Channel {
 public:
  explicit Channel(int fd) : fd_(fd) {}
  ~Channel() {
    if (fd_ >= 0) close(fd_);
  }
  void Send(const std::string& line) {
    std::string s = line + "\n";
    if (send(fd_, s.data(), s.size(), MSG_NOSIGNAL) != static_cast<ssize_t>(s.size())) {
      Die("peer connection lost while sending '" + line.substr(0, 24) + "'");
    }
  }
  std::string Recv(int timeoutMs = kWaitMs) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    for (;;) {
      size_t nl = buf_.find('\n');
      if (nl != std::string::npos) {
        std::string line = buf_.substr(0, nl);
        buf_.erase(0, nl + 1);
        return line;
      }
      auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                      deadline - std::chrono::steady_clock::now())
                      .count();
      if (left <= 0) Die("timed out waiting for the peer");
      timeval tv{static_cast<time_t>(left / 1000), static_cast<suseconds_t>((left % 1000) * 1000)};
      setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
      char tmp[4096];
      ssize_t n = recv(fd_, tmp, sizeof(tmp), 0);
      if (n == 0) Die("peer closed the connection");
      if (n > 0) buf_.append(tmp, n);
    }
  }
  // Waits for a message starting with `tag`; returns the rest of the line.
  std::string Expect(const std::string& tag, int timeoutMs = kWaitMs) {
    std::string line = Recv(timeoutMs);
    if (line.compare(0, tag.size(), tag) != 0) Die("expected '" + tag + "', got '" + line + "'");
    return line.size() > tag.size() ? line.substr(tag.size() + 1) : "";
  }

 private:
  int fd_;
  std::string buf_;
};

std::unique_ptr<Channel> Listen(int port) {
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = INADDR_ANY;
  a.sin_port = htons(port);
  if (bind(ls, reinterpret_cast<sockaddr*>(&a), sizeof(a)) != 0 || listen(ls, 1) != 0) {
    Die("cannot listen on rendezvous port " + std::to_string(port));
  }
  std::printf("receiver: waiting for the producer on port %d\n", port);
  std::fflush(stdout);
  int fd = accept(ls, nullptr, nullptr);
  close(ls);
  if (fd < 0) Die("accept failed");
  return std::make_unique<Channel>(fd);
}

std::unique_ptr<Channel> Connect(const std::string& host, int port) {
  for (int attempt = 0; attempt < 600; ++attempt) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &a.sin_addr);
    if (connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) == 0) {
      return std::make_unique<Channel>(fd);
    }
    close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  Die("cannot reach the receiver at " + host + ":" + std::to_string(port));
}

/* ---------------------------------------- buffers ---------------------------------------- */

#define HIP_OK(expr)                                                              \
  do {                                                                            \
    hipError_t e_ = (expr);                                                       \
    if (e_ != hipSuccess) Die(std::string(#expr) + ": " + hipGetErrorString(e_)); \
  } while (0)

struct Buffer {
  Buffer(IOEngine* owner, bool gpuMem, uint8_t fill) : engine(owner), gpu(gpuMem) {
    if (gpu) {
      HIP_OK(hipSetDevice(0));
      HIP_OK(hipMalloc(&data, kBufBytes));
    } else if (posix_memalign(&data, 4096, kBufBytes) != 0) {
      Die("posix_memalign failed");
    }
    Fill(fill);
    desc = engine->RegisterMemory(data, kBufBytes, gpu ? 0 : -1,
                                  gpu ? MemoryLocationType::GPU : MemoryLocationType::CPU);
  }
  ~Buffer() {
    engine->DeregisterMemory(desc);
    if (gpu)
      (void)hipFree(data);
    else
      free(data);
  }
  void Fill(uint8_t v) {
    if (gpu) {
      HIP_OK(hipMemset(data, v, kBufBytes));
      HIP_OK(hipDeviceSynchronize());
    } else {
      memset(data, v, kBufBytes);
    }
  }
  // One flag per slot: did anything land in it?
  std::vector<bool> Landed() const {
    std::vector<uint8_t> h(kBufBytes);
    if (gpu)
      HIP_OK(hipMemcpy(h.data(), data, kBufBytes, hipMemcpyDeviceToHost));
    else
      memcpy(h.data(), data, kBufBytes);
    std::vector<bool> out(kSlots);
    for (int s = 0; s < kSlots; ++s) {
      out[s] = h[s * kSlotBytes] != 0 || h[(s + 1) * kSlotBytes - 1] != 0;
    }
    return out;
  }
  IOEngine* engine;
  bool gpu;
  void* data{nullptr};
  MemoryDesc desc;
};

std::unique_ptr<IOEngine> MakeEngine(const std::string& key, const Args& a) {
  RdmaBackendConfig rc{};
  rc.qpPerTransfer = 1;
  rc.enableNotification = true;
  IOEngineConfig cfg{};
  cfg.host = a.ip;
  cfg.port = a.enginePort;
  auto e = std::make_unique<IOEngine>(key, cfg);
  e->CreateBackend(BackendType::RDMA, rc);
  return e;
}

bool WaitTerminal(TransferStatus* s, int timeoutMs) {
  auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < end) {
    if (!s->Init() && !s->InProgress()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

int Count(const std::vector<bool>& v) {
  int n = 0;
  for (bool b : v) n += b;
  return n;
}

/* ---------------------------------------- receiver --------------------------------------- */

int RunReceiver(const Args& a) {
  auto ch = Listen(a.rvPort);
  auto engine = MakeEngine(kReceiverKey, a);
  Buffer buf(engine.get(), a.gpu, 0x00);

  ch->Send("ENGINE " + PackHex(engine->GetEngineDesc()));
  ch->Send("MEM " + PackHex(buf.desc));
  engine->RegisterRemoteEngine(UnpackHex<EngineDesc>(ch->Expect("ENGINE", kSetupMs)));
  ch->Send("READY");
  std::printf("receiver: %s buffers, %d slots, producer connected\n", a.gpu ? "GPU" : "CPU",
              kSlots);

  bool pass = true;
  auto verdict = [&](bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    pass = pass && ok;
  };

  // A. one unfenced write must arrive, so the later silence means something.
  ch->Expect("A_DONE");
  verdict(buf.Landed()[0], "A. unfenced write from the remote producer landed in slot 0");
  buf.Fill(0x00);

  // B. fence mid-stream.
  ch->Send("GO_STREAM");
  auto t0 = std::chrono::steady_clock::now();
  std::vector<bool> landed;
  for (;;) {
    landed = buf.Landed();
    if (Count(landed) >= kFenceAfterSlots) break;
    if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30))
      Die("stream never started");
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  FenceResult fence = engine->FenceRemoteEngine(kProducerKey);
  std::vector<bool> atFence = buf.Landed();
  const int nAtFence = Count(atFence);
  std::printf("  fence: ok=%d fenced=%d failed=%d transfers_failed=%d, %d/%d slots had landed\n",
              fence.Ok(), fence.endpointsFenced, fence.endpointsFailed, fence.transfersFailed,
              nAtFence, kSlots - 1);
  ch->Send("FENCED");

  std::string streamDone = ch->Expect("STREAM_DONE");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::vector<bool> finalState = buf.Landed();
  int latecomers = 0;
  for (int s = 0; s < kSlots; ++s) latecomers += (finalState[s] && !atFence[s]);
  verdict(fence.Ok() && fence.endpointsFenced >= 1, "B. fence reported ok and cut the connection");
  verdict(nAtFence < kSlots - 1,
          "B. the fence landed mid-stream (the stream was not already over)");
  verdict(latecomers == 0, "B. nothing landed after the fence returned (" +
                               std::to_string(latecomers) +
                               " late slots); producer saw: " + streamDone);

  // C. The producer is still alive. Once the receiver lets it back in, its writes have
  // to notice its dead connection, rebuild through the normal handshake, and land.
  verdict(engine->UnfenceRemoteEngine(kProducerKey), "C. the producer was still marked fenced");
  buf.Fill(0x00);
  ch->Send("GO_RETRY");
  std::string retry = ch->Expect("RETRY_DONE");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  verdict(buf.Landed()[0],
          "C. a live producer rebuilt its connection and a retried write landed; "
          "producer saw: " +
              retry);

  ch->Send("BYE");
  std::printf("%s\n", pass ? "RESULT: PASS" : "RESULT: FAIL");
  return pass ? 0 : 1;
}

/* ---------------------------------------- producer --------------------------------------- */

int RunProducer(const Args& a) {
  auto ch = Connect(a.peer, a.rvPort);
  auto engine = MakeEngine(kProducerKey, a);
  engine->RegisterRemoteEngine(UnpackHex<EngineDesc>(ch->Expect("ENGINE", kSetupMs)));
  MemoryDesc remote = UnpackHex<MemoryDesc>(ch->Expect("MEM", kSetupMs));
  ch->Send("ENGINE " + PackHex(engine->GetEngineDesc()));
  ch->Expect("READY", kSetupMs);

  Buffer src(engine.get(), a.gpu, kStreamByte);
  auto write = [&](int slot, TransferStatus* st) {
    engine->Write(src.desc, slot * kSlotBytes, remote, slot * kSlotBytes, kSlotBytes, st,
                  engine->AllocateTransferUniqueId());
  };

  // A.
  {
    TransferStatus st;
    write(0, &st);
    if (!WaitTerminal(&st, 30000) || !st.Succeeded()) Die("baseline write failed: " + st.Message());
    ch->Send("A_DONE");
  }

  // B. stream without waiting, as the connector issues per-layer writes.
  ch->Expect("GO_STREAM");
  // The producer keeps posting after its connection dies, like a connector that does
  // not stop issuing layers. Once the receiver refuses it a new connection, posting
  // throws; that counts as refused, and it must never turn into a landed write.
  std::vector<std::unique_ptr<TransferStatus>> sts;
  int refused = 0;
  for (int s = 1; s < kSlots; ++s) {
    sts.push_back(std::make_unique<TransferStatus>());
    try {
      write(s, sts.back().get());
    } catch (const std::exception&) {
      ++refused;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kStreamGapMs));
  }
  int ok = 0, bad = 0, hung = 0;
  std::string firstErr;
  for (auto& st : sts) {
    if (st->Init()) continue;  // never posted: the write threw
    if (!WaitTerminal(st.get(), 90000)) {
      ++hung;
      continue;
    }
    if (st->Succeeded()) {
      ++ok;
    } else {
      ++bad;
      if (firstErr.empty()) firstErr = st->Message().substr(0, 40);
    }
  }
  ch->Expect("FENCED", 5000);  // already sent long ago; keeps the two sides in step
  ch->Send("STREAM_DONE ok=" + std::to_string(ok) + " failed=" + std::to_string(bad) + " refused=" +
           std::to_string(refused) + " hung=" + std::to_string(hung) + " [" + firstErr + "]");

  // C. new writes from the same, still-running producer, retried like a connector
  // retrying the request.
  ch->Expect("GO_RETRY");
  src.Fill(kRetryByte);
  std::string outcome = "never succeeded";
  int attempts = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kRetryBudgetMs);
  while (std::chrono::steady_clock::now() < deadline) {
    ++attempts;
    auto st = std::make_unique<TransferStatus>();
    try {
      write(0, st.get());
      if (!WaitTerminal(st.get(), 30000)) {
        outcome = "no completion in 30s";
        (void)st.release();  // the NIC may still complete it
        break;
      }
      if (st->Succeeded()) {
        outcome = "succeeded";
        break;
      }
      outcome = "failed: " + st->Message().substr(0, 60);
    } catch (const std::exception& e) {
      outcome = std::string("threw: ") + e.what();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  ch->Send("RETRY_DONE " + outcome + " after " + std::to_string(attempts) + " attempt(s)");
  ch->Expect("BYE", 30000);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a = Parse(argc, argv);
  return a.role == "receiver" ? RunReceiver(a) : RunProducer(a);
}

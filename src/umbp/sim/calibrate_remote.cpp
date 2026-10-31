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

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <msgpack.hpp>

#include "calibrate_common.h"
#include "umbp/distributed/transfer/mori_io_engine.h"
#include "umbp/sim/sim.h"

namespace {
using namespace mori::umbp;
using namespace mori::umbp::calibration;
using Clock = std::chrono::steady_clock;

void Check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

struct Buffer {
  size_t size;
  char* ptr;
  Buffer(size_t bytes, bool sim) : size(bytes) {
    ptr = static_cast<char*>(mmap(nullptr, bytes, sim ? PROT_NONE : PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    Check(ptr != MAP_FAILED, "buffer mmap failed");
    if (!sim) std::memset(ptr, 0, bytes);
  }
  ~Buffer() { munmap(ptr, size); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

// Descriptor exchange and shutdown only; control traffic is never timed.
struct Socket {
  int fd = -1;
  ~Socket() {
    if (fd >= 0) close(fd);
  }
  void IO(void* ptr, size_t bytes, bool write) {
    while (bytes) {
      const auto n = write ? send(fd, ptr, bytes, MSG_NOSIGNAL) : recv(fd, ptr, bytes, 0);
      if (n < 0 && errno == EINTR) continue;
      Check(n > 0, "control socket closed or timed out");
      ptr = static_cast<char*>(ptr) + n;
      bytes -= n;
    }
  }
  void Send(std::string data) {
    Check(data.size() <= (16 << 20), "oversized control message");
    uint32_t bytes = htonl(static_cast<uint32_t>(data.size()));
    IO(&bytes, sizeof(bytes), true);
    IO(data.data(), data.size(), true);
  }
  std::string Recv() {
    uint32_t bytes;
    IO(&bytes, sizeof(bytes), false);
    bytes = ntohl(bytes);
    Check(bytes <= (16 << 20), "oversized control message");
    std::string data(bytes, '\0');
    IO(data.data(), data.size(), false);
    return data;
  }
  std::string Exchange(std::string data, bool owner) {
    if (!owner) {
      Send(std::move(data));
      return Recv();
    }
    auto other = Recv();
    Send(std::move(data));
    return other;
  }
};

void Connect(Socket& control, const std::string& ip, uint16_t port, bool owner) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  Check(inet_pton(AF_INET, ip.c_str(), &address.sin_addr) == 1, "explicit IPv4 required");
  if (owner) {
    Socket listener;
    listener.fd = socket(AF_INET, SOCK_STREAM, 0);
    Check(listener.fd >= 0, "socket failed");
    int one = 1;
    Check(setsockopt(listener.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0,
          "SO_REUSEADDR failed");
    Check(bind(listener.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0,
          "control bind failed");
    Check(listen(listener.fd, 1) == 0, "control listen failed");
    pollfd ready{listener.fd, POLLIN, 0};
    Check(poll(&ready, 1, 180000) > 0, "waiting for caller timed out");
    control.fd = accept(listener.fd, nullptr, nullptr);
    Check(control.fd >= 0, "control accept failed");
  } else {
    const auto deadline = Clock::now() + std::chrono::seconds(60);
    do {
      control.fd = socket(AF_INET, SOCK_STREAM, 0);
      Check(control.fd >= 0, "socket failed");
      if (connect(control.fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) break;
      close(control.fd);
      control.fd = -1;
      Check(Clock::now() < deadline, "control connect timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (true);
  }
  timeval timeout{180, 0};
  int one = 1;
  Check(setsockopt(control.fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0 &&
            setsockopt(control.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0 &&
            setsockopt(control.fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0,
        "control socket setup failed");
}

void Measure(bool owner, const std::string& ip, const std::string& remote_ip, uint16_t port,
             bool training, int samples, const std::string& profile, size_t concurrency,
             size_t max_bytes, size_t gap, bool staged, std::vector<size_t> blocks,
             std::vector<size_t> batches) {
  const bool sim = !profile.empty();
  if (blocks.empty())
    blocks = training ? std::vector<size_t>{4096, 65536, 1048576, 8388608}
                      : std::vector<size_t>{8192, 131072, 2097152, 16777216};
  if (batches.empty())
    batches = training ? std::vector<size_t>{1, 8, 32} : std::vector<size_t>{1, 4, 16, 128};
  Socket control;
  Connect(control, owner ? ip : remote_ip, port, owner);
  Buffer buffer(owner ? max_bytes : 2 * max_bytes, sim);
  std::unique_ptr<TransferEngine> engine;
  const std::string id = owner ? "calibration-owner" : "calibration-caller";
  if (sim) {
    engine = std::make_unique<SimTransferEngine>(profile, id, max_bytes);
  } else {
    auto real = std::make_unique<MoriIoEngine>(id, mori::io::IOEngineConfig{ip, 0}, max_bytes);
    Check(real->Init() && real->Active(), "MoriIoEngine initialization failed");
    engine = std::move(real);
  }
  auto* directory = dynamic_cast<PeerDirectory*>(engine.get());
  Check(directory != nullptr, "missing peer directory");
  const auto registered =
      engine->RegisterMemory(buffer.ptr, buffer.size, mori::io::MemoryLocationType::CPU, -1);
  Check(registered.HasMemoryDesc(), "RDMA buffer registration failed");
  const auto desc = directory->PackedLocalEngineDesc();
  Check(directory->EnsureRemoteEngine("other", control.Exchange({desc.begin(), desc.end()}, owner)),
        "RDMA engine handshake failed");
  msgpack::sbuffer packed;
  msgpack::pack(packed, registered.mem);
  const auto data = control.Exchange({packed.data(), packed.size()}, owner);
  const auto remote = TransferRef::Remote(
      msgpack::unpack(data.data(), data.size()).get().as<mori::io::MemoryDesc>());
  if (owner) {
    const timeval unlimited{};  // The driver owns the whole-probe deadline.
    Check(setsockopt(control.fd, SOL_SOCKET, SO_RCVTIMEO, &unlimited, sizeof(unlimited)) == 0,
          "cannot clear owner receive timeout");
    Check(control.Recv() == "done", "invalid shutdown message");
    engine->Deregister(registered);
    control.Send("done");
    return;
  }
  const auto local = staged ? TransferRef::HostBytes(buffer.ptr, buffer.size) : registered;
  std::cout << "block_bytes,batch,effective_threads,concurrency,api,iteration,elapsed_ns,"
               "path,staged,segments,gap\n";
  size_t cases = 0;
  auto elapsed = [](Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
  };
  for (size_t block : blocks)
    for (size_t count : batches) {
      Check(block >= sizeof(uint64_t), "blocks must contain at least eight bytes");
      // Bound the aggregate footprint, including gaps, before multiplying dimensions.
      if (block > max_bytes / concurrency / count || gap > max_bytes / concurrency / count - block)
        continue;
      const size_t stride = block + gap, span = stride * count;
      ++cases;
      std::vector<std::vector<TransferItem>> writes(concurrency), reads(concurrency);
      for (size_t c = 0; c < concurrency; ++c)
        for (size_t k = 0; k < count; ++k) {
          const size_t offset = c * span + k * stride;
          writes[c].push_back({local, offset, remote, offset, block, k});
          reads[c].push_back({remote, offset, local, max_bytes + offset, block, k});
        }
      std::array<TransferPlanSet, 2> staging_plans;
      std::array<size_t, 2> staging_copies{};
      if (training) {
        for (bool get : {false, true}) {
          auto stage_items = get ? reads[0] : writes[0];
          for (auto& item : stage_items)
            (get ? item.dst : item.src) = TransferRef::HostBytes(buffer.ptr, buffer.size);
          auto& staging = staging_plans[get];
          staging = engine->Plan(stage_items);
          Check(staging.rejected_tags.empty() && !staging.plans.empty(),
                "rejected or empty staging plan");
          for (const auto& plan : staging.plans) {
            const auto& bounce = get ? plan.dst : plan.src;
            Check(plan.uses_bounce && bounce.HasHostPtr() && !plan.bounce_copies.empty(),
                  "not a staging plan");
            staging_copies[get] += plan.bounce_copies.size();
            for (const auto& copy : plan.bounce_copies)
              Check(copy.host != nullptr && copy.bounce_offset <= bounce.size && copy.size > 0 &&
                        copy.size <= bounce.size - copy.bounce_offset,
                    "invalid staging copy bounds");
          }
        }
      }
      for (int i = -8; i < samples; ++i) {
        if (!sim)
          for (size_t c = 0; c < concurrency; ++c)
            for (size_t k = 0; k < count; ++k) {
              const size_t offset = c * span + k * stride;
              std::memset(buffer.ptr + offset, 0x5a, block);
              const uint64_t marker =
                  (uint64_t(cases) << 32) + uint64_t(i + 8) * count * concurrency + c * count + k;
              std::memcpy(buffer.ptr + offset, &marker, sizeof(marker));
              std::memset(buffer.ptr + max_bytes + offset, 0, block);
            }
        auto output = [&](int64_t ns, const char* path, bool bounce, size_t segments) {
          if (i >= 0)
            std::cout << block << ',' << count << ",1," << concurrency << ",transfer," << i << ','
                      << ns << ',' << path << ',' << bounce << ',' << segments << ',' << gap
                      << '\n';
        };
        for (bool get : {false, true}) {
          size_t segments = 0;
          const auto geometry = engine->Plan(get ? reads[0] : writes[0]);
          for (const auto& plan : geometry.plans) segments += plan.sizes.size();
          const auto times = Concurrent(concurrency, [&](size_t c) {
            auto plan = engine->Plan(get ? reads[c] : writes[c]);
            Check(plan.rejected_tags.empty(), "rejected RDMA plan");
            const auto start = Clock::now();
            auto handle = engine->Submit(std::move(plan.plans));
            Check(bool(handle), "RDMA submit failed");
            std::vector<TransferFailure> failures;
            handle->Wait(&failures);
            const auto ns = elapsed(start);
            Check(failures.empty(), "RDMA transfer failed");
            return ns;
          });
          for (auto ns : times) output(ns, get ? "remote_read" : "remote_write", staged, segments);
        }
        if (!sim)
          for (size_t c = 0; c < concurrency; ++c)
            for (size_t k = 0; k < count; ++k) {
              const size_t offset = c * span + k * stride;
              Check(std::memcmp(buffer.ptr + offset, buffer.ptr + max_bytes + offset, block) == 0,
                    "RDMA payload mismatch");
            }
        // Staging samples follow the timed wire pair, preserving its validation sequence.
        if (training)
          for (bool get : {false, true}) {
            const auto& staging = staging_plans[get].plans;
            if (get) {
              for (const auto& plan : staging)
                for (const auto& copy : plan.bounce_copies) {
                  std::memset(copy.host, 0, copy.size);
                  std::memset(static_cast<char*>(plan.dst.host_ptr) + copy.bounce_offset, 0,
                              copy.size);
                }
              // Refresh the actual bounce buffer by RDMA before timing read copyout.
              auto wire = staging;
              for (auto& plan : wire) {
                plan.uses_bounce = false;
                plan.bounce_copies.clear();
              }
              auto handle = engine->Submit(std::move(wire));
              Check(bool(handle), "staging RDMA submit failed");
              std::vector<TransferFailure> failures;
              handle->Wait(&failures);
              Check(failures.empty(), "staging RDMA transfer failed");
            }
            // Match RunBouncePlanInline's direction, buffer and serial memcpy geometry.
            const auto start = Clock::now();
            for (const auto& plan : staging)
              for (const auto& copy : plan.bounce_copies) {
                auto* bounce =
                    static_cast<char*>((get ? plan.dst : plan.src).host_ptr) + copy.bounce_offset;
                std::memcpy(get ? copy.host : bounce, get ? bounce : copy.host, copy.size);
              }
            const auto ns = elapsed(start);
            for (const auto& plan : staging)
              for (const auto& copy : plan.bounce_copies) {
                auto* bounce =
                    static_cast<char*>((get ? plan.dst : plan.src).host_ptr) + copy.bounce_offset;
                Check(std::memcmp(bounce, copy.host, copy.size) == 0, "staging payload mismatch");
              }
            if (get)
              for (size_t k = 0; k < count; ++k)
                Check(std::memcmp(buffer.ptr + k * stride, buffer.ptr + max_bytes + k * stride,
                                  block) == 0,
                      "staging RDMA payload mismatch");
            output(ns, get ? "cpu_staging_read" : "cpu_staging", true, staging_copies[get]);
          }
      }
    }
  Check(cases != 0, "no remote workload fits --max-mib");
  engine->Deregister(registered);
  control.Send("done");
  Check(control.Recv() == "done", "shutdown acknowledgement failed");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Check(
        argc >= 7 && (std::string(argv[1]) == "owner" || std::string(argv[1]) == "caller") &&
            (std::string(argv[5]) == "train" || std::string(argv[5]) == "engine"),
        "usage: umbp_sim_remote_probe owner|caller LOCAL_IPV4 OWNER_IPV4 PORT train|engine samples "
        "[profile] [--concurrency 1|2|4] [--max-mib 64] [--block-bytes B,...] [--batches N,...] "
        "[--gap-bytes 64] [--staged]");
    const auto port = Positive(argv[4]), samples = Positive(argv[6]);
    const bool training = std::string(argv[5]) == "train";
    size_t concurrency = 1, max_mib = 64, gap = 64;
    bool staged = false;
    std::string profile;
    std::vector<size_t> blocks, batches;
    for (int i = 7; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--staged")
        staged = true;
      else if ((arg == "--concurrency" || arg == "--max-mib" || arg == "--gap-bytes") &&
               i + 1 < argc) {
        const std::string value = argv[++i];
        if (arg == "--gap-bytes")
          gap = value == "0" ? 0 : Positive(value);
        else
          (arg == "--concurrency" ? concurrency : max_mib) = Positive(value);
      } else if ((arg == "--block-bytes" || arg == "--batches") && i + 1 < argc) {
        (arg == "--block-bytes" ? blocks : batches) = Sizes(argv[++i]);
      } else if (profile.empty() && arg.rfind("--", 0) != 0)
        profile = arg;
      else
        throw std::invalid_argument("unknown or incomplete option: " + arg);
    }
    Check(port <= 65535 && samples <= std::numeric_limits<int>::max() && gap <= 4096 &&
              max_mib <= std::numeric_limits<size_t>::max() / (size_t{2} << 20) &&
              (concurrency == 1 || concurrency == 2 || concurrency == 4) &&
              (!training || (profile.empty() && concurrency == 1 && !staged)),
          "invalid port, samples, memory limit or concurrency; training is "
          "real/direct/single-caller");
    Measure(std::string(argv[1]) == "owner", argv[2], argv[3], static_cast<uint16_t>(port),
            training, static_cast<int>(samples), profile, concurrency, max_mib << 20, gap, staged,
            std::move(blocks), std::move(batches));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

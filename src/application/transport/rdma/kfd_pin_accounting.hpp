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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "infiniband/verbs.h"

namespace mori {
namespace application {

// amdgpu-dkms miscounts RDMA pins of KFD VRAM in the free memory that
// hipMemGetInfo reports and, on 6.14.x and 6.16.x, new allocations are checked
// against (no public release fixes it yet, see ROCm/mori#721):
//  - a dma-buf pin (ibv_reg_dmabuf_mr) subtracts the whole BO a second time;
//  - every peer-memory pin (ibv_reg_mr) adds the whole BO back, so N
//    peer-memory MRs on one BO over-report it N-1 times.
// The 31.x releases check new allocations against a count that never credits
// pins back, which no choice of MR type can offset.
// The count is exact while each BO that has MRs holds exactly one peer-memory
// pin: the BO's first peer-memory MR or a page-sized peer-memory anchor MR.
// Further MRs on the BO are registered through dma-buf, and before the pin
// holder goes away while other MRs remain, an anchor takes over.
//
// Only VRAM allocations of this process of at least 64 MiB are tracked. It is
// off unless MORI_KFD_PIN_FIX=1, which probes the driver on first use and acts
// only if the driver miscounts; MORI_KFD_PIN_FIX=force skips the probe. What it
// cannot fix (no peer memory, a failed anchor) is logged as an error, since mori
// shows only errors by default. Building with ENABLE_KFD_PIN_FIX=OFF leaves only
// the pass-through below.
#if defined(MORI_KFD_PIN_FIX)
class KfdPinAccounting {
 public:
  static KfdPinAccounting& Instance();

  // Brackets the registration of one data MR. While the buffer is tracked it
  // holds the accounting lock, so registrations on one BO see each other's pins.
  class Registration {
   public:
    Registration(Registration&& other) noexcept;
    Registration& operator=(Registration&&) = delete;
    ~Registration();

    // The BO already holds its pin, so another peer-memory MR would add it again.
    bool PreferDmabuf() const;
    // Call before each dma-buf attempt: pins the BO through an anchor if nothing
    // pins it yet.
    void BeforeDmabuf();
    void Commit(ibv_mr* mr, bool viaDmabuf);

   private:
    friend class KfdPinAccounting;
    Registration() = default;

    KfdPinAccounting* owner{nullptr};
    std::unique_lock<std::mutex> lock;
    uintptr_t bo{0};
    ibv_pd* pd{nullptr};
    bool committed{false};
  };

  Registration Begin(ibv_pd* pd, void* ptr, size_t size);

  // Deregisters `mr`. If it holds its BO's pin and other MRs remain on the BO,
  // an anchor takes the pin over first.
  int Deregister(ibv_mr* mr);

  // Moves anchors registered on `pd` to the PD of another MR on the same BO.
  // Call after the MRs of `pd` are deregistered and before `pd` is deallocated.
  void ReleasePd(ibv_pd* pd);

 private:
  enum class Mode { Unprobed, Off, Unknown, Exact, PerPinCredit, Unfixable };

  struct Member {
    ibv_mr* mr;
    ibv_pd* pd;
    bool peer;
  };
  struct Bo {
    size_t size{0};
    std::vector<Member> members;
    ibv_mr* anchor{nullptr};
    ibv_pd* anchorPd{nullptr};
  };

  KfdPinAccounting();
  Mode Probe(ibv_pd* pd, int device);
  static bool HasPin(const Bo& bo);
  static ibv_pd* PdForAnchor(const Bo& bo, ibv_pd* avoid);
  ibv_mr* RegisterAnchor(ibv_pd* pd, uintptr_t base);

  std::atomic<Mode> mode{Mode::Off};
  std::mutex mu;
  std::unordered_map<uintptr_t, Bo> bos;
  std::unordered_map<ibv_mr*, uintptr_t> boOfMr;
  bool anchorWarned{false};
  bool doublePinWarned{false};
};
#else
class KfdPinAccounting {
 public:
  class Registration {
   public:
    bool PreferDmabuf() const { return false; }
    void BeforeDmabuf() {}
    void Commit(ibv_mr*, bool) {}
  };

  static KfdPinAccounting& Instance() {
    static KfdPinAccounting instance;
    return instance;
  }
  Registration Begin(ibv_pd*, void*, size_t) { return {}; }
  int Deregister(ibv_mr* mr) { return ibv_dereg_mr(mr); }
  void ReleasePd(ibv_pd*) {}
};
#endif

}  // namespace application
}  // namespace mori

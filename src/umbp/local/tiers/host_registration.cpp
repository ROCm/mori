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
#include "umbp/common/host_registration.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {

// Below this, registering inline costs under ~100 ms and keeps behaviour
// deterministic, which unit tests rely on.
constexpr size_t kSyncThresholdBytes = 2ull << 30;

bool RegistrationEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("UMBP_DRAM_HOST_REGISTER");
    if (value == nullptr) return true;
    const std::string text(value);
    return text != "0" && text != "off" && text != "false";
  }();
  return enabled;
}

bool HasDevice() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return count > 0;
}

}  // namespace

HostTierRegistration::HostTierRegistration(void* base, size_t bytes)
    : base_(static_cast<char*>(base)), bytes_(bytes) {
  if (base_ == nullptr || bytes_ == 0 || !RegistrationEnabled() || !HasDevice()) {
    bytes_ = 0;
    return;
  }

  if (bytes_ <= kSyncThresholdBytes) {
    Register();
    return;
  }
  worker_ = std::thread([this]() { Register(); });
}

HostTierRegistration::~HostTierRegistration() {
  if (worker_.joinable()) worker_.join();
  if (registered_bytes_.load(std::memory_order_acquire) == 0) return;
  if (hipHostUnregister(base_) != hipSuccess) (void)hipGetLastError();
}

void HostTierRegistration::Register() {
  const auto started = std::chrono::steady_clock::now();

  // One call for the whole region, never several. hipMemcpy rejects any copy
  // whose host range spans two separate registrations with hipErrorInvalidValue,
  // so registering in chunks would break every object that happens to straddle
  // a chunk boundary -- which is most of them once the tier is large.
  //
  // hipHostRegisterPortable so every rank's device can reach the tier;
  // hipHostRegisterMapped so a kernel can dereference it directly.
  const hipError_t status =
      hipHostRegister(base_, bytes_, hipHostRegisterMapped | hipHostRegisterPortable);
  if (status != hipSuccess) {
    MORI_UMBP_WARN(
        "[DRAMTier] hipHostRegister of {} MiB failed: {}; the GPU gather path stays off and "
        "copies fall back to pageable hipMemcpy",
        bytes_ >> 20, hipGetErrorString(status));
    (void)hipGetLastError();
    return;
  }

  // The gather path hands host addresses straight to the kernel, which is only
  // valid under unified addressing. Verify rather than assume -- and verify it
  // on every device, not just whichever one this thread happens to have
  // current: the standalone server serves all ranks from one process, so a
  // batch can launch the kernel on any of them.
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices <= 0) {
    // Not being able to enumerate the devices is not the same as having
    // verified them: an empty loop below would silently publish the region as
    // kernel-addressable on every device.
    MORI_UMBP_WARN(
        "[DRAMTier] cannot enumerate devices to verify the host mapping; the GPU gather path "
        "stays off");
    (void)hipGetLastError();
    (void)hipHostUnregister(base_);
    return;
  }
  int previous = 0;
  const bool have_previous = hipGetDevice(&previous) == hipSuccess;
  if (!have_previous) (void)hipGetLastError();
  for (int device = 0; device < devices; ++device) {
    void* device_alias = nullptr;
    const bool selected = hipSetDevice(device) == hipSuccess;
    const bool matches = selected &&
                         hipHostGetDevicePointer(&device_alias, base_, 0) == hipSuccess &&
                         device_alias == base_;
    if (matches) continue;

    MORI_UMBP_WARN(
        "[DRAMTier] device {} maps the registered host region at a different address than the "
        "host does; the GPU gather path needs unified addressing and stays off",
        device);
    (void)hipGetLastError();
    if (have_previous) (void)hipSetDevice(previous);
    (void)hipHostUnregister(base_);
    return;
  }
  if (have_previous) (void)hipSetDevice(previous);

  registered_bytes_.store(bytes_, std::memory_order_release);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  MORI_UMBP_INFO("[DRAMTier] host memory registered for GPU access: {} MiB in {:.1f} s",
                 bytes_ >> 20, seconds);
}

bool HostTierRegistration::Covers(const void* ptr, size_t size) const {
  if (ptr == nullptr || bytes_ == 0) return false;
  // Compared as integers, not pointers: `ptr` may be outside this region, and
  // relational operators and subtraction on pointers into different objects are
  // undefined behaviour even when the hardware would do the obvious thing.
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t base = reinterpret_cast<uintptr_t>(base_);
  if (address < base) return false;
  const uintptr_t offset = address - base;
  const size_t registered = registered_bytes_.load(std::memory_order_acquire);
  return size <= registered && offset <= registered - size;
}

}  // namespace mori::umbp

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

  // Deliberately does not undo the registration on failure: pinning already
  // succeeded and takes every hipMemcpy on this tier off the pageable staging
  // path, so a device without an alias loses only the gather kernel.
  if (!RecordDeviceAliases()) {
    // Leaving alias_bases_ empty disables gather everywhere, which is the safe
    // answer -- an unchecked device must not be published as gatherable.
    MORI_UMBP_WARN(
        "[DRAMTier] cannot enumerate devices to map the host region; the GPU gather path stays "
        "off (the region stays registered, so copies keep the pinned path)");
    (void)hipGetLastError();
  }

  registered_bytes_.store(bytes_, std::memory_order_release);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  size_t gatherable = 0;
  for (const char* alias : alias_bases_) {
    if (alias != nullptr) ++gatherable;
  }
  MORI_UMBP_INFO(
      "[DRAMTier] host memory registered for GPU access: {} MiB in {:.1f} s ({}/{} devices "
      "gatherable)",
      bytes_ >> 20, seconds, gatherable, alias_bases_.size());
}

bool HostTierRegistration::RecordDeviceAliases() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices <= 0) return false;

  int previous = 0;
  const bool have_previous = hipGetDevice(&previous) == hipSuccess;
  if (!have_previous) (void)hipGetLastError();

  alias_bases_.assign(static_cast<size_t>(devices), nullptr);
  for (int device = 0; device < devices; ++device) {
    if (hipSetDevice(device) != hipSuccess) {
      (void)hipGetLastError();
      continue;
    }
    void* alias = nullptr;
    if (hipHostGetDevicePointer(&alias, base_, 0) != hipSuccess || alias == nullptr) {
      (void)hipGetLastError();
      MORI_UMBP_WARN(
          "[DRAMTier] device {} has no mapping for the registered host region; the "
          "GPU gather path stays off for it",
          device);
      continue;
    }

    // Only the base is queried: hipHostGetDevicePointer is documented in terms
    // of the pointer an allocation starts at, so an interior offset is outside
    // the contract and a rejection there would disable gather needlessly.
    alias_bases_[static_cast<size_t>(device)] = static_cast<char*>(alias);
  }

  if (have_previous) (void)hipSetDevice(previous);
  return true;
}

bool HostTierRegistration::GatherableOn(int device_id) const {
  if (registered_bytes_.load(std::memory_order_acquire) == 0) return false;
  if (device_id < 0 || static_cast<size_t>(device_id) >= alias_bases_.size()) return false;
  return alias_bases_[static_cast<size_t>(device_id)] != nullptr;
}

void* HostTierRegistration::DeviceAddress(const void* host_ptr, size_t size, int device_id) const {
  const size_t registered = registered_bytes_.load(std::memory_order_acquire);
  if (registered == 0) return nullptr;
  if (device_id < 0 || static_cast<size_t>(device_id) >= alias_bases_.size()) return nullptr;
  char* const alias = alias_bases_[static_cast<size_t>(device_id)];
  if (alias == nullptr || !CoversRegistered(host_ptr, size, registered)) return nullptr;
  const uintptr_t address = reinterpret_cast<uintptr_t>(host_ptr);
  const uintptr_t base = reinterpret_cast<uintptr_t>(base_);
  return alias + (address - base);
}

bool HostTierRegistration::Covers(const void* ptr, size_t size) const {
  return CoversRegistered(ptr, size, registered_bytes_.load(std::memory_order_acquire));
}

bool HostTierRegistration::CoversRegistered(const void* ptr, size_t size, size_t registered) const {
  if (ptr == nullptr || bytes_ == 0) return false;
  // Compared as integers, not pointers: `ptr` may be outside this region, and
  // relational operators and subtraction on pointers into different objects are
  // undefined behaviour even when the hardware would do the obvious thing.
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t base = reinterpret_cast<uintptr_t>(base_);
  if (address < base) return false;
  const uintptr_t offset = address - base;
  return size <= registered && offset <= registered - size;
}

}  // namespace mori::umbp

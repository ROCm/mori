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
#include <cstdint>
#include <cstdlib>
#include <string>

#include "mori/utils/mori_log.hpp"

namespace mori::umbp {
namespace {

// Below this, registering inline costs under ~100 ms and keeps behaviour
// deterministic, which unit tests rely on.
constexpr size_t kSyncThresholdBytes = 2ull << 30;

bool HasDevice() {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return count > 0;
}

}  // namespace

bool HostRegistrationEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("UMBP_DRAM_HOST_REGISTER");
    if (value == nullptr) return true;
    const std::string text(value);
    return text != "0" && text != "off" && text != "false";
  }();
  return enabled;
}

size_t HostRegistrationChunkBytes() {
  static const size_t bytes = [] {
    unsigned long long mib = 1024;
    if (const char* value = std::getenv("UMBP_DRAM_HOST_REGISTER_CHUNK_MB")) {
      char* end = nullptr;
      const unsigned long long parsed = std::strtoull(value, &end, 10);
      if (end != value && *end == '\0') mib = parsed;
    }
    // Zero asks for whole-region registrations: a boundary no address reaches.
    if (mib == 0 || mib > (1ULL << 40)) return size_t{1} << 62;
    // A power of two of at least a huge page, so pieces stay huge-page aligned.
    size_t chunk = size_t{2} << 20;
    while (chunk * 2 <= (mib << 20)) chunk *= 2;
    return chunk;
  }();
  return bytes;
}

HostTierRegistration::HostTierRegistration(void* base, size_t bytes)
    : base_(static_cast<char*>(base)), bytes_(bytes), piece_bytes_(HostRegistrationChunkBytes()) {
  if (base_ == nullptr || bytes_ == 0 || !HostRegistrationEnabled() || !HasDevice()) {
    bytes_ = 0;
    return;
  }

  // Cut at absolute multiples of the piece size, so the first and last pieces
  // may be short.
  const uintptr_t end = reinterpret_cast<uintptr_t>(base_) + bytes_;
  for (uintptr_t start = reinterpret_cast<uintptr_t>(base_); start < end;) {
    const uintptr_t next = std::min<uintptr_t>(end, (start / piece_bytes_ + 1) * piece_bytes_);
    pieces_.push_back(Piece{reinterpret_cast<char*>(start), next - start, {}});
    start = next;
  }

  if (bytes_ <= kSyncThresholdBytes) {
    Register();
    return;
  }
  worker_ = std::thread([this]() { Register(); });
}

HostTierRegistration::~HostTierRegistration() {
  if (worker_.joinable()) worker_.join();
  const size_t registered = registered_pieces_.load(std::memory_order_acquire);
  for (size_t i = 0; i < registered; ++i) {
    if (hipHostUnregister(pieces_[i].base) != hipSuccess) (void)hipGetLastError();
  }
}

void HostTierRegistration::Register() {
  const auto started = std::chrono::steady_clock::now();
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices <= 0) {
    (void)hipGetLastError();
    devices = 0;
  }
  int previous = 0;
  const bool have_previous = hipGetDevice(&previous) == hipSuccess;
  if (!have_previous) (void)hipGetLastError();

  size_t gatherable = 0;
  for (size_t i = 0; i < pieces_.size(); ++i) {
    Piece& piece = pieces_[i];
    // hipHostRegisterPortable so every rank's device can reach the tier;
    // hipHostRegisterMapped so a kernel can dereference it directly.
    const hipError_t status =
        hipHostRegister(piece.base, piece.bytes, hipHostRegisterMapped | hipHostRegisterPortable);
    if (status != hipSuccess) {
      // What is registered stays usable; the rest keeps the pageable path.
      MORI_UMBP_WARN(
          "[DRAMTier] hipHostRegister of piece {}/{} ({} MiB) failed: {}; {} MiB of the region "
          "stays on pageable hipMemcpy and off the GPU gather path",
          i + 1, pieces_.size(), piece.bytes >> 20, hipGetErrorString(status),
          (bytes_ - registered_bytes_.load(std::memory_order_relaxed)) >> 20);
      (void)hipGetLastError();
      break;
    }

    // A device without an alias loses only the gather kernel: the piece stays
    // registered, so copies keep the pinned path.
    piece.aliases.assign(static_cast<size_t>(devices), nullptr);
    for (int device = 0; device < devices; ++device) {
      if (hipSetDevice(device) != hipSuccess) {
        (void)hipGetLastError();
        continue;
      }
      // Only a piece's base is queried: hipHostGetDevicePointer is documented
      // in terms of the pointer an allocation starts at.
      void* alias = nullptr;
      if (hipHostGetDevicePointer(&alias, piece.base, 0) != hipSuccess || alias == nullptr) {
        (void)hipGetLastError();
        if (i == 0) {
          MORI_UMBP_WARN(
              "[DRAMTier] device {} has no mapping for the registered host region; the "
              "GPU gather path stays off for it",
              device);
        }
        continue;
      }
      piece.aliases[static_cast<size_t>(device)] = static_cast<char*>(alias);
    }
    if (i == 0) {
      for (const char* alias : piece.aliases) gatherable += alias != nullptr;
    }

    registered_bytes_.fetch_add(piece.bytes, std::memory_order_relaxed);
    registered_pieces_.store(i + 1, std::memory_order_release);
  }

  if (have_previous) (void)hipSetDevice(previous);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  MORI_UMBP_INFO(
      "[DRAMTier] host memory registered for GPU access: {} MiB in {:.1f} s as {} pieces of up "
      "to {} MiB ({}/{} devices gatherable)",
      registered_bytes_.load(std::memory_order_relaxed) >> 20, seconds,
      registered_pieces_.load(std::memory_order_relaxed),
      std::min<size_t>(piece_bytes_, bytes_) >> 20, gatherable, devices);
}

const HostTierRegistration::Piece* HostTierRegistration::FindPiece(const void* ptr,
                                                                   size_t size) const {
  if (ptr == nullptr || bytes_ == 0) return nullptr;
  // Compared as integers, not pointers: `ptr` may be outside this region, and
  // relational operators and subtraction on pointers into different objects are
  // undefined behaviour even when the hardware would do the obvious thing.
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t base = reinterpret_cast<uintptr_t>(base_);
  if (address < base || address - base >= bytes_) return nullptr;
  const size_t index = address / piece_bytes_ - base / piece_bytes_;
  if (index >= registered_pieces_.load(std::memory_order_acquire)) return nullptr;
  const Piece& piece = pieces_[index];
  const uintptr_t offset = address - reinterpret_cast<uintptr_t>(piece.base);
  if (size > piece.bytes - offset) return nullptr;
  return &piece;
}

bool HostTierRegistration::GatherableOn(int device_id) const {
  if (registered_pieces_.load(std::memory_order_acquire) == 0) return false;
  const auto& aliases = pieces_.front().aliases;
  if (device_id < 0 || static_cast<size_t>(device_id) >= aliases.size()) return false;
  return aliases[static_cast<size_t>(device_id)] != nullptr;
}

void* HostTierRegistration::DeviceAddress(const void* host_ptr, size_t size, int device_id) const {
  const Piece* piece = FindPiece(host_ptr, size);
  if (piece == nullptr || device_id < 0 ||
      static_cast<size_t>(device_id) >= piece->aliases.size()) {
    return nullptr;
  }
  char* const alias = piece->aliases[static_cast<size_t>(device_id)];
  if (alias == nullptr) return nullptr;
  return alias + (reinterpret_cast<uintptr_t>(host_ptr) - reinterpret_cast<uintptr_t>(piece->base));
}

}  // namespace mori::umbp

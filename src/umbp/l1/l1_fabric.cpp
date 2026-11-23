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
#include "umbp/l1/l1_fabric.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "mori/utils/hip_compat.hpp"
#include "src/io/fabric/copy_plan.hpp"
#include "src/io/fabric/vpod_topology.hpp"

namespace mori::umbp::l1 {
namespace {
namespace fabric = mori::io::fabric;
constexpr char kMagic[] = "L1FBR001";
constexpr size_t kMagicBytes = sizeof(kMagic) - 1;
constexpr size_t kHandleBytes = 64;

std::string HipError(const char* operation, hipError_t error) {
  return std::string(operation) + ": " + hipGetErrorString(error);
}

class DeviceScope {
 public:
  explicit DeviceScope(int device) {
    if (hipGetDevice(&previous_) != hipSuccess) previous_ = -1;
    ok_ = hipSetDevice(device) == hipSuccess;
  }
  ~DeviceScope() {
    if (previous_ >= 0) (void)hipSetDevice(previous_);
  }
  bool ok() const { return ok_; }

 private:
  int previous_ = -1;
  bool ok_ = false;
};

// Stable little-endian descriptor; do not serialize native C++ struct padding.
struct Descriptor {
  std::array<unsigned char, kHandleBytes> token{};
  uint64_t allocation_bytes = 0;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  fabric::VpodKey domain;
};

void Append64(std::string* out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out->push_back(static_cast<char>(value >> (i * 8)));
}

bool Read64(const std::string& in, size_t* pos, uint64_t* value) {
  if (*pos > in.size() || in.size() - *pos < 8) return false;
  *value = 0;
  for (int i = 0; i < 8; ++i) {
    *value |= uint64_t{static_cast<unsigned char>(in[(*pos)++])} << (i * 8);
  }
  return true;
}

std::string Encode(const Descriptor& d) {
  std::string out(kMagic, kMagicBytes);
  out.append(reinterpret_cast<const char*>(d.token.data()), d.token.size());
  Append64(&out, d.allocation_bytes);
  Append64(&out, d.offset);
  Append64(&out, d.bytes);
  Append64(&out, static_cast<uint64_t>(d.domain.vpodId));
  Append64(&out, d.domain.ppodId.size());
  out.append(d.domain.ppodId);
  return out;
}

bool Decode(const std::string& handle, Descriptor* d, std::string* error) {
  auto fail = [&] {
    *error = "invalid native L1 FABRIC descriptor";
    return false;
  };
  if (handle.size() < kMagicBytes + kHandleBytes ||
      handle.compare(0, kMagicBytes, kMagic, kMagicBytes) != 0)
    return fail();
  std::memcpy(d->token.data(), handle.data() + kMagicBytes, kHandleBytes);
  size_t pos = kMagicBytes + kHandleBytes;
  uint64_t vpod = 0, ppod_bytes = 0;
  if (!Read64(handle, &pos, &d->allocation_bytes) || !Read64(handle, &pos, &d->offset) ||
      !Read64(handle, &pos, &d->bytes) || !Read64(handle, &pos, &vpod) ||
      !Read64(handle, &pos, &ppod_bytes))
    return fail();
  if (d->allocation_bytes == 0 || d->allocation_bytes > std::numeric_limits<size_t>::max() ||
      d->offset > d->allocation_bytes || d->bytes == 0 ||
      d->bytes > d->allocation_bytes - d->offset || vpod > INT32_MAX || ppod_bytes > 128 ||
      ppod_bytes != handle.size() - pos)
    return fail();
  if (std::all_of(d->token.begin(), d->token.end(), [](unsigned char b) { return b == 0; })) {
    return fail();
  }
  d->domain.valid = true;
  d->domain.vpodId = static_cast<int>(vpod);
  d->domain.ppodId = handle.substr(pos);
  return true;
}

struct View {
  hipMemGenericAllocationHandle_t handle{};
  void* base = nullptr;
  uint64_t bytes = 0;
  int device = -1;
};

void Unmap(View* view) {
  if (!view->base) return;
  DeviceScope scope(view->device);
  (void)hipMemUnmap(view->base, view->bytes);
  (void)hipMemAddressFree(view->base, view->bytes);
  (void)hipMemRelease(view->handle);
  view->base = nullptr;
}

bool Map(const Descriptor& d, int device, View* view, std::string* error) {
  DeviceScope scope(device);
  if (!scope.ok()) {
    *error = "could not select FABRIC destination GPU";
    return false;
  }
  if (!fabric::DeviceSupportsFabric(device) ||
      !fabric::VpodKeySame(fabric::ReadVpodKey(device), d.domain)) {
    *error = "FABRIC memory is outside the destination GPU's active vPOD";
    return false;
  }
  hipMemFabricHandle_compat_t token;
  static_assert(sizeof(token) == kHandleBytes);
  std::memcpy(&token, d.token.data(), sizeof(token));
  hipMemGenericAllocationHandle_t allocation{};
  hipError_t err =
      hipMemImportFromShareableHandle(&allocation, &token, hipMemHandleTypeFabricCompat);
  if (err != hipSuccess) {
    *error = HipError("hipMemImportFromShareableHandle", err);
    return false;
  }
  void* base = nullptr;
  err = hipMemAddressReserve(&base, d.allocation_bytes, 0, nullptr, 0);
  if (err != hipSuccess) {
    *error = HipError("hipMemAddressReserve", err);
    (void)hipMemRelease(allocation);
    return false;
  }
  err = hipMemMap(base, d.allocation_bytes, 0, allocation, 0);
  if (err != hipSuccess) {
    *error = HipError("hipMemMap", err);
    (void)hipMemAddressFree(base, d.allocation_bytes);
    (void)hipMemRelease(allocation);
    return false;
  }
  hipMemAccessDesc access{};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device;
  access.flags = hipMemAccessFlagsProtReadWrite;
  err = hipMemSetAccess(base, d.allocation_bytes, &access, 1);
  if (err != hipSuccess) {
    *error = HipError("hipMemSetAccess", err);
    (void)hipMemUnmap(base, d.allocation_bytes);
    (void)hipMemAddressFree(base, d.allocation_bytes);
    (void)hipMemRelease(allocation);
    return false;
  }
  *view = View{allocation, base, d.allocation_bytes, device};
  return true;
}

struct Mapping {
  Descriptor descriptor;
  uint64_t offset = 0;
  int device = -1;
  void* data = nullptr;
  std::map<int, View> views;
};

struct CopyDevice {
  hipStream_t stream = nullptr;
  hipModule_t module = nullptr;
  hipFunction_t single = nullptr;
  hipFunction_t small = nullptr;
  hipFunction_t large = nullptr;
};

// A single mutex deliberately serializes import/copy/unimport in this first
// implementation. In particular, an error cannot race mapping destruction.
class FabricMemoryApi final : public MemoryApi {
 public:
  explicit FabricMemoryApi(std::string kernel_path) : kernel_path_(std::move(kernel_path)) {
    if (kernel_path_.empty()) throw std::invalid_argument("native L1 FABRIC needs a kernel path");
  }
  ~FabricMemoryApi() override {
    // Unknown completion is not a free notification. Leave the HIP mappings,
    // modules and streams intact until process teardown; pool pins are retained
    // separately by PoolManager. A new API object must not recycle these VAs.
    if (unknown_) return;
    for (auto& [token, mapping] : mappings_) {
      for (auto& [device, view] : mapping->views) Unmap(&view);
    }
    for (auto& [device, copy] : devices_) {
      DeviceScope scope(device);
      if (copy.stream) (void)hipStreamDestroy(copy.stream);
      if (copy.module) (void)hipModuleUnload(copy.module);
    }
  }

  bool SupportsFabric() const override { return true; }

  bool Allocate(int, uint64_t, void**, ExportedMemory*, std::string* error) override {
    *error = "native FABRIC L1 requires externally allocated KV HBM";
    return false;
  }
  void Free(int, void*, uint64_t) override {}  // this API never owns KV payloads

  bool Import(int device, const std::string& handle, uint64_t offset, void** base, void** mapped,
              std::string* error) override {
    Descriptor descriptor;
    if (!Decode(handle, &descriptor, error)) return false;
    if (offset >= descriptor.bytes) {
      *error = "FABRIC external offset is outside the exported view";
      return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (unknown_) {
      *error = "FABRIC copy completion unknown; restart the affected workers";
      return false;
    }
    auto mapping = std::make_unique<Mapping>();
    View view;
    if (!Map(descriptor, device, &view, error)) return false;
    mapping->descriptor = std::move(descriptor);
    mapping->offset = offset;
    mapping->device = device;
    mapping->data = static_cast<char*>(view.base) + mapping->descriptor.offset + offset;
    mapping->views.emplace(device, view);
    *base = mapping->data;
    *mapped = mapping.get();
    mappings_.emplace(mapping.get(), std::move(mapping));
    return true;
  }

  void Unimport(int, void* mapped) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (unknown_) return;
    auto it = mappings_.find(mapped);
    if (it == mappings_.end()) return;
    for (auto& [device, view] : it->second->views) Unmap(&view);
    mappings_.erase(it);
  }

  bool Copy(int dst_device, void* dst, int src_device, const void* src, uint64_t bytes,
            std::string* error) override {
    return CopyBatch(dst_device, {{dst, src, bytes, src_device}}, error);
  }
  bool CopyBatch(int dst_device, const std::vector<Fragment>& fragments,
                 std::string* error) override {
    const auto result = CopyBatchTracked(dst_device, fragments);
    *error = result.error;
    return result.ok;
  }

  CopyOutcome CopyBatchTracked(int dst_device, const std::vector<Fragment>& fragments) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (unknown_) return {false, false, "previous FABRIC copy completion unknown"};
    if (fragments.empty()) return {true, true, {}};
    DeviceScope scope(dst_device);
    if (!scope.ok()) return Failed("could not select FABRIC destination GPU");
    std::string error;
    CopyDevice* copy = GetCopyDevice(dst_device, &error);
    if (!copy) return Failed(std::move(error));

    std::vector<fabric::CopySegment> segments;
    segments.reserve(fragments.size());
    // Compare allocation-relative ranges as well as VAs: a caller can attach
    // two views of one allocation, and different mappings can alias payloads.
    using Ranges = std::vector<std::pair<uint64_t, uint64_t>>;
    std::map<std::array<unsigned char, kHandleBytes>, Ranges> destinations, sources;
    for (const auto& f : fragments) {
      if (!f.bytes) continue;
      Mapping* dst_mapping = Find(f.dst, f.bytes);
      Mapping* src_mapping = Find(f.src, f.bytes);
      if (!dst_mapping || dst_mapping->device != dst_device || !src_mapping) {
        return Failed("copy fragment is outside registered native FABRIC memory");
      }
      auto view = src_mapping->views.find(dst_device);
      if (view == src_mapping->views.end()) {
        View alias;
        if (!Map(src_mapping->descriptor, dst_device, &alias, &error)) return Failed(error);
        view = src_mapping->views.emplace(dst_device, alias).first;
      }
      const auto delta =
          reinterpret_cast<uintptr_t>(f.src) - reinterpret_cast<uintptr_t>(src_mapping->data);
      const auto source = reinterpret_cast<uintptr_t>(view->second.base) +
                          src_mapping->descriptor.offset + src_mapping->offset + delta;
      segments.push_back({source, reinterpret_cast<uintptr_t>(f.dst), f.bytes});
      const auto src_start = src_mapping->descriptor.offset + src_mapping->offset + delta;
      const auto dst_start = dst_mapping->descriptor.offset + dst_mapping->offset +
                             reinterpret_cast<uintptr_t>(f.dst) -
                             reinterpret_cast<uintptr_t>(dst_mapping->data);
      sources[src_mapping->descriptor.token].emplace_back(src_start, src_start + f.bytes);
      destinations[dst_mapping->descriptor.token].emplace_back(dst_start, dst_start + f.bytes);
    }
    // Distinct slots/regions must not produce overlapping destination writes.
    std::sort(segments.begin(), segments.end(),
              [](const auto& a, const auto& b) { return a.dst < b.dst; });
    for (size_t i = 1; i < segments.size(); ++i) {
      if (segments[i].dst - segments[i - 1].dst < segments[i - 1].size) {
        return Failed("native FABRIC copy destinations overlap");
      }
    }
    for (auto& [token, ranges] : destinations) {
      std::sort(ranges.begin(), ranges.end());
      for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
          return Failed("native FABRIC destination views alias");
        }
      }
      auto src = sources.find(token);
      if (src == sources.end()) continue;
      for (const auto& range : src->second) {
        auto next = std::lower_bound(ranges.begin(), ranges.end(), range.second,
                                     [](const auto& r, uint64_t end) { return r.first < end; });
        if (next != ranges.begin() && std::prev(next)->second > range.first) {
          return Failed("native FABRIC source and destination views overlap");
        }
      }
    }
    // These are host-side address/length metadata, passed as kernel arguments.
    // No allocation here holds any KV payload; the only payload addresses are
    // the imported source allocation and the caller's final destination slots.
    // Match the MORI-IO multi-segment copy budget; 32 blocks underfill 455 GPUs.
    auto plan = fabric::PlanCopy(std::move(segments), 1536, true, true);
    if (!plan.error.empty()) return Failed(plan.error);
    bool submitted = false;
    for (auto& dispatch : plan.dispatches) {
      const hipError_t err = std::visit(
          [&](auto& args) {
            using Args = std::decay_t<decltype(args)>;
            hipFunction_t function = copy->large;
            if constexpr (std::is_same_v<Args, fabric::FabricCopyArgs<fabric::kSingleK>>) {
              function = copy->single;
            } else if constexpr (std::is_same_v<Args, fabric::FabricCopyArgs<fabric::kSmallK>>) {
              function = copy->small;
            }
            void* params[] = {&args};
            return hipModuleLaunchKernel(function, dispatch.grid, 1, 1, 256, 1, 1, 0, copy->stream,
                                         params, nullptr);
          },
          dispatch.args);
      if (err != hipSuccess) {
        error = HipError("FABRIC copy launch", err);
        // Earlier successful launches can still read/write slots. Drain even
        // though this launch failed, before the caller releases either lease.
        if (submitted) {
          const auto drain = hipStreamSynchronize(copy->stream);
          if (drain != hipSuccess) return Unknown(error + "; " + HipError("copy drain", drain));
        }
        return Failed(error);
      }
      submitted = true;
    }
    if (submitted) {
      const auto err = hipStreamSynchronize(copy->stream);
      if (err != hipSuccess) return Unknown(HipError("FABRIC copy synchronize", err));
    }
    return {true, true, {}};
  }

  bool Synchronize(int, std::string* error) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (unknown_) {
      *error = "FABRIC completion is unknown; slots and mappings remain quarantined";
      return false;
    }
    return true;  // Copy and CopyBatch wait for their private stream before returning
  }

 private:
  static CopyOutcome Failed(std::string error) { return {false, true, std::move(error)}; }
  CopyOutcome Unknown(std::string error) {
    unknown_ = true;
    return {false, false, std::move(error)};
  }
  Mapping* Find(const void* ptr, uint64_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    for (const auto& [token, mapping] : mappings_) {
      const auto begin = reinterpret_cast<uintptr_t>(mapping->data);
      const uint64_t length = mapping->descriptor.bytes - mapping->offset;
      if (address >= begin && address - begin <= length && bytes <= length - (address - begin)) {
        return mapping.get();
      }
    }
    return nullptr;
  }
  CopyDevice* GetCopyDevice(int device, std::string* error) {
    auto found = devices_.find(device);
    if (found != devices_.end()) return &found->second;
    CopyDevice copy;
    hipError_t err = hipModuleLoad(&copy.module, kernel_path_.c_str());
    if (err == hipSuccess)
      err = hipModuleGetFunction(&copy.single, copy.module, "fabricCopyKernelSingle");
    if (err == hipSuccess)
      err = hipModuleGetFunction(&copy.small, copy.module, "fabricCopyKernelSmall");
    if (err == hipSuccess)
      err = hipModuleGetFunction(&copy.large, copy.module, "fabricCopyKernelLarge");
    if (err == hipSuccess) err = hipStreamCreateWithFlags(&copy.stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
      *error = HipError("load native FABRIC copy kernel/stream", err);
      if (copy.module) (void)hipModuleUnload(copy.module);
      return nullptr;
    }
    return &devices_.emplace(device, copy).first->second;
  }

  const std::string kernel_path_;
  std::mutex mutex_;
  bool unknown_ = false;
  std::map<void*, std::unique_ptr<Mapping>> mappings_;
  std::map<int, CopyDevice> devices_;
};
}  // namespace

bool IsFabricHandle(const std::string& handle) {
  return handle.size() >= kMagicBytes && handle.compare(0, kMagicBytes, kMagic, kMagicBytes) == 0;
}

bool ValidateFabricRange(const std::string& handle, uint64_t offset, uint64_t bytes,
                         std::string* error) {
  Descriptor descriptor;
  if (!Decode(handle, &descriptor, error)) return false;
  if (bytes == 0 || offset > descriptor.bytes || bytes > descriptor.bytes - offset) {
    *error = "native L1 pool range exceeds the exported FABRIC view";
    return false;
  }
  return true;
}

bool ExportFabricMemory(int device, void* ptr, uint64_t bytes, ExportedMemory* exported,
                        std::string* error) {
  if (!ptr || !bytes) {
    *error = "FABRIC export needs nonempty GPU memory";
    return false;
  }
  DeviceScope scope(device);
  if (!scope.ok()) {
    *error = "could not select FABRIC owner GPU";
    return false;
  }
  Descriptor descriptor;
  descriptor.domain = fabric::ReadVpodKey(device);
  if (!fabric::DeviceSupportsFabric(device) || !descriptor.domain.valid) {
    *error = "owner GPU has no active FABRIC domain: " + descriptor.domain.invalidReason;
    return false;
  }
  void* base = nullptr;
  size_t allocation_bytes = 0;
  hipError_t err =
      hipMemGetAddressRange(&base, &allocation_bytes, reinterpret_cast<hipDeviceptr_t>(ptr));
  if (err != hipSuccess) {
    *error = HipError("hipMemGetAddressRange", err);
    return false;
  }
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  const auto start = reinterpret_cast<uintptr_t>(base);
  if (!base || address < start || address - start > allocation_bytes ||
      bytes > allocation_bytes - (address - start)) {
    *error = "FABRIC exported view crosses a VMM allocation boundary";
    return false;
  }
  hipMemGenericAllocationHandle_t allocation{};
  err = hipMemRetainAllocationHandle(&allocation, ptr);
  if (err != hipSuccess) {
    *error = HipError("KV memory must use make_fabric_mem_pool; retain allocation", err);
    return false;
  }
  hipMemAllocationProp properties{};
  err = hipMemGetAllocationPropertiesFromHandle(&properties, allocation);
  if (err != hipSuccess || properties.location.type != hipMemLocationTypeDevice ||
      properties.location.id != device) {
    *error = "FABRIC export device does not own this GPU allocation";
    (void)hipMemRelease(allocation);
    return false;
  }
  hipMemFabricHandle_compat_t token;
  static_assert(sizeof(token) == kHandleBytes);
  err = hipMemExportToShareableHandle(&token, allocation, hipMemHandleTypeFabricCompat, 0);
  (void)hipMemRelease(allocation);
  if (err != hipSuccess) {
    *error = HipError("hipMemExportToShareableHandle", err);
    return false;
  }
  std::memcpy(descriptor.token.data(), &token, sizeof(token));
  descriptor.allocation_bytes = allocation_bytes;
  descriptor.offset = address - start;
  descriptor.bytes = bytes;
  exported->kind = "hip_fabric";
  exported->handle = Encode(descriptor);
  return true;
}

std::shared_ptr<MemoryApi> MakeFabricMemoryApi(const std::string& kernel_path) {
  return std::make_shared<FabricMemoryApi>(kernel_path);
}
}  // namespace mori::umbp::l1

// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include "umbp/engine/kv_allocator.h"

#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <utility>

#include "mori/utils/hip_compat.hpp"
#include "mori/utils/mori_log.hpp"

namespace mori::umbp::engine {
namespace {

hipMemAllocationHandleType HandleType(VmmHandleKind kind) {
  return kind == VmmHandleKind::kFabric ? hipMemHandleTypeFabricCompat
                                        : hipMemHandleTypePosixFileDescriptor;
}

hipMemAllocationProp AllocationProp(int device, VmmHandleKind kind) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.requestedHandleType = HandleType(kind);
  return prop;
}

class HipVmmApi final : public KvVmmApi {
 public:
  explicit HipVmmApi(std::optional<VmmHandleKind> forced_kind) : forced_kind_(forced_kind) {}

  hipError_t GetDevice(int* device) override { return hipGetDevice(device); }
  hipError_t SetDevice(int device) override { return hipSetDevice(device); }

  VmmHandleKind ProbeHandleKind(int device) override {
    if (forced_kind_.has_value()) return *forced_kind_;
    std::lock_guard<std::mutex> lock(probe_mutex_);
    auto cached = handle_kinds_.find(device);
    if (cached != handle_kinds_.end()) return cached->second;
    auto prop = AllocationProp(device, VmmHandleKind::kFabric);
    size_t granularity = 0;
    if (hipMemGetAllocationGranularity(&granularity, &prop,
                                       hipMemAllocationGranularityRecommended) != hipSuccess ||
        granularity == 0) {
      (void)hipGetLastError();
      return handle_kinds_.emplace(device, VmmHandleKind::kPosixFd).first->second;
    }
    hipMemGenericAllocationHandle_t handle{};
    if (hipMemCreate(&handle, granularity, &prop, 0) != hipSuccess) {
      (void)hipGetLastError();
      return handle_kinds_.emplace(device, VmmHandleKind::kPosixFd).first->second;
    }
    hipMemFabricHandle_compat_t blob{};
    const bool supported =
        hipMemExportToShareableHandle(&blob, handle, hipMemHandleTypeFabricCompat, 0) == hipSuccess;
    (void)hipMemRelease(handle);
    (void)hipGetLastError();
    const auto kind = supported ? VmmHandleKind::kFabric : VmmHandleKind::kPosixFd;
    handle_kinds_[device] = kind;
    return kind;
  }

  hipError_t Granularity(size_t* granularity, int device, VmmHandleKind kind) override {
    auto prop = AllocationProp(device, kind);
    return hipMemGetAllocationGranularity(granularity, &prop, hipMemAllocationGranularityMinimum);
  }
  hipError_t Create(hipMemGenericAllocationHandle_t* handle, size_t size, int device,
                    VmmHandleKind kind) override {
    auto prop = AllocationProp(device, kind);
    return hipMemCreate(handle, size, &prop, 0);
  }
  hipError_t AddressReserve(void** ptr, size_t size, size_t alignment) override {
    return hipMemAddressReserve(ptr, size, alignment, nullptr, 0);
  }
  hipError_t Map(void* ptr, size_t size, hipMemGenericAllocationHandle_t handle) override {
    return hipMemMap(ptr, size, 0, handle, 0);
  }
  hipError_t SetAccess(void* ptr, size_t size, int device) override {
    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    return hipMemSetAccess(ptr, size, &access, 1);
  }
  hipError_t Unmap(void* ptr, size_t size) override { return hipMemUnmap(ptr, size); }
  hipError_t AddressFree(void* ptr, size_t size) override { return hipMemAddressFree(ptr, size); }
  hipError_t Release(hipMemGenericAllocationHandle_t handle) override {
    return hipMemRelease(handle);
  }
  hipError_t Export(ExportedVmmHandle* out, hipMemGenericAllocationHandle_t handle,
                    VmmHandleKind kind, size_t mapped_size) override {
    out->kind = kind;
    out->mapped_size = mapped_size;
    if (kind == VmmHandleKind::kPosixFd) {
      int fd = -1;
      const hipError_t status =
          hipMemExportToShareableHandle(&fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
      if (status == hipSuccess) out->fd = fd;
      return status;
    }
    hipMemFabricHandle_compat_t blob{};
    const hipError_t status =
        hipMemExportToShareableHandle(&blob, handle, hipMemHandleTypeFabricCompat, 0);
    if (status == hipSuccess) {
      out->fabric.resize(sizeof(blob));
      std::memcpy(out->fabric.data(), &blob, sizeof(blob));
    }
    return status;
  }
  hipError_t Import(hipMemGenericAllocationHandle_t* handle,
                    const ExportedVmmHandle& exported) override {
    if (exported.kind == VmmHandleKind::kPosixFd) {
      return hipMemImportFromShareableHandleCompat(handle, exported.fd,
                                                   hipMemHandleTypePosixFileDescriptor);
    }
    if (exported.fabric.size() != sizeof(hipMemFabricHandle_compat_t)) {
      return hipErrorInvalidValue;
    }
    hipMemFabricHandle_compat_t blob{};
    std::memcpy(&blob, exported.fabric.data(), sizeof(blob));
    return hipMemImportFromShareableHandle(handle, &blob, hipMemHandleTypeFabricCompat);
  }

 private:
  std::optional<VmmHandleKind> forced_kind_;
  std::mutex probe_mutex_;
  std::unordered_map<int, VmmHandleKind> handle_kinds_;
};

class DeviceGuard {
 public:
  DeviceGuard(KvVmmApi* api, int device) : api_(api) {
    if (api_->GetDevice(&previous_) != hipSuccess) return;
    if (previous_ == device) return;
    changed_ = api_->SetDevice(device) == hipSuccess;
  }
  ~DeviceGuard() {
    if (changed_) (void)api_->SetDevice(previous_);
  }
  bool valid(int device) const { return previous_ == device || changed_; }

 private:
  KvVmmApi* api_;
  int previous_ = -1;
  bool changed_ = false;
};

}  // namespace

std::shared_ptr<KvVmmApi> MakeHipVmmApi(std::optional<VmmHandleKind> forced_kind) {
  return std::make_shared<HipVmmApi>(forced_kind);
}

void* KvVmmAllocator::Allocate(size_t size, int device) {
  if (size == 0 || device < 0) return nullptr;
  DeviceGuard guard(api_.get(), device);
  if (!guard.valid(device)) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
    return nullptr;
  }
  const VmmHandleKind kind = api_->ProbeHandleKind(device);
  size_t granularity = 0;
  if (api_->Granularity(&granularity, device, kind) != hipSuccess || granularity == 0 ||
      size > SIZE_MAX - (granularity - 1)) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
    return nullptr;
  }
  const size_t mapped = (size + granularity - 1) / granularity * granularity;
  VmmAllocation allocation;
  allocation.requested_size = size;
  allocation.mapped_size = mapped;
  allocation.device = device;
  allocation.handle_kind = kind;
  if (api_->Create(&allocation.handle, mapped, device, kind) != hipSuccess) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
    return nullptr;
  }
  allocation.handle_live = true;
  auto fail = [&]() -> void* {
    if (!CleanupNative(&allocation)) Quarantine(allocation);
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
    return nullptr;
  };
  if (api_->AddressReserve(&allocation.base, mapped, granularity) != hipSuccess ||
      allocation.base == nullptr) {
    allocation.base = nullptr;
    return fail();
  }
  allocation.address_reserved = true;
  if (api_->Map(allocation.base, mapped, allocation.handle) != hipSuccess) return fail();
  allocation.mapped = true;
  if (api_->SetAccess(allocation.base, mapped, device) != hipSuccess) return fail();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_.emplace(reinterpret_cast<uintptr_t>(allocation.base), allocation);
    allocated_bytes_ += mapped;
  }
  return allocation.base;
}

bool KvVmmAllocator::Free(void* ptr) {
  if (ptr == nullptr) return true;
  VmmAllocation allocation;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocations_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == allocations_.end()) return false;
    allocation = it->second;
    allocations_.erase(it);
  }
  DeviceGuard guard(api_.get(), allocation.device);
  const bool ok = guard.valid(allocation.device) && CleanupNative(&allocation);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ok) {
      allocated_bytes_ -= allocation.mapped_size;
    } else {
      quarantined_.push_back(allocation);
      ++failures_;
    }
  }
  return ok;
}

std::optional<VmmAllocation> KvVmmAllocator::Lookup(const void* ptr, size_t length) const {
  if (ptr == nullptr || length == 0) return std::nullopt;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  if (address > UINTPTR_MAX - length) return std::nullopt;
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& [base, allocation] : allocations_) {
    if (address >= base && address - base <= allocation.mapped_size &&
        length <= allocation.mapped_size - (address - base)) {
      return allocation;
    }
  }
  return std::nullopt;
}

std::optional<ExportedVmmHandle> KvVmmAllocator::Export(const void* ptr) const {
  if (ptr == nullptr) return std::nullopt;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(mutex_);
  const VmmAllocation* allocation = nullptr;
  for (const auto& [base, candidate] : allocations_) {
    if (address >= base && address - base < candidate.mapped_size) {
      allocation = &candidate;
      break;
    }
  }
  if (allocation == nullptr) return std::nullopt;
  ExportedVmmHandle out;
  // Keep the table lock across export so Free cannot release the generic
  // allocation handle between lookup and hipMemExportToShareableHandle.
  if (api_->Export(&out, allocation->handle, allocation->handle_kind, allocation->mapped_size) !=
      hipSuccess) {
    ++failures_;
    return std::nullopt;
  }
  return out;
}

std::optional<ImportedVmmRegion> KvVmmAllocator::Import(const ExportedVmmHandle& exported,
                                                        int device) {
  const auto fail_count = [&]() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
  };
  if (exported.mapped_size == 0 || device < 0) {
    fail_count();
    return std::nullopt;
  }
  DeviceGuard guard(api_.get(), device);
  if (!guard.valid(device)) {
    fail_count();
    return std::nullopt;
  }
  VmmAllocation allocation;
  allocation.mapped_size = exported.mapped_size;
  allocation.device = device;
  allocation.handle_kind = exported.kind;
  allocation.counted = false;
  if (api_->Import(&allocation.handle, exported) != hipSuccess) {
    fail_count();
    return std::nullopt;
  }
  allocation.handle_live = true;
  auto fail = [&]() -> std::optional<ImportedVmmRegion> {
    if (!CleanupNative(&allocation)) Quarantine(allocation);
    fail_count();
    return std::nullopt;
  };
  size_t granularity = 0;
  if (api_->Granularity(&granularity, device, exported.kind) != hipSuccess || granularity == 0) {
    return fail();
  }
  if (api_->AddressReserve(&allocation.base, exported.mapped_size, granularity) != hipSuccess ||
      allocation.base == nullptr) {
    allocation.base = nullptr;
    return fail();
  }
  allocation.address_reserved = true;
  if (api_->Map(allocation.base, exported.mapped_size, allocation.handle) != hipSuccess) {
    return fail();
  }
  allocation.mapped = true;
  if (api_->SetAccess(allocation.base, exported.mapped_size, device) != hipSuccess) return fail();
  if (api_->Release(allocation.handle) != hipSuccess) return fail();
  allocation.handle_live = false;  // the mapping keeps the physical allocation alive
  return ImportedVmmRegion{allocation.base, exported.mapped_size, device, true, true};
}

bool KvVmmAllocator::FreeImported(ImportedVmmRegion* region) {
  if (region == nullptr || region->base == nullptr) return true;
  DeviceGuard guard(api_.get(), region->device);
  VmmAllocation allocation{region->base,   region->mapped_size,      region->mapped_size,
                           region->device, VmmHandleKind::kPosixFd,  {},
                           region->mapped, region->address_reserved, false,
                           false};
  const bool ok = guard.valid(region->device) && CleanupNative(&allocation);
  if (!ok) {
    Quarantine(allocation);
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
  }
  *region = ImportedVmmRegion{};
  return ok;
}

bool KvVmmAllocator::CleanupNative(VmmAllocation* allocation) {
  if (allocation == nullptr) return true;
  if (allocation->mapped) {
    if (api_->Unmap(allocation->base, allocation->mapped_size) != hipSuccess) return false;
    allocation->mapped = false;
  }
  if (allocation->address_reserved) {
    if (api_->AddressFree(allocation->base, allocation->mapped_size) != hipSuccess) return false;
    allocation->address_reserved = false;
    allocation->base = nullptr;
  }
  if (allocation->handle_live) {
    if (api_->Release(allocation->handle) != hipSuccess) return false;
    allocation->handle_live = false;
    allocation->handle = {};
  }
  return true;
}

void KvVmmAllocator::Quarantine(VmmAllocation allocation) {
  std::lock_guard<std::mutex> lock(mutex_);
  quarantined_.push_back(std::move(allocation));
}

void KvVmmAllocator::Clear() {
  std::vector<void*> ptrs;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ptrs.reserve(allocations_.size());
    for (const auto& [base, allocation] : allocations_) {
      (void)allocation;
      ptrs.push_back(reinterpret_cast<void*>(base));
    }
  }
  for (void* ptr : ptrs) (void)Free(ptr);
}

size_t KvVmmAllocator::AllocationCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.size();
}

size_t KvVmmAllocator::QuarantinedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return quarantined_.size();
}

uint64_t KvVmmAllocator::AllocatedBytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocated_bytes_;
}

uint64_t KvVmmAllocator::FailureCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return failures_;
}

KvVmmAllocator& GlobalKvVmmAllocator() {
  static KvVmmAllocator* allocator = new KvVmmAllocator();
  return *allocator;
}

extern "C" void* mori_umbp_kv_malloc(size_t size, int device, hipStream_t /*stream*/) {
  return GlobalKvVmmAllocator().Allocate(size, device);
}

extern "C" void mori_umbp_kv_free(void* ptr, size_t /*size*/, int /*device*/,
                                  hipStream_t /*stream*/) {
  (void)GlobalKvVmmAllocator().Free(ptr);
}

}  // namespace mori::umbp::engine

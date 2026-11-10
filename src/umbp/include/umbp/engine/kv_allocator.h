// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace mori::umbp::engine {

enum class VmmHandleKind : uint8_t { kPosixFd, kFabric };

struct VmmAllocation {
  void* base = nullptr;
  size_t requested_size = 0;
  size_t mapped_size = 0;
  int device = -1;
  VmmHandleKind handle_kind = VmmHandleKind::kPosixFd;
  hipMemGenericAllocationHandle_t handle{};
  bool mapped = false;
  bool address_reserved = false;
  bool handle_live = false;
  bool counted = true;
};

struct ExportedVmmHandle {
  VmmHandleKind kind = VmmHandleKind::kPosixFd;
  int fd = -1;  // caller owns and closes
  std::vector<uint8_t> fabric;
  size_t mapped_size = 0;
};

struct ImportedVmmRegion {
  void* base = nullptr;
  size_t mapped_size = 0;
  int device = -1;
  bool mapped = false;
  bool address_reserved = false;
};

// HIP calls are injectable so the failure-unwind and device-restoration paths
// are CPU-testable. The production implementation is returned by MakeHipVmmApi.
class KvVmmApi {
 public:
  virtual ~KvVmmApi() = default;
  virtual hipError_t GetDevice(int* device) = 0;
  virtual hipError_t SetDevice(int device) = 0;
  virtual VmmHandleKind ProbeHandleKind(int device) = 0;
  virtual hipError_t Granularity(size_t* granularity, int device, VmmHandleKind kind) = 0;
  virtual hipError_t Create(hipMemGenericAllocationHandle_t* handle, size_t size, int device,
                            VmmHandleKind kind) = 0;
  virtual hipError_t AddressReserve(void** ptr, size_t size, size_t alignment) = 0;
  virtual hipError_t Map(void* ptr, size_t size, hipMemGenericAllocationHandle_t handle) = 0;
  virtual hipError_t SetAccess(void* ptr, size_t size, int device) = 0;
  virtual hipError_t Unmap(void* ptr, size_t size) = 0;
  virtual hipError_t AddressFree(void* ptr, size_t size) = 0;
  virtual hipError_t Release(hipMemGenericAllocationHandle_t handle) = 0;
  virtual hipError_t Export(ExportedVmmHandle* out, hipMemGenericAllocationHandle_t handle,
                            VmmHandleKind kind, size_t mapped_size) = 0;
  virtual hipError_t Import(hipMemGenericAllocationHandle_t* handle,
                            const ExportedVmmHandle& exported) = 0;
};

// `forced_kind` is a validation seam for exercising both HIP shareable-handle
// paths against real hardware. Production callers leave it unset and retain
// per-device capability probing.
std::shared_ptr<KvVmmApi> MakeHipVmmApi(std::optional<VmmHandleKind> forced_kind = std::nullopt);

class KvVmmAllocator {
 public:
  explicit KvVmmAllocator(std::shared_ptr<KvVmmApi> api = MakeHipVmmApi()) : api_(std::move(api)) {}
  ~KvVmmAllocator() = default;  // process-global allocator deliberately survives runtime teardown

  void* Allocate(size_t size, int device);
  bool Free(void* ptr);
  std::optional<VmmAllocation> Lookup(const void* ptr, size_t length = 1) const;
  std::optional<ExportedVmmHandle> Export(const void* ptr) const;
  std::optional<ImportedVmmRegion> Import(const ExportedVmmHandle& exported, int device);
  bool FreeImported(ImportedVmmRegion* region);
  void Clear();

  size_t AllocationCount() const;
  size_t QuarantinedCount() const;
  uint64_t AllocatedBytes() const;
  uint64_t FailureCount() const;

 private:
  bool CleanupNative(VmmAllocation* allocation);
  void Quarantine(VmmAllocation allocation);

  std::shared_ptr<KvVmmApi> api_;
  mutable std::mutex mutex_;
  std::unordered_map<uintptr_t, VmmAllocation> allocations_;
  std::vector<VmmAllocation> quarantined_;
  uint64_t allocated_bytes_ = 0;
  mutable uint64_t failures_ = 0;
};

KvVmmAllocator& GlobalKvVmmAllocator();

extern "C" void* mori_umbp_kv_malloc(size_t size, int device, hipStream_t stream);
extern "C" void mori_umbp_kv_free(void* ptr, size_t size, int device, hipStream_t stream);

}  // namespace mori::umbp::engine

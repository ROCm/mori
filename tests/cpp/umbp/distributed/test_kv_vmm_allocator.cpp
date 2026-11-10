// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <gtest/gtest.h>

#include <unordered_set>

#include "umbp/engine/kv_allocator.h"

namespace mori::umbp::engine {
namespace {

class FakeVmmApi final : public KvVmmApi {
 public:
  enum class Failure {
    kNone,
    kGranularity,
    kCreate,
    kReserve,
    kMap,
    kAccess,
    kUnmap,
    kAddressFree,
    kRelease,
    kExport,
    kImport,
  } failure;
  Failure cleanup_failure = Failure::kNone;

  explicit FakeVmmApi(Failure fail = Failure::kNone) : failure(fail) {}

  hipError_t GetDevice(int* device) override {
    *device = current_device;
    return hipSuccess;
  }
  hipError_t SetDevice(int device) override {
    current_device = device;
    device_history.push_back(device);
    return hipSuccess;
  }
  VmmHandleKind ProbeHandleKind(int) override { return VmmHandleKind::kPosixFd; }
  hipError_t Granularity(size_t* granularity, int, VmmHandleKind) override {
    if (Fails(Failure::kGranularity)) return hipErrorInvalidValue;
    *granularity = 64;
    return hipSuccess;
  }
  hipError_t Create(hipMemGenericAllocationHandle_t* handle, size_t, int, VmmHandleKind) override {
    if (Fails(Failure::kCreate)) return hipErrorOutOfMemory;
    *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(++next_handle);
    live_handles.insert(*handle);
    return hipSuccess;
  }
  hipError_t AddressReserve(void** ptr, size_t size, size_t) override {
    if (Fails(Failure::kReserve)) return hipErrorOutOfMemory;
    *ptr = new uint8_t[size];
    live_addresses.insert(*ptr);
    return hipSuccess;
  }
  hipError_t Map(void* ptr, size_t, hipMemGenericAllocationHandle_t handle) override {
    operations.push_back("map");
    if (Fails(Failure::kMap)) return hipErrorInvalidValue;
    if (live_addresses.count(ptr) == 0 || live_handles.count(handle) == 0) {
      return hipErrorInvalidValue;
    }
    mapped_addresses.insert(ptr);
    return hipSuccess;
  }
  hipError_t SetAccess(void* ptr, size_t, int) override {
    operations.push_back("access");
    if (mapped_addresses.count(ptr) == 0) return hipErrorInvalidValue;
    return Fails(Failure::kAccess) ? hipErrorInvalidValue : hipSuccess;
  }
  hipError_t Unmap(void* ptr, size_t) override {
    operations.push_back("unmap");
    ++unmaps;
    if (mapped_addresses.count(ptr) == 0 || Fails(Failure::kUnmap)) {
      return hipErrorInvalidValue;
    }
    mapped_addresses.erase(ptr);
    return hipSuccess;
  }
  hipError_t AddressFree(void* ptr, size_t) override {
    operations.push_back("address_free");
    ++address_frees;
    if (mapped_addresses.count(ptr) != 0 || Fails(Failure::kAddressFree)) {
      return hipErrorInvalidValue;
    }
    if (live_addresses.erase(ptr) != 0) delete[] static_cast<uint8_t*>(ptr);
    return hipSuccess;
  }
  hipError_t Release(hipMemGenericAllocationHandle_t handle) override {
    operations.push_back("release");
    ++releases;
    if (live_handles.count(handle) == 0 || Fails(Failure::kRelease)) {
      return hipErrorInvalidValue;
    }
    live_handles.erase(handle);
    return hipSuccess;
  }
  hipError_t Export(ExportedVmmHandle* out, hipMemGenericAllocationHandle_t, VmmHandleKind kind,
                    size_t mapped_size) override {
    if (Fails(Failure::kExport)) return hipErrorInvalidValue;
    out->kind = kind;
    out->fd = 77;
    out->mapped_size = mapped_size;
    return hipSuccess;
  }
  hipError_t Import(hipMemGenericAllocationHandle_t* handle, const ExportedVmmHandle&) override {
    if (Fails(Failure::kImport)) return hipErrorInvalidValue;
    *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(++next_handle);
    live_handles.insert(*handle);
    return hipSuccess;
  }

  bool Fails(Failure operation) const {
    return failure == operation || cleanup_failure == operation;
  }

  int current_device = 2;
  uintptr_t next_handle = 0;
  int unmaps = 0;
  int address_frees = 0;
  int releases = 0;
  std::vector<int> device_history;
  std::vector<std::string> operations;
  std::unordered_set<void*> live_addresses;
  std::unordered_set<void*> mapped_addresses;
  std::unordered_set<hipMemGenericAllocationHandle_t> live_handles;
};

TEST(KvVmmAllocatorTest, AllocatesAlignedIndependentRegionsAndRestoresDevice) {
  auto api = std::make_shared<FakeVmmApi>();
  KvVmmAllocator allocator(api);
  void* first = allocator.Allocate(65, 5);
  void* second = allocator.Allocate(7, 5);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(api->current_device, 2);
  EXPECT_EQ(allocator.AllocationCount(), 2u);
  EXPECT_EQ(allocator.AllocatedBytes(), 192u);
  auto inside = allocator.Lookup(static_cast<uint8_t*>(first) + 64, 64);
  ASSERT_TRUE(inside.has_value());
  EXPECT_EQ(inside->base, first);
  EXPECT_EQ(inside->requested_size, 65u);
  EXPECT_FALSE(allocator.Lookup(static_cast<uint8_t*>(first) + 127, 2).has_value());
  auto exported = allocator.Export(first);
  ASSERT_TRUE(exported.has_value());
  EXPECT_EQ(exported->fd, 77);
  EXPECT_EQ(exported->mapped_size, 128u);
  EXPECT_TRUE(allocator.Free(first));
  EXPECT_TRUE(allocator.Free(second));
  EXPECT_EQ(allocator.AllocationCount(), 0u);
  EXPECT_TRUE(api->live_addresses.empty());
  EXPECT_TRUE(api->live_handles.empty());
}

TEST(KvVmmAllocatorTest, EveryFailureStageUnwindsAndRestoresDevice) {
  for (auto failure :
       {FakeVmmApi::Failure::kGranularity, FakeVmmApi::Failure::kCreate,
        FakeVmmApi::Failure::kReserve, FakeVmmApi::Failure::kMap, FakeVmmApi::Failure::kAccess}) {
    auto api = std::make_shared<FakeVmmApi>(failure);
    KvVmmAllocator allocator(api);
    EXPECT_EQ(allocator.Allocate(100, 6), nullptr) << static_cast<int>(failure);
    EXPECT_EQ(api->current_device, 2) << static_cast<int>(failure);
    EXPECT_EQ(allocator.AllocationCount(), 0u);
    EXPECT_TRUE(api->live_addresses.empty()) << static_cast<int>(failure);
    EXPECT_TRUE(api->live_handles.empty()) << static_cast<int>(failure);
    EXPECT_EQ(allocator.FailureCount(), 1u);
  }
}

TEST(KvVmmAllocatorTest, ImportedMappingDropsHandleButLivesUntilExplicitUnmap) {
  auto api = std::make_shared<FakeVmmApi>();
  KvVmmAllocator allocator(api);
  ExportedVmmHandle exported;
  exported.kind = VmmHandleKind::kPosixFd;
  exported.fd = 1;
  exported.mapped_size = 128;
  auto imported = allocator.Import(exported, 4);
  ASSERT_TRUE(imported.has_value());
  EXPECT_TRUE(api->live_handles.empty());
  EXPECT_EQ(api->live_addresses.size(), 1u);
  EXPECT_TRUE(allocator.FreeImported(&*imported));
  EXPECT_EQ(imported->base, nullptr);
  EXPECT_TRUE(api->live_addresses.empty());
}

TEST(KvVmmAllocatorTest, FailedFreeRetainsNativeResourcesInQuarantine) {
  auto api = std::make_shared<FakeVmmApi>(FakeVmmApi::Failure::kNone);
  KvVmmAllocator allocator(api);
  void* ptr = allocator.Allocate(64, 4);
  ASSERT_NE(ptr, nullptr);
  api->failure = FakeVmmApi::Failure::kUnmap;
  EXPECT_FALSE(allocator.Free(ptr));
  EXPECT_EQ(allocator.AllocationCount(), 0u);
  EXPECT_EQ(allocator.QuarantinedCount(), 1u);
  EXPECT_EQ(allocator.AllocatedBytes(), 64u);
  EXPECT_EQ(api->live_addresses.size(), 1u);
  EXPECT_EQ(api->live_handles.size(), 1u);
}

TEST(KvVmmAllocatorTest, ExportAndImportFailuresDoNotInventMappingsOrLeakHandles) {
  auto api = std::make_shared<FakeVmmApi>();
  KvVmmAllocator allocator(api);
  void* ptr = allocator.Allocate(64, 4);
  ASSERT_NE(ptr, nullptr);
  api->failure = FakeVmmApi::Failure::kExport;
  EXPECT_FALSE(allocator.Export(ptr).has_value());
  EXPECT_EQ(api->mapped_addresses.size(), 1u);
  EXPECT_EQ(api->live_handles.size(), 1u);
  api->failure = FakeVmmApi::Failure::kNone;
  EXPECT_TRUE(allocator.Free(ptr));

  ExportedVmmHandle exported;
  exported.kind = VmmHandleKind::kPosixFd;
  exported.fd = 1;
  exported.mapped_size = 128;
  api->failure = FakeVmmApi::Failure::kImport;
  EXPECT_FALSE(allocator.Import(exported, 4).has_value());
  EXPECT_TRUE(api->mapped_addresses.empty());
  EXPECT_TRUE(api->live_addresses.empty());
  EXPECT_TRUE(api->live_handles.empty());
}

TEST(KvVmmAllocatorTest, FailedImportCleanupStopsAtFirstUnsafeStepAndQuarantinesRemainder) {
  auto api = std::make_shared<FakeVmmApi>(FakeVmmApi::Failure::kAccess);
  api->cleanup_failure = FakeVmmApi::Failure::kUnmap;
  KvVmmAllocator allocator(api);
  ExportedVmmHandle exported;
  exported.kind = VmmHandleKind::kPosixFd;
  exported.fd = 1;
  exported.mapped_size = 128;
  EXPECT_FALSE(allocator.Import(exported, 4).has_value());
  EXPECT_EQ(allocator.QuarantinedCount(), 1u);
  EXPECT_EQ(api->mapped_addresses.size(), 1u);
  EXPECT_EQ(api->live_addresses.size(), 1u);
  EXPECT_EQ(api->live_handles.size(), 1u);
  ASSERT_GE(api->operations.size(), 3u);
  EXPECT_EQ(api->operations[api->operations.size() - 1], "unmap");
}

TEST(KvVmmAllocatorTest, PartialFreeCleanupRecordsExactRemainingNativeState) {
  for (auto cleanup_failure : {FakeVmmApi::Failure::kAddressFree, FakeVmmApi::Failure::kRelease}) {
    auto api = std::make_shared<FakeVmmApi>();
    KvVmmAllocator allocator(api);
    void* ptr = allocator.Allocate(64, 4);
    ASSERT_NE(ptr, nullptr);
    api->failure = cleanup_failure;
    EXPECT_FALSE(allocator.Free(ptr));
    EXPECT_EQ(allocator.QuarantinedCount(), 1u);
    EXPECT_TRUE(api->mapped_addresses.empty());
    if (cleanup_failure == FakeVmmApi::Failure::kAddressFree) {
      EXPECT_EQ(api->live_addresses.size(), 1u);
      EXPECT_EQ(api->live_handles.size(), 1u);
    } else {
      EXPECT_TRUE(api->live_addresses.empty());
      EXPECT_EQ(api->live_handles.size(), 1u);
    }
  }
}

}  // namespace
}  // namespace mori::umbp::engine

// (C) Sebastian Aaltonen 2023
// MIT License (see file: LICENSE)
//
// Migrated to umbp namespace for UMBP SPDK integration.
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace umbp::offset_allocator {

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
using NodeIndex = uint32;

class OffsetAllocator;
class __Allocator;

static constexpr uint32 NUM_TOP_BINS = 32;
static constexpr uint32 BINS_PER_LEAF = 8;
static constexpr uint32 TOP_BINS_INDEX_SHIFT = 3;
static constexpr uint32 LEAF_BINS_INDEX_MASK = 0x7;
static constexpr uint32 NUM_LEAF_BINS = NUM_TOP_BINS * BINS_PER_LEAF;

struct OffsetAllocation {
    static constexpr uint32 NO_SPACE = 0xffffffff;

   private:
    uint32 offset = NO_SPACE;
    NodeIndex metadata = NO_SPACE;

   public:
    OffsetAllocation(uint32 offset_param, NodeIndex metadata_param)
        : offset(offset_param), metadata(metadata_param) {}
    uint64_t getOffset() const { return static_cast<uint64_t>(offset); }
    bool isNoSpace() const { return offset == NO_SPACE; }

    friend class __Allocator;
};

struct OffsetAllocStorageReport {
    uint64_t totalFreeSpace;
    uint64_t largestFreeRegion;
};

struct OffsetAllocStorageReportFull {
    struct Region {
        uint64_t size;
        uint64_t count;
    };
    Region freeRegions[NUM_LEAF_BINS];
};

class OffsetAllocationHandle {
   public:
    OffsetAllocationHandle()
        : m_allocation(OffsetAllocation::NO_SPACE, OffsetAllocation::NO_SPACE),
          real_base(0),
          requested_size(0) {}

    OffsetAllocationHandle(std::shared_ptr<OffsetAllocator> allocator,
                           OffsetAllocation allocation, uint64_t base,
                           uint64_t size);

    OffsetAllocationHandle(OffsetAllocationHandle&& other) noexcept;
    OffsetAllocationHandle& operator=(OffsetAllocationHandle&& other) noexcept;

    OffsetAllocationHandle(const OffsetAllocationHandle&) = delete;
    OffsetAllocationHandle& operator=(const OffsetAllocationHandle&) = delete;

    ~OffsetAllocationHandle();

    bool isValid() const { return !m_allocator.expired(); }
    uint64_t address() const { return real_base; }
    void* ptr() const { return reinterpret_cast<void*>(address()); }
    uint64_t size() const { return requested_size; }

   private:
    std::weak_ptr<OffsetAllocator> m_allocator;
    OffsetAllocation m_allocation;
    uint64_t real_base;
    uint64_t requested_size;
};

struct OffsetAllocatorMetrics {
    uint64_t allocated_size_;
    uint64_t allocated_num_;
    uint64_t largest_free_region_;
    uint64_t total_free_space_;
    const uint64_t capacity;
};

std::ostream& operator<<(std::ostream& os, const OffsetAllocatorMetrics& metrics);

class OffsetAllocator : public std::enable_shared_from_this<OffsetAllocator> {
   public:
    static std::shared_ptr<OffsetAllocator> create(
        uint64_t base, size_t size, uint32 init_capacity = 128 * 1024,
        uint32 max_capacity = (1 << 20));

    static std::shared_ptr<OffsetAllocator> createAligned(
        uint64_t base, size_t size, size_t min_alignment,
        uint32 init_capacity = 128 * 1024,
        uint32 max_capacity = (1 << 20));

    OffsetAllocator(const OffsetAllocator&) = delete;
    OffsetAllocator& operator=(const OffsetAllocator&) = delete;
    OffsetAllocator(OffsetAllocator&&) noexcept = delete;
    OffsetAllocator& operator=(OffsetAllocator&&) noexcept = delete;

    ~OffsetAllocator() = default;

    [[nodiscard]]
    std::optional<OffsetAllocationHandle> allocate(size_t size);

    std::vector<std::optional<OffsetAllocationHandle>>
        batch_allocate(const std::vector<size_t>& sizes);

    [[nodiscard]]
    OffsetAllocStorageReport storageReport() const;

    [[nodiscard]]
    OffsetAllocStorageReportFull storageReportFull() const;

    [[nodiscard]]
    OffsetAllocatorMetrics get_metrics() const;

   private:
    friend class OffsetAllocationHandle;

    void freeAllocation(const OffsetAllocation& allocation, uint64_t size);

    [[nodiscard]]
    OffsetAllocatorMetrics get_metrics_internal() const;

    std::unique_ptr<__Allocator> m_allocator;
    uint64_t m_base;
    uint64_t m_multiplier_bits;
    uint64_t m_capacity;
    mutable std::mutex m_mutex;

    uint64_t m_allocated_size = 0;
    uint64_t m_allocated_num = 0;

    OffsetAllocator(uint64_t base, size_t size, uint32 init_capacity,
                    uint32 max_capacity);
    OffsetAllocator(uint64_t base, size_t size, uint64_t multiplier_bits,
                    std::unique_ptr<__Allocator> allocator);
};

class __Allocator {
   public:
    __Allocator(uint32 size, uint32 init_capacity, uint32 max_capacity);
    __Allocator(__Allocator&& other);
    ~__Allocator() = default;
    void reset();

    OffsetAllocation allocate(uint32 size);
    void free(OffsetAllocation allocation);

    uint32 allocationSize(OffsetAllocation allocation) const;
    OffsetAllocStorageReport storageReport() const;
    OffsetAllocStorageReportFull storageReportFull() const;

   private:
    uint32 insertNodeIntoBin(uint32 size, uint32 dataOffset);
    void removeNodeFromBin(uint32 nodeIndex);

    struct Node {
        static constexpr NodeIndex unused = 0xffffffff;
        uint32 dataOffset = 0;
        uint32 dataSize = 0;
        NodeIndex binListPrev = unused;
        NodeIndex binListNext = unused;
        NodeIndex neighborPrev = unused;
        NodeIndex neighborNext = unused;
        bool used = false;
    };

    uint32 m_size;
    uint32 m_current_capacity;
    uint32 m_max_capacity;
    uint32 m_freeStorage;

    uint32 m_usedBinsTop;
    uint8 m_usedBins[NUM_TOP_BINS];
    NodeIndex m_binIndices[NUM_LEAF_BINS];

    std::vector<Node> m_nodes;
    std::vector<NodeIndex> m_freeNodes;
    uint32 m_freeOffset;
};

}  // namespace umbp::offset_allocator

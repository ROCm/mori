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
// (C) Sebastian Aaltonen 2023
// MIT License (see file: LICENSE)
//
// Migrated to umbp namespace for UMBP SPDK integration.

#include "umbp/storage/spdk/offset_allocator.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>

#ifdef DEBUG
#include <assert.h>
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace umbp::offset_allocator {

inline uint32 lzcnt_nonzero(uint32 v) {
#ifdef _MSC_VER
  unsigned long retVal;
  _BitScanReverse(&retVal, v);
  return 31 - retVal;
#else
  return __builtin_clz(v);
#endif
}

inline uint32 tzcnt_nonzero(uint32 v) {
#ifdef _MSC_VER
  unsigned long retVal;
  _BitScanForward(&retVal, v);
  return retVal;
#else
  return __builtin_ctz(v);
#endif
}

namespace SmallFloat {
static constexpr uint32 MANTISSA_BITS = 3;
static constexpr uint32 MANTISSA_VALUE = 1 << MANTISSA_BITS;
static constexpr uint32 MANTISSA_MASK = MANTISSA_VALUE - 1;
static constexpr uint64_t MAX_BIN_SIZE = 4026531840ull;  // 3.75GB

uint32 uintToFloatRoundUp(uint32 size) {
  uint32 exp = 0;
  uint32 mantissa = 0;
  if (size < MANTISSA_VALUE) {
    mantissa = size;
  } else {
    uint32 leadingZeros = lzcnt_nonzero(size);
    uint32 highestSetBit = 31 - leadingZeros;
    uint32 mantissaStartBit = highestSetBit - MANTISSA_BITS;
    exp = mantissaStartBit + 1;
    mantissa = (size >> mantissaStartBit) & MANTISSA_MASK;
    uint32 lowBitsMask = (1 << mantissaStartBit) - 1;
    if ((size & lowBitsMask) != 0) mantissa++;
  }
  return (exp << MANTISSA_BITS) + mantissa;
}

uint32 uintToFloatRoundDown(uint32 size) {
  uint32 exp = 0;
  uint32 mantissa = 0;
  if (size < MANTISSA_VALUE) {
    mantissa = size;
  } else {
    uint32 leadingZeros = lzcnt_nonzero(size);
    uint32 highestSetBit = 31 - leadingZeros;
    uint32 mantissaStartBit = highestSetBit - MANTISSA_BITS;
    exp = mantissaStartBit + 1;
    mantissa = (size >> mantissaStartBit) & MANTISSA_MASK;
  }
  return (exp << MANTISSA_BITS) | mantissa;
}

uint32 floatToUint(uint32 floatValue) {
  uint32 exponent = floatValue >> MANTISSA_BITS;
  uint32 mantissa = floatValue & MANTISSA_MASK;
  if (exponent == 0) {
    return mantissa;
  } else {
    return (mantissa | MANTISSA_VALUE) << (exponent - 1);
  }
}
}  // namespace SmallFloat

uint32 findLowestSetBitAfter(uint32 bitMask, uint32 startBitIndex) {
  uint32 maskBeforeStartIndex = (1 << startBitIndex) - 1;
  uint32 maskAfterStartIndex = ~maskBeforeStartIndex;
  uint32 bitsAfter = bitMask & maskAfterStartIndex;
  if (bitsAfter == 0) return OffsetAllocation::NO_SPACE;
  return tzcnt_nonzero(bitsAfter);
}

// __Allocator implementation
__Allocator::__Allocator(uint32 size, uint32 init_capacity, uint32 max_capacity)
    : m_size(size),
      m_current_capacity(init_capacity),
      m_max_capacity(std::max(init_capacity, max_capacity)) {
  if (sizeof(NodeIndex) == 2) {
    ASSERT(m_max_capacity <= 65536);
  }
  reset();
}

__Allocator::__Allocator(__Allocator&& other)
    : m_size(other.m_size),
      m_current_capacity(other.m_current_capacity),
      m_max_capacity(other.m_max_capacity),
      m_freeStorage(other.m_freeStorage),
      m_usedBinsTop(other.m_usedBinsTop),
      m_nodes(std::move(other.m_nodes)),
      m_freeNodes(std::move(other.m_freeNodes)),
      m_freeOffset(other.m_freeOffset) {
  memcpy(m_usedBins, other.m_usedBins, sizeof(uint8) * NUM_TOP_BINS);
  memcpy(m_binIndices, other.m_binIndices, sizeof(NodeIndex) * NUM_LEAF_BINS);
  other.m_nodes.clear();
  other.m_freeNodes.clear();
  other.m_freeOffset = 0;
  other.m_current_capacity = 0;
  other.m_max_capacity = 0;
  other.m_usedBinsTop = 0;
}

void __Allocator::reset() {
  m_freeStorage = 0;
  m_usedBinsTop = 0;
  m_freeOffset = 0;
  for (uint32 i = 0; i < NUM_TOP_BINS; i++) m_usedBins[i] = 0;
  for (uint32 i = 0; i < NUM_LEAF_BINS; i++) m_binIndices[i] = Node::unused;
  m_nodes.clear();
  m_freeNodes.clear();
  m_nodes.reserve(m_max_capacity);
  m_freeNodes.reserve(m_max_capacity);
  m_nodes.resize(m_current_capacity);
  m_freeNodes.resize(m_current_capacity);
  for (uint32 i = 0; i < m_current_capacity; i++) {
    m_freeNodes[i] = i;
  }
  insertNodeIntoBin(m_size, 0);
}

OffsetAllocation __Allocator::allocate(uint32 size) {
  if (m_freeOffset == m_max_capacity) {
    return OffsetAllocation(OffsetAllocation::NO_SPACE, OffsetAllocation::NO_SPACE);
  }
  if (m_freeOffset == m_current_capacity) {
    m_freeNodes.push_back(m_current_capacity);
    m_nodes.emplace_back();
    m_current_capacity++;
  }

  uint32 minBinIndex = SmallFloat::uintToFloatRoundUp(size);
  uint32 minTopBinIndex = minBinIndex >> TOP_BINS_INDEX_SHIFT;
  uint32 minLeafBinIndex = minBinIndex & LEAF_BINS_INDEX_MASK;
  uint32 topBinIndex = minTopBinIndex;
  uint32 leafBinIndex = OffsetAllocation::NO_SPACE;

  if (m_usedBinsTop & (1 << topBinIndex)) {
    leafBinIndex = findLowestSetBitAfter(m_usedBins[topBinIndex], minLeafBinIndex);
  }

  if (leafBinIndex == OffsetAllocation::NO_SPACE) {
    topBinIndex = findLowestSetBitAfter(m_usedBinsTop, minTopBinIndex + 1);
    if (topBinIndex == OffsetAllocation::NO_SPACE) {
      return OffsetAllocation(OffsetAllocation::NO_SPACE, OffsetAllocation::NO_SPACE);
    }
    leafBinIndex = tzcnt_nonzero(m_usedBins[topBinIndex]);
  }

  uint32 binIndex = (topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex;
  uint32 nodeIndex = m_binIndices[binIndex];
  Node& node = m_nodes[nodeIndex];
  uint32 nodeTotalSize = node.dataSize;

#ifdef OFFSET_ALLOCATOR_NOT_ROUND_UP
  uint32 roundupSize = size;
#else
  uint32 roundupSize = SmallFloat::floatToUint(minBinIndex);
#endif

  node.dataSize = roundupSize;
  node.used = true;
  m_binIndices[binIndex] = node.binListNext;
  if (node.binListNext != Node::unused) m_nodes[node.binListNext].binListPrev = Node::unused;
  m_freeStorage -= nodeTotalSize;

  if (m_binIndices[binIndex] == Node::unused) {
    m_usedBins[topBinIndex] &= ~(1 << leafBinIndex);
    if (m_usedBins[topBinIndex] == 0) {
      m_usedBinsTop &= ~(1 << topBinIndex);
    }
  }

  uint32 reminderSize = nodeTotalSize - roundupSize;
  if (reminderSize > 0) {
    uint32 newNodeIndex = insertNodeIntoBin(reminderSize, node.dataOffset + roundupSize);
    if (node.neighborNext != Node::unused) m_nodes[node.neighborNext].neighborPrev = newNodeIndex;
    m_nodes[newNodeIndex].neighborPrev = nodeIndex;
    m_nodes[newNodeIndex].neighborNext = node.neighborNext;
    node.neighborNext = newNodeIndex;
  }

  return OffsetAllocation(node.dataOffset, nodeIndex);
}

void __Allocator::free(OffsetAllocation allocation) {
  ASSERT(allocation.metadata != OffsetAllocation::NO_SPACE);
  if (m_nodes.empty()) return;

  uint32 nodeIndex = allocation.metadata;
  Node& node = m_nodes[nodeIndex];
  ASSERT(node.used == true);

  uint32 offset = node.dataOffset;
  uint32 size = node.dataSize;

  if ((node.neighborPrev != Node::unused) && (m_nodes[node.neighborPrev].used == false)) {
    Node& prevNode = m_nodes[node.neighborPrev];
    offset = prevNode.dataOffset;
    size += prevNode.dataSize;
    removeNodeFromBin(node.neighborPrev);
    ASSERT(prevNode.neighborNext == nodeIndex);
    node.neighborPrev = prevNode.neighborPrev;
  }

  if ((node.neighborNext != Node::unused) && (m_nodes[node.neighborNext].used == false)) {
    Node& nextNode = m_nodes[node.neighborNext];
    size += nextNode.dataSize;
    removeNodeFromBin(node.neighborNext);
    ASSERT(nextNode.neighborPrev == nodeIndex);
    node.neighborNext = nextNode.neighborNext;
  }

  uint32 neighborNext = node.neighborNext;
  uint32 neighborPrev = node.neighborPrev;

  m_freeNodes[--m_freeOffset] = nodeIndex;

  uint32 combinedNodeIndex = insertNodeIntoBin(size, offset);

  if (neighborNext != Node::unused) {
    m_nodes[combinedNodeIndex].neighborNext = neighborNext;
    m_nodes[neighborNext].neighborPrev = combinedNodeIndex;
  }
  if (neighborPrev != Node::unused) {
    m_nodes[combinedNodeIndex].neighborPrev = neighborPrev;
    m_nodes[neighborPrev].neighborNext = combinedNodeIndex;
  }
}

uint32 __Allocator::insertNodeIntoBin(uint32 size, uint32 dataOffset) {
  uint32 binIndex = SmallFloat::uintToFloatRoundDown(size);
  uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
  uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;

  if (m_binIndices[binIndex] == Node::unused) {
    m_usedBins[topBinIndex] |= 1 << leafBinIndex;
    m_usedBinsTop |= 1 << topBinIndex;
  }

  uint32 topNodeIndex = m_binIndices[binIndex];
  uint32 nodeIndex = m_freeNodes[m_freeOffset++];

  m_nodes[nodeIndex] = {.dataOffset = dataOffset, .dataSize = size, .binListNext = topNodeIndex};
  if (topNodeIndex != Node::unused) m_nodes[topNodeIndex].binListPrev = nodeIndex;
  m_binIndices[binIndex] = nodeIndex;
  m_freeStorage += size;

  return nodeIndex;
}

void __Allocator::removeNodeFromBin(uint32 nodeIndex) {
  Node& node = m_nodes[nodeIndex];
  if (node.binListPrev != Node::unused) {
    m_nodes[node.binListPrev].binListNext = node.binListNext;
    if (node.binListNext != Node::unused) m_nodes[node.binListNext].binListPrev = node.binListPrev;
  } else {
    uint32 binIndex = SmallFloat::uintToFloatRoundDown(node.dataSize);
    uint32 topBinIndex = binIndex >> TOP_BINS_INDEX_SHIFT;
    uint32 leafBinIndex = binIndex & LEAF_BINS_INDEX_MASK;
    m_binIndices[binIndex] = node.binListNext;
    if (node.binListNext != Node::unused) m_nodes[node.binListNext].binListPrev = Node::unused;
    if (m_binIndices[binIndex] == Node::unused) {
      m_usedBins[topBinIndex] &= ~(1 << leafBinIndex);
      if (m_usedBins[topBinIndex] == 0) {
        m_usedBinsTop &= ~(1 << topBinIndex);
      }
    }
  }
  m_freeNodes[--m_freeOffset] = nodeIndex;
  m_freeStorage -= node.dataSize;
}

uint32 __Allocator::allocationSize(OffsetAllocation allocation) const {
  if (allocation.metadata == OffsetAllocation::NO_SPACE) return 0;
  if (m_nodes.empty()) return 0;
  return m_nodes[allocation.metadata].dataSize;
}

OffsetAllocStorageReport __Allocator::storageReport() const {
  uint32 largestFreeRegion = 0;
  uint32 freeStorage = 0;
  if (m_freeOffset < m_max_capacity) {
    freeStorage = m_freeStorage;
    if (m_usedBinsTop) {
      uint32 topBinIndex = 31 - lzcnt_nonzero(m_usedBinsTop);
      uint32 leafBinIndex = 31 - lzcnt_nonzero(m_usedBins[topBinIndex]);
      largestFreeRegion =
          SmallFloat::floatToUint((topBinIndex << TOP_BINS_INDEX_SHIFT) | leafBinIndex);
      ASSERT(freeStorage >= largestFreeRegion);
    }
  }
  return {.totalFreeSpace = freeStorage, .largestFreeRegion = largestFreeRegion};
}

OffsetAllocStorageReportFull __Allocator::storageReportFull() const {
  OffsetAllocStorageReportFull report;
  for (uint32 i = 0; i < NUM_LEAF_BINS; i++) {
    uint32 count = 0;
    uint32 nodeIndex = m_binIndices[i];
    while (nodeIndex != Node::unused) {
      nodeIndex = m_nodes[nodeIndex].binListNext;
      count++;
    }
    report.freeRegions[i] = {.size = SmallFloat::floatToUint(i), .count = count};
  }
  return report;
}

// OffsetAllocationHandle implementation
OffsetAllocationHandle::OffsetAllocationHandle(std::shared_ptr<OffsetAllocator> allocator,
                                               OffsetAllocation allocation, uint64_t base,
                                               uint64_t size)
    : m_allocator(std::move(allocator)),
      m_allocation(allocation),
      real_base(base),
      requested_size(size) {}

OffsetAllocationHandle::OffsetAllocationHandle(OffsetAllocationHandle&& other) noexcept
    : m_allocator(std::move(other.m_allocator)),
      m_allocation(other.m_allocation),
      real_base(other.real_base),
      requested_size(other.requested_size) {
  other.m_allocation = {OffsetAllocation::NO_SPACE, OffsetAllocation::NO_SPACE};
  other.real_base = 0;
  other.requested_size = 0;
}

OffsetAllocationHandle& OffsetAllocationHandle::operator=(OffsetAllocationHandle&& other) noexcept {
  if (this != &other) {
    auto allocator = m_allocator.lock();
    if (allocator) {
      allocator->freeAllocation(m_allocation, requested_size);
    }
    m_allocator = std::move(other.m_allocator);
    m_allocation = other.m_allocation;
    real_base = other.real_base;
    requested_size = other.requested_size;
    other.m_allocation = {OffsetAllocation::NO_SPACE, OffsetAllocation::NO_SPACE};
    other.real_base = 0;
    other.requested_size = 0;
  }
  return *this;
}

OffsetAllocationHandle::~OffsetAllocationHandle() {
  auto allocator = m_allocator.lock();
  if (allocator) {
    allocator->freeAllocation(m_allocation, requested_size);
  }
}

static uint64_t calculateMultiplier(size_t size) {
  static constexpr uint64_t MAX_BIN_SIZE = 4026531840ull;
  uint64_t multiplier_bits = 0;
  for (; MAX_BIN_SIZE < (size >> multiplier_bits); multiplier_bits++) {
  }
  return multiplier_bits;
}

// Thread-safe OffsetAllocator implementation
std::shared_ptr<OffsetAllocator> OffsetAllocator::create(uint64_t base, size_t size,
                                                         uint32 init_capacity,
                                                         uint32 max_capacity) {
  return std::shared_ptr<OffsetAllocator>(
      new OffsetAllocator(base, size, init_capacity, max_capacity));
}

OffsetAllocator::OffsetAllocator(uint64_t base, size_t size, uint32 init_capacity,
                                 uint32 max_capacity)
    : m_base(base), m_multiplier_bits(calculateMultiplier(size)), m_capacity(size) {
  m_allocator =
      std::make_unique<__Allocator>(size >> m_multiplier_bits, init_capacity, max_capacity);
}

OffsetAllocator::OffsetAllocator(uint64_t base, size_t size, uint64_t multiplier_bits,
                                 std::unique_ptr<__Allocator> allocator)
    : m_allocator(std::move(allocator)),
      m_base(base),
      m_multiplier_bits(multiplier_bits),
      m_capacity(size) {}

std::shared_ptr<OffsetAllocator> OffsetAllocator::createAligned(uint64_t base, size_t size,
                                                                size_t min_alignment,
                                                                uint32 init_capacity,
                                                                uint32 max_capacity) {
  uint64_t min_bits = 0;
  while ((1ULL << min_bits) < min_alignment) min_bits++;
  uint64_t bits = std::max(calculateMultiplier(size), min_bits);
  auto allocator =
      std::make_unique<__Allocator>(static_cast<uint32>(size >> bits), init_capacity, max_capacity);
  return std::shared_ptr<OffsetAllocator>(
      new OffsetAllocator(base, size, bits, std::move(allocator)));
}

std::optional<OffsetAllocationHandle> OffsetAllocator::allocate(size_t size) {
  if (size == 0) return std::nullopt;

  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_allocator) return std::nullopt;

  size_t fake_size =
      m_multiplier_bits > 0
          ? ((size + (static_cast<uint64_t>(1) << m_multiplier_bits) - 1u) >> m_multiplier_bits)
          : size;

  if (fake_size > SmallFloat::MAX_BIN_SIZE) return std::nullopt;

  OffsetAllocation allocation = m_allocator->allocate(fake_size);
  if (allocation.isNoSpace()) return std::nullopt;

  m_allocated_size += size;
  m_allocated_num++;

  return OffsetAllocationHandle(shared_from_this(), allocation,
                                m_base + (allocation.getOffset() << m_multiplier_bits), size);
}

std::vector<std::optional<OffsetAllocationHandle>> OffsetAllocator::batch_allocate(
    const std::vector<size_t>& sizes) {
  std::vector<std::optional<OffsetAllocationHandle>> results;
  results.reserve(sizes.size());

  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_allocator) {
    for (size_t i = 0; i < sizes.size(); ++i) results.emplace_back(std::nullopt);
    return results;
  }

  for (size_t size : sizes) {
    if (size == 0) {
      results.push_back(std::nullopt);
      continue;
    }

    size_t fake_size =
        m_multiplier_bits > 0
            ? ((size + (static_cast<uint64_t>(1) << m_multiplier_bits) - 1u) >> m_multiplier_bits)
            : size;

    if (fake_size > SmallFloat::MAX_BIN_SIZE) {
      results.push_back(std::nullopt);
      break;
    }

    OffsetAllocation allocation = m_allocator->allocate(fake_size);
    if (allocation.isNoSpace()) {
      results.push_back(std::nullopt);
      break;
    }

    m_allocated_size += size;
    m_allocated_num++;

    results.emplace_back(
        OffsetAllocationHandle(shared_from_this(), allocation,
                               m_base + (allocation.getOffset() << m_multiplier_bits), size));
  }

  return results;
}

OffsetAllocStorageReport OffsetAllocator::storageReport() const {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_allocator) return {0, 0};
  OffsetAllocStorageReport report = m_allocator->storageReport();
  return {report.totalFreeSpace << m_multiplier_bits,
          report.largestFreeRegion << m_multiplier_bits};
}

OffsetAllocStorageReportFull OffsetAllocator::storageReportFull() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_allocator) {
    OffsetAllocStorageReportFull report{};
    return report;
  }
  OffsetAllocStorageReportFull report = m_allocator->storageReportFull();
  for (uint32 i = 0; i < NUM_LEAF_BINS; i++) {
    report.freeRegions[i] = {.size = report.freeRegions[i].size << m_multiplier_bits,
                             .count = report.freeRegions[i].count};
  }
  return report;
}

OffsetAllocatorMetrics OffsetAllocator::get_metrics_internal() const {
  if (!m_allocator) return {0, 0, 0, 0, m_capacity};
  OffsetAllocStorageReport basic_report = m_allocator->storageReport();
  return {
      m_allocated_size,
      m_allocated_num,
      basic_report.largestFreeRegion << m_multiplier_bits,
      basic_report.totalFreeSpace << m_multiplier_bits,
      m_capacity,
  };
}

OffsetAllocatorMetrics OffsetAllocator::get_metrics() const {
  std::lock_guard<std::mutex> guard(m_mutex);
  return get_metrics_internal();
}

void OffsetAllocator::freeAllocation(const OffsetAllocation& allocation, uint64_t size) {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_allocator) {
    m_allocator->free(allocation);
    m_allocated_size -= size;
    m_allocated_num--;
  }
}

static std::string byte_size_str(uint64_t bytes) {
  char buf[64];
  if (bytes >= (1ULL << 30))
    std::snprintf(buf, sizeof(buf), "%.1fGB", bytes / (double)(1ULL << 30));
  else if (bytes >= (1ULL << 20))
    std::snprintf(buf, sizeof(buf), "%.1fMB", bytes / (double)(1ULL << 20));
  else if (bytes >= (1ULL << 10))
    std::snprintf(buf, sizeof(buf), "%.1fKB", bytes / (double)(1ULL << 10));
  else
    std::snprintf(buf, sizeof(buf), "%luB", static_cast<unsigned long>(bytes));
  return buf;
}

std::ostream& operator<<(std::ostream& os, const OffsetAllocatorMetrics& metrics) {
  double utilization =
      metrics.capacity > 0 ? (double)metrics.allocated_size_ / metrics.capacity * 100.0 : 0.0;
  os << "OffsetAllocatorMetrics{"
     << "allocated=" << byte_size_str(metrics.allocated_size_)
     << ", allocs=" << metrics.allocated_num_ << ", capacity=" << byte_size_str(metrics.capacity)
     << ", utilization=" << std::fixed << std::setprecision(1) << utilization << "%"
     << ", free_space=" << byte_size_str(metrics.total_free_space_)
     << ", largest_free=" << byte_size_str(metrics.largest_free_region_) << "}";
  return os;
}

}  // namespace umbp::offset_allocator

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
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */
#pragma once

#include <hip/hip_runtime_api.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "hsa/hsa_ext_amd.h"
#include "hsakmt/hsakmt.h"
#include "hsakmt/hsakmttypes.h"
#include "mori/core/transport/sdma/anvil_device.hpp"

namespace anvil {

class SdmaQueue {
 public:
  SdmaQueue(uint32_t localNodeId, uint32_t engineId);
  ~SdmaQueue();

  SdmaQueueDeviceHandle* deviceHandle() const;

 private:
  uint64_t* cachedWptr_;
  uint64_t* committedWptr_;
  void* queueBuffer_;
  HsaQueueResource queue_;
  SdmaQueueDeviceHandle* deviceHandle_;
};

class AnvilLib {
 private:
  // Make constructor private
  AnvilLib() = default;

 public:
  ~AnvilLib();
  // access to singleton
  static AnvilLib& getInstance();

  AnvilLib(const AnvilLib&) = delete;
  AnvilLib& operator=(const AnvilLib&) = delete;

 public:
  void init();
  // srcNode/dstNode are KFD topology node ids (== HSA_AGENT_INFO_NODE), a
  // host-global GPU identity that does NOT depend on HIP_VISIBLE_DEVICES. This
  // is the correct key for a peer even when the peer GPU is not in this
  // process's HIP device list. Channels for a pair are shared process-wide.
  bool connect(int srcNode, int dstNode, int numChannels = 1);

  // Get the SDMA queue for a given src/dst node pair and channel index.
  SdmaQueue* getSdmaQueue(int srcNode, int dstNode, int channelIdx = 0);

  // Map a HIP device ordinal to its KFD node id. For single-process callers
  // (e.g. the examples) that only have HIP device ids; multi-process collectives
  // should exchange node ids out-of-band instead (see Context::KfdNodeId).
  static uint32_t nodeForHipDevice(int hipDev);

  // Resolve the KFD topology node id of the given HIP device WITHOUT initializing
  // HSA. The KFD node id (the directory index under /sys/class/kfd/kfd/topology/nodes)
  // is identical to what HSA_AGENT_INFO_NODE / hsaKmtGetNodeProperties report,
  // and is a host-global identity independent of HIP_VISIBLE_DEVICES.
  static int kfdNodeIdForHipDevice(int hipDev);

 private:
  /*
   * OAM MAP
   * src\dst    0  1 2 3 4 5 6 7
   * 0         0 7 6 1 2 4 5 3
   * 1         7 0 1 5 4 2 3 6
   * 2         5 1 0 6 7 3 2 4
   * 3         1 6 5 0 3 7 4 2
   * 4         2 4 7 3 0 5 6 1
   * 5         4 2 3 7 6 0 1 5
   * 6         5 3 2 4 6 1 0 7
   * 7         3 6 4 2 1 5 7 0
   */
  std::array<std::array<int, 8>, 8> mi300xOamMap = {{{0, 7, 6, 1, 2, 4, 5, 3},
                                                     {7, 0, 1, 5, 4, 2, 3, 6},
                                                     {5, 1, 0, 6, 7, 3, 2, 4},
                                                     {1, 6, 5, 0, 3, 7, 4, 2},
                                                     {2, 4, 7, 3, 0, 5, 6, 1},
                                                     {4, 2, 3, 7, 6, 0, 1, 5},
                                                     {5, 3, 2, 4, 6, 1, 0, 7},
                                                     {3, 6, 4, 2, 1, 5, 7, 0}}};

  // xGMI physical (OAM) id for a KFD node, read from the GPU's PCI sysfs.
  int getOamId(int node);

  int getSdmaEngineId(int srcNode, int dstNode);

  // Bitmask of SDMA engine ids KFD recommends for the src->dst xGMI link to
  // reach maximum bandwidth (sysfs recommended_sdma_engine_id_mask). Returns 0
  // if the link or property is unavailable, in which case callers fall back to
  // the static OAM map.
  uint32_t getRecommendedEngineMask(int srcNode, int dstNode);

  // Bitmask of the general (CPU-link, non-xGMI) SDMA engines. Zero if the node
  // reports no CPU link. Used to spread loopback channels on gfx1250.
  uint32_t getHostLinkEngineMask(int srcNode);

  // True for gfx1250 (gfx12.5), the only arch that spreads loopback channels.
  bool isGfx1250(int node);

  struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
      return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 16);
    }
  };
  using SdmaQueueVector = std::vector<std::unique_ptr<SdmaQueue>>;

  std::once_flag init_flag;
  std::mutex channels_mutex_;
  std::unordered_map<std::pair<int, int>, SdmaQueueVector, PairHash> sdma_channels_;
};

extern AnvilLib& anvil;

inline void checkHipError(hipError_t err, const char* msg, const char* file, int line) {
  if (err != hipSuccess) {
    std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
              << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_HIP_ERROR(cmd) anvil::checkHipError((cmd), #cmd, __FILE__, __LINE__)
// Hardware cap on SDMA channels per GPU pair on CDNA (4 queues/engine ×
// 2 recommended engines). Requests above this are clamped, not failed.
inline constexpr int kMaxSdmaChannelsPerPair = 8;

inline int GetSdmaNumChannels(int defaultVal = 2) {
  const char* env = std::getenv("MORI_SDMA_NUM_CHANNELS");
  if (env != nullptr) {
    int val = std::atoi(env);
    if (val >= 1) return val < kMaxSdmaChannelsPerPair ? val : kMaxSdmaChannelsPerPair;
  }
  return defaultVal;
}

}  // namespace anvil

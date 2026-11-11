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
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mori {
namespace moe {
// GPU-free EP (MORI_EP_SIM_ENABLE=1): dispatch and combine launch no kernel and move no token.
// Each rank draws its tokens' routing from MORI_EP_SIM_ROUTING, and the ranks of the EP group (all
// on this host) exchange it through shared memory, so each rank learns which tokens it receives.
// An op takes a fixed MORI_EP_SIM_DISPATCH_US / MORI_EP_SIM_COMBINE_US and, as with the real
// cross-device barrier, ends on every rank at the latest arrival + latency.
namespace sim {

bool IsEnabled();

// The expert distributions of MORI_EP_SIM_ROUTING, one per MoE layer; dispatch calls cycle through
// the layers.
class Routing {
 public:
  explicit Routing(int numExperts);
  ~Routing();
  // topk distinct experts per token into ids[numTokens * topk], from the layer of dispatch `call`.
  void Draw(uint64_t call, uint64_t seed, int numTokens, int topk, int32_t* ids) const;

 private:
  struct Layer;
  std::vector<Layer> layers_;
};

class EpSim {
 public:
  // `group` names the shared memory and is the same on every rank; "" only when worldSize == 1.
  // asyncSend: AsyncLL, whose send returns before the op ends and whose recv waits for it.
  EpSim(const std::string& group, int rank, int worldSize, int maxTokens, int topk,
        int expertsPerRank, int maxRecv, bool asyncSend);
  ~EpSim();
  EpSim(const EpSim&) = delete;
  EpSim& operator=(const EpSim&) = delete;

  // Marks this rank's arrival at the next Dispatch/Combine. Call it first, so that the caller's own
  // work before the op runs inside the op's latency, as host work overlaps a kernel on a GPU.
  void Arrive();
  // Draws the expert ids of this rank's numTokens tokens from the next layer's distribution and
  // exchanges them with every rank. Writes the ids of the tokens this rank receives to recvIds
  // ([maxRecv, topk], by source rank then token) and their count per local expert to
  // expertCount; returns how many tokens it receives.
  int Dispatch(int numTokens, int32_t* recvIds, int32_t* expertCount);
  // Returns the rows combine gives back: the token count of this rank's last dispatch.
  int Combine();
  // Ends a Dispatch/Combine call: waits for the op, except for an AsyncLL send.
  void EndCall() const {
    if (!asyncSend_) Wait();
  }
  // Sleeps until the last op ends on every rank.
  void Wait() const;

 private:
  struct Slot;
  Slot* SlotOf(uint64_t seq, int rank) const;
  // Begin stamps this rank's arrival into its slot; Publish shares the slot and waits for all.
  Slot* Begin(int phase, double latencyUs, int numTokens);
  void Publish(const Slot* mine);
  // Draws this rank's routing into scratch_: ids, then per destination rank its tokens and hits.
  void Route(int numTokens);

  int rank_, world_, maxTokens_, topk_, expertsPerRank_, maxRecv_;
  bool asyncSend_;
  double dispatchUs_, combineUs_;
  Routing routing_;
  std::vector<int32_t> scratch_;  // the slot's contents, built here and copied in at once
  std::string name_;
  size_t bytes_{0}, slotBytes_{0};
  char* base_{nullptr};
  uint64_t seq_{0}, dispatches_{0};
  int lastTokens_{0};
  int64_t deadlineNs_{0}, arrivalNs_{0};
};

// Runtime mode (MORI_EP_SIM_MODE=runtime, sim_rt.cpp): the op is built and launched as on a GPU,
// on a simulated HIP runtime that calls mori's host handlers for the EP kernels.
namespace rt {

// A simulated runtime is loaded (it exports the symbols sim_rt.cpp looks up).
bool Available();

// One op. Map entries: source = srcRank * srcStride + token; destination = destRank * destStride
// + slot, or world * destStride for a dropped pair.
struct OpSpec {
  int version;  // 1 = EpDispatchCombineOp, 2 = dispatch_combine_v2 (flydsl)
  int rank, world, gpuPerNode, maxTok, topk, expertsPerRank, maxRecv;
  int64_t srcStride, destStride;
  double dispRowBytes, combRowBytes;  // v2 (v1: from the kernel)
  uintptr_t outIdx, srcMap;           // v2 (v1: from the kernel arguments)
  std::vector<uintptr_t> keys;        // device pointers that find the op in its kernels' arguments
  std::string group;                  // the same on every rank, unique per op and job
};
void Register(const OpSpec& spec);
// Counters of every op of this process (JSON); also written to MORI_EP_SIM_STATS_DIR.
std::string Stats();

}  // namespace rt
}  // namespace sim
}  // namespace moe
}  // namespace mori

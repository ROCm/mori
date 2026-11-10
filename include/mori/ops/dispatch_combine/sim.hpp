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

namespace mori {
namespace moe {
// GPU-free EP (MORI_EP_SIM_ENABLE=1): dispatch and combine launch no kernel and move no token.
// The ranks of one EP group, all on this host, exchange their routing through shared memory, so
// each rank learns which tokens it would receive; each op takes a fixed MORI_EP_SIM_DISPATCH_US /
// MORI_EP_SIM_COMBINE_US and, as with the real cross-device barrier, ends on every rank at the
// latest arrival + latency.
namespace sim {

bool IsEnabled();

class EpSim {
 public:
  // `group` names the shared memory and is the same on every rank; "" only when worldSize == 1.
  EpSim(const std::string& group, int rank, int worldSize, int maxTokens, int topk,
        int expertsPerRank, int maxRecv);
  ~EpSim();
  EpSim(const EpSim&) = delete;
  EpSim& operator=(const EpSim&) = delete;

  // ids: [numTokens, topk] global expert ids, -1 = unrouted. Waits for every rank's dispatch, then
  // writes the ids of the tokens this rank receives to recvIds ([maxRecv, topk], by source rank
  // then token) and their count per local expert to expertCount; returns how many it received.
  int Dispatch(const int32_t* ids, int numTokens, int32_t* recvIds, int32_t* expertCount);
  // Waits for every rank's combine.
  void Combine();
  // Sleeps until the last Dispatch/Combine ends on every rank.
  void Wait() const;

 private:
  struct Slot;
  Slot* SlotOf(uint64_t seq, int rank) const;
  // Begin stamps this rank's arrival into its slot; Finish publishes the slot and waits for all.
  Slot* Begin(int phase, double latencyUs, int numTokens);
  void Finish(const Slot* mine);

  int rank_, world_, maxTokens_, topk_, expertsPerRank_, maxRecv_;
  double dispatchUs_, combineUs_;
  std::string name_;
  size_t bytes_{0}, slotBytes_{0};
  char* base_{nullptr};
  uint64_t seq_{0};
  int64_t deadlineNs_{0};
};

}  // namespace sim
}  // namespace moe
}  // namespace mori

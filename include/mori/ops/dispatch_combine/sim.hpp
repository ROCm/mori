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

#include <hip/library_types.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "mori/ops/dispatch_combine/dispatch_combine.hpp"

namespace mori {
namespace moe {

// Hardware-free EP dispatch/combine (MORI_EP_SIM_ENABLE=1). No kernel runs and no token moves:
// each op sleeps the calling thread for a modelled duration. The duration comes from measured
// latency curves (MORI_EP_SIM_CATALOG, entry picked by MORI_EP_SIM_GPU), scaled by this call's
// routing load; ranks of one EP group meet in shared memory so that, as with the real
// cross-device barrier, every rank finishes at max over ranks of (arrival + its own time).
namespace sim {

bool IsEnabled();

enum class Phase : int { Dispatch = 0, Combine = 1 };

// What a call costs on the wire; the model needs nothing else.
struct OpDesc {
  Phase phase{Phase::Dispatch};
  hipDataType dtype{HIP_R_16BF};
  int hiddenDim{0};
  int numTokens{0};
  int scaleBytes{0};     // dispatch: per-token scale payload
  bool zeroCopy{false};  // combine: input is the registered buffer (use_external_inp_buf=0)
  // [numTokens, numExpertPerToken] expert ids readable by the host, -1 = unrouted;
  // nullptr (or MORI_EP_SIM_ROUTING=uniform) = the expected load of uniform routing.
  const int32_t* indices{nullptr};
};

struct Estimate {
  double latencyUs{0};
  double tokens{0};    // the catalog's token axis value the latency was read at
  std::string source;  // catalog entry, or why the analytic model was used
};

struct OpStats {
  uint64_t seq{0};
  Phase phase{Phase::Dispatch};
  double modelUs{0};    // this rank's own duration
  double elapsedUs{0};  // arrival to wake-up, including waiting for the slowest rank
  double loadTokens{0};
  int recvTokens{0};
  std::string source;
};

class EpSim {
 public:
  explicit EpSim(const EpDispatchCombineConfig& config);
  ~EpSim();
  EpSim(const EpSim&) = delete;
  EpSim& operator=(const EpSim&) = delete;

  // Joins the EP group's rendezvous named `group` (all ranks pass the same name) and loads the
  // timing model; `defaultCatalog` is used when MORI_EP_SIM_CATALOG is unset. Ops call
  // Start("", "") themselves when it was not called. join=false only loads the model.
  void Start(const std::string& group, const std::string& defaultCatalog, bool join = true);

  // Post publishes this rank's arrival and load; Wait sleeps until the op ends on all ranks and
  // returns the tokens this rank received (dispatch) or reduced (combine). Run = Post + Wait;
  // the split form is AsyncLL's send/recv pair, so work between the two hides the transfer.
  void Post(const OpDesc& op);
  int Wait();
  int Run(const OpDesc& op) {
    Post(op);
    return Wait();
  }
  bool HasPending() const { return pending_; }
  bool UsesIndices() const { return useIndices_; }

  // Per local expert, tokens received by the last finished dispatch.
  const std::vector<int>& LocalExpertCount() const { return localExpertCount_; }
  const OpStats& LastStats() const { return last_; }
  int LastDispatchTokens() const { return lastDispatchTokens_; }

  // The modelled duration of one op when every rank carries `tokens` tokens with uniform
  // routing, i.e. the catalog's own operating point. For tools and tests; no rendezvous.
  Estimate EstimateUniform(const OpDesc& op) const;

 private:
  struct Model;
  struct Shared;
  struct Posted {
    uint64_t seq{0};
    Phase phase{Phase::Dispatch};
    OpDesc op;
    std::chrono::steady_clock::time_point arrival;
    std::vector<double> rankLoad;    // tokens this rank sends to each rank
    std::vector<double> expertLoad;  // tokens this rank routes to each expert
  };

  void EnsureStarted();
  void CountLoad(const OpDesc& op, std::vector<double>* rankLoad,
                 std::vector<double>* expertLoad) const;
  Estimate Duration(const OpDesc& op, double sendIntra, double sendInter, double recvIntra,
                    double recvInter) const;
  void WriteTrace(const OpStats& s, int64_t arrivalNs, int64_t endNs);

  EpDispatchCombineConfig config_;
  int numExperts_;
  int nodeSize_;
  bool started_{false};
  bool sync_{true};
  bool useIndices_{true};
  double timeScale_{1.0};
  std::chrono::nanoseconds spinTail_{0};
  std::chrono::seconds timeout_{300};

  std::unique_ptr<Model> model_;
  std::unique_ptr<Shared> shared_;

  uint64_t seq_{0};
  bool pending_{false};
  Posted posted_;
  int lastDispatchTokens_{0};
  std::vector<int> localExpertCount_;
  OpStats last_;
  std::FILE* trace_{nullptr};
};

// One simulated op on a sim handle, for the C++ launch API and the Python binding. AsyncLL only
// posts (its recv half calls handle.sim->Wait()); the other kernels also wait. Returns the
// tokens received (dispatch) or reduced (combine), or -1 when only posted.
int SimDispatch(EpDispatchCombineHandle& handle, const int32_t* indices, int numTokens,
                hipDataType dtype, int hiddenDim, bool hasScales);
int SimCombine(EpDispatchCombineHandle& handle, const int32_t* indices, int numTokens,
               hipDataType dtype, int hiddenDim, bool zeroCopy);

}  // namespace sim
}  // namespace moe
}  // namespace mori

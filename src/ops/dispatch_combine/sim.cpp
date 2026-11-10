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
#include "mori/ops/dispatch_combine/sim.hpp"

#include <sys/prctl.h>

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

#include "mori/utils/env_utils.hpp"

namespace mori {
namespace moe {
namespace sim {

namespace {

using Clock = std::chrono::steady_clock;

double LatencyUs(const char* name) {
  const char* raw = std::getenv(name);
  char* end = nullptr;
  const double us = raw ? std::strtod(raw, &end) : -1;
  if (raw == nullptr || end == raw || *end != '\0' || !(us >= 0)) {
    throw std::runtime_error(std::string("MORI_EP_SIM_ENABLE=1 needs ") + name +
                             " set to a latency in microseconds");
  }
  return us;
}

}  // namespace

bool IsEnabled() { return env::IsEnvVarEnabled("MORI_EP_SIM_ENABLE"); }

int64_t Deadline(Phase phase) {
  const double us =
      LatencyUs(phase == Phase::Dispatch ? "MORI_EP_SIM_DISPATCH_US" : "MORI_EP_SIM_COMBINE_US");
  const auto end = Clock::now() + std::chrono::nanoseconds(static_cast<int64_t>(us * 1e3));
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
}

void SleepUntil(int64_t deadlineNs) {
  // The default 50 us timer slack, and the scheduler's wake-up delay, would dominate a 30 us op:
  // sleep with 1 ns slack to 20 us short of the deadline, then spin.
  thread_local const bool fineSlack = (prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL), true);
  (void)fineSlack;
  constexpr auto kSpinTail = std::chrono::microseconds(20);
  const Clock::time_point end{std::chrono::nanoseconds(deadlineNs)};
  if (end - Clock::now() > kSpinTail) std::this_thread::sleep_until(end - kSpinTail);
  while (Clock::now() < end) {
  }
}

}  // namespace sim
}  // namespace moe
}  // namespace mori

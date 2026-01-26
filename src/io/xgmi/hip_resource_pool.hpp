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

#include <hip/hip_runtime.h>

#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

namespace mori {
namespace io {

class StreamPool {
 public:
  explicit StreamPool(int numStreamsPerDevice = 64);
  ~StreamPool();

  StreamPool(const StreamPool&) = delete;
  StreamPool& operator=(const StreamPool&) = delete;
  StreamPool(StreamPool&&) = delete;
  StreamPool& operator=(StreamPool&&) = delete;

  hipStream_t GetNextStream(int deviceId);

 private:
  bool InitializeStreamsForDevice(int deviceId);

  int numStreamsPerDevice_;
  std::mutex mutex_;
  std::unordered_map<int, std::vector<hipStream_t>> streams_;
  std::unordered_map<int, int> currentStreamIdx_;
};

class EventPool {
 public:
  explicit EventPool(int numEventsPerDevice = 64);
  ~EventPool();

  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;
  EventPool(EventPool&&) = delete;
  EventPool& operator=(EventPool&&) = delete;

  hipEvent_t GetEvent(int deviceId);
  void PutEvent(hipEvent_t event, int deviceId);

 private:
  hipEvent_t CreateEvent();
  hipEvent_t CreateEventForDevice(int deviceId);
  bool InitializeEventsForDevice(int deviceId);

  int numEventsPerDevice_;
  std::mutex mutex_;
  std::unordered_map<int, std::queue<hipEvent_t>> eventPools_;
};

}  // namespace io
}  // namespace mori

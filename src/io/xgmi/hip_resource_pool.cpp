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
#include "src/io/xgmi/hip_resource_pool.hpp"

#include "mori/io/logging.hpp"

namespace mori {
namespace io {

/* ---------------------------------------------------------------------------------------------- */
/*                                           StreamPool                                           */
/* ---------------------------------------------------------------------------------------------- */

StreamPool::StreamPool(int numStreamsPerDevice) : numStreamsPerDevice_(numStreamsPerDevice) {
  if (numStreamsPerDevice_ <= 0) {
    numStreamsPerDevice_ = 64;
  }
}

StreamPool::~StreamPool() {
  for (auto& deviceEntry : streams_) {
    for (auto stream : deviceEntry.second) {
      if (stream != nullptr) {
        hipStreamDestroy(stream);
      }
    }
  }
}

hipStream_t StreamPool::GetNextStream(int deviceId) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (streams_.find(deviceId) == streams_.end()) {
    if (!InitializeStreamsForDevice(deviceId)) {
      return nullptr;
    }
  }

  auto& deviceStreams = streams_[deviceId];
  if (deviceStreams.empty()) {
    return nullptr;
  }

  auto& currentIdx = currentStreamIdx_[deviceId];
  hipStream_t stream = deviceStreams[currentIdx];
  currentIdx = (currentIdx + 1) % static_cast<int>(deviceStreams.size());

  return stream;
}

bool StreamPool::InitializeStreamsForDevice(int deviceId) {
  hipError_t err = hipSetDevice(deviceId);
  if (err != hipSuccess) {
    MORI_IO_ERROR("StreamPool: Failed to set device {}: {}", deviceId, hipGetErrorString(err));
    return false;
  }

  std::vector<hipStream_t> deviceStreams;
  deviceStreams.reserve(numStreamsPerDevice_);

  for (int i = 0; i < numStreamsPerDevice_; ++i) {
    hipStream_t stream;
    err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (err != hipSuccess) {
      MORI_IO_ERROR("StreamPool: Failed to create stream for device {}: {}", deviceId,
                    hipGetErrorString(err));
      for (auto s : deviceStreams) {
        hipStreamDestroy(s);
      }
      return false;
    }
    deviceStreams.push_back(stream);
  }

  streams_[deviceId] = std::move(deviceStreams);
  currentStreamIdx_[deviceId] = 0;
  MORI_IO_TRACE("StreamPool: Initialized {} streams for device {}", numStreamsPerDevice_, deviceId);
  return true;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                            EventPool                                           */
/* ---------------------------------------------------------------------------------------------- */

EventPool::EventPool(int numEventsPerDevice) : numEventsPerDevice_(numEventsPerDevice) {
  if (numEventsPerDevice_ <= 0) {
    numEventsPerDevice_ = 64;
  }
}

EventPool::~EventPool() {
  for (auto& deviceEntry : eventPools_) {
    while (!deviceEntry.second.empty()) {
      hipEvent_t event = deviceEntry.second.front();
      deviceEntry.second.pop();
      if (event != nullptr) {
        hipEventDestroy(event);
      }
    }
  }
}

hipEvent_t EventPool::GetEvent(int deviceId) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (eventPools_.find(deviceId) == eventPools_.end()) {
    eventPools_[deviceId] = std::queue<hipEvent_t>();
    if (!InitializeEventsForDevice(deviceId)) {
      return nullptr;
    }
  }

  auto& pool = eventPools_[deviceId];

  if (pool.empty()) {
    return CreateEventForDevice(deviceId);
  }

  hipEvent_t event = pool.front();
  pool.pop();
  return event;
}

void EventPool::PutEvent(hipEvent_t event, int deviceId) {
  if (event == nullptr) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  eventPools_[deviceId].push(event);
}

hipEvent_t EventPool::CreateEvent() {
  hipEvent_t event;
  hipError_t err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
  if (err != hipSuccess) {
    MORI_IO_ERROR("EventPool: Failed to create event: {}", hipGetErrorString(err));
    return nullptr;
  }
  return event;
}

hipEvent_t EventPool::CreateEventForDevice(int deviceId) {
  hipError_t err = hipSetDevice(deviceId);
  if (err != hipSuccess) {
    MORI_IO_ERROR("EventPool: Failed to set device {}: {}", deviceId, hipGetErrorString(err));
    return nullptr;
  }

  return CreateEvent();
}

bool EventPool::InitializeEventsForDevice(int deviceId) {
  hipError_t err = hipSetDevice(deviceId);
  if (err != hipSuccess) {
    MORI_IO_ERROR("EventPool: Failed to set device {}: {}", deviceId, hipGetErrorString(err));
    return false;
  }

  auto& pool = eventPools_[deviceId];
  for (int i = 0; i < numEventsPerDevice_; ++i) {
    hipEvent_t event = CreateEvent();
    if (event == nullptr) {
      return false;
    }
    pool.push(event);
  }
  MORI_IO_TRACE("EventPool: Initialized {} events for device {}", numEventsPerDevice_, deviceId);
  return true;
}

}  // namespace io
}  // namespace mori

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
// CPU-only HIP stand-ins: no device or provider is initialized.
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>

#include "src/io/xgmi/transfer_completion.hpp"

using namespace mori::io;
using namespace std::chrono_literals;

namespace {
struct Event {
  bool recorded{false};
  hipError_t query{hipErrorNotReady};
};
hipError_t recordResult = hipSuccess, streamResult = hipErrorNotReady;
int unrecordedQueries = 0, freedMetadata = 0, destroyedEvents = 0, records = 0;
int currentDevice = 0;
Event* newestEvent = nullptr;
std::promise<void>* creatingEvent = nullptr;
std::shared_future<void> releaseCreate;
std::promise<void>* synchronizing = nullptr;
std::shared_future<void> releaseSynchronize;
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

extern "C" {
hipError_t hipGetDevice(int* device) {
  *device = currentDevice;
  return hipSuccess;
}
hipError_t hipSetDevice(int device) {
  currentDevice = device;
  return hipSuccess;
}
const char* hipGetErrorString(hipError_t) { return "stub HIP error"; }
hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned) {
  if (creatingEvent) {
    creatingEvent->set_value();
    releaseCreate.wait();
  }
  newestEvent = new Event;
  *event = reinterpret_cast<hipEvent_t>(newestEvent);
  return hipSuccess;
}
hipError_t hipEventDestroy(hipEvent_t event) {
  ++destroyedEvents;
  delete reinterpret_cast<Event*>(event);
  return hipSuccess;
}
hipError_t hipEventRecord(hipEvent_t event, hipStream_t) {
  ++records;
  if (recordResult == hipSuccess) reinterpret_cast<Event*>(event)->recorded = true;
  return recordResult;
}
hipError_t hipEventQuery(hipEvent_t event) {
  auto* state = reinterpret_cast<Event*>(event);
  if (!state->recorded) {
    ++unrecordedQueries;
    return hipSuccess;
  }
  return state->query;
}
hipError_t hipEventSynchronize(hipEvent_t event) {
  if (synchronizing) {
    synchronizing->set_value();
    releaseSynchronize.wait();
  }
  return reinterpret_cast<Event*>(event)->query;
}
hipError_t hipStreamQuery(hipStream_t) { return streamResult; }
hipError_t hipGetLastError() { return hipSuccess; }
hipError_t hipFree(void* address) {
  ++freedMetadata;
  std::free(address);
  return hipSuccess;
}
}

namespace {
struct Fixture {
  std::unique_ptr<TransferStatus> status = std::make_unique<TransferStatus>();
  std::shared_ptr<EventPool> events = std::make_shared<EventPool>(1);
  std::shared_ptr<StreamPool> streams = std::make_shared<StreamPool>(1);
  std::shared_ptr<std::atomic<size_t>> pending = std::make_shared<std::atomic<size_t>>(0);
  std::shared_ptr<TransferCompletion> operation;
  Fixture() {
    recordResult = hipSuccess;
    streamResult = hipErrorNotReady;
    unrecordedQueries = freedMetadata = records = 0;
    operation = TransferCompletion::Begin(status.get(), events, streams, pending);
    operation->stream = reinterpret_cast<hipStream_t>(1);
    operation->eventDevice = 0;
    operation->event = events->GetEvent(0);
  }
  void Submitted() {
    operation->submitted = true;
    operation->deviceMeta = std::malloc(8);
    operation->hostMeta.resize(8);
  }
  void Fail() {
    status->Update(StatusCode::ERR_GPU_OP, "queued failure");
    operation->Seal();
  }
};

void SuccessAndReuse() {
  Fixture f;
  f.Submitted();
  Require(hipEventRecord(f.operation->event, f.operation->stream) == hipSuccess,
          "event recording must succeed");
  reinterpret_cast<Event*>(f.operation->event)->query = hipSuccess;
  f.operation->eventRecorded = true;
  f.operation->Seal();
  Require(f.status->WaitQuiesced(10) && f.status->Succeeded(),
          "successful event proves completion");
  Require(f.pending->load() == 0 && freedMetadata == 1, "successful resources recycled once");
  f.status->SetCode(StatusCode::INIT);
  Require(f.status->Quiesced(), "INIT preserves certified success");
}

void PreSubmissionFailure() {
  Fixture f;
  f.Fail();
  Require(f.status->Quiesced() && f.status->Failed(), "pre-submit failure is quiesced");
  Require(records == 0 && f.pending->load() == 0, "pre-submit failure needs no fence");
}

void QueuedFailureAndFreshFence() {
  Fixture f;
  f.Submitted();
  auto original = f.operation->event;
  f.Fail();
  Require(f.status->Failed() && freedMetadata == 0 && f.pending->load() == 1,
          "failure publishes before drain and retains metadata");
  Require(reinterpret_cast<hipEvent_t>(newestEvent) != original && records == 1,
          "failure records a separate fence");
  Require(f.status->WaitQuiescence(1) == QuiescenceState::Pending && unrecordedQueries == 0,
          "never query the unrecorded original event");
  newestEvent->query = hipSuccess;
  Require(f.status->WaitQuiesced(10) && freedMetadata == 1 && f.pending->load() == 0,
          "fresh fence evidence releases queued failure resources");
}

void FenceRecordFailureUsesStream() {
  Fixture f;
  f.Submitted();
  recordResult = hipErrorInvalidValue;
  f.Fail();
  Require(f.status->Quiescence() == QuiescenceState::Pending && unrecordedQueries == 0,
          "failed fence record uses stream query only");
  streamResult = hipSuccess;
  Require(f.status->Quiesced() && freedMetadata == 1, "stream evidence releases resources");
}

void QueryErrorRetainsUntilExit() {
  Fixture f;
  f.Submitted();
  recordResult = hipErrorInvalidValue;
  f.Fail();
  streamResult = hipErrorInvalidValue;
  Require(f.status->Quiescence() == QuiescenceState::Unprovable, "sticky error is unprovable");
  std::weak_ptr<TransferCompletion> operation = f.operation;
  std::weak_ptr<EventPool> pool = f.events;
  f.status.reset();
  f.operation.reset();
  f.events.reset();
  f.streams.reset();
  Require(!operation.expired() && !pool.expired() && freedMetadata == 0 && f.pending->load() == 1,
          "unprovable resources survive status and engine ownership");
}

void PendingSurvivesStatusDestruction() {
  Fixture f;
  f.Submitted();
  f.Fail();
  std::weak_ptr<TransferCompletion> operation = f.operation;
  f.status.reset();
  f.operation.reset();
  f.events.reset();
  f.streams.reset();
  Require(!operation.expired() && f.pending->load() == 1 && freedMetadata == 0,
          "unqueried failure retained to process exit");
}

void WaitAndQueryContention() {
  Fixture f;
  f.Submitted();
  Require(hipEventRecord(f.operation->event, f.operation->stream) == hipSuccess,
          "event recording must succeed");
  reinterpret_cast<Event*>(f.operation->event)->query = hipSuccess;
  f.operation->eventRecorded = true;
  f.operation->Seal();
  std::promise<void> entered, resume;
  synchronizing = &entered;
  releaseSynchronize = resume.get_future().share();
  auto waiter = std::async(std::launch::async, [&] { f.operation->FinalizeBlocking(); });
  entered.get_future().wait();
  auto start = std::chrono::steady_clock::now();
  auto proof = f.status->Quiescence();
  auto elapsed = std::chrono::steady_clock::now() - start;
  resume.set_value();
  waiter.get();
  synchronizing = nullptr;
  Require(proof == QuiescenceState::Pending && elapsed < 100ms, "query must not wait on eventMu");
  Require(f.status->Quiesced(), "blocking completion certifies exactly once");
}

void QueryDoesNotWaitOnPool() {
  Fixture f;
  f.Submitted();
  Require(hipEventRecord(f.operation->event, f.operation->stream) == hipSuccess,
          "event recording must succeed");
  reinterpret_cast<Event*>(f.operation->event)->query = hipSuccess;
  f.operation->eventRecorded = true;
  f.operation->Seal();
  std::promise<void> entered, resume;
  creatingEvent = &entered;
  releaseCreate = resume.get_future().share();
  auto creator = std::async(std::launch::async, [&] { return f.events->GetEvent(0); });
  entered.get_future().wait();
  auto start = std::chrono::steady_clock::now();
  auto proof = f.status->Quiescence();
  auto elapsed = std::chrono::steady_clock::now() - start;
  resume.set_value();
  auto spare = creator.get();
  creatingEvent = nullptr;
  f.events->PutEvent(spare, 0);
  Require(proof == QuiescenceState::Pending && elapsed < 100ms,
          "query must not wait on pool mutex");
  Require(f.status->Quiesced() && freedMetadata == 1, "pool contention retries cleanup once");
}

void ResultOnlyBackends() {
  TransferStatus status;
  status.TrackResultQuiescence();
  status.SetCode(StatusCode::SUCCESS);
  Require(!status.Quiesced(), "public SUCCESS cannot manufacture completion evidence");
  status.SetCode(StatusCode::INIT);
  Require(!status.Quiesced(), "public INIT cannot erase an incomplete operation");
  status.SetResultCode(StatusCode::SUCCESS);
  status.SetCode(StatusCode::INIT);
  Require(status.Quiesced(), "result proof survives INIT before first query");
  status.TrackResultQuiescence(true);
  status.Update(StatusCode::ERR_RDMA_OP, "external failure");
  Require(!status.Quiesced(), "public Update cannot settle a backend operation");
  status.UpdateResult(StatusCode::ERR_RDMA_OP, "simulated failure");
  Require(status.Quiesced(), "SIM never accesses payload, including failures");
  status.TrackResultQuiescence();
  status.UpdateResult(StatusCode::ERR_GPU_OP, "unsupported failure");
  Require(status.Quiescence() == QuiescenceState::Unprovable, "unaudited errors are unprovable");
}
}  // namespace

int main() {
  try {
    SuccessAndReuse();
    PreSubmissionFailure();
    QueuedFailureAndFreshFence();
    FenceRecordFailureUsesStream();
    QueryErrorRetainsUntilExit();
    PendingSurvivesStatusDestruction();
    WaitAndQueryContention();
    QueryDoesNotWaitOnPool();
    ResultOnlyBackends();
    std::cout << "PASS: 9 XGMI/result quiescence CPU cases\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}

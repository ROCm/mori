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
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include "mori/io/backend.hpp"
#include "mori/io/engine.hpp"
#include "mori/io/logging.hpp"

using namespace mori::io;

namespace {
inline void CheckHip(hipError_t e, const char* msg) {
  if (e != hipSuccess) {
    std::cerr << "HIP error (" << msg << "): " << hipGetErrorString(e) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

bool WaitForCompletion(TransferStatus* st, int maxMs = 3000, int sleepUs = 1000) {
  // Spin/poll; backend updates status asynchronously.
  int waited = 0;
  while (st->Code() == StatusCode::IN_PROGRESS && waited < maxMs * 1000) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleepUs));
    waited += sleepUs;
  }
  return st->Code() == StatusCode::SUCCESS;
}
}  // namespace

int main(int argc, char** argv) {
  SetLogLevel("info");

  // Allow overriding ports by CLI: tcp_example [portA] [portB]
  uint16_t portA = 34567;
  uint16_t portB = 34568;
  if (argc > 1) portA = static_cast<uint16_t>(std::stoi(argv[1]));
  if (argc > 2) portB = static_cast<uint16_t>(std::stoi(argv[2]));

  IOEngineConfig cfgA{"127.0.0.1", portA};
  IOEngineConfig cfgB{"127.0.0.1", portB};
  IOEngine engineA("engineA", cfgA);
  IOEngine engineB("engineB", cfgB);

  TcpBackendConfig tcfg;  // default: 4 worker threads, preconnect true
  engineA.CreateBackend(BackendType::TCP, tcfg);
  engineB.CreateBackend(BackendType::TCP, tcfg);

  EngineDesc descA = engineA.GetEngineDesc();
  EngineDesc descB = engineB.GetEngineDesc();
  engineA.RegisterRemoteEngine(descB);
  engineB.RegisterRemoteEngine(descA);

  const size_t bufSize = 1024;
  std::vector<char> hostA(bufSize, 0), hostB(bufSize, 0), hostRoundtrip(bufSize, 0);
  const char* msg = "Hello TCP Backend";
  std::memcpy(hostA.data(), msg, std::strlen(msg) + 1);

  void* devA = nullptr;
  void* devB = nullptr;
  void* devRoundtrip = nullptr;  // for read-back into engineA after remote write
  CheckHip(hipMalloc(&devA, bufSize), "hipMalloc devA");
  CheckHip(hipMalloc(&devB, bufSize), "hipMalloc devB");
  CheckHip(hipMalloc(&devRoundtrip, bufSize), "hipMalloc devRoundtrip");
  CheckHip(hipMemset(devB, 0, bufSize), "hipMemset devB");
  CheckHip(hipMemset(devRoundtrip, 0, bufSize), "hipMemset devRoundtrip");
  CheckHip(hipMemcpy(devA, hostA.data(), bufSize, hipMemcpyHostToDevice), "H2D copy devA");

  auto memA = engineA.RegisterMemory(devA, bufSize, 0, MemoryLocationType::GPU);
  auto memB = engineB.RegisterMemory(devB, bufSize, 0, MemoryLocationType::GPU);
  auto memRoundtrip = engineA.RegisterMemory(devRoundtrip, bufSize, 0, MemoryLocationType::GPU);

  // Demonstrate WRITE: engineA -> engineB (memA to memB)
  TransferStatus writeStatus;
  auto writeTid = engineA.AllocateTransferUniqueId();
  writeStatus.SetCode(StatusCode::IN_PROGRESS);
  engineA.Write(memA, 0, memB, 0, std::strlen(msg) + 1, &writeStatus, writeTid);

  // Define cleanup lambda after resources & before any early returns
  auto cleanup = [&]() {
    engineA.DeregisterMemory(memA);
    engineA.DeregisterMemory(memRoundtrip);
    engineB.DeregisterMemory(memB);
    CheckHip(hipFree(devA), "hipFree devA");
    CheckHip(hipFree(devB), "hipFree devB");
    CheckHip(hipFree(devRoundtrip), "hipFree devRoundtrip");
  };

  if (!WaitForCompletion(&writeStatus)) {
    std::cerr << "Write did not complete successfully. Code=" << (unsigned)writeStatus.Code()
              << " Message=" << writeStatus.Message() << std::endl;
    cleanup();
    return 1;
  }

  CheckHip(hipMemcpy(hostB.data(), devB, bufSize, hipMemcpyDeviceToHost), "D2H copy devB");
  std::cout << "[After Write] EngineB buffer: '" << hostB.data() << "'" << std::endl;

  // Demonstrate READ: engineA reads back from engineB's memB into its own memRoundtrip
  TransferStatus readStatus;
  auto readTid = engineA.AllocateTransferUniqueId();
  readStatus.SetCode(StatusCode::IN_PROGRESS);
  engineA.Read(memRoundtrip, 0, memB, 0, std::strlen(msg) + 1, &readStatus, readTid);
  if (!WaitForCompletion(&readStatus)) {
    std::cerr << "Read did not complete successfully. Code=" << (unsigned)readStatus.Code()
              << " Message=" << readStatus.Message() << std::endl;
    cleanup();
    return 1;
  }

  CheckHip(hipMemcpy(hostRoundtrip.data(), devRoundtrip, bufSize, hipMemcpyDeviceToHost),
           "D2H copy devRoundtrip");
  std::cout << "[After Read] EngineA roundtrip buffer: '" << hostRoundtrip.data() << "'";
  if (std::string(hostRoundtrip.data()) == msg)
    std::cout << " (validation OK)";
  else
    std::cout << " (validation FAILED)";
  std::cout << std::endl;

  if (auto sessOpt = engineA.CreateSession(memA, memB)) {
    auto& sess = *sessOpt;
    TransferStatus sessWriteStatus;
    auto sessTid = sess.AllocateTransferUniqueId();
    sessWriteStatus.SetCode(StatusCode::IN_PROGRESS);
    const char* msg2 = "SessionWrite";
    size_t len2 = std::strlen(msg2) + 1;
    // Overwrite beginning of memA, then write to remote
    std::memset(hostA.data(), 0, bufSize);
    std::memcpy(hostA.data(), msg2, len2);
    CheckHip(hipMemcpy(devA, hostA.data(), bufSize, hipMemcpyHostToDevice), "H2D copy devA msg2");
    sess.Write(0, 0, len2, &sessWriteStatus, sessTid);
    if (WaitForCompletion(&sessWriteStatus)) {
      CheckHip(hipMemcpy(hostB.data(), devB, bufSize, hipMemcpyDeviceToHost), "D2H copy devB2");
      std::cout << "[Session Write] EngineB buffer now: '" << hostB.data() << "'" << std::endl;
    } else {
      std::cout << "[Session Write] Incomplete status code=" << (unsigned)sessWriteStatus.Code()
                << std::endl;
    }
  } else {
    std::cout << "Session creation not available / failed (continuing)." << std::endl;
  }

  cleanup();
  return 0;
}

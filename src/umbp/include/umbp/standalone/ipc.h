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
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace mori::umbp::standalone {

constexpr uint32_t kFdRegistrationMagic = 0x554d4250;  // "UMBP"
constexpr uint32_t kFdRegistrationVersion = 1;
constexpr size_t kClientIdBytes = 64;

struct FdRegistrationMessage {
  uint32_t magic = kFdRegistrationMagic;
  uint32_t version = kFdRegistrationVersion;
  char client_id[kClientIdBytes] = {};
  uint64_t worker_base = 0;
  uint64_t size = 0;
};

std::string UnixPathFromGrpcAddress(const std::string& address);
std::string DeriveFdSocketPath(const std::string& grpc_address);
std::string DefaultStandaloneAddress();

bool EnsureParentDirectory(const std::string& path, std::string* error = nullptr);

int SendFdRegistration(const std::string& socket_path, int fd, const std::string& client_id,
                       uintptr_t worker_base, size_t size, int timeout_ms, std::string* error);
int RecvFdRegistration(int socket_fd, FdRegistrationMessage* message, std::string* error);

bool SendStatus(int socket_fd, int32_t status, std::string* error = nullptr);
bool RecvStatus(int socket_fd, int32_t* status, std::string* error = nullptr);

}  // namespace mori::umbp::standalone

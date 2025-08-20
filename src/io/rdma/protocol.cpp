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
#include "src/io/rdma/protocol.hpp"

#include <msgpack.hpp>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace io {

Protocol::Protocol(application::TCPEndpointHandle eph) : ep(eph) {}

Protocol::~Protocol() {}

MessageHeader Protocol::ReadMessageHeader() {
  MessageHeader hdr;
  SYSCALL_RETURN_ZERO(ep.Recv(&hdr.type, sizeof(hdr.type)));
  SYSCALL_RETURN_ZERO(ep.Recv(&hdr.len, sizeof(hdr.len)));
  hdr.len = ntohl(hdr.len);
  return hdr;
}

void Protocol::WriteMessageHeader(const MessageHeader& hdr) {
  SYSCALL_RETURN_ZERO(ep.Send(&hdr.type, sizeof(hdr.type)));
  uint32_t len = htonl(hdr.len);
  SYSCALL_RETURN_ZERO(ep.Send(&len, sizeof(len)));
}

MessageRegEndpoint Protocol::ReadMessageRegEndpoint(size_t len) {
  std::vector<char> buf(len);
  SYSCALL_RETURN_ZERO(ep.Recv(buf.data(), len));
  auto out = msgpack::unpack(buf.data(), len);
  return out.get().as<MessageRegEndpoint>();
}

void Protocol::WriteMessageRegEndpoint(const MessageRegEndpoint& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::RegEndpoint, len});
  SYSCALL_RETURN_ZERO(ep.Send(buf.data(), buf.size()));
}

MessageAskMemoryRegion Protocol::ReadMessageAskMemoryRegion(size_t len) {
  std::vector<char> buf(len);
  SYSCALL_RETURN_ZERO(ep.Recv(buf.data(), len));
  auto out = msgpack::unpack(buf.data(), len);
  return out.get().as<MessageAskMemoryRegion>();
}

void Protocol::WriteMessageAskMemoryRegion(const MessageAskMemoryRegion& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::AskMemoryRegion, len});
  SYSCALL_RETURN_ZERO(ep.Send(buf.data(), buf.size()));
}

}  // namespace io
}  // namespace mori

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

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include "mori/application/utils/check.hpp"

namespace mori {
namespace io {

namespace {

// A peer that disconnects must not take the ENGINE down with it.
//
// Every socket call in this file used to go through SYSCALL_RETURN_ZERO
// (`check.hpp:56`), which prints and then `exit(-1)`s the whole process. That
// was survivable only while TCPEndpoint::Recv could not report an error:
// before `99768347` it stored recv()'s return in a `size_t`, so `n < 0` was
// dead and a mid-message EOF came back as 0 == "success". Fixing that made the
// error path REACHABLE -- and reachable meant `exit(-1)`, i.e. a peer closing
// its connection at the wrong moment kills the local inference engine.
//
// It also silently defeated `486a1b53`. That commit wrapped
// HandleControlPlaneProtocol in try/catch precisely so one bad connection
// costs that CONNECTION and not the engine, but `exit(-1)` is not catchable,
// so the handler could never fire on the EOF/error path -- which is the
// teardown/reconnect path a PD role switch actually runs, since a flip tears
// down and re-establishes the peer's endpoints.
//
// So: throw. `486a1b53`'s handler already turns a throw into
// DropControlPlaneConn(fd), which is the intended behaviour, and it is the
// same channel msgpack::unpack_error and std::length_error already use from
// the lines just below these calls.
void CheckSyscall(int ret, const char* what) {
  if (ret == 0) return;
  int err = errno;
  throw std::runtime_error(std::string("mori::io control-plane socket ") + what +
                           " failed: " + std::strerror(err) +
                           " (errno=" + std::to_string(err) +
                           "); dropping this connection");
}

}  // namespace

Protocol::Protocol(application::TCPEndpointHandle eph) : ep(eph) {}

Protocol::~Protocol() {}

MessageHeader Protocol::ReadMessageHeader() {
  MessageHeader hdr;
  CheckSyscall(ep.Recv(&hdr.type, sizeof(hdr.type)), "recv");
  CheckSyscall(ep.Recv(&hdr.len, sizeof(hdr.len)), "recv");
  hdr.len = ntohl(hdr.len);
  return hdr;
}

void Protocol::WriteMessageHeader(const MessageHeader& hdr) {
  CheckSyscall(ep.Send(&hdr.type, sizeof(hdr.type)), "send");
  uint32_t len = htonl(hdr.len);
  CheckSyscall(ep.Send(&len, sizeof(len)), "send");
}

MessageRegEndpoint Protocol::ReadMessageRegEndpoint(size_t len) {
  std::vector<char> buf(len);
  CheckSyscall(ep.Recv(buf.data(), len), "recv");
  auto out = msgpack::unpack(buf.data(), len);
  return out.get().as<MessageRegEndpoint>();
}

void Protocol::WriteMessageRegEndpoint(const MessageRegEndpoint& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::RegEndpoint, len});
  CheckSyscall(ep.Send(buf.data(), buf.size()), "send");
}

MessageAskMemoryRegion Protocol::ReadMessageAskMemoryRegion(size_t len) {
  std::vector<char> buf(len);
  CheckSyscall(ep.Recv(buf.data(), len), "recv");
  auto out = msgpack::unpack(buf.data(), len);
  return out.get().as<MessageAskMemoryRegion>();
}

void Protocol::WriteMessageAskMemoryRegion(const MessageAskMemoryRegion& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::AskMemoryRegion, len});
  CheckSyscall(ep.Send(buf.data(), buf.size()), "send");
}

}  // namespace io
}  // namespace mori

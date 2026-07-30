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

// Largest control-plane message we will allocate for.
//
// Every message on this channel is a msgpack of a handful of ints plus an
// EngineKey and one RdmaEndpointHandle / RdmaMemoryRegion -- hundreds of
// bytes. 1 MiB is four orders of magnitude of headroom and still bounds the
// allocation to something a control-plane thread can absorb.
constexpr uint32_t kMaxControlMessageBytes = 1u << 20;

MessageHeader Protocol::ReadMessageHeader() {
  MessageHeader hdr;
  CheckSyscall(ep.Recv(&hdr.type, sizeof(hdr.type)), "recv");
  CheckSyscall(ep.Recv(&hdr.len, sizeof(hdr.len)), "recv");
  hdr.len = ntohl(hdr.len);

  // VALIDATE THE HEADER HERE, where both the server loop and the two client
  // call sites go through, rather than at the call sites.
  //
  // `486a1b53`'s own message named the unbounded `hdr.len` as the root hazard
  // and `3c6bc1a1` reworked this file without fixing it: `len` came straight
  // off the wire into `std::vector<char> buf(len)` two functions below, so a
  // truncated, byte-swapped or garbage header asks for up to 4 GiB. The type
  // was checked only by `assert(hdr.type == ...)` at the call sites.
  // CORRECTION (review #64-1): that assert was NOT compiled out -- this project
  // does not build with -DNDEBUG (CMakeLists.txt:4 is a bare "-O3"; 0 of 49
  // compile_commands.json entries carry NDEBUG; `__assert_fail` is in the
  // shipped backend_impl.cpp.o). So the check was live and `abort()`ed the
  // process rather than being absent. The length bound below is unaffected by
  // that correction -- nothing ever bounded `len`.
  //
  // This is not a hypothetical peer being hostile. A PD role switch tears the
  // control plane down and re-establishes it on every flip, so a half-written
  // header from a peer that flipped mid-message is the ordinary case on the
  // path this campaign exists to exercise.
  //
  // Throwing is the established channel: on the server thread `486a1b53`'s
  // handler turns it into DropControlPlaneConn(fd); on the two client call
  // sites (BuildRdmaConn, AskRemoteMemoryRegion) it unwinds through the
  // TcpEndpointGuard, which closes the fd -- see COORD for what that thread
  // does with it, since there is no handler on that chain yet.
  if (hdr.type != MessageType::RegEndpoint && hdr.type != MessageType::AskMemoryRegion) {
    throw std::runtime_error(
        "mori::io control-plane: unknown message type " +
        std::to_string(static_cast<unsigned>(hdr.type)) + "; dropping this connection");
  }
  if (hdr.len > kMaxControlMessageBytes) {
    throw std::runtime_error(
        "mori::io control-plane: message length " + std::to_string(hdr.len) +
        " exceeds the maximum of " + std::to_string(kMaxControlMessageBytes) +
        " bytes; refusing to allocate, dropping this connection");
  }
  return hdr;
}

MessageHeader Protocol::ReadMessageHeader(MessageType expected) {
  // Read + validate as usual (unknown type, oversized len), THEN pin the type.
  // Ordering matters: an unknown type must still report as unknown rather than
  // as a mismatch, because the two mean different things to whoever reads the
  // log -- unknown is a garbage/desynchronized stream, mismatch is a peer that
  // is talking a protocol we understand at the wrong moment.
  MessageHeader hdr = ReadMessageHeader();
  if (hdr.type != expected) {
    throw std::runtime_error(
        "mori::io control-plane: expected message type " +
        std::to_string(static_cast<unsigned>(expected)) + " but the peer sent " +
        std::to_string(static_cast<unsigned>(hdr.type)) +
        "; refusing to unpack one message as another, dropping this connection");
  }
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

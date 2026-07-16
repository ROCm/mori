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

#include <cerrno>
#include <cstring>
#include <msgpack.hpp>
#include <sstream>

namespace mori {
namespace io {
namespace {

void ThrowRecvError(const char* op, size_t expected, size_t received, int rc, int err) {
  std::ostringstream os;
  os << op << " failed: expected " << expected << " byte(s), received " << received;
  if (rc > 0) {
    os << " before EOF";
  } else {
    os << ", errno=" << err << " (" << strerror(err) << ")";
  }
  throw ProtocolError(os.str());
}

void ReadExact(application::TCPEndpoint* ep, void* buf, size_t len, const char* op) {
  size_t consumed = 0;
  int rc = ep->Recv(buf, len, &consumed);
  int err = errno;
  if (rc != 0) ThrowRecvError(op, len, consumed, rc, err);
}

void WriteExact(application::TCPEndpoint* ep, const void* buf, size_t len, const char* op) {
  if (ep->Send(buf, len) != 0) {
    int err = errno;
    std::ostringstream os;
    os << op << " failed while writing " << len << " byte(s): errno=" << err << " ("
       << strerror(err) << ")";
    throw ProtocolError(os.str());
  }
}

void ValidateControlMessageLength(size_t len, const char* messageName) {
  if (len > kMaxControlMessageBytes) {
    std::ostringstream os;
    os << messageName << " length " << len << " exceeds control-plane limit "
       << kMaxControlMessageBytes;
    throw ProtocolError(os.str());
  }
}

}  // namespace

const char* MessageTypeName(MessageType type) {
  switch (type) {
    case MessageType::RegEndpoint:
      return "RegEndpoint";
    case MessageType::AskMemoryRegion:
      return "AskMemoryRegion";
    default:
      return "Unknown";
  }
}

void ExpectMessage(const MessageHeader& hdr, MessageType expected, const std::string& context) {
  if (hdr.type == expected) return;

  std::ostringstream os;
  os << "unexpected control-plane message";
  if (!context.empty()) os << " (" << context << ")";
  os << ": expected " << MessageTypeName(expected) << '(' << static_cast<int>(expected) << ')'
     << ", observed " << MessageTypeName(hdr.type) << '(' << static_cast<int>(hdr.type) << ')'
     << ", len=" << hdr.len;
  throw ProtocolError(os.str());
}

Protocol::Protocol(application::TCPEndpointHandle eph) : ep(eph) {}

Protocol::~Protocol() {}

MessageHeader Protocol::ReadMessageHeader() {
  MessageHeader hdr{};
  ReadExact(&ep, &hdr.type, sizeof(hdr.type), "ReadMessageHeader(type)");
  ReadExact(&ep, &hdr.len, sizeof(hdr.len), "ReadMessageHeader(len)");
  hdr.len = ntohl(hdr.len);
  return hdr;
}

void Protocol::WriteMessageHeader(const MessageHeader& hdr) {
  WriteExact(&ep, &hdr.type, sizeof(hdr.type), "WriteMessageHeader(type)");
  uint32_t len = htonl(hdr.len);
  WriteExact(&ep, &len, sizeof(len), "WriteMessageHeader(len)");
}

MessageRegEndpoint Protocol::ReadMessageRegEndpoint(size_t len) {
  ValidateControlMessageLength(len, "MessageRegEndpoint");
  std::vector<char> buf(len);
  ReadExact(&ep, buf.data(), len, "ReadMessageRegEndpoint(body)");
  try {
    auto out = msgpack::unpack(buf.data(), len);
    return out.get().as<MessageRegEndpoint>();
  } catch (const std::exception& e) {
    throw ProtocolError(std::string("failed to decode MessageRegEndpoint: ") + e.what());
  }
}

void Protocol::WriteMessageRegEndpoint(const MessageRegEndpoint& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::RegEndpoint, len});
  WriteExact(&ep, buf.data(), buf.size(), "WriteMessageRegEndpoint(body)");
}

MessageAskMemoryRegion Protocol::ReadMessageAskMemoryRegion(size_t len) {
  ValidateControlMessageLength(len, "MessageAskMemoryRegion");
  std::vector<char> buf(len);
  ReadExact(&ep, buf.data(), len, "ReadMessageAskMemoryRegion(body)");
  try {
    auto out = msgpack::unpack(buf.data(), len);
    return out.get().as<MessageAskMemoryRegion>();
  } catch (const std::exception& e) {
    throw ProtocolError(std::string("failed to decode MessageAskMemoryRegion: ") + e.what());
  }
}

void Protocol::WriteMessageAskMemoryRegion(const MessageAskMemoryRegion& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::AskMemoryRegion, len});
  WriteExact(&ep, buf.data(), buf.size(), "WriteMessageAskMemoryRegion(body)");
}

}  // namespace io
}  // namespace mori

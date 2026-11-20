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

#include <cstddef>
#include <msgpack.hpp>
#include <stdexcept>
#include <string>

#include "mori/io/common.hpp"
#include "mori/io/msgpack_adaptor.hpp"
#include "src/io/rdma/backend_impl.hpp"

namespace mori {
namespace io {
/* ---------------------------------------------------------------------------------------------- */
/*                                             Message                                            */
/* ---------------------------------------------------------------------------------------------- */
enum class MessageType : uint8_t {
  RegEndpoint = 0,
  AskMemoryRegion = 1,
  ReadViaWrite = 2,
  ReadViaWriteReply = 3,
};

struct MessageHeader {
  MessageType type;
  uint32_t len;
};

struct MessageRegEndpoint {
  EngineKey ekey;
  TopoKeyPair topo;
  int devId;
  application::RdmaEndpointHandle eph;
  int nicRank{0};
  int railId{-1};  // index in availDevices for rail affinity; -1 = unset (backward compat)
  MSGPACK_DEFINE(ekey, topo, devId, eph, nicRank, railId);
};

struct MessageAskMemoryRegion {
  EngineKey ekey;
  int devId;
  MemoryUniqueId id;
  application::RdmaMemoryRegion mr;
  MSGPACK_DEFINE(ekey, devId, id, mr);
};

struct MessageBuildConn {
  EngineKey key;
  MSGPACK_DEFINE(key);
};

// Reader asks the owner of ownerMemId to WRITE [ownerOffsets, sizes) into readerMem.
struct MessageReadViaWrite {
  EngineKey readerKey;
  std::string readerHost;
  int readerPort{0};
  MemoryDesc readerMem;
  MemoryUniqueId ownerMemId{0};
  SizeVec readerOffsets;
  SizeVec ownerOffsets;
  SizeVec sizes;
  TransferUniqueId id{0};
  bool notify{false};
  MSGPACK_DEFINE(readerKey, readerHost, readerPort, readerMem, ownerMemId, readerOffsets,
                 ownerOffsets, sizes, id, notify);
};

struct MessageReadViaWriteReply {
  int code{0};
  std::string message;
  MSGPACK_DEFINE(code, message);
};

inline constexpr size_t kMaxControlMessageBytes = 1 << 20;

class ProtocolError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

const char* MessageTypeName(MessageType type);
void ExpectMessage(const MessageHeader& hdr, MessageType expected, const std::string& context = {});

/* ---------------------------------------------------------------------------------------------- */
/*                                            Protocol                                            */
/* ---------------------------------------------------------------------------------------------- */
class Protocol {
 public:
  Protocol(application::TCPEndpointHandle);
  ~Protocol();

  MessageHeader ReadMessageHeader();
  void WriteMessageHeader(const MessageHeader&);

  MessageRegEndpoint ReadMessageRegEndpoint(size_t len);
  void WriteMessageRegEndpoint(const MessageRegEndpoint&);

  MessageAskMemoryRegion ReadMessageAskMemoryRegion(size_t len);
  void WriteMessageAskMemoryRegion(const MessageAskMemoryRegion&);

  MessageReadViaWrite ReadMessageReadViaWrite(size_t len);
  void WriteMessageReadViaWrite(const MessageReadViaWrite&);

  MessageReadViaWriteReply ReadMessageReadViaWriteReply(size_t len);
  void WriteMessageReadViaWriteReply(const MessageReadViaWriteReply&);

 private:
  application::TCPEndpoint ep;
};

}  // namespace io
}  // namespace mori

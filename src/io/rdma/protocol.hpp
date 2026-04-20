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

#include <msgpack.hpp>

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
  AskMemoryLayout = 2,
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
  int requestedRemoteDevId{-1};
  MSGPACK_DEFINE(ekey, topo, devId, eph, requestedRemoteDevId);
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

struct RdmaRemoteMemoryChunkWire {
  uint64_t offset;
  uint64_t addr;
  uint32_t rkey;
  uint64_t length;
  MSGPACK_DEFINE(offset, addr, rkey, length);
};

struct MessageAskMemoryLayoutRequest {
  EngineKey ekey;
  MemoryUniqueId id;
  MSGPACK_DEFINE(ekey, id);
};

struct MessageAskMemoryLayoutResponse {
  EngineKey ekey;
  MemoryUniqueId id;
  StatusCode code;
  std::string message;
  int rdmaDevId;
  std::vector<RdmaRemoteMemoryChunkWire> chunks;
  MSGPACK_DEFINE(ekey, id, code, message, rdmaDevId, chunks);
};

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

  MessageAskMemoryLayoutRequest ReadMessageAskMemoryLayoutRequest(size_t len);
  void WriteMessageAskMemoryLayoutRequest(const MessageAskMemoryLayoutRequest&);

  MessageAskMemoryLayoutResponse ReadMessageAskMemoryLayoutResponse(size_t len);
  void WriteMessageAskMemoryLayoutResponse(const MessageAskMemoryLayoutResponse&);

 private:
  application::TCPEndpoint ep;
};

}  // namespace io
}  // namespace mori

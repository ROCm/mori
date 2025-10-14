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
  RebuildRequest = 2,
  RebuildAck = 3,
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
  MSGPACK_DEFINE(ekey, topo, devId, eph);
};

struct MessageAskMemoryRegion {
  EngineKey ekey;
  int devId;
  MemoryUniqueId id;
  application::RdmaMemoryRegion mr;
  MSGPACK_DEFINE(ekey, devId, id, mr);
};

// QP rebuild handshake messages
// RebuildRequest: one side proposes rebuilding a specific QP (old_qpn). It may optionally
// pre-create a new local QP and include its new_qpn so peer can connect to it; in the minimal
// version we let peer allocate first, so new_qpn can be 0 meaning 'not allocated yet'.
// generation is the expected next generation (old_generation+1) for validation.
struct MessageRebuildRequest {
  EngineKey ekey;                // remote engine key (target of the request)
  uint32_t old_qpn;              // qpn being replaced (as seen by requester)
  uint32_t requester_new_qpn;    // optional new local qpn (0 if not created yet)
  uint32_t expected_generation;  // requester expectation for new generation
  MSGPACK_DEFINE(ekey, old_qpn, requester_new_qpn, expected_generation);
};

// RebuildAck: peer responds with its newly created QP number and echoes fields for validation.
// status: 0 success, non-zero error codes (TBD). If status!=0 new_qpn fields may be 0.
struct MessageRebuildAck {
  EngineKey ekey;  // echo
  uint32_t old_qpn;
  uint32_t responder_new_qpn;    // peer's new qpn (if success)
  uint32_t requester_new_qpn;    // echo back to confirm which proposal matched
  uint32_t expected_generation;  // echo
  uint32_t status;               // 0 success
  MSGPACK_DEFINE(ekey, old_qpn, responder_new_qpn, requester_new_qpn, expected_generation, status);
};

struct MessageBuildConn {
  EngineKey key;
  MSGPACK_DEFINE(key);
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

  MessageRebuildRequest ReadMessageRebuildRequest(size_t len);
  void WriteMessageRebuildRequest(const MessageRebuildRequest&);

  MessageRebuildAck ReadMessageRebuildAck(size_t len);
  void WriteMessageRebuildAck(const MessageRebuildAck&);

 private:
  application::TCPEndpoint ep;
};

}  // namespace io
}  // namespace mori

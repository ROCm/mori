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
  MSGPACK_DEFINE(ekey, topo, devId, eph, nicRank);
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

/* ---------------------------------------------------------------------------------------------- */
/*                                            Protocol                                            */
/* ---------------------------------------------------------------------------------------------- */
class Protocol {
 public:
  Protocol(application::TCPEndpointHandle);
  ~Protocol();

  // `expected`, when given, requires the header to carry exactly that type and
  // throws otherwise. The server loop, which legitimately accepts either type
  // and dispatches on it, omits it; the two CLIENT call sites are replying to
  // a request they just wrote and know precisely what must come back, so they
  // pass it. Those sites used to check with `assert(hdr.type == ...)`.
  // CORRECTION (review #64-1): I previously wrote here that the assert was
  // compiled out under -DNDEBUG. That is FALSE for this build -- CMakeLists.txt:4
  // sets CMAKE_CXX_FLAGS_RELEASE to a bare "-O3", and NDEBUG appears in 0 of 49
  // compile_commands.json entries. The assert was LIVE, so a client handed the
  // WRONG-but-valid message type did not silently unpack one struct as another:
  // it `abort()`ed the whole engine. The throw is still the right answer -- an
  // abort in a control-plane thread is uncatchable and kills a serving instance,
  // where a throw costs only that connection. That is the case `19b718f3`
  // motivated itself with (a peer
  // that flipped role mid-message has both kinds in flight on one channel) and
  // only half-discharged: it made an UNDEFINED type throw and left a
  // WELL-DEFINED but wrong one passing. Review #62 item 5.
  MessageHeader ReadMessageHeader();
  MessageHeader ReadMessageHeader(MessageType expected);
  void WriteMessageHeader(const MessageHeader&);

  MessageRegEndpoint ReadMessageRegEndpoint(size_t len);
  void WriteMessageRegEndpoint(const MessageRegEndpoint&);

  MessageAskMemoryRegion ReadMessageAskMemoryRegion(size_t len);
  void WriteMessageAskMemoryRegion(const MessageAskMemoryRegion&);

 private:
  application::TCPEndpoint ep;
};

}  // namespace io
}  // namespace mori

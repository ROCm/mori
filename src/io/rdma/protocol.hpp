#pragma once

#include <msgpack.hpp>

#include "mori/io/meta_data.hpp"
#include "mori/io/msgpack_adaptor.hpp"
#include "src/io/rdma/backend_impl_v1.hpp"

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
  MSGPACK_DEFINE(ekey, topo, devId, eph);
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
/*                                            Protocal                                            */
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

 private:
  application::TCPEndpoint ep;
};

}  // namespace io
}  // namespace mori
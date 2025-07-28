#pragma once

#include <msgpack.hpp>

#include "mori/io/meta_data.hpp"
#include "mori/io/msgpack_adaptor.hpp"

namespace mori {
namespace io {
/* ---------------------------------------------------------------------------------------------- */
/*                                             Message                                            */
/* ---------------------------------------------------------------------------------------------- */
enum class MessageType : uint8_t {
  RegEngine = 0,
};

struct MessageHeader {
  MessageType type;
  uint32_t len;
};

struct MessageRegEngine {
  EngineDesc engineDesc;

  // TODO: protocol for generic backend info, we can use a msgpack vector with each backend info be
  // a triple of (backend_type, bytes)
  application::RdmaEndpointHandle rdmaEph;

  MSGPACK_DEFINE(engineDesc, rdmaEph);
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

  MessageRegEngine ReadMessageRegEngine(size_t len);
  void WriteMessageRegEngine(const MessageRegEngine&);

 private:
  application::TCPEndpoint ep;
};

}  // namespace io
}  // namespace mori
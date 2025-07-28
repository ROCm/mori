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

MessageRegEngine Protocol::ReadMessageRegEngine(size_t len) {
  std::vector<char> buf(len);
  SYSCALL_RETURN_ZERO(ep.Recv(buf.data(), len));
  auto out = msgpack::unpack(buf.data(), len);
  return out.get().as<MessageRegEngine>();
}

void Protocol::WriteMessageRegEngine(const MessageRegEngine& msg) {
  msgpack::sbuffer buf;
  msgpack::pack(buf, msg);
  uint32_t len = static_cast<uint32_t>(buf.size());
  WriteMessageHeader({MessageType::RegEngine, len});
  SYSCALL_RETURN_ZERO(ep.Send(buf.data(), buf.size()));
}

}  // namespace io
}  // namespace mori
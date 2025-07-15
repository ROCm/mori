
#include "mori/application/utils/serdes.hpp"

#include <arpa/inet.h>

namespace mori {
namespace application {

RdmaEndpointHandlePacker::RdmaEndpointHandlePacker() {}

RdmaEndpointHandlePacker::~RdmaEndpointHandlePacker() {}

size_t RdmaEndpointHandlePacker::PackedSizeCompact() const {
  return sizeof(decltype(RdmaEndpointHandle::qpn)) +
         sizeof(decltype(InfiniBandEndpointHandle::lid)) +
         sizeof(decltype(EthernetEndpointHandle::gid));
}
void RdmaEndpointHandlePacker::PackCompact(const RdmaEndpointHandle& handle, void* packed) {
  reinterpret_cast<uint32_t*>(packed)[0] = htonl(handle.qpn);
  reinterpret_cast<uint32_t*>(packed)[1] = htonl(handle.ib.lid);
  memcpy(reinterpret_cast<uint32_t*>(packed) + 2, handle.eth.gid, sizeof(handle.eth.gid));
}

void RdmaEndpointHandlePacker::UnpackCompact(RdmaEndpointHandle& handle, void* packed) {
  handle.qpn = ntohl(reinterpret_cast<uint32_t*>(packed)[0]);
  handle.ib.lid = ntohl(reinterpret_cast<uint32_t*>(packed)[1]);
  memcpy(handle.eth.gid, reinterpret_cast<uint32_t*>(packed) + 2, sizeof(handle.eth.gid));
}

}  // namespace application
}  // namespace mori
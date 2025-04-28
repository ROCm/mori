#pragma once

namespace mori {
namespace core {
namespace transport {
namespace ibgda {

enum ProviderType {
  MLX5 = 1,
  BNXTRE = 2,
};

struct IbgdaOpCtrlSeg {
  uint32_t qpn;
};

struct IbgdaOpDataSeg {
  void* ptr;
  uint32_t key;
  size_t bytes;
};

template <ProviderType ProdType>
class IbgdaTransport {
 public:
  void RdmaSend(IbgdaOpCtrlSeg ctrl, IbgdaOpDataSeg data);
  void RdmaRecv(IbgdaOpCtrlSeg ctrl, IbgdaOpDataSeg data);
  void RdmaWrite(IbgdaOpCtrlSeg ctrl, IbgdaOpDataSeg local_data, IbgdaOpDataSeg remote_data);
  void RdmaRead(IbgdaOpCtrlSeg ctrl, IbgdaOpDataSeg local_data, IbgdaOpDataSeg remote_data);
  // TODO: add atomic
};

}  // namespace ibgda
}  // namespace transport
}  // namespace core
}  // namespace mori
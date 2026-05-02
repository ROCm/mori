#include "mori/xshmem/xshmem_types.hpp"

namespace mori {
namespace xshmem {
namespace gda {

struct XshmemGda_NoSignal {};
struct XshmemGda_NoCounter {};

struct XshmemGda_SignalInc {
  XshmemGdaSignal_t signalId;
  __device__ inline XshmemGda_SignalInc(XshmemGdaSignal_t id) : signalId(id) {}
};

struct XshmemGda_SignalAdd {
  XshmemGdaSignal_t signalId;
  uint64_t value;
  __device__ inline XshmemGda_SignalAdd(XshmemGdaSignal_t id, uint64_t val)
      : signalId(id), value(val) {}
};

struct XshmemGda_CounterInc {
  XshmemGdaCounter_t counterId;
  __device__ inline XshmemGda_CounterInc(XshmemGdaCounter_t id) : counterId(id) {}
};

struct XshmemGdaCtx {
  int rank;
  int worldSize;
  void* handle;
  int contextId;
};

typedef void* XshmemWindow_t;
typedef void* XshmemGdaRequest_t;

typedef uint32_t XshmemGdaSignal_t;
typedef uint32_t XshmemGdaCounter_t;

typedef enum XshmemGdaSignalOp_t {
  XshmemGdaSignalInc = 0,
  XshmemGdaSignalAdd,
} XshmemGdaSignalOp_t;

struct XshmemGda {
  XshmemDevComm const& comm;
  uint32_t contextId;
  XshmemGdaCtx ctx;  // diff from nccl gin
  void* _gdaHandle;

  // Constructor
  __device__ inline XshmemGda(XshmemDevComm const&, int contextIndex);

  // Data transfer operations
  template <typename RemoteAction = XshmemGda_NoSignal, typename LocalAction = XshmemGda_NoCounter>
  __device__ inline void put(int peer, XshmemWindow_t dstWin, size_t dstOffset,
                             XshmemWindow_t srcWin, size_t srcOffset, size_t bytes,
                             RemoteAction remoteAction = XshmemGda_NoSignal{},
                             LocalAction localAction = XshmemGda_NoCounter{});

  template <typename T, typename RemoteAction = XshmemGda_NoSignal>
  __device__ inline void putValue(int peer, XshmemWindow_t dstWin, size_t dstOffset, T value,
                                  RemoteAction remoteAction = XshmemGda_NoSignal{});

  __device__ inline void get(int peer, XshmemWindow_t remoteWin, size_t remoteOffset,
                             XshmemWindow_t localWin, size_t localOffset, size_t bytes);

  // Signal operations
  template <typename RemoteAction>
  __device__ inline void signal(int peer, RemoteAction remoteAction);

  __device__ inline uint64_t readSignal(XshmemGdaSignal_t signalId, int bits = 64);

  __device__ inline void waitSignal(XshmemGdaSignal_t signalId, uint64_t least, int bits = 64);

  __device__ inline void resetSignal(XshmemGdaSignal_t signalId);

  // Counter operations
  __device__ inline uint64_t readCounter(XshmemGdaCounter_t counterId, int bits = 56);

  __device__ inline void waitCounter(XshmemGdaCounter_t counterId, uint64_t least, int bits = 56);

  __device__ inline void resetCounter(XshmemGdaCounter_t counterId);

  // Completion operations
  __device__ inline void flush();

  __device__ inline void flushAsync(int peer, XshmemGdaRequest_t* outRequest);

  __device__ inline void wait(XshmemGdaRequest_t& request);
};

}  // namespace gda
}  // namespace xshmem
}  // namespace mori
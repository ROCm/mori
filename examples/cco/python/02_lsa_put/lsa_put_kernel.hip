// CCO LSA put kernel — intra-node P2P write via the flat VA.
//
// Each rank writes a sentinel (its own rank index) into every other LSA
// peer's receive buffer, then the host validates that all sentinels arrived.

#include <hip/hip_runtime.h>
#include "mori/cco/cco.hpp"

using namespace mori::cco;

extern "C" __global__ void lsa_put_kernel(
    ccoDevComm* devComm,
    ccoWindowDevice* win,
    uint64_t my_buf_off,
    uint64_t peer_buf_off)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // LSA flat-VA addressing:
    //   peer_va = win->winBase + (peerLsaRank * stride4G << 32) + byte_offset
    uint64_t stride = (uint64_t)win->stride4G << 32;

    uint64_t my_rank = (uint64_t)devComm->lsaRank;
    for (int peer = 0; peer < devComm->lsaSize; peer++) {
        if (peer == devComm->lsaRank)
            continue;

        uint64_t* peer_slot = (uint64_t*)(
            win->winBase + (uint64_t)peer * stride + peer_buf_off
        );
        *peer_slot = my_rank;
    }

    __threadfence_system();
}

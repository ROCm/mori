// CCO LSA put kernel — intra-node P2P write via the flat VA.
//
// Each rank writes a sentinel (its own rank index) into every other LSA
// peer's receive buffer, then the host validates that all sentinels arrived.
//
// Kernel args:
//   devComm    : ccoDevComm by value (lives in kernel-arg / constant memory)
//   win        : ccoWindowDevice* (GPU pointer; holds winBase + stride4G)
//   my_buf_off : byte offset of my send-value within my window slot
//   peer_buf_off: byte offset of the receive-value within peer's window slot

#include "mori/cco/cco.hpp"

using namespace mori::cco;

extern "C" __global__ void lsa_put_kernel(
    ccoDevComm devComm,
    ccoWindowDevice* win,
    uint64_t my_buf_off,
    uint64_t peer_buf_off)
{
    // Single-thread kernel: only thread 0 of block 0 does the puts.
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;

    // LSA flat-VA addressing:
    //   peer_va = win->winBase + (peerLsaRank * stride4G << 32) + byte_offset
    // win->stride4G = perRankSize >> 32  (perRankSize is 4 GB-aligned)
    uint64_t stride = (uint64_t)win->stride4G << 32;

    // Write my rank value into each peer's receive slot.
    uint64_t my_rank = (uint64_t)devComm.lsaRank;
    for (int peer = 0; peer < devComm.lsaSize; peer++) {
        if (peer == devComm.lsaRank)
            continue;

        uint64_t* peer_slot = (uint64_t*)(
            win->winBase + (uint64_t)peer * stride + peer_buf_off
        );
        // Store my rank so the peer can verify it arrived from the right source.
        *peer_slot = my_rank;
    }

    // System-scope fence: flush writes before the host-side barrier_all.
    __threadfence_system();
}

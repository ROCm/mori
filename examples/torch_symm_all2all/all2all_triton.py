# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The same one-shot all-to-all as ``all2all_kernel.hip``, written in Triton.

Nothing here is mori-specific. It is the pointer-array idiom torch's own symmetric-memory
Triton kernels use: pass ``hdl.buffer_ptrs_dev`` -- the address of a device array of
``world_size`` peer base addresses -- as a plain integer, cast it to a pointer inside the
kernel, and load the peer you want out of it::

    peers = peer_ptrs.to(tl.pointer_type(tl.uint64))
    dst   = tl.load(peers + peer).to(tl.pointer_type(tl.int32))

That is the whole interface to the backend, and it is why a Triton kernel needs no
extension build: torch already hands out the array, so no C++ is involved.

Run it through ``all2all.py --kernel triton``.
"""

import functools

import torch
import triton
import triton.language as tl

# 256 lanes on a wave64 part, 4 int32 per lane -- the dwordx4 store the HIP kernel does.
BLOCK = 1024
NUM_WARPS = 4


# do_not_specialize: Triton turns an int argument that happens to equal 1 into a constexpr,
# and rank 1 would then get a plain Python int where the kernel wants a value to cast.
@triton.jit(do_not_specialize=["chunk_elems", "rank_id", "blocks_per_peer"])
def _all2all_push_ptrs(
    send_ptr,  # *i32, local, world_size*chunk_elems
    peer_ptrs,  # i64 address of the world_size-entry peer pointer array
    chunk_elems,  # i32 elements each peer receives from us
    rank_id,
    blocks_per_peer,
    BLOCK: tl.constexpr,
):
    """Push our chunk for every peer into that peer's window, in our own slot.

    Grid is (world_size * blocks_per_peer,) and flattens (peer, slice of chunk): one
    block per peer would leave all but world_size CUs idle with too few writes in
    flight to cover interconnect latency.
    """
    pid = tl.program_id(0)
    peer = pid // blocks_per_peer
    sub = pid % blocks_per_peer

    # peers[peer] -- the same load the HIP kernel does, spelled as an int-to-pointer cast.
    peers = peer_ptrs.to(tl.pointer_type(tl.uint64))
    dst = tl.load(peers + peer).to(tl.pointer_type(tl.int32))

    # 64-bit from here on: chunk_elems*world_size overflows i32 past 8 GiB of window.
    chunk = chunk_elems.to(tl.int64)
    dst += rank_id.to(tl.int64) * chunk  # the slot that peer reserves for us
    src = send_ptr + peer.to(tl.int64) * chunk  # what we owe that peer

    span = blocks_per_peer * BLOCK
    for start in range(sub * BLOCK, chunk_elems, span):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < chunk_elems
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


@functools.cache
def _blocks_per_peer(chunk_elems: int, world_size: int, device) -> int:
    """Two blocks per CU split across peers, capped by how much there is to slice.

    Cached because it is on the launch path, where microseconds are the whole story:
    ``get_device_properties`` costs ~0.9 us, against a Triton launch of ~9-13 us.
    """
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    bpp = max(1, cus * 2 // max(1, world_size))
    return min(bpp, max(1, chunk_elems // BLOCK))


def all2all_push_ptrs(send, peers_dev, chunk_bytes, rank_id, world_size):
    """Signature-compatible with the HIP ``all2all_push_ptrs``, so callers can swap them.

    ``peers_dev`` is ``hdl.buffer_ptrs_dev``. The caller barriers afterwards; this kernel
    only writes.
    """
    if not send.is_contiguous():
        raise ValueError("send must be contiguous")
    if send.dtype != torch.int32:
        raise ValueError(f"send must be int32, got {send.dtype}")
    if chunk_bytes % 4:
        raise ValueError(f"chunk_bytes must be a multiple of 4, got {chunk_bytes}")
    chunk_elems = chunk_bytes // 4
    if send.numel() != world_size * chunk_elems:
        raise ValueError(
            f"send holds {send.numel()} elements, expected {world_size * chunk_elems}"
        )

    bpp = _blocks_per_peer(chunk_elems, world_size, send.device)
    _all2all_push_ptrs[(world_size * bpp,)](
        send,
        peers_dev,
        chunk_elems,
        rank_id,
        bpp,
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )

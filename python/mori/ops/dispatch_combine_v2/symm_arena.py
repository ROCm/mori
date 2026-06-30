"""SymmArena — one cco symmetric window carved into named sub-regions.

DeepEP-EPv2's ElasticBuffer puts all dispatch/combine staging into a single
contiguous symmetric buffer; SymmArena is the cco analogue. It allocates ONE
cco window (one ccoWindowRegister => peers P2P-reachable over the flat VA) and
lays out the symmetric buffers at fixed, aligned offsets inside it.

Device side: pass the single window handle (uint64) + the (compile-time) region
offsets; a kernel reaches peer pe's copy of region R via
    cco.Window(handle).lsa_ptr(pe, offset_of(R) + elem*esize)

Local (per-rank, non-peer-accessed) metadata — counters, barriers, the routing
map — are NOT arena regions; they are plain device tensors (see RoutingHandle).
"""
from dataclasses import dataclass, field


def _align_up(x, a):
    return (x + a - 1) // a * a


class SymmArena:
    _ALIGN = 256  # 256B: isolates peers' P2P stores to separate cache lines

    def __init__(self, comm, regions):
        """regions: list of (name, nbytes). Lifetime tied to `comm`."""
        self._comm = comm
        self._offsets = {}
        self._sizes = {}
        off = 0
        for name, nbytes in regions:
            off = _align_up(off, self._ALIGN)
            self._offsets[name] = off
            self._sizes[name] = nbytes
            off += nbytes
        self._total = max(_align_up(off, self._ALIGN), self._ALIGN)
        self._win = comm.alloc_window(self._total)

    @property
    def handle(self):
        return self._win.handle

    @property
    def total_bytes(self):
        return self._total

    def offset(self, name):
        return self._offsets[name]

    def local_ptr(self, name):
        return self._win.local_ptr + self._offsets[name]

    def __repr__(self):
        return (f"<SymmArena total={self._total}B regions="
                f"{ {k: (self._offsets[k], self._sizes[k]) for k in self._offsets} }>")


@dataclass
class RoutingHandle:
    """Per-rank routing metadata produced by dispatch, consumed by combine
    (analogue of mori's EpDispatchRoutingHandle / DeepEP's EPHandle).

    Holds device tensors (local, not symmetric): the (token,k)->dest flat-index
    map, total received count, and the per-source token count snapshot. Can be
    cached and replayed across decode iterations when routing is unchanged.
    """
    cur_tok: int
    total_recv = None        # device int32[1]
    dest_tok_map = None      # device int32[cur_tok * topk]
    meta: dict = field(default_factory=dict)

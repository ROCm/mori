"""FSDP2 forward-prefetch scheduling helper.

Composed with the deferred landing fence and the deep-pipe path, forward-prefetch
lets the CU-free SDMA-intra / RDMA-inter big-AG fill land behind forward GEMM
compute for the giant embed/lm_head all-gathers.

This is FSDP scheduling only: it drives torch's FSDP2
set_modules_to_forward_prefetch. It issues no collective payload and touches no
transport -- the bulk stays on the SDMA-intra reassembly + RDMA-inter ring, with
zero CU payload.

Bit-exact safety: depth is hard-clamped to 1 unless unsafe_deep=True. At depth 1
exactly one big AG is in flight, so the deferred landing fence (the copy-out
host-wait) still covers it. depth>=2 puts two AGs in flight, so the deferred fence
covers only one and the loss drifts; the deeper setting is reachable only via the
explicit unsafe opt-in and is never a shipped config.

All entry points are no-ops unless the caller enables it (the harness gates this on
MORI_FSDP_FWD_PREFETCH), so importing/shipping this module changes nothing.
"""

from typing import Callable, Iterable, List, Optional

import torch


_DECODER_LAYER_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers")


def iter_decoder_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """Yield the decoder/transformer block submodules of a HF-style causal LM."""
    for path in _DECODER_LAYER_PATHS:
        obj: Optional[torch.nn.Module] = model
        for name in path.split("."):
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            yield from obj
            return
    raise RuntimeError(
        "Could not find decoder layers. Pass a Qwen/Qwen2-like model or extend "
        "mori.ccl.fsdp_prefetch._DECODER_LAYER_PATHS for this architecture."
    )


def apply_forward_prefetch(
    model: torch.nn.Module,
    depth: int = 1,
    root_chain: bool = True,
    unsafe_deep: bool = False,
    layer_iter: Optional[Callable[[torch.nn.Module], Iterable[torch.nn.Module]]] = None,
) -> int:
    """Wire explicit FSDP2 forward-prefetch on a fully_shard-wrapped model.

    depth       : how many following decoder layers each layer prefetches. HARD-
                  CLAMPED to 1 unless unsafe_deep=True (depth>=2 drifts loss; the
                  deferred landing fence covers only one in-flight big AG).
    root_chain  : also chain the ROOT model (giant embed/lm_head AGs) into the last
                  `depth` layers' forward-prefetch set, so those long-pole AGs get a
                  real land-behind-compute window (Team A T17 mechanism).
    layer_iter  : override for locating decoder layers (defaults to iter_decoder_layers).

    Returns the number of layers wired (0 => model had no FSDP2 prefetch hook, no-op).
    """
    depth = max(1, int(depth))
    if depth > 1 and not unsafe_deep:
        depth = 1
    it = layer_iter or iter_decoder_layers
    layers: List[torch.nn.Module] = list(it(model))
    wired = 0
    for i, layer in enumerate(layers):
        nxt = list(layers[i + 1 : i + 1 + depth])
        if root_chain and i >= len(layers) - depth:
            nxt = nxt + [model]
        if nxt and hasattr(layer, "set_modules_to_forward_prefetch"):
            layer.set_modules_to_forward_prefetch(nxt)
            wired += 1
    return wired

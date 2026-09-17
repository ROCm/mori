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
import importlib
import os

import torch

from .dispatch_combine import (
    EpDispatchCombineKernelType,
    EpDispatchCombineConfig,
    EpDispatchCombineOp as _MoriEpDispatchCombineOp,
)
from .adaptive_dispatch_combine import (
    AdaptiveEpDispatchCombineOp,
    initialize_kiwi_lci_from_torch_process_group,
)
from .local_expert_count import (
    launch_local_expert_count,
)


def _adaptive_kiwi_threshold():
    value = os.environ.get("MORI_EP_KIWI_MAX_TOKENS")
    if value is None:
        return None
    try:
        threshold = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"MORI_EP_KIWI_MAX_TOKENS must be a non-negative integer, got {value!r}"
        ) from exc
    if threshold < 0:
        raise RuntimeError(
            f"MORI_EP_KIWI_MAX_TOKENS must be a non-negative integer, got {threshold}"
        )
    return threshold


def _vllm_combine_dtype(config):
    value = os.environ.get("MORI_EP_KIWI_COMBINE_DTYPE", "bfloat16").lower()
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if value not in dtypes:
        raise RuntimeError(
            "MORI_EP_KIWI_COMBINE_DTYPE must be bfloat16 or float32, "
            f"got {value!r}"
        )
    dtype = dtypes[value]
    if dtype.itemsize != config.max_token_type_size:
        raise RuntimeError(
            f"MORI_EP_KIWI_COMBINE_DTYPE={value} has itemsize {dtype.itemsize}, "
            f"but MORI config max_token_type_size={config.max_token_type_size}"
        )
    return dtype


def _vllm_global_num_tokens(local_num_tokens):
    try:
        from vllm.forward_context import get_forward_context

        dp_metadata = get_forward_context().dp_metadata
    except (ImportError, AssertionError):
        return local_num_tokens
    if dp_metadata is None:
        return local_num_tokens
    return int(dp_metadata.num_tokens_across_dp_cpu.max().item())


class EpDispatchCombineOp(_MoriEpDispatchCombineOp):
    """Construct normal MORI, or an opt-in adaptive MORI/Kiwi operator."""

    def __new__(cls, config):
        threshold = _adaptive_kiwi_threshold()
        if threshold is None:
            return super().__new__(cls)
        return AdaptiveEpDispatchCombineOp(
            config,
            kiwi_max_num_tokens=threshold,
            mori_op=_MoriEpDispatchCombineOp(config),
            dispatch_dtype=config.data_type,
            combine_dtype=_vllm_combine_dtype(config),
            num_blocks=0,
            selection_num_tokens_fn=_vllm_global_num_tokens,
        )


# dispatch_combine_v2 (FlyDSL) requires the optional `flydsl` dependency, so it
# is imported lazily: `import mori.ops` stays usable without flydsl installed,
# and `mori.ops.dispatch_combine_v2` resolves on first access.
_LAZY_SUBMODULES = {"dispatch_combine_v2"}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(globals().keys()) + sorted(_LAZY_SUBMODULES)

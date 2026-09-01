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
"""InterNodeV1 / InterNodeV1LL on the CCO kernels the v2 JIT plans compile.

The internode counterpart to ``test_dispatch_combine_v2_intranode.py``, and the
CCO counterpart to ``test_dispatch_combine_internode_v1.py`` -- which covers the
same two kernels on the shmem AOT path and is the model this file follows: the
same worker-pool fan-out, the same analytic golden from
``EpDispatchCombineTestCase``, the same ``_KERNELS`` geometry pairs. Expectations
are computed analytically, so a pass means "matches the intended semantics",
not "matches the other backend".

``backend`` is the axis this file adds. ``cco`` builds the handle on the CCO
communicator and redirects every InterNodeV1 launch to a v2 JIT plan (see the
launch-redirect section below); ``shmem`` is the stock AOT path, present as a
control -- if a case fails on both, suspect the shape or the harness rather than
the CCO port.

Three modes, in one file, because they want opposite settings:

``correctness``
    One checked round per case. Default; this is the mode that gates a merge.

``stress`` (``MORI_INTERNODE_TEST_STRESS=1``)
    Hundreds of rounds, checked once at the start and then run unchecked. Finds
    what a single round cannot: chunk flags that are never reset, recv counters
    that drift, send-slot reuse. The shmem suite does the same thing in
    ``test_dispatch_combine_internode.py``.

``bench`` (``MORI_INTERNODE_TEST_BENCH=1``)
    Latency and algorithmic bandwidth for both backends. **Asserts nothing about
    speed.** This machine's absolute throughput moves ~25% between batches with
    no code change, so a threshold would be a coin flip; the pair is printed
    (use ``-s``) and the comparison left to a human. Only same-invocation pairs
    are meaningful.

One host means one CCO node. CCO derives its node grouping from the physical
topology, so ``gpu_per_node`` stays at the whole world here: lowering it changes
only EP's idea of a node, and the RDMA path then issues puts to peers RAIL left
without a QP. The intra-node half of these kernels is what runs; the network
path needs two hosts and a driver that is machine-specific glue.
"""
import os
import statistics

import pytest
import torch
import torch.distributed as dist

import mori
from mori.ops.dispatch_combine_v2.ep_plans import EP_INTERNODE_PLANS, INTERNODE_DTYPES
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    _all_data_types,
    assert_worker_results,
    cross_dtype_hidden_dims,
    cross_dtype_skip_reason,
    run_ep_dispatch_combine_test,
)

# kernel -> (type, block_num, rdma_block_num, warp_num_per_block). The pairs
# test_dispatch_combine_internode_v1.py ships: the bandwidth table uses V1, the
# latency table V1LL, and the two want different grids.
_KERNELS = {
    "v1": (mori.ops.EpDispatchCombineKernelType.InterNodeV1, 96, 64, 8),
    "v1_ll": (mori.ops.EpDispatchCombineKernelType.InterNodeV1LL, 256, 128, 8),
}

# InterNodeV1 is what the reference bench selects at >=4096 tokens and V1LL
# below it, so each kernel is exercised at a count it actually ships for.
_BENCH_TOKENS = {"v1": 4096, "v1_ll": 128}

# CCO's LL path needs thousands of rounds to reach steady state where shmem
# needs under ten; a few hundred reads a transient. V1 at 4096 tokens covers the
# same cumulative work in far fewer rounds, which is why the two differ.
_BENCH_WARMUP = {"v1": 20, "v1_ll": 2000}

_WORLD_SIZE = 8


def _enabled(name):
    return os.environ.get(name, "0").strip().lower() not in ("", "0", "false", "no")


_stress_only = pytest.mark.skipif(
    not _enabled("MORI_INTERNODE_TEST_STRESS"),
    reason="stress mode is opt-in: set MORI_INTERNODE_TEST_STRESS=1",
)
_bench_only = pytest.mark.skipif(
    not _enabled("MORI_INTERNODE_TEST_BENCH"),
    reason="bench mode is opt-in: set MORI_INTERNODE_TEST_BENCH=1",
)


# ---------------------------------------------------------------------------
# launch redirect: the only caller the CCO internode kernels have
# ---------------------------------------------------------------------------
#
# v1 internode exists twice over, as two implementations sharing only a handle
# and an args struct:
#
#   shmem  src/ops/dispatch_combine/internode_v1.cpp, entries
#          ``EpDispatchInterNodeV1Kernel_<dt>``, which dispatch_combine.py drives.
#   cco    src/ops/dispatch_combine_v2/ep_internode_kernel.hpp, entries
#          ``mori_ep_internode_*`` -- what the JIT v2 plans compile, and what has
#          no caller anywhere else in the tree.
#
# ``EpDispatchCombineOp._launch_multi`` is the single funnel every InterNodeV1
# launch goes through, so redirecting it to the plans leaves argument building,
# routing and output tensors on the existing code. It is swapped for the whole
# process rather than per call: the cco kernels need a cco-backed handle, and the
# shmem kernels address the shmem heap, so on such a handle they fault.


# Read once: these sit on the per-launch path, which runs as often as the kernels
# do, and an os.environ lookup there is pure overhead in the common case where
# none of them is set.
_TRACE_PATH = os.environ.get("MORI_INTERNODE_TRACE")
_TRACE_CFG = bool(os.environ.get("MORI_INTERNODE_TRACE_CFG"))
_TRACE_SYNC = bool(os.environ.get("MORI_INTERNODE_TRACE_SYNC"))


def _trace(msg):
    """Progress markers to a file, enabled by MORI_INTERNODE_TRACE.

    The workers this runs in are spawned by the test's process manager, which
    keeps no handle on their stdout; when one dies the buffered output goes with
    it, so a stage that hangs or faults leaves no trace at all on stdout.
    """
    if not _TRACE_PATH:
        return
    with open(f"{_TRACE_PATH}.{os.environ.get('RANK', os.getpid())}", "a") as f:
        f.write(msg + "\n")


# The AOT entry base name of each pass, mapped to the plan that supplies it.
_PASS_BY_ENTRY = {
    "EpDispatchCopyToStaging": "copystaging",
    "EpDispatchInterNodeV1Kernel": "dispatch",
    "EpDispatchInterNodeV1KernelLowLatency": "dispatch_ll",
    "EpCombineSync": "combinesync",
    "EpCombineSyncBarrier": "combinesyncbarrier",
    "EpCombineInterNodeV1Kernel": "combine",
    "EpCombineInterNodeV1KernelLowLatency": "combine_ll",
    "EpCombineAll": "combineall",
}
# Longest-first so "..._fp8_ocp" is not read as dtype "ocp", and so
# "...V1Kernel" does not swallow "...V1KernelLowLatency".
_DTYPES_LONGEST_FIRST = sorted(INTERNODE_DTYPES, key=len, reverse=True)

# Cfg field -> the get_handle_info key reporting the same thing, for the
# consistency dump. Fields the handle does not report are simply absent.
_HANDLE_INFO_KEY = {
    "worldSize": "world_size",
    "scaleDim": "scale_dim",
    "scaleTypeSize": "scale_type_size",
    "maxTokenTypeSize": "max_token_type_size",
    "maxNumInpTokenPerRank": "max_num_inp_token_per_rank",
    "numExpertPerRank": "num_expert_per_rank",
    "numExpertPerToken": "num_expert_per_token",
    "gpuPerNode": "gpu_per_node",
    "rdmaBlockNum": "rdma_block_num",
    "quantType": "quant_type",
}


def _split_entry(func_name):
    """``EpCombineAll_bf16`` -> ``("combineall", "bf16")``; None if not a v1 pass."""
    for dtype in _DTYPES_LONGEST_FIRST:
        if not func_name.endswith("_" + dtype):
            continue
        base = func_name[: -len(dtype) - 1]
        if base in _PASS_BY_ENTRY:
            return _PASS_BY_ENTRY[base], dtype
    return None


def _dev_comm_host_ptr(op):
    """Address of the host-side ``ccoDevComm``, which the args hold by value.

    ``DevComm.ptr`` is the device-side copy, for kernels taking a pointer; this
    has to be the host struct, because ``EpInterNodeCcoArgs`` embeds one.
    """
    cached = op.__dict__.get("_internode_dev_comm")
    if cached is not None:
        return cached
    if op._cco_comm is None:
        raise RuntimeError(
            "the cco v1 internode kernels need a cco-backed handle; "
            "set MORI_EP_COMM=cco before building the op"
        )
    from mori.cco import cco as _cco
    from mori.cco.communicator import DevCommHandle

    # These requirements are what the kernels are written against, and since the
    # C++ side has no devComm of its own this is the only place they are chosen.
    # The defaults are not close enough to work: v1 talks only to its own local
    # rank on other nodes, so it asks for RAIL rather than the CROSSNODE default,
    # and ccoGda picks its QP as contextId % numQpPerPe, so a smaller context
    # count than numQpPerPe silently collapses the stripes onto fewer QPs.
    # A production path would need to make the same two choices.
    reqs = _cco.DevCommRequirements()
    reqs.gda_connection_type = _cco.GDA_CONNECTION_RAIL
    reqs.gda_context_count = max(1, op.config.num_qp_per_pe)

    handle = DevCommHandle(op._cco_comm, requirements=reqs)
    op.__dict__["_internode_dev_comm_handle"] = handle  # keep it alive
    # lsa_size is CCO's own view of how many ranks share a node, which comes from
    # the physical topology and not from config.gpuPerNode. Under RAIL there are
    # QPs only to cross-node peers, so lsa_size == world_size means none exist.
    _trace(
        f"  devComm world_size={handle.world_size} lsa_size={handle.lsa_size} "
        f"rank={handle.rank} lsa_rank={handle.lsa_rank}"
    )
    ptr = handle._dev_comm.host_ptr
    op.__dict__["_internode_dev_comm"] = ptr
    return ptr


def _plan_for(op, pass_name, dtype, block_num, warp_per_block, mp_count, want=None):
    """One plan per (pass, dtype, geometry). Geometry is outside the JIT cache
    key, so the geometry variants share one compiled binary.

    ``want`` is the (grid, block, smem) dispatch_combine.py computed. It is
    checked against EpInterNodeGeometry once per plan rather than once per launch:
    reading ``plan.info`` parses the whole Cfg out of a key=value blob, and the
    two arithmetics cannot start agreeing or stop agreeing between launches of
    the same plan.
    """
    key = (pass_name, dtype, block_num, warp_per_block, mp_count)
    cache = op.__dict__.setdefault("_internode_plans", {})
    plan = cache.get(key)
    if plan is not None:
        return plan

    info = op._handle_info
    cfg = op.config
    # Shape comes from what C++ resolved on the handle rather than from the
    # Python config: these become compiled-in constants that the kernel's
    # EpInterNodeBindConfig writes back over args.config, so a constant disagreeing
    # with what the launch carries would silently change behaviour. hiddenDim is
    # the exception, overridden per call by build_args from the input tensor.
    plan = EP_INTERNODE_PLANS[pass_name](
        worldSize=info["world_size"],
        hiddenDim=op.__dict__.get("_internode_hidden_dim") or info["hidden_dim"],
        scaleDim=info["scale_dim"],
        scaleTypeSize=info["scale_type_size"],
        maxTokenTypeSize=info["max_token_type_size"],
        maxNumInpTokenPerRank=info["max_num_inp_token_per_rank"],
        numExpertPerRank=info["num_expert_per_rank"],
        numExpertPerToken=info["num_expert_per_token"],
        maxTotalRecvTokens=cfg.max_total_recv_tokens,
        gpuPerNode=info["gpu_per_node"],
        numQpPerPe=cfg.num_qp_per_pe,
        quantType=int(info["quant_type"]),  # the handle reports the code, not the name
        dtype=dtype,
        blockNum=block_num,
        warpPerBlock=warp_per_block,
        rdmaBlockNum=info["rdma_block_num"],
        mpCount=mp_count,
    )
    got = plan.info
    if _TRACE_CFG:
        # The kernel's EpInterNodeBindConfig overwrites args.config with the constants
        # compiled in, so any field where the two disagree is a silent behaviour
        # change rather than an error.
        hi = op._handle_info
        for k, v in sorted(got.items()):
            mine = hi.get(_HANDLE_INFO_KEY.get(k, ""), "-")
            flag = "" if str(mine) in ("-", str(v)) else "  <== DIFFERS"
            _trace(f"    cfg {k}={v} handle={mine}{flag}")
    # Geometry is arrived at twice -- once by dispatch_combine.py, once by
    # EpInterNodeGeometry -- so a disagreement means the host arithmetic drifted.
    if want is not None:
        have = (got["gridX"], got["blockX"], got["sharedBytes"])
        if want != have:
            raise AssertionError(
                f"{pass_name}: geometry drift, dispatch_combine.py wants "
                f"grid/block/smem {want}, EpInterNodeGeometry says {have}"
            )
    cache[key] = plan
    return plan


def _install_jit_redirect():
    """Route every InterNodeV1 launch in this process to the JIT v2 plans.

    Pair with _uninstall_jit_redirect(): the patch is on the class, so leaving it
    in place would have a later shmem case launch cco plans against a shmem
    handle.
    """
    import mori.ops.dispatch_combine as dc

    Op = dc.EpDispatchCombineOp
    if getattr(Op, "_internode_installed", False):
        return
    orig_launch_multi = Op._launch_multi
    Op._internode_saved = {
        "_launch_multi": orig_launch_multi,
        "dispatch": Op.dispatch,
        "combine": Op.combine,
    }

    def _launch_multi(self, func_names, grids, blocks, shared_mems, stream, args_ptr):
        passes = [_split_entry(n) for n in func_names]
        if any(p is None for p in passes):
            return orig_launch_multi(
                self, func_names, grids, blocks, shared_mems, stream, args_ptr
            )
        mp_count = self._handle_info["multi_processor_count"]
        _trace(f"launch_multi enter {[p for p, _ in passes]}")
        dev_comm = _dev_comm_host_ptr(self)
        _trace(f"  dev_comm={dev_comm:#x}")
        for (pass_name, dtype), grid, block, smem in zip(
            passes, grids, blocks, shared_mems
        ):
            wpb = max(1, block // self._warp_size)
            plan = _plan_for(
                self, pass_name, dtype, grid, wpb, mp_count, (grid, block, smem)
            )
            _trace(f"  plan ready {pass_name}")
            _trace(f"  launching {pass_name} grid={grid} block={block} smem={smem}")
            plan.launch(raw=args_ptr, devComm=dev_comm, stream=stream)
            _trace(f"  launched {pass_name}")
            if _TRACE_SYNC:
                # Kernel errors are asynchronous, so without a sync per pass a
                # fault is reported against whichever call happens to sync next
                # -- or takes the process down with no attribution at all.
                try:
                    torch.cuda.synchronize()
                    _trace(f"  synced {pass_name} ok")
                except Exception as exc:
                    _trace(f"  synced {pass_name} FAILED {type(exc).__name__}: {exc}")
                    raise

    Op._launch_multi = _launch_multi

    # build_args takes the hidden dim from the input tensor, and the kernel gets
    # it as a compiled constant, so the plan must be built for the same value.
    for name in ("dispatch", "combine"):
        orig = getattr(Op, name)

        def wrapper(self, input, *a, _orig=orig, **kw):
            self.__dict__["_internode_hidden_dim"] = input.size(1)
            return _orig(self, input, *a, **kw)

        setattr(Op, name, wrapper)

    Op._internode_installed = True


def _uninstall_jit_redirect():
    """Undo _install_jit_redirect(), returning the op to the stock AOT launch path.

    The cached plans and devComm need no cleanup: they hang off the op instance,
    not the class, so they go when the op does.
    """
    import mori.ops.dispatch_combine as dc

    Op = dc.EpDispatchCombineOp
    saved = getattr(Op, "_internode_saved", None)
    if not getattr(Op, "_internode_installed", False) or saved is None:
        return
    Op._launch_multi = saved["_launch_multi"]
    Op.dispatch = saved["dispatch"]
    Op.combine = saved["combine"]
    Op._internode_installed = False
    Op._internode_saved = None


def _install_backend(backend):
    """Select the implementation. Must precede the op: the backend decides what
    the handle can host, and the JIT redirect has to be in place for the first
    launch. The shmem branch uninstalls rather than merely not installing,
    because a cco case earlier in the session has already patched the class.
    """
    if backend == "cco":
        os.environ["MORI_EP_COMM"] = "cco"
        _install_jit_redirect()
    else:
        os.environ.pop("MORI_EP_COMM", None)
        _uninstall_jit_redirect()


def _make_config(
    rank,
    world_size,
    kernel,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    scale_dim=0,
    scale_type_size=1,
    quant_type="none",
    combine_data_type=None,
):
    kernel_type, block_num, rdma_block_num, warp_num_per_block = _KERNELS[kernel]
    _, _, config_hidden_dim = cross_dtype_hidden_dims(
        hidden_dim, data_type, combine_data_type or data_type
    )
    return mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=config_hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=2,
        block_num=block_num,
        rdma_block_num=rdma_block_num,
        warp_num_per_block=warp_num_per_block,
        kernel_type=kernel_type,
        # One node here, and it has to match LOCAL_WORLD_SIZE -- see the module
        # docstring.
        gpu_per_node=world_size,
        # The field defaults to 1, which starves the cross-node path by ~1.5x.
        # It also sizes the CCO side, where gdaContextCount == numQpPerPe, so
        # both backends are measured at the value the published bench uses.
        num_qp_per_pe=2,
        quant_type=quant_type,
    )


def _fanout(manager, func, args):
    for _ in range(_WORLD_SIZE):
        manager.task_queue.put((func, args))
    assert_worker_results(manager, _WORLD_SIZE)


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------


def _worker_correctness(
    rank,
    backend,
    kernel,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
    scale_dim,
    scale_type_size,
):
    _install_backend(backend)
    config = _make_config(
        rank=rank,
        world_size=_WORLD_SIZE,
        kernel=kernel,
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
    )
    run_ep_dispatch_combine_test(
        config, EpDispatchCombineTestCase, use_max_token_num=True
    )


@pytest.mark.parametrize("backend", ("cco", "shmem"))
@pytest.mark.parametrize("kernel", tuple(_KERNELS))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (32, 128))
@pytest.mark.parametrize("scale_dim, scale_type_size", ((0, 1), (32, 4)))
def test_internode_correctness(
    torch_dist_process_manager,
    backend,
    kernel,
    data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    scale_dim,
    scale_type_size,
):
    """dispatch + combine must match the analytic golden."""
    _fanout(
        torch_dist_process_manager,
        _worker_correctness,
        [
            backend,
            kernel,
            data_type,
            hidden_dim,
            max_num_inp_token_per_rank,
            32,  # num_experts_per_rank
            8,  # num_experts_per_token
            scale_dim,
            scale_type_size,
        ],
    )


def _worker_cross_dtype(
    rank,
    backend,
    kernel,
    data_type,
    combine_data_type,
    hidden_dim,
    max_num_inp_token_per_rank,
    quant_type,
):
    _install_backend(backend)
    dispatch_hidden_dim, combine_hidden_dim, _ = cross_dtype_hidden_dims(
        hidden_dim, data_type, combine_data_type
    )
    config = _make_config(
        rank=rank,
        world_size=_WORLD_SIZE,
        kernel=kernel,
        data_type=data_type,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=32,
        num_experts_per_token=8,
        quant_type=quant_type,
        combine_data_type=combine_data_type,
    )
    run_ep_dispatch_combine_test(
        config,
        EpDispatchCombineTestCase,
        use_max_token_num=True,
        combine_data_type=combine_data_type,
        combine_hidden_dim=combine_hidden_dim,
        dispatch_hidden_dim=dispatch_hidden_dim,
    )


@pytest.mark.parametrize("backend", ("cco", "shmem"))
@pytest.mark.parametrize("kernel", tuple(_KERNELS))
@pytest.mark.parametrize("data_type", _all_data_types())
@pytest.mark.parametrize("quant_type", ("none", "fp8_direct_cast"))
def test_internode_cross_dtype(
    torch_dist_process_manager, backend, kernel, data_type, quant_type
):
    """FP8/FP4 dispatch with a BF16 combine -- the pairing the tuning matrix
    ships, and the one the CCO path's FP8 and LL branches were never exercised
    on before this port."""
    combine_data_type = torch.bfloat16
    skip = cross_dtype_skip_reason(quant_type, data_type, combine_data_type)
    if skip:
        pytest.skip(skip)

    _fanout(
        torch_dist_process_manager,
        _worker_cross_dtype,
        [backend, kernel, data_type, combine_data_type, 7168, 128, quant_type],
    )


def _worker_small_shapes(rank, backend, kernel, tokens):
    _install_backend(backend)
    config = _make_config(
        rank=rank,
        world_size=_WORLD_SIZE,
        kernel=kernel,
        data_type=torch.bfloat16,
        hidden_dim=4096,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=32,
        num_experts_per_token=8,
    )
    run_ep_dispatch_combine_test(
        config, EpDispatchCombineTestCase, use_max_token_num=True
    )


@pytest.mark.parametrize("backend", ("cco", "shmem"))
@pytest.mark.parametrize("kernel", tuple(_KERNELS))
@pytest.mark.parametrize("tokens", (1, 32))
def test_internode_small_shapes(torch_dist_process_manager, backend, kernel, tokens):
    """One token per rank is where an off-by-one in the chunk count or an empty
    chunk-flag range shows up; the shipping shapes never produce a single-token
    chunk."""
    _fanout(torch_dist_process_manager, _worker_small_shapes, [backend, kernel, tokens])


# ---------------------------------------------------------------------------
# stress
# ---------------------------------------------------------------------------


def _round(case, op, test_data, check=False):
    """One dispatch+combine, mirroring EpDispatchCombineTestCase.run_test_once
    but with ``call_reset=True`` so the counters are returned to their initial
    state -- a loop that leaves them dirty measures the second round against the
    first round's leftovers."""
    (_, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales) = test_data
    rank = case.config.rank
    (
        dispatch_output,
        dispatch_weights,
        _,
        dispatch_indices,
        dispatch_recv_num_token,
    ) = op.dispatch(
        all_rank_input[rank],
        all_rank_weights[rank],
        all_rank_scales[rank],
        all_rank_indices[rank],
    )
    # Read it here, between the two calls: combine reuses this counter, so after
    # combine it reads back as zero.
    total_recv = dispatch_recv_num_token[0].item()
    if check:
        num_experts = case.config.num_experts_per_rank * case.config.world_size
        max_expert = dispatch_indices[:total_recv].max().item()
        assert max_expert < num_experts, (
            f"rank[{rank}] dispatch returned expert id {max_expert} "
            f">= {num_experts}"
        )
    combine_input = case._get_combine_input(op, dispatch_output, num_token=total_recv)
    op.combine(
        combine_input,
        dispatch_weights,
        all_rank_indices[rank],
        call_reset=True,
    )
    case.sync()
    return total_recv


def _worker_stress(rank, backend, kernel, tokens, reps):
    _install_backend(backend)
    config = _make_config(
        rank=rank,
        world_size=_WORLD_SIZE,
        kernel=kernel,
        data_type=torch.bfloat16,
        hidden_dim=7168,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=32,
        num_experts_per_token=8,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    case = EpDispatchCombineTestCase(config)
    test_data = case.gen_test_data(use_max_token_num=True)

    # One fully checked round first, so a path that silently moves nothing
    # cannot pass this test by being fast.
    case.run_test_once(op, test_data, check_results=True)

    for i in range(reps):
        _round(case, op, test_data, check=True)
        if rank == 0 and (i + 1) % 50 == 0:
            print(f"[stress] {backend}/{kernel} {i + 1}/{reps}", flush=True)


@_stress_only
@pytest.mark.parametrize("backend", ("cco", "shmem"))
@pytest.mark.parametrize("kernel", tuple(_KERNELS))
def test_internode_stress(torch_dist_process_manager, backend, kernel):
    """Many rounds on one op. What breaks at round 300 and not at round 1 is
    state that survives a round, and none of it is visible to a single-shot
    check."""
    _fanout(
        torch_dist_process_manager,
        _worker_stress,
        [backend, kernel, _BENCH_TOKENS[kernel], 300],
    )


@_stress_only
@pytest.mark.parametrize("backend", ("cco", "shmem"))
def test_internode_large_tokens(torch_dist_process_manager, backend):
    """4096 tokens/rank, the largest shape that completes.

    Deliberately not 8192: cco at 7168/8192 runs past a 1500s timeout while
    shmem finishes the same config in ~30s, still unexplained. Raise this when
    that is fixed -- leaving it at 4096 keeps the test from being a slow way to
    rediscover a known hang. Few rounds, because at this size the one checked
    round dominates the cost.
    """
    _fanout(torch_dist_process_manager, _worker_stress, [backend, "v1", 4096, 20])


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------


def _worker_bench(rank, backend, kernel, data_type, tokens, warmup, iters):
    _install_backend(backend)
    config = _make_config(
        rank=rank,
        world_size=_WORLD_SIZE,
        kernel=kernel,
        data_type=data_type,
        hidden_dim=7168,
        max_num_inp_token_per_rank=tokens,
        num_experts_per_rank=32,
        num_experts_per_token=8,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    case = EpDispatchCombineTestCase(config)
    test_data = case.gen_test_data(use_max_token_num=True)

    # Checked once before timing: a measurement of a path that drops tokens is
    # worse than no measurement.
    case.run_test_once(op, test_data, check_results=True)

    (_, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales) = test_data
    total_recv = _round(case, op, test_data)

    # Routing is fixed by the test data, so the combine input is resolved once
    # rather than per round -- in zero-copy mode that call is a device copy, and
    # timing it would attribute a harness cost to the kernel.
    d_us, c_us = [], []
    for i in range(warmup + iters):
        d0, d1 = torch.cuda.Event(True), torch.cuda.Event(True)
        c0, c1 = torch.cuda.Event(True), torch.cuda.Event(True)

        d0.record()
        out, weights, _, _, _ = op.dispatch(
            all_rank_input[rank],
            all_rank_weights[rank],
            all_rank_scales[rank],
            all_rank_indices[rank],
        )
        d1.record()
        combine_input = case._get_combine_input(op, out, num_token=total_recv)
        c0.record()
        op.combine(combine_input, weights, all_rank_indices[rank], call_reset=True)
        c1.record()
        case.sync()

        if i >= warmup:
            d_us.append(d0.elapsed_time(d1) * 1000.0)
            c_us.append(c0.elapsed_time(c1) * 1000.0)

    elem = torch.tensor([], dtype=data_type).element_size()

    def bw(us):
        # Algorithmic bandwidth, the bench_dispatch_combine.py definition:
        # received payload over elapsed time, covering both the XGMI and the
        # RDMA leg.
        return total_recv * config.hidden_dim * elem / (us * 1e-6) / 1e9

    mine = {
        "rank": rank,
        "d": statistics.mean(d_us),
        "c": statistics.mean(c_us),
        "recv": total_recv,
    }
    gathered = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, mine)
    if rank == 0:
        d = statistics.mean(g["d"] for g in gathered)
        c = statistics.mean(g["c"] for g in gathered)
        # Mean and worst both matter: these are collective passes, so a round is
        # not over until the slowest rank finishes.
        print(
            f"[bench] {backend:5s} {kernel:5s} {str(data_type).split('.')[-1]:14s} "
            f"tokens={tokens} recv={total_recv} "
            f"dispatch={d:8.1f}us (worst {max(g['d'] for g in gathered):8.1f}) "
            f"{bw(d):6.1f} GB/s  "
            f"combine={c:8.1f}us (worst {max(g['c'] for g in gathered):8.1f}) "
            f"{bw(c):6.1f} GB/s",
            flush=True,
        )


@_bench_only
@pytest.mark.parametrize("kernel", tuple(_KERNELS))
@pytest.mark.parametrize(
    "data_type",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(
            torch.float8_e4m3fn,
            id="fp8",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available()
                or not torch.cuda.get_device_properties(0)
                .gcnArchName.split(":")[0]
                .startswith("gfx95"),
                reason="float8_e4m3fn (OCP) is gfx950-only",
            ),
        ),
    ],
)
def test_internode_bench(torch_dist_process_manager, kernel, data_type):
    """Latency and algorithmic bandwidth for both backends, printed as a pair.

    No speed assertion, and both backends run inside one test rather than as two
    parametrized cases: absolute numbers here move ~25% between batches with no
    code change, so the only defensible comparison is cco against shmem measured
    close together. Use ``-s`` to see the lines.

    Two things these numbers are not. They are **one host**, so nNodes=1 and no
    RDMA leg runs -- they do not reproduce, and must not be compared against,
    the two-node figures. And ``shmem``/``v1_ll``/combine carries a fixed ~1.39ms
    here (against cco's ~55us) that does not scale with tokens and does not
    appear on ``v1``; it looks like the local-peer handling these kernels
    inherited rather than a real cco win, so read that one cell with suspicion.
    """
    for backend in ("shmem", "cco"):
        _fanout(
            torch_dist_process_manager,
            _worker_bench,
            [
                backend,
                kernel,
                data_type,
                _BENCH_TOKENS[kernel],
                _BENCH_WARMUP[kernel],
                50,
            ],
        )


# ---------------------------------------------------------------------------
# two-host entry, under torchrun
# ---------------------------------------------------------------------------


def _spawn_worker(rank, fn, args):
    """Install the redirect in the worker, then hand off to the harness.

    The harness fans out with torch.multiprocessing.spawn, and a spawned worker
    is a fresh interpreter that re-imports mori -- so the class patch the parent
    made is not there. MORI_EP_COMM travels with the environment and does arrive,
    which is the dangerous half: the worker would build a cco-backed handle and
    then launch the shmem AOT entries at it, and those address the shmem heap, so
    the result is a fault at 0x0 rather than a clean failure.
    """
    _install_jit_redirect()
    return fn(rank, *args)


if __name__ == "__main__":
    # Everything above is one host: the cases fan out through the process manager
    # and _WORLD_SIZE is a single node's worth of ranks, so no RDMA leg runs. The
    # cross-node numbers need two hosts, and the harness that knows how to drive
    # them is the shmem one in examples/ -- it owns the CLI, the shapes and the
    # result tables. All this entry adds is the cco redirect.
    #
    #   torchrun --nnodes=2 --node_rank=N --nproc_per_node=1 \
    #     tests/python/ops/dispatch_combine_v2/test_dispatch_combine_v2_internode.py \
    #     --kernel-type v1 --cmd bench --dtype bf16 --max-tokens 4096
    #
    # nproc_per_node is 1 by design: the harness reads WORLD_SIZE as the node
    # count and spawns gpu_per_node workers of its own on top of it.
    import runpy
    import sys

    import torch.multiprocessing

    _repo_root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):  # dispatch_combine_v2 -> ops -> python -> tests -> root
        _repo_root = os.path.dirname(_repo_root)

    # sys.path[0] under torchrun is this file's own directory, not the repo root,
    # so the import below needs the root put there first. Doing it here rather
    # than leaving it to PYTHONPATH keeps the entry usable from any launcher.
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Self, under the package name. Anything handed to spawn is pickled as
    # module + qualname, and this file's __name__ here is "__main__" -- whose
    # counterpart in the worker is the harness, not this module, so a
    # "__main__._spawn_worker" reference would not resolve there.
    from tests.python.ops.dispatch_combine_v2 import (
        test_dispatch_combine_v2_internode as _self,
    )

    os.environ["MORI_EP_COMM"] = "cco"

    _real_spawn = torch.multiprocessing.spawn

    def _spawn(fn, args=(), nprocs=1, **kwargs):
        return _real_spawn(_self._spawn_worker, args=(fn, args), nprocs=nprocs, **kwargs)

    torch.multiprocessing.spawn = _spawn

    # run_name="__main__" because the harness keeps its driver under its own
    # __main__ guard; importing it would only parse argv and define classes. Our
    # argv is already the harness's argv, and argparse ignores argv[0].
    runpy.run_path(
        os.path.join(
            _repo_root,
            "examples",
            "ops",
            "dispatch_combine",
            "test_dispatch_combine_internode.py",
        ),
        run_name="__main__",
    )

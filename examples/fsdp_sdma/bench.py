#!/usr/bin/env python3
"""
Benchmark Qwen-style FSDP2 training with native torch allgather vs MORI SDMA allgather.

Example:
  torchrun --nproc_per_node=8 examples/fsdp/bench_qwen7b_allgather.py --mode native
  MORI_ENABLE_SDMA=1 torchrun --nproc_per_node=8 examples/fsdp/bench_qwen7b_allgather.py --mode mori
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist


def _use_local_fsdp() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fsdp_dir = repo_root / "fsdp"
    spec = importlib.util.spec_from_file_location(
        "torch.distributed.fsdp",
        fsdp_dir / "__init__.py",
        submodule_search_locations=[str(fsdp_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local FSDP package from {fsdp_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["torch.distributed.fsdp"] = module
    sys.modules["fsdp"] = module
    spec.loader.exec_module(module)


# Disabled for PyTorch migration validation: use installed torch + /root/wuyl/pytorch FSDP overlay.
# _use_local_fsdp()
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

import torch.distributed.fsdp as _fsdp_pkg
import torch.distributed.fsdp._fully_shard._fully_shard as _fsdp_fully_shard_mod
import torch.distributed.fsdp._fully_shard._fsdp_param_group as _fsdp_param_group_mod
import torch.distributed.fsdp._fully_shard._mori_sdma_allgather as _mori_sdma_mod
print("[verify-fsdp] fsdp package:", _fsdp_pkg.__file__, flush=True)
print("[verify-fsdp] fully_shard module:", _fsdp_fully_shard_mod.__file__, flush=True)
print("[verify-fsdp] param_group module:", _fsdp_param_group_mod.__file__, flush=True)
print("[verify-fsdp] mori sdma module:", _mori_sdma_mod.__file__, flush=True)


_BENCHMARK_SEED = 1234


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen7B FSDP2 training benchmark for native vs MORI SDMA allgather"
    )
    parser.add_argument("--mode", choices=("native", "mori"), required=True)
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Optional HF model/config path. If omitted, a built-in Qwen2-7B config is used.",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--backend", default="nccl")
    parser.add_argument("--reshard-root", action="store_true")
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Rank 0 prints an aggregate every N measured steps.",
    )
    parser.add_argument(
        "--profile-dir",
        default=None,
        help=(
            "Optional directory for per-rank PyTorch profiler Chrome traces. "
            "Profiling starts after warmup and uses measured-step indices."
        ),
    )
    parser.add_argument(
        "--profile-start-step",
        type=int,
        default=0,
        help="Measured step index at which profiling starts after warmup.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=5,
        help="Number of measured steps to include in the profiler trace.",
    )
    return parser.parse_args()


def _configure_mode(mode: str) -> None:
    if mode == "mori":
        os.environ.setdefault("MORI_ENABLE_SDMA", "1")
        os.environ["MORI_FSDP_ENABLE_SDMA"] = "1"
    else:
        os.environ.pop("MORI_FSDP_ENABLE_SDMA", None)


def _init_distributed(backend: str) -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=backend)
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    return rank, local_rank, world_size, device


def _init_mori_shmem_if_needed(mode: str) -> None:
    if mode != "mori":
        return
    import mori.shmem as shmem

    shmem.shmem_torch_process_group_init("default")
    if shmem.shmem_mype() != dist.get_rank() or shmem.shmem_npes() != dist.get_world_size():
        raise RuntimeError(
            "MORI SHMEM PE mapping must match the FSDP process group for this benchmark"
        )


def _finalize_mori_shmem_if_needed(mode: str) -> None:
    if mode != "mori":
        return
    import mori.shmem as shmem

    shmem.shmem_finalize()


def _get_torch_dtype(dtype: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]


def _load_qwen_model(model_name_or_path: str | None, dtype: torch.dtype):
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        from transformers.models.qwen2 import Qwen2Config
    except ImportError as exc:
        raise RuntimeError(
            "This benchmark requires transformers. Install it in the runtime environment."
        ) from exc

    if model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        config = Qwen2Config(
            vocab_size=152064,
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            max_position_embeddings=32768,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            tie_word_embeddings=False,
            use_cache=False,
        )
    config.use_cache = False
    try:
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    return model, config


def _iter_decoder_layers(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        for name in path.split("."):
            obj = getattr(obj, name, None)
            if obj is None:
                break
        if obj is not None:
            yield from obj
            return
    raise RuntimeError(
        "Could not find decoder layers. Pass a Qwen/Qwen2-like model or extend "
        "_iter_decoder_layers() for this architecture."
    )


def _apply_fsdp2(model: torch.nn.Module, dtype: torch.dtype, reshard_root: bool) -> None:
    mp_policy = (
        MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
        if dtype != torch.float32
        else MixedPrecisionPolicy()
    )
    shards = []
    for layer in _iter_decoder_layers(model):
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
        shards.append(layer)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_root)
    shards.append(model)
    # Our branch decouples the backend from env auto-wiring, so opt in
    # explicitly via the public set_custom_all_gather API (one per param group).
    if os.environ.get("MORI_FSDP_ENABLE_HIER"):
        from mori_allgather import MoriAllGather

        ag = MoriAllGather()
        if os.environ.get("MORI_FSDP_ROOT_ONLY"):
            model.set_custom_all_gather(ag)
        else:
            for m in shards:
                m.set_custom_all_gather(ag)
    elif os.environ.get("MORI_FSDP_ENABLE_SDMA"):
        from torch.distributed.fsdp._fully_shard._mori_sdma_allgather import (
            MoriSdmaAllGather,
        )
        zc = os.environ.get("MORI_FSDP_ZERO_COPY_OUTPUT", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        ag = MoriSdmaAllGather(zero_copy_output=zc)
        if os.environ.get("MORI_FSDP_ROOT_ONLY"):
            model.set_custom_all_gather(ag)
        else:
            for m in shards:
                m.set_custom_all_gather(ag)

    # MORI_FSDP_FWD_PREFETCH=D (default OFF): explicit forward-prefetch depth.
    # The residual gap is the CU-free SDMA-intra fill on the big AGs sitting on the
    # serial all_gather_stream with too small an overlap window. Issue each decoder
    # layer's AG D layers earlier from the CPU so the CU-free SDMA/RDMA fill overlaps
    # more forward GEMM compute. Since the per-layer AGs
    # (reshard_after_forward=True) free and recycle their buffer, depth is capped to
    # what the deferred landing fence covers.
    # Shipped-safe guard: depth>=2 races the deferred copy-out fence (two AGs in
    # flight, so the deferred landing fence covers only one) and the loss drifts, so
    # depth is hard-clamped to 1 unless MORI_FSDP_FWD_PREFETCH_UNSAFE=1 explicitly
    # opts into the drifting deeper depth for measurement only (never shipped).
    _fwd_pf = os.environ.get("MORI_FSDP_FWD_PREFETCH", "").strip()
    if _fwd_pf and _fwd_pf not in ("0", "false", "False"):
        depth = max(1, int(_fwd_pf))
        if depth > 1 and os.environ.get("MORI_FSDP_FWD_PREFETCH_UNSAFE", "") not in (
            "1", "true", "True", "yes", "on",
        ):
            depth = 1
        layers = list(_iter_decoder_layers(model))
        for i, layer in enumerate(layers):
            nxt = layers[i + 1 : i + 1 + depth]
            if nxt:
                layer.set_modules_to_forward_prefetch(nxt)

    # LEVER (MORI_FSDP_BIG_PREFETCH=1, default OFF): target the ACTUAL long pole --
    # the giant embed/lm_head AG (the 54/46 profile: it IS the whole step). Decoder
    # FWD_PREFETCH above helped 7% but leaves the giant AG exposed. Requires
    # MORI_FSDP_SPLIT_ROOT so embed_tokens + lm_head are their OWN fully_shard units;
    # then wire the LAST decoder layer to forward-prefetch the lm_head group, issuing
    # the giant lm_head AG from the CPU during the last transformer layer's forward
    # GEMM -- manufacturing the compute window the transport levers cannot reach.
    # Keeps ONE big AG in flight at a time (last-layer decoder AG has already landed
    # + resharded before lm_head runs) so the deferred landing fence stays valid.
    if os.environ.get("MORI_FSDP_BIG_PREFETCH", "") not in ("", "0", "false", "False"):
        layers = list(_iter_decoder_layers(model))
        lm_head = getattr(model, "lm_head", None)
        if layers and lm_head is not None and hasattr(lm_head, "set_modules_to_forward_prefetch"):
            layers[-1].set_modules_to_forward_prefetch([lm_head])
        # backward: layers[0] runs last in the backward pass before the root/embed
        # group is re-gathered -- prefetch the embed group's backward AG into it.
        embed = getattr(getattr(model, "model", model), "embed_tokens", None)
        if layers and embed is not None and hasattr(layers[0], "set_modules_to_backward_prefetch"):
            layers[0].set_modules_to_backward_prefetch([embed])


def _estimate_training_tflops(num_params: int, tokens: int, step_time_s: float) -> float:
    # Dense transformer training is commonly approximated as 6 FLOPs per
    # parameter per token. This is a model-level estimate, not a profiler count.
    return 6.0 * num_params * tokens / step_time_s / 1e12


def _make_batches(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    total_steps: int,
    device: torch.device,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batches = []
    for _ in range(total_steps):
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
            generator=generator,
        )
        batches.append((input_ids, input_ids.clone()))
    return batches


def _run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> float:
    input_ids, labels = batch
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def main() -> None:
    args = _parse_args()
    if args.profile_dir is not None:
        if args.profile_start_step < 0:
            raise ValueError("--profile-start-step must be non-negative")
        if args.profile_steps <= 0:
            raise ValueError("--profile-steps must be positive when --profile-dir is set")
    _configure_mode(args.mode)
    rank, local_rank, world_size, device = _init_distributed(args.backend)
    _init_mori_shmem_if_needed(args.mode)

    dtype = _get_torch_dtype(args.dtype)
    torch.manual_seed(_BENCHMARK_SEED + rank)
    torch.cuda.manual_seed_all(_BENCHMARK_SEED + rank)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if dtype in (torch.bfloat16, torch.float16)
        else nullcontext()
    )

    if rank == 0:
        print("Loading Qwen model config and initializing weights...", flush=True)
    model, config = _load_qwen_model(args.model_name_or_path, dtype)
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(
            f"Applying FSDP2 layer-by-layer sharding "
            f"(num_params={num_params:,})...",
            flush=True,
        )
    _apply_fsdp2(model, dtype=dtype, reshard_root=args.reshard_root)
    model.train()
    if rank == 0:
        print("FSDP2 model is ready; starting optimizer setup and benchmark.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dist.barrier()
    torch.cuda.synchronize(device)

    measured_times: list[float] = []
    measured_losses: list[float] = []
    total_steps = args.warmup + args.steps
    batches = _make_batches(
        config.vocab_size,
        args.micro_batch_size,
        args.seq_len,
        total_steps,
        device,
        seed=_BENCHMARK_SEED + 100_000 + rank,
    )
    dist.barrier()
    torch.cuda.synchronize(device)
    profiler = None
    profiled_steps = 0
    for step in range(total_steps):
        measured_step_for_profile = step - args.warmup
        should_start_profile = (
            args.profile_dir is not None
            and profiler is None
            and profiled_steps == 0
            and measured_step_for_profile == args.profile_start_step
        )
        if should_start_profile:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            profiler.__enter__()

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with autocast_ctx:
            loss = _run_step(
                model,
                optimizer,
                batches[step],
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        if profiler is not None:
            profiler.step()
            profiled_steps += 1
            if profiled_steps >= args.profile_steps:
                trace_path = Path(args.profile_dir) / f"trace_rank{rank}.json"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                profiler.__exit__(None, None, None)
                profiler.export_chrome_trace(str(trace_path))
                profiler = None
                if rank == 0:
                    print(
                        f"Exported PyTorch profiler traces to {args.profile_dir}",
                        flush=True,
                    )

        if step >= args.warmup:
            measured_step = step - args.warmup
            measured_times.append(elapsed)
            measured_losses.append(loss)
            should_print = (
                rank == 0
                and args.print_every > 0
                and (
                    (measured_step + 1) % args.print_every == 0
                    or measured_step + 1 == args.steps
                )
            )
            if should_print:
                recent_times = measured_times[-args.print_every :]
                tokens = args.micro_batch_size * args.seq_len * world_size
                avg_time = sum(recent_times) / len(recent_times)
                tflops = _estimate_training_tflops(num_params, tokens, avg_time)
                print(
                    f"steps={measured_step - len(recent_times) + 1}-{measured_step} "
                    f"mode={args.mode} avg_time_s={avg_time:.6f} "
                    f"min_time_s={min(recent_times):.6f} "
                    f"max_time_s={max(recent_times):.6f} "
                    f"tokens_per_s={tokens / avg_time:.2f} "
                    f"tflops={tflops:.2f} "
                    f"tflops_per_gpu={tflops / world_size:.2f} "
                    f"loss={loss:.6f}",
                    flush=True,
                )

    if profiler is not None:
        trace_path = Path(args.profile_dir) / f"trace_rank{rank}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(str(trace_path))
        if rank == 0:
            print(f"Exported PyTorch profiler traces to {args.profile_dir}", flush=True)

    local = torch.tensor(
        [
            sum(measured_times),
            min(measured_times),
            max(measured_times),
            len(measured_times),
        ],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(local, op=dist.ReduceOp.SUM)
    avg_step_time = local[0].item() / local[3].item()
    avg_tokens_per_s = (
        args.micro_batch_size * args.seq_len * world_size / avg_step_time
    )
    avg_tflops = _estimate_training_tflops(
        num_params,
        args.micro_batch_size * args.seq_len * world_size,
        avg_step_time,
    )

    if rank == 0:
        summary = {
            "mode": args.mode,
            "world_size": world_size,
            "seq_len": args.seq_len,
            "micro_batch_size": args.micro_batch_size,
            "steps": args.steps,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "seed": _BENCHMARK_SEED,
            "num_params": num_params,
            "avg_step_time_s": avg_step_time,
            "avg_tokens_per_s": avg_tokens_per_s,
            "avg_tflops": avg_tflops,
            "avg_tflops_per_gpu": avg_tflops / world_size,
            "last_loss": measured_losses[-1] if measured_losses else None,
            "mori_fsdp_enable_sdma": os.environ.get("MORI_FSDP_ENABLE_SDMA"),
            "mori_fsdp_zero_copy_output": os.environ.get("MORI_FSDP_ZERO_COPY_OUTPUT"),
            "mori_enable_sdma": os.environ.get("MORI_ENABLE_SDMA"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    _finalize_mori_shmem_if_needed(args.mode)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

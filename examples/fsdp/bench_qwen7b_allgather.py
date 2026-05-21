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


_use_local_fsdp()
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


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
        default=1,
        help="Rank 0 prints every N measured steps.",
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
    for layer in _iter_decoder_layers(model):
        fully_shard(layer, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(model, mp_policy=mp_policy, reshard_after_forward=reshard_root)


def _make_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    return input_ids, labels


def _run_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> float:
    input_ids, labels = _make_batch(vocab_size, batch_size, seq_len, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def main() -> None:
    args = _parse_args()
    _configure_mode(args.mode)
    rank, local_rank, world_size, device = _init_distributed(args.backend)
    _init_mori_shmem_if_needed(args.mode)

    dtype = _get_torch_dtype(args.dtype)
    torch.manual_seed(1234 + rank)
    torch.cuda.manual_seed_all(1234 + rank)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if dtype in (torch.bfloat16, torch.float16)
        else nullcontext()
    )

    if rank == 0:
        print("Loading Qwen model config and initializing weights...", flush=True)
    model, config = _load_qwen_model(args.model_name_or_path, dtype)
    if rank == 0:
        print("Applying FSDP2 layer-by-layer sharding...", flush=True)
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
    for step in range(total_steps):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with autocast_ctx:
            loss = _run_step(
                model,
                optimizer,
                config.vocab_size,
                args.micro_batch_size,
                args.seq_len,
                device,
            )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        if step >= args.warmup:
            measured_step = step - args.warmup
            measured_times.append(elapsed)
            measured_losses.append(loss)
            if rank == 0 and measured_step % args.print_every == 0:
                tokens = args.micro_batch_size * args.seq_len * world_size
                print(
                    f"step={measured_step} mode={args.mode} "
                    f"time_s={elapsed:.6f} tokens_per_s={tokens / elapsed:.2f} "
                    f"loss={loss:.6f}",
                    flush=True,
                )

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

    if rank == 0:
        summary = {
            "mode": args.mode,
            "world_size": world_size,
            "seq_len": args.seq_len,
            "micro_batch_size": args.micro_batch_size,
            "steps": args.steps,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "avg_step_time_s": avg_step_time,
            "avg_tokens_per_s": avg_tokens_per_s,
            "last_loss": measured_losses[-1] if measured_losses else None,
            "mori_fsdp_enable_sdma": os.environ.get("MORI_FSDP_ENABLE_SDMA"),
            "mori_enable_sdma": os.environ.get("MORI_ENABLE_SDMA"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    _finalize_mori_shmem_if_needed(args.mode)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

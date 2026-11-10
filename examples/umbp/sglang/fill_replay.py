# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Generate new deterministic prefixes and require UMBP DRAM hits on replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

# The image keeps the HTTP smoke script beside this file; a checkout keeps it
# in examples/umbp/. Support both layouts without duplicating the helper.
sys.path.append(str(Path(__file__).resolve().parent.parent))
from sglang_linker_smoke import request_json, save


def make_workload(tokenizer, prefixes, input_tokens, seed):
    rng = random.Random(seed)
    words = (
        "cache memory page request compute token document model buffer process "
        "reader writer record state layer index pool address storage batch"
    ).split()
    rows = []
    for number in range(prefixes):
        marker = f"Record {number:04d} / {rng.getrandbits(128):032x}. "
        parts = [marker, "Read the following technical notes and describe the cache. "]
        # More text than needed, then truncate the encoded IDs exactly.
        for _ in range(input_tokens):
            parts.append(rng.choice(words) + " ")
        ids = tokenizer.encode("".join(parts), add_special_tokens=True)
        if len(ids) < input_tokens:
            raise RuntimeError("Tokenizer produced too few tokens for this workload")
        ids = ids[:input_tokens]
        if any(type(token) is not int or token < 0 for token in ids):
            raise RuntimeError("Tokenizer produced invalid input IDs")
        rows.append(
            {
                "id": f"prefix-{number:04d}",
                "input_ids": ids,
                "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
        )
    if len({tuple(row["input_ids"][:256]) for row in rows}) != prefixes:
        raise RuntimeError("Prefixes must differ within the first token page")
    return rows


def one_request(url, row, timeout, max_new_tokens):
    start = time.perf_counter()
    response = request_json(
        url + "/generate",
        timeout,
        {
            "input_ids": row["input_ids"],
            "stream": False,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
    )
    meta = response.get("meta_info") or {}
    if (meta.get("finish_reason") or {}).get("type") == "abort":
        raise RuntimeError(f"{row['id']}: generation aborted")
    if not isinstance(response.get("text"), str) or not response["text"].strip():
        raise RuntimeError(f"{row['id']}: empty generated text")
    if meta.get("completion_tokens") != max_new_tokens:
        raise RuntimeError(f"{row['id']}: output token count mismatch")
    if meta.get("prompt_tokens") != len(row["input_ids"]):
        raise RuntimeError(f"{row['id']}: input token count mismatch")
    details = meta.get("cached_tokens_details")
    if not isinstance(details, dict):
        raise RuntimeError(f"{row['id']}: missing cached_tokens_details")
    for name in ("device", "host", "storage"):
        if type(details.get(name, 0)) is not int or details.get(name, 0) < 0:
            raise RuntimeError(f"{row['id']}: invalid {name} cache count")
    return {
        "id": row["id"],
        "input_sha256": row["sha256"],
        "elapsed_s": time.perf_counter() - start,
        "response": response,
    }


def run_stage(args, name, rows):
    save(
        args.out / f"{name}-server-info-before.json",
        request_json(args.url + "/get_server_info", args.timeout),
    )
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(one_request, args.url, row, args.timeout, args.max_new_tokens)
            for row in rows
        ]
        try:
            for future in as_completed(futures):
                result = future.result()
                save(args.out / f"{name}-{result['id']}.json", result)
                results.append(result)
                print(
                    json.dumps(
                        {
                            "stage": name,
                            "id": result["id"],
                            "cached_tokens_details": result["response"]["meta_info"][
                                "cached_tokens_details"
                            ],
                        }
                    ),
                    flush=True,
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise
    details = [
        result["response"]["meta_info"]["cached_tokens_details"] for result in results
    ]
    stat = {
        "stage": name,
        "requests": len(results),
        "wall_s": time.perf_counter() - start,
        "cache_tokens": {
            key: sum(detail.get(key, 0) for detail in details)
            for key in ("device", "host", "storage")
        },
        "cache_hit_requests": {
            key: sum(detail.get(key, 0) > 0 for detail in details)
            for key in ("device", "host", "storage")
        },
    }
    save(args.out / f"{name}-summary.json", stat)
    return stat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("UMBP_TEST_URL", "http://127.0.0.1:18000")
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, default=24)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--seed", type=int, default=352644285)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    if args.prefixes < 2 or args.input_tokens < 256 or args.input_tokens % 256:
        parser.error("prefixes >= 2 and page-aligned input-tokens >= 256 are required")
    if not 1 <= args.concurrency <= 8 or args.max_new_tokens <= 0 or args.timeout <= 0:
        parser.error(
            "concurrency must be 1..8; token limit and timeout must be positive"
        )
    args.url = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        from transformers import AutoTokenizer

        info = request_json(args.url + "/get_server_info", args.timeout)
        save(args.out / "server-info.json", info)
        for key, expected in {
            "enable_unified_cache_external_linker": True,
            "unified_cache_external_linker_backend": "mori",
            "enable_hierarchical_cache": False,
            "tp_size": 4,
            "max_total_tokens": 65536,
        }.items():
            if info.get(key) != expected:
                raise RuntimeError(f"Server setting mismatch: {key}={info.get(key)!r}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model), local_files_only=True, trust_remote_code=True
        )
        rows = make_workload(tokenizer, args.prefixes, args.input_tokens, args.seed)
        save(
            args.out / "workload.json",
            {
                "kind": "new-functional-workload",
                "seed": args.seed,
                "input_tokens": args.input_tokens,
                "rows": rows,
            },
        )
        summary = [run_stage(args, "fill", rows)]
        time.sleep(3)  # Allow asynchronous cache writeback to settle.
        summary.append(run_stage(args, "replay", rows))
        save(args.out / "summary.json", summary)
        if summary[-1]["cache_hit_requests"]["host"] == 0:
            raise RuntimeError(
                "Replay had no DRAM hit; GPU hits alone do not verify UMBP restore. "
                "See model/standalone logs and workload size."
            )
        save(
            args.out / "complete.json",
            {
                "ok": True,
                "kind": "new-functional-workload",
                "requests": 2 * args.prefixes,
                "replay_dram_hit_requests": summary[-1]["cache_hit_requests"]["host"],
            },
        )
        return 0
    except Exception as exc:
        save(args.out / "error.json", {"ok": False, "error": str(exc)})
        print(f"Fill/replay failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Send two requests to an existing SGLang Mori Direct Linker server."""

import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def request_json(url, timeout, body=None):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url", default=os.environ.get("UMBP_TEST_URL", "http://127.0.0.1:18000")
    )
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="Explain what a KV cache does during language model inference.",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_new_tokens <= 0 or not args.prompt.strip():
        parser.error("timeout, token limit and prompt must be positive/nonempty")
    url = args.url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        info = request_json(url + "/get_server_info", args.timeout)
        save(args.out / "server-info.json", info)
        expected = {
            "enable_unified_cache_external_linker": True,
            "unified_cache_external_linker_backend": "mori",
            "enable_hierarchical_cache": False,
        }
        for key, value in expected.items():
            if info.get(key) != value:
                raise RuntimeError(
                    f"Server setting mismatch: {key}={info.get(key)!r}, expected {value!r}"
                )

        body = {
            "text": args.prompt,
            "stream": False,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
            },
        }
        results = []
        for number in (1, 2):
            start = time.perf_counter()
            response = request_json(url + "/generate", args.timeout, body)
            elapsed = time.perf_counter() - start
            save(args.out / f"response-{number}.json", response)
            meta = response.get("meta_info") or {}
            tokens = meta.get("completion_tokens")
            if (
                not isinstance(response.get("text"), str)
                or not response["text"].strip()
            ):
                raise RuntimeError(f"Request {number}: empty response text")
            if type(tokens) is not int or tokens <= 0:
                raise RuntimeError(f"Request {number}: invalid completion token count")
            if (meta.get("finish_reason") or {}).get("type") == "abort":
                raise RuntimeError(f"Request {number}: generation aborted")
            result = {
                "request": number,
                "elapsed_s": elapsed,
                "completion_tokens": tokens,
                "cached_tokens": meta.get("cached_tokens"),
                "cached_tokens_details": meta.get("cached_tokens_details"),
            }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        save(
            args.out / "complete.json", {"ok": True, "requests": 2, "results": results}
        )
        return 0
    except Exception as exc:
        save(args.out / "error.json", {"ok": False, "error": str(exc)})
        print(f"Smoke check failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

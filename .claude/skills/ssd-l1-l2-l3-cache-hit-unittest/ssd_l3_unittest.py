#!/usr/bin/env python3
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
"""L1/L2/L3 cache-tier isolation unit test -- N rounds, single long-lived
engine (no restart between rounds).

Per round:
  1. Flush ALL 3 tiers: /flush_cache (L1+L2) + /clear_hicache_storage_backend
     (L3, via the real HTTP endpoint -- confirmed working, no restart needed).
  2. Send A -> genuine cold recompute (cached_tokens should be 0).
  3. Wait for the async write-through ack to L3 to drain.
  4. Flush L1/L2 only (leave L3 populated).
  5. Resend A -> expect an L3-only hit.
  -> loop back to step 1 for the next round.

No "B" eviction-driver request is used: flushing L1/L2 directly forces the
L3-only path for the resend.

Uses raw input_ids (not text) so token counts are exact.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

from transformers import AutoTokenizer

MODEL_PATH = "/agent_ci/models/Kimi-K2.6-MXFP4"


def build_ids(tok, n_tokens, unit):
    reps = (n_tokens // 4) + 100
    while True:
        ids = tok(unit * reps, add_special_tokens=False)["input_ids"]
        if len(ids) >= n_tokens:
            return ids[:n_tokens]
        reps *= 2


def send(endpoint, ids, label):
    payload = {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    t1 = time.time()
    meta = body.get("meta_info", {})
    print(
        f"[{label}] len={len(ids)} wall_time={t1 - t0:.3f}s "
        f"cached_tokens={meta.get('cached_tokens')} "
        f"e2e_latency={meta.get('e2e_latency')}"
    )
    return body


def _post_with_retry(url, label, max_retries=10, retry_wait=1.0):
    req = urllib.request.Request(url, data=b"", method="POST")
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode()
            print(f"[{label}] (attempt {attempt}) {body.strip()}")
            return
        except urllib.error.HTTPError as e:
            print(
                f"[{label}] (attempt {attempt}) HTTP {e.code}, retrying "
                f"in {retry_wait}s (pending async backup/prefetch bookkeeping "
                f"not yet drained)"
            )
            time.sleep(retry_wait)
    raise RuntimeError(f"{label} failed after {max_retries} attempts")


def flush_cache(base_url, **kw):
    _post_with_retry(base_url + "/flush_cache", "flush_cache", **kw)


def clear_l3(base_url, **kw):
    _post_with_retry(base_url + "/clear_hicache_storage_backend", "clear_l3", **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True, help="http://<ip>:30000/generate")
    p.add_argument("--a-len", type=int, default=4096)
    p.add_argument("--settle-secs", type=float, default=3.0)
    p.add_argument("--rounds", type=int, default=10)
    args = p.parse_args()

    base_url = args.endpoint.rsplit("/generate", 1)[0]

    print(f"Loading tokenizer from {MODEL_PATH} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    a_ids = build_ids(tok, args.a_len, "Alpha request filler segment marker. ")

    for r in range(1, args.rounds + 1):
        print(f"\n################ ROUND {r}/{args.rounds} ################")

        print(
            "\n=== Flushing all 3 tiers (L1+L2 via flush_cache, L3 via "
            "clear_hicache_storage_backend) ==="
        )
        flush_cache(base_url)
        clear_l3(base_url)
        time.sleep(1.0)

        print(
            f"\n=== Step 1: send A (expect genuine cold recompute, "
            f"len={len(a_ids)}) ==="
        )
        send(args.endpoint, a_ids, f"r{r}-A-cold")

        print(
            f"\n=== Waiting {args.settle_secs}s for async write-through ack "
            f"to L3 to drain ==="
        )
        time.sleep(args.settle_secs)

        print("\n=== Flushing L1/L2 only (leave L3 populated) ===")
        flush_cache(base_url)
        time.sleep(1.0)

        print(f"\n=== Step 2: resend A (expect L3-only hit, len={len(a_ids)}) ===")
        send(args.endpoint, a_ids, f"r{r}-A-L3hit")


if __name__ == "__main__":
    main()

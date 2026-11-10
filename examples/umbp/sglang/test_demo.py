# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""CPU tests for new workload validation; no model packages or GPUs required."""

from contextlib import redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import fill_replay
import preflight


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return [1] + [ord(char) + 2 for char in text]


TRANSFORMERS = types.SimpleNamespace(
    AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply(
            {
                "enable_unified_cache_external_linker": True,
                "unified_cache_external_linker_backend": self.server.backend,
                "enable_hierarchical_cache": False,
                "tp_size": 4,
                "max_total_tokens": 65536,
            }
        )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        signature = hashlib.sha256(json.dumps(body["input_ids"]).encode()).hexdigest()
        with self.server.seen_lock:
            replay = signature in self.server.seen
            self.server.seen.add(signature)
        self.reply(
            {
                "text": "Cache response",
                "meta_info": {
                    "completion_tokens": body["sampling_params"]["max_new_tokens"],
                    "prompt_tokens": len(body["input_ids"]),
                    "finish_reason": {"type": self.server.finish_reason},
                    "cached_tokens_details": {
                        "device": 0,
                        "host": 256 if replay and self.server.host_hits else 0,
                        "storage": 0,
                    },
                },
            }
        )


class DemoTests(unittest.TestCase):
    def test_workload_is_deterministic_and_distinct(self):
        first = fill_replay.make_workload(Tokenizer(), 24, 4096, 17)
        self.assertEqual(first, fill_replay.make_workload(Tokenizer(), 24, 4096, 17))
        self.assertNotEqual(first, fill_replay.make_workload(Tokenizer(), 24, 4096, 18))
        self.assertEqual(len({tuple(row["input_ids"][:256]) for row in first}), 24)
        self.assertTrue(all(len(row["input_ids"]) == 4096 for row in first))

    def run_client(self, *, backend="mori", host_hits=True, finish_reason="length"):
        with tempfile.TemporaryDirectory() as directory:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.backend = backend
            server.host_hits = host_hits
            server.finish_reason = finish_reason
            server.seen = set()
            server.seen_lock = threading.Lock()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            output = Path(directory) / "client"
            argv = [
                "fill_replay.py",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                directory,
                "--out",
                str(output),
                "--prefixes",
                "2",
                "--input-tokens",
                "256",
            ]
            try:
                with (
                    patch("sys.argv", argv),
                    patch.dict("sys.modules", {"transformers": TRANSFORMERS}),
                    patch.object(fill_replay.time, "sleep"),
                    redirect_stdout(io.StringIO()),
                ):
                    code = fill_replay.main()
                complete = output / "complete.json"
                result = json.loads(
                    (
                        complete if complete.exists() else output / "error.json"
                    ).read_text()
                )
                return code, result, len(server.seen)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_dram_replay_passes_and_records_completion(self):
        code, result, requests = self.run_client()
        self.assertEqual(code, 0)
        self.assertEqual(result["requests"], 4)
        self.assertEqual(result["replay_dram_hit_requests"], 2)
        self.assertEqual(requests, 2)

    def test_gpu_only_replay_does_not_pass_dram_check(self):
        code, result, _ = self.run_client(host_hits=False)
        self.assertEqual(code, 1)
        self.assertIn("no DRAM hit", result["error"])

    def test_wrong_backend_rejected_before_generation(self):
        code, result, requests = self.run_client(backend="other")
        self.assertEqual(code, 1)
        self.assertIn("setting mismatch", result["error"])
        self.assertEqual(requests, 0)

    def test_aborted_generation_fails(self):
        code, result, _ = self.run_client(finish_reason="abort")
        self.assertEqual(code, 1)
        self.assertIn("aborted", result["error"])

    def test_missing_weight_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text('{"model_type":"deepseek_v4"}')
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"layer":"missing.safetensors"}}'
            )
            with patch.dict("sys.modules", {"transformers": TRANSFORMERS}):
                with self.assertRaises(FileNotFoundError):
                    preflight.model_check(root)


if __name__ == "__main__":
    unittest.main()

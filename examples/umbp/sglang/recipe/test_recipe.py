"""CPU-only recipe tests. No Torch, Mori, model, Docker, or GPU commands."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run
import replay


class RecipeTests(unittest.TestCase):
    def command(self, mode, **overrides):
        env = run.environment(os.environ, 'kernel_sync', Path('/validated'))
        env.update(UMBP_E2E_ADDRESS='unix:///tmp/test.sock', **overrides)
        return shlex.split(run.capture_with_env(
            ['bash', str(run.ROOT / 'launch-model.sh'), mode, '--dry-run'], env))

    def test_modes_use_exclusive_backends(self):
        native, direct = self.command('hicache'), self.command('umbp')
        self.assertIn('--enable-hierarchical-cache', native)
        self.assertNotIn('--enable-unified-cache-external-linker', native)
        self.assertNotIn('--enable-hierarchical-cache', direct)
        self.assertIn('--enable-unified-cache-external-linker', direct)
        extra = json.loads(direct[direct.index('--hicache-storage-backend-extra-config') + 1])
        self.assertEqual(extra, {'ssd_enabled': False, 'standalone_auto_start': False})
        for command in (native, direct):
            self.assertNotIn('--speculative-algorithm', command)
            self.assertNotIn('--enable-prefill-cp', command)
            self.assertEqual(command[command.index('--hicache-ratio') + 1], '58.7890625')
            self.assertEqual(command[command.index('--max-total-tokens') + 1], '65536')
            self.assertEqual(command[command.index('--chunked-prefill-size') + 1], '512')

    def test_model_path_is_one_argument(self):
        command = self.command('umbp', MODEL_PATH='/models/model with spaces', PORT='18234')
        self.assertEqual(command[command.index('--model-path') + 1], '/models/model with spaces')
        self.assertEqual(command[command.index('--port') + 1], '18234')

    def test_agentx_gpu_capacity(self):
        command = self.command('umbp', GPU_KV_TOKENS='4194304', CHUNKED_PREFILL_SIZE='16384')
        self.assertEqual(command[command.index('--max-total-tokens') + 1], '4194304')
        self.assertEqual(command[command.index('--chunked-prefill-size') + 1], '16384')
        for mode, budget in (('hicache', '4194304'), ('umbp', '65537')):
            result = subprocess.run([sys.executable, str(run.ROOT / 'run.py'), mode,
                                     '--source', '/unused', '--out', '/unused',
                                     '--gpu-kv-tokens', budget, '--dry-run'],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_invalid_prefill_chunk(self):
        result = subprocess.run([sys.executable, str(run.ROOT / 'run.py'), 'umbp',
                                 '--source', '/unused', '--out', '/unused',
                                 '--chunked-prefill-size', '513', '--dry-run'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_inherited_modes_are_cleared(self):
        parent = dict(UMBP_MASTER_ADDRESS='unwanted', UMBP_DISTRIBUTED_MEDIUM='HBM',
                      AMD_SERIALIZE_COPY='3', HIP_LAUNCH_BLOCKING='1',
                      SGLANG_SIMULATE_ACC_LEN='3.77', NCCL_MIN_NCHANNELS='112',
                      HIP_VISIBLE_DEVICES='2', PRIVATE_TOKEN='do-not-print')
        sync = run.environment(parent, 'kernel_sync', '/validated')
        async_env = run.environment(parent, 'async', '/validated')
        self.assertEqual(sync['AMD_SERIALIZE_KERNEL'], '3')
        self.assertEqual(sync['HIP_LAUNCH_BLOCKING'], '1')
        self.assertNotIn('AMD_SERIALIZE_COPY', sync)
        for key in run.SYNC:
            self.assertNotIn(key, async_env)
        for key in ('UMBP_MASTER_ADDRESS', 'UMBP_DISTRIBUTED_MEDIUM', 'NCCL_MIN_NCHANNELS',
                    'SGLANG_SIMULATE_ACC_LEN', 'HIP_VISIBLE_DEVICES'):
            self.assertNotIn(key, sync)

    def test_dry_run_has_no_runtime_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'must-not-exist'
            env = dict(os.environ, PRIVATE_TOKEN='never-show-this-token')
            result = subprocess.run([sys.executable, str(run.ROOT / 'run.py'), 'umbp',
                                     '--source', '/not-installed', '--out', str(out), '--dry-run'],
                                    env=env, text=True, capture_output=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan['nominal_payload_bytes'], 128 << 30)
            self.assertFalse(out.exists())
            self.assertNotIn('never-show-this-token', result.stdout)

    def test_amd_smi_schema_and_unknown_pid(self):
        data = [{'process_list': [{'process_info': 'No running processes detected'}]} for _ in range(4)]
        data[2]['process_list'] = [{'process_info': {'pid': 1234}}]
        with patch.object(run, 'capture', return_value=json.dumps(data)):
            self.assertEqual(run.gpu_pids(), {1234})
        with patch.object(run, 'capture', return_value='{}'):
            with self.assertRaises(RuntimeError):
                run.gpu_pids()
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertFalse(run.owns(99999999))

    def test_fault_deltas_ignore_timestamps(self):
        before = run.messages('<4>[ 1.0] old warning\n')
        after = run.messages('<4>[ 2.0] old warning\n<3>[ 3.0] GPU memory fault\n')
        self.assertEqual(list(after - before), ['GPU memory fault'])
        self.assertTrue(any(run.FAULT.search(line) for line in after - before))

    def test_replay_uses_hashes_when_ids_repeat(self):
        rows = []
        for i in range(84):
            ids = [i + 1] * 4096
            row = dict(id=f'prefix-{i % 80}', input_ids=ids,
                       sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            if i < 8:
                row['expected'] = '1234'
            rows.append(row)
        workload = dict(fits=rows[4:28], pressure=rows[28:])
        with patch.object(Path, 'read_text', side_effect=[json.dumps(workload), json.dumps(rows[:4])]):
            stages = replay.load_inputs(Path('workload'), Path('correctness'))
        self.assertEqual(sum(len(r) for _, r, _ in stages), 168)
        self.assertEqual(sum(sum('expected' in r for r in batch) for _, batch, _ in stages), 16)
        workload['fits'][0]['sha256'] = 'bad-hash'
        with patch.object(Path, 'read_text', side_effect=[json.dumps(workload), json.dumps(rows[:4])]):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                replay.load_inputs(Path('workload'), Path('correctness'))

    def test_truncated_stream_is_not_success(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_lines.return_value = [b'data: {"text":"1234", "meta_info":{"completion_tokens":2}}']
        with patch.object(replay.requests, 'post', return_value=response):
            result = replay.one('http://unused', dict(id='code', sha256='input', input_ids=[1], expected='1234'))
        self.assertFalse(result['ok'])
        self.assertIn('Incomplete SSE', result['error'])

    def test_cleanup_terminates_owned_process_group(self):
        child = subprocess.Popen([sys.executable, '-c',
                                  'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
                                  'print("ready", flush=True); time.sleep(60)'],
                                 stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            run.stop_processes([child], grace_seconds=0.05)
            self.assertEqual(child.returncode, -9)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
            child.stdout.close()


if __name__ == '__main__':
    unittest.main()

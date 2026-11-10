"""Run the TP4 DRAM recipe inside a dedicated container with host PID visibility."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
CAPACITY = 128 << 30
SYNC = ('AMD_SERIALIZE_KERNEL', 'AMD_SERIALIZE_COPY', 'HIP_LAUNCH_BLOCKING')
FAULT = re.compile(r'GPU memory fault|page fault|VM_L2_PROTECTION|ring .*timeout|GPU reset', re.I)
BINARY_HASH = '7a1eaff3d0c9ce4c4b8d1c853b888151d38666acac93a9d0b4f86f686dbc4c62'


def environment(parent, execution, source):
    env = {k: v for k, v in parent.items()
           if not k.startswith(('UMBP_', 'SGLANG_SIMULATE_ACC_'))
           and k not in (*SYNC, 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                         'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS',
                         'FAULT_AUDIT_DIR', 'FAULT_BLOCK_MORI')}
    env.update(ROCR_VISIBLE_DEVICES='0,1,2,3', HSA_DISABLE_COREDUMP_ON_EXCEPTION='1',
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               PYTHONPATH=f'{source}:/sgl-workspace/aiter',
               UMBP_DRAM_CAPACITY=str(CAPACITY), UMBP_DRAM_USE_HUGEPAGES='0',
               UMBP_SSD_ENABLED='0', MORI_UMBP_LOG_LEVEL='INFO')
    if execution == 'kernel_sync':
        env.update(AMD_SERIALIZE_KERNEL='3', HIP_LAUNCH_BLOCKING='1')
    return env


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def capture(command):
    return subprocess.check_output(command, text=True, timeout=20)


def gpu_pids():
    data = json.loads(capture(['amd-smi', 'process', '--json']))
    if not isinstance(data, list) or len(data) != 4:
        raise RuntimeError('Expected the validated four-device amd-smi JSON format')
    return {int(p['process_info']['pid']) for g in data for p in g['process_list']
            if p.get('process_info') != 'No running processes detected'}


def owns(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == os.getpid():
            return True
        seen.add(pid)
        try:
            status = Path(f'/proc/{pid}/status').read_text()
        except FileNotFoundError:
            return False
        pid = int(re.search(r'^PPid:\s+(\d+)', status, re.M)[1])
    return False


def messages(raw):
    return Counter(re.sub(r'^<\d+>\[\s*[\d.]+\]\s*', '', line) for line in raw.splitlines())


def kernel_log():
    return capture(['dmesg', '--raw', '--nopager'])


def stop_processes(children, grace_seconds=15):
    # Include groups whose leaders already exited, since workers may remain.
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    for child in reversed(children):
        try:
            child.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    stuck = []
    for child in reversed(children):
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            stuck.append(child.pid)
    if stuck:
        raise RuntimeError(f'Owned processes did not exit after SIGKILL: {stuck}')


def wait_idle(out, quiet_seconds):
    devices = sorted(p for p in Path('/sys/bus/pci/devices').glob('*/mem_info_vram_used')
                     if (p.parent / 'vendor').read_text().strip() == '0x1002')
    if len(devices) != 4:
        raise RuntimeError('Expected four AMD GPU memory counters in sysfs')
    start, quiet_since = time.monotonic(), None
    with (out / 'idle.jsonl').open('x') as log:
        while True:
            now = time.monotonic()
            pids = gpu_pids()
            vram = [int(p.read_text()) for p in devices]
            quiet_since = (quiet_since if quiet_since is not None else now) if not pids and max(vram) < 1 << 30 else None
            elapsed = now - quiet_since if quiet_since is not None else 0
            row = dict(epoch=time.time(), pids=sorted(pids), vram=vram, quiet_seconds=elapsed)
            log.write(json.dumps(row) + '\n')
            log.flush()
            print(json.dumps(row), flush=True)
            if elapsed >= quiet_seconds:
                return
            if now - start >= 1800:
                raise TimeoutError('No idle window within 30 minutes')
            time.sleep(5)


def validate(info, model_log, standalone_log, mode, gpu_kv_tokens=65536, chunked_prefill_size=512):
    expected = dict(enable_hierarchical_cache=mode == 'hicache',
                    enable_unified_cache_external_linker=mode == 'umbp',
                    hicache_storage_backend=None, tp_size=4, dp_size=1, ep_size=1,
                    max_total_tokens=gpu_kv_tokens, chunked_prefill_size=chunked_prefill_size,
                    random_seed=352644285, hicache_ratio=58.7890625,
                    enable_prefill_cp=False, disable_overlap_schedule=False,
                    disable_cuda_graph=True, speculative_algorithm=None, kv_cache_dtype='bfloat16')
    for key, value in expected.items():
        if key not in info or info[key] != value:
            raise RuntimeError(f'Server setting mismatch: {key}={info.get(key)!r}, expected {value!r}')
    if mode == 'umbp':
        if info.get('unified_cache_external_linker_backend') != 'mori':
            raise RuntimeError('Expected the Mori Direct Linker')
        ranks = re.findall(r'TP(\d+)\] UMBPDirectLinker topology=StandaloneProcess\+\S+ '
                           r'ranged_io=yes rank_replicated=True suffix=cp0_pp0', model_log)
        attached = re.findall(r'TP(\d+)\] Tree cache initialized:.*hicache_attached=False', model_log)
        sizes = re.findall(r'Init tier=\d+ source=HostPageMemorySource loc=1 device=-1 buffers=1 total_bytes=(\d+)', standalone_log)
        if sorted(ranks) != ['0', '1', '2', '3'] or sorted(attached) != ['0', '1', '2', '3']:
            raise RuntimeError('Direct Linker was not confirmed on all four ranks')
        if sizes != [str(CAPACITY)] or 'source=HbmPageMemorySource' in standalone_log:
            raise RuntimeError('Expected exactly one 128 GiB DRAM standalone pool')
        if 'host memory for V4' in model_log:
            raise RuntimeError('Unexpected native HiCache allocation in the Direct arm')
        return dict(payload_bytes=CAPACITY, medium='DRAM', host_cache='UMBP Direct')
    pools = {}
    pattern = (r'TP(\d+)\] Allocating [\d.]+ GB host memory for V4 paged pool '
               r"'([^']+)' \(layers=(\d+), pages=(\d+), item_bytes=(\d+), layout=page_first\)")
    for rank, name, layers, pages, item in re.findall(pattern, model_log):
        key = (rank, name)
        if key in pools:
            raise RuntimeError('Duplicate host pool allocation')
        pools[key] = tuple(map(int, (layers, pages, item)))
    layout = {'deepseek_v4_c4': (30, 15050, 65536),
              'deepseek_v4_c4_indexer': (30, 15050, 8448),
              'deepseek_v4_c128': (31, 15050, 2048)}
    if pools != {(str(rank), name): shape for rank in range(4) for name, shape in layout.items()}:
        raise RuntimeError('Native HiCache capacity differs from the validated V4 layout')
    attached = re.findall(r'TP(\d+)\] Tree cache initialized:.*hicache_attached=True', model_log)
    if sorted(attached) != ['0', '1', '2', '3'] or 'UMBPDirectLinker topology=' in model_log:
        raise RuntimeError('Native HiCache was not confirmed on all four ranks')
    if 'host memory for V4 state pool' in model_log:
        raise RuntimeError('Unexpected additional host state pool')
    return dict(payload_bytes=sum(a * b * c for a, b, c in pools.values()),
                medium='DRAM', host_cache='native HiCache')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('hicache', 'umbp'))
    parser.add_argument('--source', type=Path, required=True, help='Validated Python snapshot containing sglang/')
    parser.add_argument('--manifest', type=Path, help='Historical source-manifest.json; required for execution')
    parser.add_argument('--out', type=Path, required=True, help='New run directory')
    parser.add_argument('--execution', choices=('kernel_sync', 'async'), default='kernel_sync')
    parser.add_argument('--binary', default='/sgl-workspace/mori/python/mori/umbp_standalone_server')
    parser.add_argument('--idle-seconds', type=int, default=300)
    parser.add_argument('--timeout', type=int, default=3600, help='Maximum seconds after the idle gate')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gpu-kv-tokens', type=int, default=65536,
                        help='Direct arm logical GPU KV budget; use a larger pool for AgentX')
    parser.add_argument('--chunked-prefill-size', type=int, default=512,
                        help='Prefill chunk tokens; 16384 follows the TP4 AgentX recipe')
    parser.add_argument('--client', nargs=argparse.REMAINDER, help='Optional benchmark command after readiness')
    args = parser.parse_args()
    if not 60 <= args.idle_seconds < 1800 or args.timeout <= 0:
        parser.error('idle-seconds must be 60..1799; timeout must be positive')
    if args.client == []:
        parser.error('--client requires a command')
    if args.gpu_kv_tokens < 65536 or args.gpu_kv_tokens % 256:
        parser.error('--gpu-kv-tokens must be a page-aligned integer >= 65536')
    if args.chunked_prefill_size <= 0 or args.chunked_prefill_size % 256:
        parser.error('--chunked-prefill-size must be a positive multiple of 256')
    if args.mode == 'hicache' and args.gpu_kv_tokens != 65536:
        parser.error('Native HiCache ratio must be recalculated before changing the GPU KV budget')
    source, out = args.source.resolve(), args.out.resolve()
    env = environment(os.environ, args.execution, source)
    env['GPU_KV_TOKENS'] = str(args.gpu_kv_tokens)
    env['CHUNKED_PREFILL_SIZE'] = str(args.chunked_prefill_size)
    address = 'unix:///tmp/umbp-dry-run/server.sock'
    env['UMBP_E2E_ADDRESS'] = address
    command = ['bash', str(ROOT / 'launch-model.sh'), args.mode]
    preview = capture_with_env(command + ['--dry-run'], env)
    plan = dict(mode=args.mode, execution=args.execution, source=str(source),
                model_command=preview.strip(), standalone_command=[args.binary, address] if args.mode == 'umbp' else None,
                nominal_payload_bytes=CAPACITY, gpu_kv_tokens=args.gpu_kv_tokens,
                chunked_prefill_size=args.chunked_prefill_size, client=args.client,
                environment={k: v for k, v in env.items() if k in SYNC or k.startswith('UMBP_')
                             or k in ('PYTHONPATH', 'ROCR_VISIBLE_DEVICES')})
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.manifest:
        parser.error('--manifest is required for execution')
    manifest = json.loads(args.manifest.read_text())['integrated']
    if not manifest:
        raise RuntimeError('Empty source manifest')
    for name, expected in manifest.items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source manifest mismatch: {name}')
    if hashlib.sha256(Path(args.binary).read_bytes()).hexdigest() != BINARY_HASH:
        raise RuntimeError('Mori standalone binary differs from the validated build')
    for port in (int(env.get('PORT', '18000')), int(env.get('NCCL_PORT', '19500'))):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'plan.json', plan)
    save(out / 'source.json', dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                                 binary_sha256=BINARY_HASH))
    children, required, returncodes = [], [], []
    error, baseline, started = None, None, None
    private = None

    def stop_signal(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    def spawn(argv, name, permanent=False):
        with (out / name).open('ab') as log:
            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        if permanent:
            required.append(child)
        return child

    def check():
        if time.monotonic() - started > args.timeout:
            raise TimeoutError('Run timeout')
        if any(p.poll() is not None for p in required):
            raise RuntimeError('A required server exited; see process logs')
        foreign = [pid for pid in gpu_pids() if not owns(pid)]
        if foreign:
            raise RuntimeError(f'Foreign GPU processes: {foreign}')
        delta = messages(kernel_log()) - baseline
        if any(FAULT.search(line) for line in delta):
            raise RuntimeError('New GPU fault in kernel log')
        log = out / 'model.log'
        if log.exists() and re.search(r'Memory access fault|Memory Fault Error|Scheduler hit an exception|Traceback \(most recent call last\)', log.read_text(errors='replace')):
            raise RuntimeError('Model logged a fatal error')

    def finish(child, seconds):
        deadline = time.monotonic() + seconds
        while child.poll() is None:
            check()
            if time.monotonic() > deadline:
                raise TimeoutError('Probe/client timeout')
            time.sleep(1)
        check()
        # Retire short-lived probes now; their PIDs may be reused during serving.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        children.remove(child)
        returncodes.append(child.returncode)
        return child.returncode

    try:
        kernel_log()  # Verify permission before waiting for resources.
        wait_idle(out, args.idle_seconds)
        before = kernel_log()
        (out / 'dmesg-before.txt').write_text(before)
        baseline, started = messages(before), time.monotonic()
        (out / 'metrics-before.json').write_text(capture(['amd-smi', 'metric', '--ecc', '--mem-usage', '--json']))
        if finish(spawn([sys.executable, str(ROOT / 'probe.py'), 'gpu'], 'preflight.log'), 120):
            raise RuntimeError('GPU preflight failed')
        private = tempfile.TemporaryDirectory(prefix='umbp-tp4-', dir='/tmp')
        address = f'unix://{private.name}/server.sock'
        env['UMBP_E2E_ADDRESS'] = address
        if args.mode == 'umbp':
            spawn([args.binary, address], 'standalone.log', permanent=True)
            deadline = time.monotonic() + 180
            while True:
                probe = spawn([sys.executable, str(ROOT / 'probe.py'), 'standalone', '--address', address], 'standalone-probe.log')
                status = finish(probe, 20)
                if status == 0:
                    break
                if status != 75:
                    raise RuntimeError('Standalone client failed; see standalone-probe.log')
                if time.monotonic() > deadline:
                    raise TimeoutError('Standalone readiness timeout')
                time.sleep(1)
        save(out / 'runtime.json', dict(address=address if args.mode == 'umbp' else None,
                                      hip_environment={k: env.get(k) for k in SYNC}))
        spawn(command, 'model.log', permanent=True)
        url = f"http://127.0.0.1:{env.get('PORT', '18000')}"
        deadline = time.monotonic() + 1200
        while True:
            check()
            try:
                with urllib.request.urlopen(url + '/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Model readiness timeout')
            time.sleep(2)
        with urllib.request.urlopen(url + '/get_server_info', timeout=30) as response:
            info = json.load(response)
        save(out / 'server-info.json', info)
        standalone_log = (out / 'standalone.log').read_text() if args.mode == 'umbp' else ''
        capacity = validate(info, (out / 'model.log').read_text(), standalone_log, args.mode,
                            args.gpu_kv_tokens, args.chunked_prefill_size)
        save(out / 'capacity.json', capacity)
        save(out / 'ready.json', dict(url=url, **capacity))
        print(f'READY {args.mode}: {url}; logs: {out}', flush=True)
        if args.client:
            env['UMBP_TEST_URL'] = url
            if finish(spawn(args.client, 'client.log'), args.timeout):
                raise RuntimeError('Benchmark client failed')
            check()
        else:
            while True:
                check()
                time.sleep(2)
    except Exception as exc:
        error = repr(exc)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            stop_processes(children)
            if private is not None:
                private.cleanup()
        except Exception as exc:
            error = f'{error}; cleanup: {exc!r}' if error else repr(exc)
        try:
            after = kernel_log()
            (out / 'dmesg-after.txt').write_text(after)
            if baseline is not None:
                delta = messages(after) - baseline
                (out / 'dmesg-new.txt').write_text('\n'.join(delta.elements()))
                if any(FAULT.search(line) for line in delta):
                    error = error or 'New GPU fault in final kernel log'
            (out / 'metrics-after.json').write_text(capture(['amd-smi', 'metric', '--ecc', '--mem-usage', '--json']))
            save(out / 'processes-after.json', sorted(gpu_pids()))
        except Exception as exc:
            error = error or repr(exc)
        save(out / 'state.json', dict(error=error, returncodes=returncodes + [p.returncode for p in children]))
    if error:
        print(error, file=sys.stderr)
    return 1 if error else 0


def capture_with_env(command, env):
    return subprocess.check_output(command, env=env, text=True, timeout=20)


if __name__ == '__main__':
    raise SystemExit(main())

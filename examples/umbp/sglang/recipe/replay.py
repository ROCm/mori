"""Replay the archived 168-request workload against an already verified server."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import requests


def load_inputs(workload_path, correctness_path):
    workload = json.loads(workload_path.read_text())
    correctness = json.loads(correctness_path.read_text())
    if set(workload) != {'fits', 'pressure'} or [len(workload[k]) for k in ('fits', 'pressure')] != [24, 56]:
        raise ValueError('Expected the saved 24/56-request results/workload.json')
    if len(correctness) != 4:
        raise ValueError('Expected four cold/GPU-hit retrieval requests')
    rows = correctness + workload['fits'] + workload['pressure']
    for row in rows:
        ids = row['input_ids']
        if len(ids) != 4096 or any(type(x) is not int or x < 0 for x in ids):
            raise ValueError(f"Invalid input IDs: {row['id']}")
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != row['sha256']:
            raise ValueError(f"Input hash mismatch: {row['id']}")
    if len({r['sha256'] for r in rows}) != 84:
        raise ValueError('Expected 84 unique prefixes')
    if any('expected' not in row for row in correctness) or sum('expected' in r for r in rows) != 8:
        raise ValueError('Expected eight distinct retrieval inputs (16 checks with replay)')
    return [('correctness-cold', correctness, 1), ('correctness-gpu-hit', correctness, 1),
            ('fits-fill', workload['fits'], 2), ('fits-reuse', workload['fits'], 2),
            ('pressure-fill', workload['pressure'], 2), ('pressure-reuse', workload['pressure'], 2)]


def one(url, row):
    limit = 16 if 'expected' in row else 32
    start, first, last, done = time.perf_counter(), None, None, False
    result = dict(id=row['id'], sha256=row['sha256'], input_tokens=len(row['input_ids']))
    body = dict(input_ids=row['input_ids'], stream=True, sampling_params=dict(
        temperature=0, max_new_tokens=limit, ignore_eos='expected' not in row))
    try:
        with requests.post(url + '/generate', json=body, stream=True, timeout=(10, 240)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                last = json.loads(payload)
                if first is None and last.get('meta_info', {}).get('completion_tokens', 0) > 0:
                    first = time.perf_counter() - start
        if last is None or not done:
            raise RuntimeError('Incomplete SSE response')
        meta, text = last.get('meta_info', {}), last.get('text') or ''
        if 'expected' in row:
            valid = re.findall(r'(?<!\d)\d{4}(?!\d)', text) == [row['expected']]
            result.update(expected=row['expected'], answer_ok=valid)
        else:
            valid = meta.get('completion_tokens') == limit and len(text.strip()) >= 16
        result.update(meta=meta, text=text, ok=valid and (meta.get('finish_reason') or {}).get('type') != 'abort')
    except Exception as exc:
        result.update(ok=False, error=repr(exc))
    return dict(result, ttft_s=first, e2e_s=time.perf_counter() - start)


def percentile(values, q):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (pos - low)


def snapshot(url, out, name):
    for endpoint in ('get_server_info', 'metrics'):
        response = requests.get(url + '/' + endpoint, timeout=30)
        response.raise_for_status()
        (out / f'{name}-{endpoint}.txt').write_text(response.text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default=os.environ.get('UMBP_TEST_URL', 'http://127.0.0.1:18000'))
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--correctness', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stages = load_inputs(args.workload, args.correctness)
    args.out.mkdir(parents=True, exist_ok=False)
    for path, name in ((args.workload, 'workload.json'), (args.correctness, 'correctness.json')):
        (args.out / name).write_bytes(path.read_bytes())
    summary = []
    for name, rows, concurrency in stages:
        snapshot(args.url, args.out, name + '-before')
        start, results = time.perf_counter(), []
        with (args.out / f'{name}.jsonl').open('x') as log, ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(one, args.url, row) for row in rows]
            for future in as_completed(futures):
                if future.cancelled():
                    continue
                row = future.result()
                results.append(row)
                log.write(json.dumps(row) + '\n')
                log.flush()
                if not row['ok']:
                    for pending in futures:
                        pending.cancel()
        wall = time.perf_counter() - start
        good = [r for r in results if r['ok']]
        details = [r['meta'].get('cached_tokens_details') or {} for r in good]
        for detail in details:
            if any(type(detail.get(k, 0)) is not int or detail.get(k, 0) < 0 for k in ('device', 'host', 'storage')):
                raise RuntimeError('Invalid cache token counts')
        stat = dict(stage=name, requested=len(rows), completed=len(results), ok=len(good),
                    concurrency=concurrency, wall_s=wall,
                    output_tokens_s=sum(r['meta'].get('completion_tokens', 0) for r in good) / wall,
                    cache_tokens={k: sum(d.get(k, 0) for d in details) for k in ('device', 'host', 'storage')},
                    cache_hit_requests={k: sum(d.get(k, 0) > 0 for d in details) for k in ('device', 'host', 'storage')})
        for field in ('ttft_s', 'e2e_s'):
            for q in (.5, .95):
                stat[f'{field[:-2]}_p{int(q * 100)}_s'] = percentile([r[field] for r in good], q)
        summary.append(stat)
        (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        snapshot(args.url, args.out, name + '-after')
        print(json.dumps(stat), flush=True)
        if len(good) != len(rows):
            raise RuntimeError(f'{name}: response/retrieval check failed')
        time.sleep(3)
    (args.out / 'complete.json').write_text(json.dumps(dict(requests=168, retrieval_checks=16)) + '\n')


if __name__ == '__main__':
    main()

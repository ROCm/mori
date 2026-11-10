"""Short-lived GPU health and standalone readiness probes, outside timed requests."""
import argparse


def gpu():
    import gc
    import torch

    if torch.cuda.device_count() != 4:
        raise RuntimeError('This recipe requires exactly four visible GPUs')
    for device in range(4):
        for _ in range(3):
            x = torch.arange(4096, dtype=torch.int32, device=f'cuda:{device}')
            if not torch.equal((x * 3 + 7).cpu(), torch.arange(4096, dtype=torch.int32) * 3 + 7):
                raise RuntimeError(f'GPU {device} copy/compute check failed')
            del x
            gc.collect()
            torch.cuda.empty_cache()
    print(f'GPU preflight passed; torch={torch.__version__}, HIP={torch.version.hip}')


def standalone(address):
    import mori.umbp as u

    config = u.UMBPConfig()
    config.ssd.enabled = False
    process = u.UMBPStandaloneProcessConfig()
    process.address = address
    process.auto_start = False
    process.startup_timeout_ms = 10000
    config.standalone_process = process
    try:
        client = u.UMBPClient(config)
    except RuntimeError as exc:
        if str(exc).startswith('StandaloneProcessClient: server is not ready at '):
            return 75
        raise
    del client
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('gpu', 'standalone'))
    parser.add_argument('--address')
    args = parser.parse_args()
    if args.mode == 'gpu':
        gpu()
    else:
        if not args.address:
            parser.error('--address is required for standalone')
        raise SystemExit(standalone(args.address))

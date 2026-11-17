# torch symm_mem conformance

Runs PyTorch's own `test/distributed/test_symmetric_memory.py` with every
`torch.distributed._symmetric_memory` call routed to the MORI backend, and checks the
result against a list of documented exceptions (`expected.tsv`).

It is the test that answers "is MORI a complete torch symm_mem backend": the file is
torch's, unmodified, at the commit of the installed torch.

## How MORI is forced

`site/sitecustomize.py` runs at start-up in every interpreter that has this directory on
`PYTHONPATH` and `MORI_CONFORMANCE=1` set, including the worker processes torch's
distributed tests spawn. It registers the MORI backend and calls
`symm_mem.set_backend("MORI")`. If that fails it raises, so a broken build can never fall
back to torch's own backend and pass. It acts only in the environment under test (matched
by `sys.prefix`), and not in processes started while it runs: the backend's JIT probes the
GPU with a Python tool, which would otherwise load the backend again and recurse.

`run_conformance.py` sets this up, collects the test ids, and runs each id in its own
pytest process and process session: torch's `MultiProcContinuousTest` poisons a whole
class after one failure, and a hung peer must not take the next test down with it. After
each test it kills whatever is left in that session and records it as a leftover.

The runner turns core files off for itself and every test. A kernel that times out traps,
and with core files on, ROCm writes a GPU core file of gigabytes and aborts the process
instead of raising the error torch's timeout tests wait for.

## Views

torch skips most of this file on ROCm. The view decides which of those skips are lifted:

| view | lifted | why |
|---|---|---|
| A | none | what torch's own ROCm CI runs |
| B | `skip_if_rocm_ver_lessthan_multiprocess` (ROCm < 10.1 gates), `skip_if_rocm_arch_multiprocess` (MI350 / gfx950) and the `lazy_skip_if` gates whose reason names MI350 or ROCm 10.1 | these gate torch's own backend on CI runner problems, not on a missing feature |
| C | B, plus every `skip_if_rocm_multiprocess` | the full contract a backend can be held to on ROCm |

Platform skips stay in every view: no multicast on ROCm, NCCL symmetric memory and
torchcomms not built for ROCm, device-side assertions not enabled in ROCm builds.

The default is C. A view C run also reports view B: each test records whether it ran only
because C lifted its `skip_if_rocm_multiprocess`, and B counts exactly those as skipped.

## Running

Needs torch (ROCm), mori with its torch extension (prebuilt or JIT), `pytest`,
`pytest-timeout` and `expecttest`, and at least two GPUs. The tests use every visible
GPU, so a lane's GPU list sets the world size; `expected.tsv` was measured at world 4.

```bash
# test file fetched for the installed torch's git commit
python tests/python/allocator/torch_conformance/run_conformance.py \
    --fetch --out /tmp/conformance --gpus 0,1,2,3

# a local copy of the file, two lanes of four GPUs, view B only
python run_conformance.py --test-file ~/pytorch/test/distributed/test_symmetric_memory.py \
    --out /tmp/conformance --view B --gpus 0,1,2,3 --gpus 4,5,6,7
```

Useful options:

- `--python PATH` runs the tests in another environment than the runner's.
- `-k SUBSTRING` / `--ids FILE` select tests; `--resume` keeps the results already in `OUT`.
- `--wait-idle PCT` waits before each test until the lane's GPUs are below `PCT`% busy
  (shared machines).
- `--backend CUDA` runs the same file on torch's own backend, as a control.
- `--as-cuda` makes `get_backend()` answer `"CUDA"` for the forced backend. Some tests of
  a generic contract are gated on that name (the `test_stream_serialization_*` family);
  use it only together with `-k`, since other `"CUDA"`-gated tests assert the internals
  of torch's own backend.
- `--baseline OLD/summary.tsv` lists every status change against an earlier run.
- `--report-only` rebuilds the report from `OUT` (after editing `expected.tsv`).

Set `MORI_SYMM_FORCE_JIT=1` and a fresh `TORCH_EXTENSIONS_DIR` to test the backend
sources rather than a prebuilt module.

## Output

`OUT/summary.tsv` has one row per test: status (`PASSED`, `FAILED`, `ERROR`, `SKIPPED`,
`TIMEOUT` for a test killed at `--per-test-timeout`, `NO_RESULT` when pytest wrote no
result), whether view C lifted its skip, seconds, leftover processes, and the most
telling error line or the skip reason. Per-test logs and JUnit XML are in `OUT/logs`.

`OUT/report.txt` has the counts per view, the expected failures and skips by category,
and three lists that need action:

- **UNEXPECTED failures**: a failure no line of `expected.tsv` covers.
- **UNEXPECTED skips**: a skip no line covers. A test that stops running is lost
  coverage, not a pass.
- **expected to fail but passed**: delete the line.

The exit status is 1 when either unexpected list is non-empty.

## expected.tsv

One line per documented exception, seven tab-separated fields:

```
status  torch  view  test  cause  category  note
```

- `status`: `FAIL` (failed or errored), `SKIP`, `TIMEOUT` or `ANY`.
- `torch`: glob on `torch.__version__`, e.g. `2.15.*` or `*`.
- `view`: the views the line applies to, e.g. `AB`, or `*` for all. A line never excuses
  a result in a view it does not name.
- `test`: glob on `Class::test_name`.
- `cause`: regular expression searched in the error line or skip reason (`.` for any).
- `category` and `note`: why, for the report.

The first matching line wins. Keep lines narrow: a broad pattern would also hide a new
failure.

## Exception categories

| category | what it covers |
|---|---|
| `kept-by-view` | torch's own ROCm skips that views A and B keep, scoped to those views by the `view` field |
| `rocm-no-*` | features ROCm builds of torch do not have: multicast, cuda-python, NCCL symmetric memory, torchcomms, device-side assertions, async matmul |
| `gated-on-backend-name` | tests that skip unless the backend is named `"CUDA"`; `--as-cuda` runs the generic ones on MORI |
| `cuda-backend-internal` | tests that pin `TORCH_SYMMMEM=CUDA` or assert how torch's own backend allocates its tables |
| `cuda-only-api` | torch utilities that exist for CUDA only (NVLink connectivity detection) |
| `torch-name-dispatch` | `symm_mem.get` and friends pick their implementation by backend name and reject any other |
| `torch-build` | an op the torch wheel under test was built without |
| `rocshmem-only`, `world-size`, `not-native-allocator` | tests for another backend or setup |

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
import os
import sys
import time
import torch
import torch.distributed as dist
import socket
from multiprocessing import Queue
import mori
import traceback


# Where a worker's SIGUSR1 stack dump lands. One file per rank -- see the long
# comment in TorchDistProcessManager._worker for why a shared stderr does not
# work at world 8. Overridable so a run can keep its dumps beside its log.
STACK_DUMP_DIR = os.environ.get("MORI_TEST_STACK_DUMP_DIR", "/tmp")


def stack_dump_path(rank, pid=None):
    """Path rank `rank`'s faulthandler dump is written to.

    Keyed on the WORKER's pid, which both sides can name: the worker passes
    nothing (defaults to its own getpid()), and the parent -- which is the
    consumer, in assert_worker_results' timeout branch -- passes
    manager.processes[rank].pid. Keying on the pid and not on the rank alone
    matters because the pool is session-scoped and gets restarted: two
    successive worker sets would otherwise write the same eight paths, and a
    stale dump from a previous test read as the current wedge is worse than no
    dump at all.
    """
    pid = os.getpid() if pid is None else pid
    return os.path.join(STACK_DUMP_DIR, f"mori_stack_pid{pid}_rank{rank}.txt")


# The worker's dump file, held at module scope so the fd outlives _worker's
# frame -- faulthandler keeps the raw fd and writes nothing if it is closed.
_STACK_DUMP_FILE = None


str_to_dtype = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "half": torch.float16,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def string_to_dtype(s):
    s = s.lower()
    if s not in str_to_dtype:
        raise ValueError(f"Unknown dtype string: {s}")
    return str_to_dtype[s]


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def data_type_supported(dtype):
    arch = torch.cuda.get_device_capability(0)
    arch_int = int("".join(map(str, arch)))
    if dtype is torch.float8_e4m3fnuz:
        return arch_int == 94
    if dtype is torch.float8_e4m3fn:
        return arch_int >= 95
    if dtype is torch.float4_e2m1fn_x2:
        return arch_int >= 95
    return True


class TorchDistContext:
    def __init__(
        self,
        rank,
        world_size,
        master_addr="localhost",
        master_port="12335",
        device_id=None,
        backend="cpu:gloo,cuda:nccl",
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.device_id = device_id if device_id is not None else self.rank
        self.backend = backend

    def __enter__(self):
        if self.master_addr is not None:
            os.environ["MASTER_ADDR"] = self.master_addr
        if self.master_port is not None:
            os.environ["MASTER_PORT"] = str(self.master_port)

        torch.cuda.set_device(self.device_id)
        device = torch.device("cuda", self.device_id)

        dist.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )

        world_group = torch.distributed.group.WORLD
        assert world_group is not None
        torch._C._distributed_c10d._register_process_group("default", world_group)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not dist.is_initialized():
            return
        # This barrier is teardown convenience only -- every test already
        # reports its own result through the manager's result queue before we
        # get here, so nothing observable depends on it. Left unguarded it is
        # actively harmful: the worker loop calls mori.shmem.shmem_finalize()
        # just before this, and on some ranks the following device barrier
        # then raises `HIP error: invalid argument`. That rank leaves __exit__
        # by exception, never reaching destroy_process_group(), while its
        # peers block in the same barrier forever -- so the parent's join()
        # never returns and pytest never prints its summary. Every run then
        # looks like a hang in the code under test. Swallow it, and always
        # destroy the group.
        try:
            dist.barrier()
        except BaseException as exc:  # noqa: BLE001 - teardown must not strand peers
            print(
                f"[rank {self.rank}] teardown barrier failed, continuing to "
                f"destroy_process_group: {type(exc).__name__}: {exc}",
                flush=True,
            )
        try:
            dist.destroy_process_group()
        except BaseException as exc:  # noqa: BLE001
            print(
                f"[rank {self.rank}] destroy_process_group failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )


class TorchDistProcessManager:
    def __init__(self, init_mori_shmem=True):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.processes = []
        self.init_mori_shmem = init_mori_shmem

    @staticmethod
    def _worker(rank, world_size, port, init_shmem, task_queue, result_queue):
        # Make a wedged worker able to say WHERE it is wedged, on demand.
        #
        # The dominant failure shape in this suite is: every rank ALIVE
        # (exitcode None), 0/N reported, the parent times out after 300s, and
        # nothing anywhere names the line the ranks are stopped on. That has
        # been re-derived from the parent's view across six turns of this
        # campaign and the answer has never once come from the inference -- it
        # came from reading source or from a real stack. So make the stack
        # obtainable: on SIGUSR1 every thread of this worker dumps its Python
        # frames to stderr, which pytest -s passes through to the run log.
        # faulthandler's handler is async-signal-safe (raw write(2), no
        # allocation), so it still fires from a thread blocked in a gloo
        # collective -- which is exactly when it is wanted.
        #
        # It cannot unwind a rank spinning inside a HIP kernel; that shows as
        # the launching Python frame. But distinguishing "wedged in python/
        # gloo teardown" from "spinning in device code" is precisely the open
        # question behind review #47-2, and the launching frame answers it.
        #
        # PER-RANK FILE, not the shared stderr. The first version of this
        # dumped to the inherited stderr, and at world 8 that is unreadable:
        # T25d's log holds 32 `Thread 0x` blocks from 8 ranks writing the same
        # fd concurrently, byte-interleaved, with embedded NULs -- `tr -d
        # '\000'` is needed even to grep it, and no single rank's stack can be
        # reassembled from it. faulthandler makes no attempt to write a dump
        # atomically (it emits one raw write per LINE), so interleaving is
        # guaranteed, not unlucky. An instrument built to name the cause of a
        # wedge that cannot be read at the world size the wedge happens at is
        # not an instrument.
        #
        # The fd must stay open for the life of the process: faulthandler
        # stores the raw fd, and writing to a closed one silently produces
        # nothing. So bind it to a module-global rather than a local, and never
        # close it.
        import faulthandler
        import signal

        faulthandler.enable()
        if hasattr(faulthandler, "register"):
            global _STACK_DUMP_FILE
            try:
                _STACK_DUMP_FILE = open(stack_dump_path(rank), "w")
            except OSError:
                _STACK_DUMP_FILE = None
            faulthandler.register(
                signal.SIGUSR1,
                file=_STACK_DUMP_FILE if _STACK_DUMP_FILE is not None else sys.stderr,
                all_threads=True,
                chain=False,
            )

        with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
            if init_shmem:
                mori.shmem.shmem_torch_process_group_init("default")
            while True:
                task = task_queue.get()
                if task == "STOP":
                    if init_shmem:
                        mori.shmem.shmem_finalize()
                    break
                func, args = task
                try:
                    result = func(rank, *args)
                    result_queue.put((rank, result))
                except BaseException:
                    # BaseException, not Exception: pytest's Skipped/Failed and
                    # SystemExit derive from BaseException, and letting one
                    # escape here exits the loop without ever reporting, which
                    # blocks the parent's collective get() forever.
                    result_queue.put((rank, traceback.format_exc()))

    def start_workers(self, world_size):
        port = get_free_port()
        self.processes = [
            torch.multiprocessing.Process(
                target=self._worker,
                args=(
                    rank,
                    world_size,
                    port,
                    self.init_mori_shmem,
                    self.task_queue,
                    self.result_queue,
                ),
            )
            for rank in range(world_size)
        ]
        for p in self.processes:
            p.start()

    def shutdown(self):
        # Bounded, because this is session teardown: an unbounded join() on a
        # rank wedged in a collective (a peer already dead, a device barrier
        # that raised) hangs pytest AFTER every test has passed, so the run
        # produces no summary line and reads as a failure of the code under
        # test. Terminate the stragglers and name them instead.
        timeout = float(os.environ.get("MORI_TEST_SHUTDOWN_TIMEOUT", "60"))
        for _ in range(len(self.processes)):
            self.task_queue.put("STOP")
        deadline = time.monotonic() + timeout
        for p in self.processes:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
        straggling = [i for i, p in enumerate(self.processes) if p.is_alive()]
        if straggling:
            print(
                f"[manager] shutdown: ranks {straggling} did not exit within "
                f"{timeout:.0f}s; terminating. This does NOT invalidate results "
                f"already reported -- they were queued before teardown.",
                flush=True,
            )
            for i in straggling:
                self.processes[i].terminate()
            for i in straggling:
                self.processes[i].join(timeout=10)
                if self.processes[i].is_alive():
                    self.processes[i].kill()

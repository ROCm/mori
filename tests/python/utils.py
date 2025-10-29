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
import torch
import torch.distributed as dist
import socket
from datetime import timedelta
from multiprocessing import Queue
import mori
import traceback


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


class KeyErrorMessage(str):
    r"""str subclass that returns itself in repr"""

    def __repr__(self):
        return self


class ExceptionWrapper:
    r"""Wraps an exception plus traceback to communicate across processes"""

    def __init__(self, rank, exc_info=None):
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = f"in rank {rank}"

    def reraise(self):
        r"""Reraises the wrapped exception in the current thread"""
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"
        if self.exc_type is KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            raise self.exc_type(message=msg)
        try:
            exception = self.exc_type(msg)
        except Exception:
            # If the exception takes multiple arguments or otherwise can't
            # be constructed, don't try to instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception


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
            timeout=timedelta(seconds=10),
        )

        world_group = torch.distributed.group.WORLD
        assert world_group is not None
        torch._C._distributed_c10d._register_process_group("default", world_group)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


class TorchDistProcessManager:
    def __init__(self, init_mori_shmem=True):
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.processes = []
        self.init_mori_shmem = init_mori_shmem
        self.on_error = False

    @staticmethod
    def _worker(rank, world_size, port, init_shmem, task_queue, result_queue):
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
                except Exception:
                    result_queue.put((rank, ExceptionWrapper(rank=rank)))

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
        for idx in range(len(self.processes)):
            self.task_queue.put("STOP")
        for p in self.processes:
            if self.on_error and p.is_alive():
                p.terminate()
            p.join()

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
"""The JIT build lock and outputs: a build that dies must not leave the lock held
or a partial output behind. CPU only."""

import errno
import multiprocessing as mp
import os
import signal
import threading
import time

import pytest

import mori.jit.core as jit_core
from mori.jit.core import FileBaton, _atomic_output

_CTX = mp.get_context("fork")


def _hold(lock_path, ready, release):
    with FileBaton(lock_path):
        ready.set()
        release.wait(60)


def _hold_until_killed(lock_path, ready):
    FileBaton(lock_path).__enter__()
    ready.set()
    time.sleep(120)


def _enter_in_thread(baton):
    entered = threading.Event()

    def run():
        baton.__enter__()
        entered.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, entered


def test_leftover_lock_file_does_not_block(tmp_path):
    # What a killed build (or the old O_EXCL lock) leaves behind.
    lock = tmp_path / ".kernel.lock"
    lock.touch()
    start = time.monotonic()
    with FileBaton(lock) as baton:
        assert not baton.skipped
    assert time.monotonic() - start < 5


def test_lock_is_released_when_the_holder_is_killed(tmp_path):
    lock = str(tmp_path / ".kernel.lock")
    ready = _CTX.Event()
    holder = _CTX.Process(target=_hold_until_killed, args=(lock, ready))
    holder.start()
    assert ready.wait(30)
    os.kill(holder.pid, signal.SIGKILL)
    holder.join(30)
    start = time.monotonic()
    with FileBaton(lock) as baton:
        assert not baton.skipped
    assert time.monotonic() - start < 5


def test_waiter_blocks_until_the_holder_releases(tmp_path):
    lock = str(tmp_path / ".kernel.lock")
    ready, release = _CTX.Event(), _CTX.Event()
    holder = _CTX.Process(target=_hold, args=(lock, ready, release))
    holder.start()
    try:
        assert ready.wait(30)
        baton = FileBaton(lock)
        thread, entered = _enter_in_thread(baton)
        assert not entered.wait(2)
        release.set()
        assert entered.wait(30)
        assert not baton.skipped
        baton.__exit__(None, None, None)
        thread.join(5)
    finally:
        release.set()
        holder.join(30)


def test_waiter_skips_once_the_target_exists(tmp_path):
    lock = str(tmp_path / ".kernel.lock")
    target = tmp_path / "kernel.hsaco"
    ready, release = _CTX.Event(), _CTX.Event()
    holder = _CTX.Process(target=_hold, args=(lock, ready, release))
    holder.start()
    try:
        assert ready.wait(30)
        baton = FileBaton(lock, wait_for=target)
        thread, entered = _enter_in_thread(baton)
        assert not entered.wait(1)
        target.touch()
        assert entered.wait(30)
        assert baton.skipped
        baton.__exit__(None, None, None)
        thread.join(5)
    finally:
        release.set()
        holder.join(30)


def test_lock_error_names_the_lock_file_and_closes_it(tmp_path, monkeypatch):
    # What NFS returns for an exclusive lock on the read-only fallback descriptor.
    def flock(fd, op):
        raise OSError(errno.EBADF, os.strerror(errno.EBADF))

    monkeypatch.setattr(jit_core.fcntl, "flock", flock)
    lock = tmp_path / ".kernel.lock"
    fds = len(os.listdir("/proc/self/fd"))
    with pytest.raises(OSError, match=str(lock)):
        FileBaton(lock).__enter__()
    assert len(os.listdir("/proc/self/fd")) == fds


def test_failed_build_leaves_no_output(tmp_path):
    out = tmp_path / "kernel.hsaco"
    with pytest.raises(RuntimeError):
        with _atomic_output(out) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("build killed")
    assert list(tmp_path.iterdir()) == []


def test_output_appears_only_when_complete(tmp_path):
    out = tmp_path / "kernel.hsaco"
    with _atomic_output(out) as tmp:
        tmp.write_bytes(b"part")
        assert not out.exists()
        tmp.write_bytes(b"complete")
    assert out.read_bytes() == b"complete"
    assert list(tmp_path.iterdir()) == [out]

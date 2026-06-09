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
import threading
import time

from mori.cpp import TransferStatus
from mori.io import IOEngine, IOEngineConfig, StatusCode


def make_status_in_progress():
    status = TransferStatus()
    status.SetCode(StatusCode.IN_PROGRESS)
    return status


def test_wait_for_blocks_and_releases_gil():
    status = make_status_in_progress()

    def updater():
        time.sleep(0.05)
        status.Update(StatusCode.SUCCESS, "")

    other_thread_ran = []

    def sibling():
        time.sleep(0.01)
        other_thread_ran.append(time.perf_counter())

    updater_thread = threading.Thread(target=updater)
    sibling_thread = threading.Thread(target=sibling)
    updater_thread.start()
    sibling_thread.start()

    start = time.perf_counter()
    rc = status.WaitFor(timeout_ms=500)
    elapsed = time.perf_counter() - start

    updater_thread.join()
    sibling_thread.join()

    assert rc == StatusCode.SUCCESS
    assert 0.03 < elapsed < 0.30
    assert len(other_thread_ran) == 1


def test_wait_all_wrapper_success():
    engine = IOEngine("py_wait_all_success", IOEngineConfig(host="127.0.0.1", port=0))
    statuses = [make_status_in_progress() for _ in range(3)]

    def complete(status, delay):
        time.sleep(delay)
        status.Update(StatusCode.SUCCESS, "")

    threads = [
        threading.Thread(target=complete, args=(status, 0.01 * (idx + 1)))
        for idx, status in enumerate(statuses)
    ]
    for thread in threads:
        thread.start()

    rc = engine.wait_all(statuses, timeout_ms=500)
    for thread in threads:
        thread.join()

    assert rc == StatusCode.SUCCESS
    assert all(status.Succeeded() for status in statuses)


def test_wait_all_zero_timeout_failure_wins():
    engine = IOEngine(
        "py_wait_all_zero_failure", IOEngineConfig(host="127.0.0.1", port=0)
    )
    success = TransferStatus()
    in_progress = make_status_in_progress()
    failed = TransferStatus()
    success.SetCode(StatusCode.SUCCESS)
    failed.SetCode(StatusCode.ERR_RDMA_OP)

    rc = engine.wait_all([success, in_progress, failed], timeout_ms=0)
    assert rc == StatusCode.ERR_RDMA_OP

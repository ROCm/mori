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

"""Unit tests for new multi-session BatchRead/BatchWrite API.

This focuses on the new engine-level overloaded APIs that take vectors of:
  - memory descriptors (one per session)
  - offset lists (one list per session)
  - size lists (one list per session)
  - status pointers (returned; one per session)
  - transfer unique ids (one per session)

Existing tests in `test_engine.py` already cover the single-session batch form
where a single memory descriptor pair is supplied with per-transfer offsets.
Here we validate that multiple independent session pairs can be issued in a
single BatchRead / BatchWrite call and each completes successfully.
"""

import pytest
import torch

from tests.python.utils import get_free_port
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    RdmaBackendConfig,
    set_log_level,
)


# -----------------------------------------------------------------------------
# Helpers / Fixtures
# -----------------------------------------------------------------------------


def create_connected_engine_pair(
    name_prefix, qp_per_transfer=1, post_batch_size=-1, num_worker_threads=1
):
    """Create two RDMA-enabled IOEngines and register each other.

    Returns (initiator, target).
    """
    config = IOEngineConfig(host="127.0.0.1", port=get_free_port())
    initiator = IOEngine(key=f"{name_prefix}_initiator", config=config)
    config.port = get_free_port()
    target = IOEngine(key=f"{name_prefix}_target", config=config)

    be_cfg = RdmaBackendConfig(
        qp_per_transfer=qp_per_transfer,
        post_batch_size=post_batch_size,
        num_worker_threads=num_worker_threads,
    )
    initiator.create_backend(BackendType.RDMA, be_cfg)
    target.create_backend(BackendType.RDMA, be_cfg)

    initiator_desc = initiator.get_engine_desc()
    target_desc = target.get_engine_desc()
    initiator.register_remote_engine(target_desc)
    target.register_remote_engine(initiator_desc)

    return initiator, target


@pytest.fixture(scope="module")
def pre_connected_engine_pair():
    set_log_level("info")
    normal = create_connected_engine_pair(
        "multi_normal", qp_per_transfer=2, num_worker_threads=1
    )
    multhd = create_connected_engine_pair(
        "multi_multhd", qp_per_transfer=2, num_worker_threads=2
    )
    engines = {
        "normal": normal,
        "multhd": multhd,
    }
    yield engines
    # Cleanup references (explicit deregistration not strictly necessary here)
    del normal, multhd


def wait_status(status):
    while status.InProgress():
        pass


def wait_inbound_status(engine, remote_engine_key, transfer_uid):
    while True:
        target_side_status = engine.pop_inbound_transfer_status(
            remote_engine_key, transfer_uid
        )
        if target_side_status:
            return target_side_status


# -----------------------------------------------------------------------------
# Multi-session batch tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("engine_type", ("normal", "multhd"))
@pytest.mark.parametrize("op_type", ("read", "write"))
def test_multi_session_batch_read_write(
    pre_connected_engine_pair, engine_type, op_type
):
    """Issue a single multi-session BatchRead/BatchWrite with >1 memory pair.

    Layout:
      - For each session i we allocate independent tensors on device0 (initiator)
        and device1 (target) of length BATCH_SIZE * BUFFER_SIZE bytes.
      - We register each tensor to obtain MemoryDesc pairs.
      - We build vectors of (mem, offsets[], sizes[]) per session and call
        engine.batch_read/write with all sessions at once.
      - We then wait on each returned TransferStatus and validate data movement.
    """

    initiator, target = pre_connected_engine_pair[engine_type]

    NUM_SESSIONS = 3
    BATCH_SIZE = 4
    BUFFER_SIZE = 256  # bytes per transfer within a session
    TOTAL_SIZE = BATCH_SIZE * BUFFER_SIZE

    # Allocate tensors and register memory for each session.
    initiator_tensors = []
    target_tensors = []
    initiator_mems = []
    target_mems = []

    device0 = torch.device("cuda", 0)
    device1 = torch.device("cuda", 1)

    for i in range(NUM_SESSIONS):
        it = torch.randn(TOTAL_SIZE).to(device0, dtype=torch.uint8)
        tt = torch.randn(TOTAL_SIZE).to(device1, dtype=torch.uint8)
        initiator_tensors.append(it)
        target_tensors.append(tt)
        initiator_mems.append(initiator.register_torch_tensor(it))
        target_mems.append(target.register_torch_tensor(tt))

    # Build per-session batch parameters.
    # Offsets inside a session: contiguous segments.
    per_session_offsets = [
        [j * BUFFER_SIZE for j in range(BATCH_SIZE)] for _ in range(NUM_SESSIONS)
    ]
    per_session_sizes = [
        [BUFFER_SIZE for _ in range(BATCH_SIZE)] for _ in range(NUM_SESSIONS)
    ]

    # Allocate unique transfer IDs per session.
    transfer_ids = [initiator.allocate_transfer_uid() for _ in range(NUM_SESSIONS)]

    # Call batch_read / batch_write with vectors of descriptors.
    if op_type == "read":
        # Read: localDest <- remoteSrc (initiator receives remote data)
        statuses = initiator.batch_read(
            initiator_mems,
            per_session_offsets,
            target_mems,
            per_session_offsets,
            per_session_sizes,
            transfer_ids,
        )
    else:
        statuses = initiator.batch_write(
            initiator_mems,
            per_session_offsets,
            target_mems,
            per_session_offsets,
            per_session_sizes,
            transfer_ids,
        )

    assert len(statuses) == NUM_SESSIONS, "Expected one status per session"

    initiator_key = initiator.get_engine_desc().key

    # Wait & validate each session independently.
    for i in range(NUM_SESSIONS):
        st = statuses[i]
        wait_status(st)
        inbound = wait_inbound_status(target, initiator_key, transfer_ids[i])
        assert (
            st.Succeeded()
        ), f"Initiator status failed for session {i}: {st.Message()}"
        assert (
            inbound.Succeeded()
        ), f"Target status failed for session {i}: {inbound.Message()}"

        if op_type == "read":
            # After read, initiator tensor should equal original target tensor.
            assert torch.equal(
                initiator_tensors[i].cpu(), target_tensors[i].cpu()
            ), f"Data mismatch (read) on session {i}"
        else:
            # After write, target tensor should equal original initiator tensor.
            assert torch.equal(
                initiator_tensors[i].cpu(), target_tensors[i].cpu()
            ), f"Data mismatch (write) on session {i}"

    # Cleanup registrations.
    for m in initiator_mems:
        initiator.deregister_memory(m)
    for m in target_mems:
        target.deregister_memory(m)

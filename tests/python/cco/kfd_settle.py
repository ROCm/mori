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
"""Run a multi-rank launch and return once KFD has released its processes.

A rank's SDMA queues are freed when KFD tears its process down, which can lag the
process exit, so a launch started right after the previous one can find the
engines full (anvil.cpp). Waiting for the launch's own KFD processes to go away
replaces a fixed sleep before every launch.
"""

import os
import subprocess
import time

_KFD_PROC = "/sys/class/kfd/kfd/proc"


def kfd_pids():
    try:
        return set(os.listdir(_KFD_PROC))
    except OSError:
        return set()


def run(cmd, settle_timeout=20, **kwargs):
    """subprocess.run, then wait up to settle_timeout s for its KFD processes to go."""
    before = kfd_pids()
    result = subprocess.run(cmd, **kwargs)
    deadline = time.monotonic() + settle_timeout
    while kfd_pids() - before and time.monotonic() < deadline:
        time.sleep(0.1)
    return result

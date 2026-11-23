#!/usr/bin/env python3
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

"""Serve native FABRIC L1 only; coordinate owner/peer drain before shutdown."""

import argparse
import signal
import threading
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", required=True, help="Peer-reachable HOST:PORT")
    parser.add_argument(
        "--peer", action="append", default=[], help="Other tray's HOST:PORT"
    )
    parser.add_argument(
        "--kernel", required=True, type=Path, help="Compiled fabric_copy.hsaco"
    )
    args = parser.parse_args()
    if len(args.peer) > 8 or len(set(args.peer)) != len(args.peer):
        parser.error("use at most eight distinct peers")
    if not args.kernel.is_file():
        parser.error("--kernel must name an existing compiled FABRIC copy module")

    import mori_umbp_l1

    if getattr(mori_umbp_l1, "NATIVE_L1_FABRIC_VERSION", None) != 1:
        parser.error("install the matching MORI native L1 FABRIC Python module")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    server = mori_umbp_l1.start_fabric_server(
        args.listen, args.peer, str(args.kernel.resolve())
    )
    print(f"Native FABRIC L1 listening on {args.listen}; peers={args.peer}", flush=True)
    try:
        while not stop.wait(1):
            pass
    finally:
        # This stops RPC serving. It cannot prove that a lost peer has drained;
        # workers must retain their allocations until coordinated teardown.
        del server


if __name__ == "__main__":
    main()

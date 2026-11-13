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
"""xgmi-netmon: per-link XGMI bandwidth of AMD Instinct GPUs, pushed to VictoriaMetrics.

  xgmi_netmon.py list
  xgmi_netmon.py push --url http://127.0.0.1:28428 [-d 60]

Source: /sys/class/drm/cardN/device/gpu_metrics (MI355X; world-readable), v1.8 (fixed struct)
or v1.9 (self-describing, offsets parsed at startup). Fields used:
  xgmi_read_data_acc[8], xgmi_write_data_acc[8]   u64, KB, cumulative
  xgmi_link_width / xgmi_link_speed                u16, lanes / Gbps
  firmware_timestamp                               u64, 10 ns, PMFW time of the snapshot

PMFW refreshes the table every 2 ms (measured on MI355X). Each GPU is read every 2 ms on the
wall-clock grid, like ainic-netmon, and a sample is kept only when firmware_timestamp changed
(the host and PMFW clocks are not phase-locked, so some 2 ms reads see no new snapshot and some
intervals span two snapshots). The rate of an interval is

    delta_acc * 1024 / (delta_firmware_timestamp * 10 ns)

i.e. the interval comes from the firmware clock, not from when the host happened to read.
A missed snapshot loses no bytes (the counters are cumulative); the interval just gets longer.
A counter that decreased (driver reload) gives no point for that interval.

Pushed series (labels host, gpu, and peer/dir per link; dir = wr: sent by this GPU on the link,
rd: received; both ends of a link agree). Values are link bytes: a torch peer copy shows ~1.12x its
payload as wr (sender) / rd (receiver) and ~0.57x in the other direction (MI355X, measured):
  xgmi_bytes                     bytes in the interval
  xgmi_rate_bytes_per_second     bytes / interval
  xgmi_interval_seconds          interval length, per GPU
Point time = wall clock when the new snapshot was first read (<= ~2.2 ms after the snapshot).
"""
import argparse
import os
import signal
import socket
import struct
import sys
import threading
import time
import urllib.request

NUM_LINKS = 8
# gpu_metrics v1.9 (amdgpu 6.16.13+): header, int attr_count, then packed
# [u64 encoding, inst values] per field; encoding = unit<<24 | type<<20 | id<<10 | inst.
V19_TYPE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 8, 7: 8}
V19_ID = {"xgmi_width": 18, "xgmi_speed": 19, "rd": 29, "wr": 30, "fw": 32}


def layout(raw):
    """Byte offsets of the fields we need: {rd, wr, fw, xgmi_width, xgmi_speed}."""
    frev, crev = raw[2], raw[3]
    if (frev, crev) == (1, 8):
        return {"rd": 0x088, "wr": 0x0C8, "fw": 288, "xgmi_width": 72, "xgmi_speed": 74}
    if (frev, crev) != (1, 9):
        sys.exit(f"gpu_metrics v{frev}.{crev} not supported (known: v1.8, v1.9)")
    want = {v: k for k, v in V19_ID.items()}
    off, out = 8, {}
    for _ in range(struct.unpack_from("<i", raw, 4)[0]):
        (enc,) = struct.unpack_from("<Q", raw, off)
        typ, aid, inst = (enc >> 20) & 0xF, (enc >> 10) & 0x3FF, enc & 0x3FF
        off += 8
        if aid in want:
            name = want[aid]
            # rd/wr/fw must be u64 (8 links / 1 value), widths u16; anything else is a new layout
            exp = {"rd": (6, NUM_LINKS), "wr": (6, NUM_LINKS), "fw": (6, 1)}.get(
                name, (2, 1)
            )
            if (typ, inst) != exp:
                sys.exit(
                    f"gpu_metrics v1.9 field {name}: type {typ} x{inst}, expected type {exp[0]} x{exp[1]}"
                )
            out[name] = off
        off += V19_TYPE_SIZE[typ] * inst
    missing = set(V19_ID) - set(out)
    if missing:
        sys.exit(f"gpu_metrics v1.9 lacks {sorted(missing)}")
    return out


def discover():
    """GPUs with XGMI, in card order: [{idx, card, pci, path, link_max, links: {port: peer_idx}, ...}]."""
    drm = "/sys/class/drm"
    cards = sorted(
        (
            e
            for e in os.listdir(drm)
            if e.startswith("card")
            and e[4:].isdigit()
            and os.path.exists(f"{drm}/{e}/device/xgmi_device_id")
        ),
        key=lambda e: int(e[4:]),
    )
    gpus, node_to_idx = [], {}
    for idx, card in enumerate(cards):
        dev = f"{drm}/{card}/device"
        pairs = []  # (src_node, src_port, dst_node), lines like "01:05 ->  02:05"
        for line in open(f"{dev}/xgmi_port_num").read().split("\n"):
            if "->" in line:
                src, dst = line.split("->")
                s_node, s_port = (int(x) for x in src.strip().split(":"))
                pairs.append((s_node, s_port, int(dst.strip().split(":")[0])))
        if pairs:
            node_to_idx[pairs[0][0]] = idx
        pci = os.path.basename(os.path.realpath(dev))
        raw = open(f"{dev}/gpu_metrics", "rb").read()
        lo = layout(raw)
        (width,) = struct.unpack_from("<H", raw, lo["xgmi_width"])
        (gbps,) = struct.unpack_from("<H", raw, lo["xgmi_speed"])
        start = min(lo["rd"], lo["wr"], lo["fw"])
        end = max(lo["rd"] + 8 * NUM_LINKS, lo["wr"] + 8 * NUM_LINKS, lo["fw"] + 8)
        gpus.append(
            {
                "idx": idx,
                "card": card,
                "pci": pci,
                "path": f"{dev}/gpu_metrics",
                "pairs": pairs,
                "link_max": width * gbps * 1e9 / 8,  # bytes/s per direction
                "slice": slice(start, end),  # the bytes a sample keeps
                "rd_at": lo["rd"] - start,
                "wr_at": lo["wr"] - start,
                "fw_at": lo["fw"] - start,
            }
        )
    for g in gpus:
        g["links"] = {
            port: node_to_idx[dst]
            for _, port, dst in g.pop("pairs")
            if dst in node_to_idx
        }
    return gpus


def cmd_list(_args):
    gpus = discover()
    if not gpus:
        sys.exit("no GPU with XGMI found")
    print(f"{'gpu':>3}  {'card':<7} {'pci':<13} {'link max/dir':>12}  port->peer")
    for g in gpus:
        links = " ".join(f"{p}->GPU{q}" for p, q in sorted(g["links"].items()))
        print(
            f"{g['idx']:>3}  {g['card']:<7} {g['pci']:<13} {g['link_max'] / 1e9:>9.1f} GB/s  {links}"
        )


class Sampler(threading.Thread):
    """Reads every GPU each period on the wall-clock grid; appends (gpu_idx, wall_ns, bytes)
    on a new snapshot."""

    def __init__(self, gpus, period_ns):
        super().__init__(daemon=True)
        self.fds = [os.open(g["path"], os.O_RDONLY) for g in gpus]
        self.slices = [g["slice"] for g in gpus]
        self.fw_at = [g["fw_at"] for g in gpus]
        self.p_ns = period_ns
        self.buf = []
        self.stop = threading.Event()
        self.late = 0  # rounds that started more than one period late

    def run(self):
        last = [b""] * len(self.fds)
        p_ns = self.p_ns
        # Rounds start on the wall-clock grid (multiples of the period in CLOCK_REALTIME), as in
        # ainic-netmon, so samples of different hosts and tools line up given NTP/PTP.
        to_real = time.time_ns() - time.monotonic_ns()

        def next_grid():
            return ((time.monotonic_ns() + to_real) // p_ns + 1) * p_ns - to_real

        t_next = next_grid()
        while not self.stop.is_set():
            delay = t_next - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            elif delay < -p_ns:
                self.late += 1
                t_next = next_grid()
                continue
            for i, fd in enumerate(self.fds):
                raw = os.pread(fd, 4096, 0)[self.slices[i]]
                fw = raw[self.fw_at[i] : self.fw_at[i] + 8]
                if fw != last[i]:
                    last[i] = fw
                    self.buf.append((i, time.time_ns(), raw))
            t_next += p_ns

    def take(self):
        batch, self.buf = self.buf, []  # atomic swap under the GIL
        return batch


class Pusher:
    """Turns snapshots into interval points and POSTs them as Prometheus text with timestamps."""

    def __init__(self, gpus, host, url):
        self.url = url.rstrip("/") + "/api/v1/import/prometheus"
        self.prev = [None] * len(gpus)
        self.rel = [(g["rd_at"], g["wr_at"], g["fw_at"]) for g in gpus]
        self.labels = []  # per gpu: [(port, rd_labels, wr_labels)]
        self.gpu_labels = []
        for g in gpus:
            base = f'host="{host}",gpu="{g["idx"]}"'
            self.gpu_labels.append("{" + base + "}")
            self.labels.append(
                [
                    (
                        p,
                        f'{{{base},peer="{q}",dir="rd"}}',
                        f'{{{base},peer="{q}",dir="wr"}}',
                    )
                    for p, q in sorted(g["links"].items())
                ]
            )
        self.intervals = []
        self.points = self.pushed_points = self.failed_points = self.resets = 0

    def lines(self, batch):
        out = []
        for i, wall_ns, raw in batch:
            rd_at, wr_at, fw_at = self.rel[i]
            rd = struct.unpack_from("<8Q", raw, rd_at)
            wr = struct.unpack_from("<8Q", raw, wr_at)
            (fw,) = struct.unpack_from("<Q", raw, fw_at)
            prev, self.prev[i] = self.prev[i], (fw, rd, wr)
            if prev is None:
                continue
            dt = (fw - prev[0]) * 1e-8
            if dt <= 0:
                continue
            ts = wall_ns // 1_000_000
            self.intervals.append(dt)
            out.append(f"xgmi_interval_seconds{self.gpu_labels[i]} {dt:.8f} {ts}")
            for port, l_rd, l_wr in self.labels[i]:
                for cur, old, lab in ((rd, prev[1], l_rd), (wr, prev[2], l_wr)):
                    d = cur[port] - old[port]
                    if d < 0:
                        self.resets += 1
                        continue
                    b = d * 1024
                    out.append(f"xgmi_bytes{lab} {b} {ts}")
                    out.append(f"xgmi_rate_bytes_per_second{lab} {b / dt:.6g} {ts}")
        return out

    def push(self, batch):
        out = self.lines(batch)
        if not out:
            return
        self.points += len(out)
        req = urllib.request.Request(
            self.url, data=("\n".join(out) + "\n").encode(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                r.read()
            self.pushed_points += len(out)
        except OSError as e:
            self.failed_points += len(out)
            print(f"push failed ({len(out)} points dropped): {e}", file=sys.stderr)


def cmd_push(args):
    gpus = discover()
    if not gpus:
        sys.exit("no GPU with XGMI found")
    host = args.host or socket.gethostname().split(".")[0]
    pusher = Pusher(gpus, host, args.url)
    try:  # refuse to start if VictoriaMetrics is not there
        urllib.request.urlopen(args.url.rstrip("/") + "/health", timeout=2).read()
    except OSError as e:
        sys.exit(f"VictoriaMetrics at {args.url} not reachable: {e}")
    sys.setswitchinterval(
        0.0002
    )  # the push thread must not hold the GIL for 5 ms at a time
    sampler = Sampler(gpus, int(args.period_ms * 1e6))
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    sampler.start()
    t_end = time.monotonic() + args.duration if args.duration else None
    print(
        f"pushing {len(gpus)} GPUs from {host} to {args.url} (Ctrl-C stops)",
        file=sys.stderr,
    )
    while not stop.wait(1.0):
        pusher.push(sampler.take())
        if t_end and time.monotonic() >= t_end:
            break
    sampler.stop.set()
    sampler.join()
    pusher.push(sampler.take())

    iv = sorted(pusher.intervals)
    if iv:

        def q(f):  # quantile in ms
            return iv[min(len(iv) - 1, int(f * len(iv)))] * 1e3

        long_n = sum(1 for x in iv if x > 3e-3)
        print(
            f"intervals: {len(iv)}  median {q(.5):.3f} ms  p99 {q(.99):.3f} ms  max {iv[-1] * 1e3:.3f} ms  "
            f">3 ms: {long_n}",
            file=sys.stderr,
        )
    print(
        f"points: {pusher.points}  pushed {pusher.pushed_points}  failed {pusher.failed_points}  "
        f"counter decreases {pusher.resets}  late sampling rounds {sampler.late}",
        file=sys.stderr,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="GPUs, XGMI ports and peers")
    p = sub.add_parser("push", help="sample and push to VictoriaMetrics")
    p.add_argument("--url", default="http://127.0.0.1:28428")
    p.add_argument(
        "-d",
        "--duration",
        type=float,
        default=0,
        help="seconds (default: until Ctrl-C)",
    )
    p.add_argument(
        "--period-ms",
        type=float,
        default=2.0,
        help="read period per GPU (default 2 ms, as ainic-netmon)",
    )
    p.add_argument("--host", help="host label (default: short hostname)")
    args = ap.parse_args()
    {"list": cmd_list, "push": cmd_push}[args.cmd](args)


if __name__ == "__main__":
    main()

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

import json
import warnings
from collections import defaultdict


def _parse_trace_events(trace_buffer):
    """Parse trace event stream: [ts0, meta0, ts1, meta1, ...]
    Meta encoding: [warpId:16][slot:14][type:2]
    Returns list of (ts, warp_id, slot, event_type)
    """
    events = []

    if trace_buffer.is_cuda:
        trace_buffer = trace_buffer.cpu()

    # We iterate warp by warp
    num_elements = trace_buffer.numel()
    warp_stride = 8192

    for base in range(0, num_elements, warp_stride):
        warp_buffer = trace_buffer[base : base + warp_stride]

        # Circular buffer handling:
        # 1. Read all pairs (ts, meta) that are non-zero
        # 2. Sort by timestamp to restore order

        warp_events = []
        for i in range(0, warp_stride, 2):
            ts = warp_buffer[i].item()
            meta = warp_buffer[i + 1].item()

            if ts == 0:
                continue

            warp_events.append((ts, meta))

        # Sort by timestamp (clock64 is monotonic)
        warp_events.sort(key=lambda x: x[0])

        for ts, meta in warp_events:
            # Decode meta field
            # Bits 0-1:   EventType (0=BEGIN, 1=END, 2=INSTANT)
            # Bits 2-15:  SlotEnum
            # Bits 16-31: WarpId

            event_type = meta & 0x3
            slot = (meta >> 2) & 0x3FFF
            warp_id = (meta >> 16) & 0xFFFF

            events.append((ts, warp_id, slot, event_type))

    # Sort all events across all warps (optional but good for global timeline)
    events.sort(key=lambda x: x[0])
    return events


def _sanitize_events(raw_events, drop_orphan_ends=True, drop_orphan_begins=True):
    """
    Remove orphaned END/BEGIN events so Perfetto timelines stay consistent.
    - drop_orphan_ends: drop END without a matching BEGIN
    - drop_orphan_begins: drop BEGIN without a matching END
    Returns a filtered list sorted by timestamp.
    """
    if not raw_events:
        return []

    raw_events.sort(key=lambda x: x[0])

    filtered = []
    # Stack per (warp, slot) holds indices of BEGINs in `filtered`
    stacks = defaultdict(list)
    # Track which BEGIN indices should be removed (if left unmatched)
    begin_to_drop = set()

    for idx, (ts, warp_id, slot, event_type) in enumerate(raw_events):
        key = (warp_id, slot)
        if event_type == 0:  # BEGIN
            stacks[key].append(len(filtered))
            filtered.append((ts, warp_id, slot, event_type))
        elif event_type == 1:  # END
            if stacks[key]:
                stacks[key].pop()
                filtered.append((ts, warp_id, slot, event_type))
            else:
                if not drop_orphan_ends:
                    filtered.append((ts, warp_id, slot, event_type))
        else:
            # INSTANT is always kept
            filtered.append((ts, warp_id, slot, event_type))

    if drop_orphan_begins:
        for key, stack in stacks.items():
            for begin_idx in stack:
                begin_to_drop.add(begin_idx)

        if begin_to_drop:
            filtered = [ev for i, ev in enumerate(filtered) if i not in begin_to_drop]

    filtered.sort(key=lambda x: x[0])
    return filtered


def export_to_perfetto(
    trace_buffer,
    slot_map,
    filename,
    gpu_freq_ghz=1.7,
    validate_pairs=True,
    sanitize_orphans=True,
):
    """
    Export profiling buffer to Perfetto JSON trace format.
    Args:
        trace_buffer: torch.Tensor (int64), the raw debug time buffer
        slot_map: dict or object with attributes, mapping slot names to integer IDs.
                  If it's an object/module (like InterNodeV1Slots), we'll extract attributes.
        filename: str, output filename (e.g., 'trace.json')
        gpu_freq_ghz: float, GPU frequency in GHz for timestamp conversion.
                      Default 1.7 GHz (MI300 approx).
        validate_pairs: bool, if True, validate and report unpaired BEGIN/END events.
    The generated JSON can be loaded in https://ui.perfetto.dev/
    """

    # Resolve slot map if it's an object/module
    if not isinstance(slot_map, dict):
        id_to_name = {}
        for name in dir(slot_map):
            if not name.startswith("_"):
                try:
                    value = getattr(slot_map, name)
                    if isinstance(value, int):
                        id_to_name[value] = name
                except (AttributeError, TypeError):
                    continue
    else:
        # Check if keys are int (id->name) or str (name->id)
        if not slot_map:
            id_to_name = {}
        else:
            first_key = next(iter(slot_map))
            if isinstance(first_key, str):
                id_to_name = {v: k for k, v in slot_map.items()}
            else:
                id_to_name = slot_map

    raw_events = _parse_trace_events(trace_buffer)

    if not raw_events:
        warnings.warn(
            "No trace events found in buffer. The profiler might not have been used."
        )
        return

    # Optionally drop orphaned BEGIN/END so Perfetto doesn't show gaps
    if sanitize_orphans:
        sanitized_events = _sanitize_events(raw_events)
        if not sanitized_events:
            warnings.warn("All events were dropped during sanitization.")
            return
    else:
        sanitized_events = sorted(raw_events, key=lambda x: x[0])

    # Use relative timestamps (subtract first timestamp after sanitization)
    initial_ts = sanitized_events[0][0]

    # Validate BEGIN/END pairing on the sanitized stream to spot remaining issues
    if validate_pairs:
        from collections import defaultdict

        stacks = defaultdict(list)
        orphaned_ends = []

        for ts, warp_id, slot, event_type in sanitized_events:
            key = (warp_id, slot)
            if event_type == 0:  # BEGIN
                stacks[key].append(ts)
            elif event_type == 1:  # END
                if stacks[key]:
                    stacks[key].pop()
                else:
                    orphaned_ends.append(
                        (warp_id, id_to_name.get(slot, f"slot_{slot}"), ts)
                    )

        orphaned_begins = []
        for key, stack in stacks.items():
            for ts in stack:
                orphaned_begins.append(
                    (key[0], id_to_name.get(key[1], f"slot_{key[1]}"), ts)
                )

        if orphaned_ends:
            warnings.warn(
                f"Found {len(orphaned_ends)} END events without matching BEGIN "
                f"after sanitization. First few: {orphaned_ends[:3]}"
            )
        if orphaned_begins:
            warnings.warn(
                f"Found {len(orphaned_begins)} BEGIN events without matching END "
                f"after sanitization. First few: {orphaned_begins[:3]}"
            )

    trace_events = []

    for ts, warp_id, slot, event_type in sanitized_events:
        # Convert cycles to microseconds (relative to start)
        # Formula: (cycles - initial_cycles) / (freq_ghz * 1000)
        ts_us = (ts - initial_ts) / (gpu_freq_ghz * 1000.0)

        name = id_to_name.get(slot, f"Unknown_Slot_{slot}")

        # Perfetto Phase types: 'B' (Begin), 'E' (End), 'i' (Instant)
        if event_type == 0:
            ph = "B"
        elif event_type == 1:
            ph = "E"
        elif event_type == 2:
            ph = "i"
        else:
            warnings.warn(f"Unknown event type {event_type}, treating as instant")
            ph = "i"

        event = {
            "name": name,
            "cat": "gpu_kernel",
            "ph": ph,
            "ts": ts_us,
            "pid": 0,  # Process ID (single process)
            "tid": warp_id,  # Thread ID = Warp ID
        }

        # Add scope for instant events
        if event_type == 2:
            event["s"] = "t"  # Scope: thread

        trace_events.append(event)

    # Perfetto expects "displayTimeUnit" to match actual "ts" unit
    trace_data = {
        "traceEvents": trace_events,
        "displayTimeUnit": "ms",  # Perfetto will display in ms, ts is in us
    }

    with open(filename, "w") as f:
        json.dump(trace_data, f, indent=2)

    print(f"Trace exported to {filename} ({len(trace_events)} events)")

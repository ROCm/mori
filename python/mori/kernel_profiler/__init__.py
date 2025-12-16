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
import torch
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

        # Iterate events in this warp
        for i in range(0, warp_stride, 2):
            ts = warp_buffer[i].item()
            meta = warp_buffer[i + 1].item()

            if ts == 0 and meta == 0:
                # End of events for this warp
                break

            # Skip invalid entries (should not happen but be defensive)
            if ts == 0:
                continue

            # Decode meta field
            # Bits 0-1:   EventType (0=BEGIN, 1=END, 2=INSTANT)
            # Bits 2-15:  SlotEnum
            # Bits 16-31: WarpId

            event_type = meta & 0x3
            slot = (meta >> 2) & 0x3FFF
            warp_id = (meta >> 16) & 0xFFFF

            events.append((ts, warp_id, slot, event_type))

    return events


def export_to_perfetto(
    trace_buffer, slot_map, filename, gpu_freq_ghz=1.7, validate_pairs=True
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

    print("Parsing trace events...")
    raw_events = _parse_trace_events(trace_buffer)

    if not raw_events:
        warnings.warn(
            "No trace events found in buffer. The profiler might not have been used."
        )
        return

    # Sort by timestamp to ensure correct ordering
    raw_events.sort(key=lambda x: x[0])

    # Use relative timestamps (subtract first timestamp)
    initial_ts = raw_events[0][0]

    # Validate BEGIN/END pairing if requested
    if validate_pairs:
        from collections import defaultdict

        stacks = defaultdict(list)
        orphaned_ends = []

        for ts, warp_id, slot, event_type in raw_events:
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
                f"Found {len(orphaned_ends)} END events without matching BEGIN. "
                f"Buffer may have overflowed. First few: {orphaned_ends[:3]}"
            )
        if orphaned_begins:
            warnings.warn(
                f"Found {len(orphaned_begins)} BEGIN events without matching END. "
                f"Kernel may have been interrupted. First few: {orphaned_begins[:3]}"
            )

    trace_events = []

    print(f"Generating {len(raw_events)} trace events...")

    for ts, warp_id, slot, event_type in raw_events:
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


def diagnose_buffer(trace_buffer, gpu_freq_ghz=1.7):
    """
    Diagnose profiling buffer usage and potential issues.

    Args:
        trace_buffer: torch.Tensor (int64), the raw debug time buffer
        gpu_freq_ghz: float, GPU frequency in GHz for time conversion

    Returns:
        dict with diagnostic information
    """
    events = _parse_trace_events(trace_buffer)

    if not events:
        return {"status": "empty", "message": "No events found in buffer"}

    # Group by warp
    warp_events = defaultdict(list)
    for ts, warp_id, slot, event_type in events:
        warp_events[warp_id].append((ts, slot, event_type))

    # Calculate statistics
    total_warps = len(warp_events)
    events_per_warp = {wid: len(evts) for wid, evts in warp_events.items()}
    max_events = max(events_per_warp.values())
    min_events = min(events_per_warp.values())
    avg_events = sum(events_per_warp.values()) / total_warps

    # Check for buffer overflow (approaching 4096 events = 8192 int64s)
    MAX_EVENTS = 4096
    warps_near_full = [
        (wid, cnt) for wid, cnt in events_per_warp.items() if cnt > MAX_EVENTS * 0.9
    ]

    # Time span
    all_ts = [ts for ts, _, _, _ in events]
    time_span_cycles = max(all_ts) - min(all_ts)
    time_span_ms = time_span_cycles / (gpu_freq_ghz * 1000 * 1000)

    result = {
        "status": "ok",
        "total_events": len(events),
        "total_warps": total_warps,
        "events_per_warp": {"min": min_events, "max": max_events, "avg": avg_events},
        "time_span_ms": time_span_ms,
        "warnings": [],
    }

    if warps_near_full:
        result["warnings"].append(
            f"{len(warps_near_full)} warp(s) have >90% buffer usage. "
            f"Consider increasing MAX_TRACE_EVENTS_PER_WARP. "
            f"Warps: {[wid for wid, _ in warps_near_full[:5]]}"
        )

    if time_span_ms > 1000:
        result["warnings"].append(
            f"Time span is very large ({time_span_ms:.0f}ms). "
            f"This might indicate multiple kernel calls or very long execution."
        )

    return result

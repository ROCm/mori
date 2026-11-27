// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include "mori/metrics/metrics.hpp"

// ---------------------------------------------------------------------------
//  The one observability seam the refactored data plane publishes through.
//
//  The rule this header exists to enforce: A COMPONENT IS OBSERVED, IT DOES NOT
//  OBSERVE ITSELF.  A storage medium and a transfer engine each get their
//  generic metrics from a decorator that sits on their interface, not from
//  calls they make — so a NEW backend or a NEW engine is instrumented the
//  moment it is composed in, with no metrics code of its own and no dashboard
//  edit.  See instrumented_backend.h and CompositeTransferEngine.
//
//  What a component may still publish for itself is what its interface cannot
//  show from outside — the drive's own read outcomes, a staging arena's
//  occupancy.  Those ride SampleMetrics() below, and they go out under a
//  GENERIC metric name with the specifics in a label (event=/state=), never
//  under a name that spells the medium.  A metric named mori_umbp_ssd_* forces
//  a per-medium panel; mori_umbp_backend_medium_events_total{tier="SSD"} shows
//  up in the panel that was already there.
//
//  Nothing here knows about gRPC or the master: a component fills in samples,
//  MetricPublisher diffs them and hands deltas to a sink.  That is what makes
//  the whole path unit-testable without standing up a server.
// ---------------------------------------------------------------------------

namespace mori::umbp {

// ---------------------------------------------------------------------------
//  Metric-name vocabulary
//
//  Backend-agnostic by construction: every name below is shared by every
//  medium, and the medium is a LABEL.  A dashboard panel written as
//  `sum by (tier) (rate(mori_umbp_backend_bytes_total{op="commit"}[$__rate_interval]))`
//  gains a new series the day a new backend registers and needs no edit.
// ---------------------------------------------------------------------------

// --- Generic storage-backend metrics (emitted by InstrumentedBackend) -------
// Labels on all of them: tier=<HBM|DRAM|SSD|...>, backend=<MediumBackend::Name>

// op=<allocate|commit|abort|resolve|evict>, status=<ok|exists|no_space|miss|failed>
// Counted per ENTRY (one key), not per batch call — the batch shape is the
// wire's business, the key count is the workload.
#define MORI_UMBP_METRIC_BACKEND_OPS_TOTAL "mori_umbp_backend_ops_total"
#define MORI_UMBP_METRIC_BACKEND_OPS_TOTAL_HELP                                              \
  "Storage-backend slot-lifecycle operations by op and status, counted per key. Emitted by " \
  "the instrumentation decorator for every medium, so a new backend appears here with no "   \
  "code of its own"

// op=<allocate|commit|abort|resolve|evict>: one batch call, whatever its size.
// Pair with _ops_total for average batch depth, which is the knob that decides
// whether a medium is being driven efficiently.
#define MORI_UMBP_METRIC_BACKEND_BATCHES_TOTAL "mori_umbp_backend_batches_total"
#define MORI_UMBP_METRIC_BACKEND_BATCHES_TOTAL_HELP \
  "Storage-backend batch calls by op (one per call regardless of batch size)"

// op=<commit|resolve|evict>.  commit = bytes landed in the medium, resolve =
// bytes handed out to readers, evict = bytes freed.  rate() gives per-medium
// write / read / reclaim bandwidth in one panel.
#define MORI_UMBP_METRIC_BACKEND_BYTES_TOTAL "mori_umbp_backend_bytes_total"
#define MORI_UMBP_METRIC_BACKEND_BYTES_TOTAL_HELP \
  "Bytes committed to / resolved from / freed in a storage backend, by op"

// op=<allocate|commit|abort|resolve|evict>.  Seconds spent INSIDE the backend
// call.  rate(seconds_total)/rate(batches_total) is mean call latency; a
// histogram would need a second transport and this is the number that moves.
#define MORI_UMBP_METRIC_BACKEND_OP_SECONDS_TOTAL "mori_umbp_backend_op_seconds_total"
#define MORI_UMBP_METRIC_BACKEND_OP_SECONDS_TOTAL_HELP                       \
  "Cumulative seconds spent inside storage-backend calls, by op. Divide by " \
  "mori_umbp_backend_batches_total for mean call latency, or by "            \
  "mori_umbp_backend_ops_total for mean per-key latency"

// --- Medium-specific detail, under generic names ---------------------------
// A medium reports what its interface cannot show from outside.  The medium
// names the EVENT, never the metric, so every medium lands in the same panel.
// Labels: tier, backend, event=<medium's own string>

#define MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL "mori_umbp_backend_medium_events_total"
#define MORI_UMBP_METRIC_BACKEND_MEDIUM_EVENTS_TOTAL_HELP                                  \
  "Medium-internal events a backend chose to publish (drive read outcomes, single-flight " \
  "coalescing, eviction rounds, staging pressure), keyed by event"

#define MORI_UMBP_METRIC_BACKEND_MEDIUM_BYTES_TOTAL "mori_umbp_backend_medium_bytes_total"
#define MORI_UMBP_METRIC_BACKEND_MEDIUM_BYTES_TOTAL_HELP                                    \
  "Medium-internal byte counters (e.g. bytes that actually reached the device, as opposed " \
  "to the logical bytes in mori_umbp_backend_bytes_total), keyed by event"

// Gauge.  Labels: tier, backend, state=<medium's own string>
#define MORI_UMBP_METRIC_BACKEND_MEDIUM_STATE "mori_umbp_backend_medium_state"
#define MORI_UMBP_METRIC_BACKEND_MEDIUM_STATE_HELP \
  "Medium-internal live state a backend chose to publish (arena occupancy, queue depth)"

// --- Generic transfer-layer metrics (emitted by CompositeTransferEngine) ----
// Labels: engine=<TransferEngine::Name>, direction=<push|pull|local>

// status=<ok|failed>.  One count per PLAN — the unit an engine actually posts.
#define MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL "mori_umbp_transfer_ops_total"
#define MORI_UMBP_METRIC_TRANSFER_OPS_TOTAL_HELP                                            \
  "Transfer plans posted by engine, direction and outcome. engine=\"none\" with "           \
  "status=\"rejected\" counts items no engine could carry, which is a routing bug and not " \
  "a transfer failure"

#define MORI_UMBP_METRIC_TRANSFER_BYTES_TOTAL "mori_umbp_transfer_bytes_total"
#define MORI_UMBP_METRIC_TRANSFER_BYTES_TOTAL_HELP \
  "Bytes posted through a transfer engine, by engine and direction"

// Wall time from Submit to the completion of Wait, charged to the engine that
// carried the plans.  rate(bytes)/rate(seconds) is achieved bandwidth.
#define MORI_UMBP_METRIC_TRANSFER_SECONDS_TOTAL "mori_umbp_transfer_seconds_total"
#define MORI_UMBP_METRIC_TRANSFER_SECONDS_TOTAL_HELP                                \
  "Cumulative seconds transfers were in flight (submit to settled), by engine and " \
  "direction. Divide mori_umbp_transfer_bytes_total by this for achieved bandwidth"

// ---------------------------------------------------------------------------
//  Sample types and the publisher
//
//  Now in mori/metrics/metrics.hpp; the names are kept for UMBP code and its tests.
// ---------------------------------------------------------------------------

using MetricKind = ::mori::metrics::MetricKind;
using MetricLabels = ::mori::metrics::MetricLabels;
using MetricSample = ::mori::metrics::MetricSample;
using MetricSource = ::mori::metrics::MetricSource;
using MetricPublisher = ::mori::metrics::MetricPublisher;

}  // namespace mori::umbp

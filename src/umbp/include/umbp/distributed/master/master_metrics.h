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

// ---------------------------------------------------------------------------
// Prometheus metric names and help strings for the UMBP master server.
//
// All metric identifiers and their descriptions are centralised here so that
// dashboards, alerts, and tests can refer to a single source of truth.
// ---------------------------------------------------------------------------

// --- External KV API call counters -----------------------------------------

#define MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL "mori_umbp_external_kv_report_total"
#define MORI_UMBP_METRIC_EXT_KV_REPORT_TOTAL_HELP \
  "Total number of ReportExternalKvBlocks API calls received by the master"

#define MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL "mori_umbp_external_kv_revoke_total"
#define MORI_UMBP_METRIC_EXT_KV_REVOKE_TOTAL_HELP \
  "Total number of RevokeExternalKvBlocks API calls received by the master"

#define MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL "mori_umbp_external_kv_match_total"
#define MORI_UMBP_METRIC_EXT_KV_MATCH_TOTAL_HELP \
  "Total number of MatchExternalKv API calls received by the master"

// --- External KV block count counters (for average-per-call computation) ---

#define MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL "mori_umbp_external_kv_report_blocks_total"
#define MORI_UMBP_METRIC_EXT_KV_REPORT_BLOCKS_TOTAL_HELP \
  "Total number of KV blocks received across all ReportExternalKvBlocks calls"

#define MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL "mori_umbp_external_kv_revoke_blocks_total"
#define MORI_UMBP_METRIC_EXT_KV_REVOKE_BLOCKS_TOTAL_HELP \
  "Total number of KV blocks revoked across all RevokeExternalKvBlocks calls"

// --- External KV match block counters (for avg match num and hit rate) ------

#define MORI_UMBP_METRIC_EXT_KV_MATCH_QUERIED_BLOCKS_TOTAL \
  "mori_umbp_external_kv_match_queried_blocks_total"
#define MORI_UMBP_METRIC_EXT_KV_MATCH_QUERIED_BLOCKS_TOTAL_HELP \
  "Total number of KV blocks queried across all MatchExternalKv calls"

#define MORI_UMBP_METRIC_EXT_KV_MATCH_MATCHED_BLOCKS_TOTAL \
  "mori_umbp_external_kv_match_matched_blocks_total"
#define MORI_UMBP_METRIC_EXT_KV_MATCH_MATCHED_BLOCKS_TOTAL_HELP \
  "Total number of KV blocks matched across all MatchExternalKv calls"

// --- Per-node live external KV block count (gauge) -------------------------
// The full metric name is the prefix concatenated with the node_id.
// The full help string is the prefix concatenated with the node_id.

#define MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT_PREFIX "mori_umbp_external_kv_live_count_"
#define MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT_HELP_PREFIX "Live external KV block count for node "
#define MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT "mori_umbp_external_kv_live_count"
#define MORI_UMBP_METRIC_EXT_KV_LIVE_COUNT_HELP "Live external KV block count"

// --- Per-client live KV key count (reported by clients in heartbeat) -------
// Labels: node=<node_id>, tier=<hbm|dram|ssd>

#define MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT "mori_umbp_client_kv_live_count"
#define MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT_HELP \
  "Live KV key count owned by this client, reported by the client (per tier)"

#define MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT_TOTAL "mori_umbp_client_kv_live_count_total"
#define MORI_UMBP_METRIC_CLIENT_KV_LIVE_COUNT_TOTAL_HELP \
  "Total live KV key count owned by this client (sum across tiers)"

// --- Alive client count (gauge) --------------------------------------------

#define MORI_UMBP_METRIC_CLIENT_COUNT "mori_umbp_client_count"
#define MORI_UMBP_METRIC_CLIENT_COUNT_HELP "Number of alive clients registered with the master"

// --- Per-client tier capacity gauges ---------------------------------------
// Full name: prefix + sanitized_node_id + "_" + tier (hbm|dram|ssd)

#define MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL_PREFIX "mori_umbp_client_capacity_total_bytes_"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL_HELP_PREFIX "Total capacity bytes for client "
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL "mori_umbp_client_capacity_total_bytes"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL_HELP "Total capacity bytes"

#define MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL_PREFIX "mori_umbp_client_capacity_available_bytes_"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL_HELP_PREFIX "Available capacity bytes for client "
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL "mori_umbp_client_capacity_available_bytes"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL_HELP "Available capacity bytes"

#define MORI_UMBP_METRIC_CLIENT_CAPACITY_USED_PREFIX "mori_umbp_client_capacity_used_bytes_"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_USED_HELP_PREFIX "Used capacity bytes for client "
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_USED "mori_umbp_client_capacity_used_bytes"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_USED_HELP "Used capacity bytes (total - available)"

#define MORI_UMBP_METRIC_CLIENT_CAPACITY_UTILIZATION_PREFIX \
  "mori_umbp_client_capacity_utilization_ratio_"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_UTILIZATION_HELP_PREFIX \
  "Capacity utilization ratio for client "
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_UTILIZATION "mori_umbp_client_capacity_utilization_ratio"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_UTILIZATION_HELP \
  "Capacity utilization ratio (used / total) in [0,1]"

// --- Per-client logical tier capacity gauges -------------------------------
// The capacity gauges above key on the medium, so a policy that places two
// DRAM backends in different logical tiers reports both as a single DRAM
// series and the hierarchy stops being readable: per-tier occupancy, whether a
// watermark drain reached its low watermark, and which tiers accept new PUTs
// all disappear into the sum. These carry the tier name the policy declared.
// Labels: node=<node_id>, logical_tier=<policy tier name>, tier=<hbm|dram|ssd>

#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_TOTAL \
  "mori_umbp_client_logical_tier_capacity_total_bytes"
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_TOTAL_HELP \
  "Total capacity bytes of a logical tier on this client's pool"

#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_AVAIL \
  "mori_umbp_client_logical_tier_capacity_available_bytes"
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_AVAIL_HELP \
  "Available capacity bytes of a logical tier on this client's pool"

#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_USED \
  "mori_umbp_client_logical_tier_capacity_used_bytes"
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_USED_HELP \
  "Used capacity bytes of a logical tier (total - available)"

#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_UTILIZATION \
  "mori_umbp_client_logical_tier_capacity_utilization_ratio"
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_CAPACITY_UTILIZATION_HELP \
  "Utilization ratio (used / total) of a logical tier, in [0,1]"

// A tier that stopped accepting PUTs is indistinguishable from one nothing
// happens to be writing to unless the eligibility itself is published.
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_PUT_ELIGIBLE \
  "mori_umbp_client_logical_tier_put_eligible"
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_PUT_ELIGIBLE_HELP \
  "1 when a logical tier currently accepts new PUTs, 0 when it does not"

// The ratio the offload decision reads. utilization_ratio above aggregates the
// tier's backends, and on the entry tier it aggregates every tier reachable by
// offload, so comparing either against a watermark is meaningless. This is the
// series to plot against high_watermark and low_watermark.
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_PEAK_UTILIZATION \
  "mori_umbp_client_logical_tier_peak_member_utilization_ratio"
#define MORI_UMBP_METRIC_CLIENT_LOGICAL_PEAK_UTILIZATION_HELP \
  "Highest utilization among a logical tier's backends, in [0,1]; what the watermarks compare"

// --- Logical tier transitions ----------------------------------------------
// Whether an offload or a promotion ever ran is otherwise only visible to a
// process that links the tier benchmark, so a served workload cannot tell a
// working tier graph from one whose every migration fails.

#define MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS "mori_umbp_client_tier_transitions_total"
#define MORI_UMBP_METRIC_CLIENT_TIER_TRANSITIONS_HELP \
  "Logical tier transitions on this client's pool, by outcome"

#define MORI_UMBP_METRIC_CLIENT_TIER_OFFLOADED_BYTES "mori_umbp_client_tier_offloaded_bytes_total"
#define MORI_UMBP_METRIC_CLIENT_TIER_OFFLOADED_BYTES_HELP \
  "Bytes moved to a later tier by offload"

#define MORI_UMBP_METRIC_CLIENT_TIER_PROMOTED_BYTES "mori_umbp_client_tier_promoted_bytes_total"
#define MORI_UMBP_METRIC_CLIENT_TIER_PROMOTED_BYTES_HELP \
  "Bytes moved to an earlier tier by promote-on-read"

// Labelled by logical tier, so which tier served a read stays visible even
// though the per-backend labels collapse every instance of a medium into one
// series.
#define MORI_UMBP_METRIC_CLIENT_TIER_READ_HITS "mori_umbp_client_tier_read_hits_total"
#define MORI_UMBP_METRIC_CLIENT_TIER_READ_HITS_HELP \
  "Reads served by each logical tier of this client's pool"

// --- Per-client RPC call counters ------------------------------------------
// Full name: prefix + sanitized_node_id

#define MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_PREFIX "mori_umbp_client_route_put_total_"
#define MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_HELP_PREFIX "Total RoutePut calls targeting client "
#define MORI_UMBP_METRIC_CLIENT_ROUTE_PUT "mori_umbp_client_route_put_total"
#define MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_HELP "Total RoutePut calls targeting client"

#define MORI_UMBP_METRIC_CLIENT_ROUTE_GET_PREFIX "mori_umbp_client_route_get_total_"
#define MORI_UMBP_METRIC_CLIENT_ROUTE_GET_HELP_PREFIX "Total RouteGet hits served by client "
#define MORI_UMBP_METRIC_CLIENT_ROUTE_GET "mori_umbp_client_route_get_total"
#define MORI_UMBP_METRIC_CLIENT_ROUTE_GET_HELP "Total RouteGet hits served by client"

#define MORI_UMBP_METRIC_CLIENT_LOOKUP_PREFIX "mori_umbp_client_lookup_total_"
#define MORI_UMBP_METRIC_CLIENT_LOOKUP_HELP_PREFIX "Total Lookup (exists) hits for keys on client "
#define MORI_UMBP_METRIC_CLIENT_LOOKUP "mori_umbp_client_lookup_total"
#define MORI_UMBP_METRIC_CLIENT_LOOKUP_HELP "Total Lookup (exists) hits for keys on client"

// --- Per-client batch RPC call counters ------------------------------------
// Full name: prefix + sanitized_node_id

#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_PREFIX "mori_umbp_client_batch_route_put_total_"
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_HELP_PREFIX \
  "Total BatchRoutePut entries targeting client "
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT "mori_umbp_client_batch_route_put_total"
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_HELP "Total BatchRoutePut entries targeting client"

#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_PREFIX "mori_umbp_client_batch_route_get_total_"
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_HELP_PREFIX \
  "Total BatchRouteGet hits served by client "
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET "mori_umbp_client_batch_route_get_total"
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_HELP "Total BatchRouteGet hits served by client"

// --- Per-client traffic byte counters (reported by clients) ----------------

#define MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL "mori_umbp_client_outbound_put_bytes_total"
#define MORI_UMBP_METRIC_CLIENT_OUTBOUND_PUT_BYTES_TOTAL_HELP \
  "Total bytes written by this client (outbound) split by local/remote traffic"

#define MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL "mori_umbp_client_outbound_get_bytes_total"
#define MORI_UMBP_METRIC_CLIENT_OUTBOUND_GET_BYTES_TOTAL_HELP \
  "Total bytes fetched by this client (outbound reads) split by local/remote traffic"

#define MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL "mori_umbp_client_inbound_put_bytes_total"
#define MORI_UMBP_METRIC_CLIENT_INBOUND_PUT_BYTES_TOTAL_HELP \
  "Total bytes received by this client (inbound writes) split by local/remote traffic"

#define MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL "mori_umbp_client_inbound_get_bytes_total"
#define MORI_UMBP_METRIC_CLIENT_INBOUND_GET_BYTES_TOTAL_HELP \
  "Total bytes delivered to this client (inbound reads) split by local/remote traffic"

#define MORI_UMBP_METRIC_RANGED_REMOTE_INSTALL_FAILURES_TOTAL \
  "mori_umbp_ranged_remote_install_failures_total"
#define MORI_UMBP_METRIC_RANGED_REMOTE_INSTALL_FAILURES_TOTAL_HELP \
  "Remote ranged objects that could not be synchronously installed in the local medium"

// --- Heartbeat / event-shipping counters (master-as-advisor) ----------------

#define MORI_UMBP_METRIC_HEARTBEAT_EVENTS_APPLIED_TOTAL "mori_umbp_heartbeat_events_applied_total"
#define MORI_UMBP_METRIC_HEARTBEAT_EVENTS_APPLIED_TOTAL_HELP \
  "KvEvents applied to GlobalBlockIndex via heartbeat"

#define MORI_UMBP_METRIC_HEARTBEAT_SEQ_GAP_TOTAL "mori_umbp_heartbeat_seq_gap_total"
#define MORI_UMBP_METRIC_HEARTBEAT_SEQ_GAP_TOTAL_HELP \
  "Heartbeats rejected due to seq gap (full sync requested)"

// --- Per-client batch bandwidth histograms (reported by clients) -----------

#define MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH "mori_umbp_client_batch_put_bandwidth_gibps"
#define MORI_UMBP_METRIC_CLIENT_BATCH_PUT_BANDWIDTH_HELP                                           \
  "BatchPut e2e call bandwidth in GiB/s (successful bytes only, split by client and local/remote " \
  "traffic)"

#define MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH "mori_umbp_client_batch_get_bandwidth_gibps"
#define MORI_UMBP_METRIC_CLIENT_BATCH_GET_BANDWIDTH_HELP                                           \
  "BatchGet e2e call bandwidth in GiB/s (successful bytes only, split by client and local/remote " \
  "traffic)"

// Ranged siblings.  Kept as their own series rather than folded into the two
// above: a ranged call moves a SUBSET of each object, so mixing them would make
// the whole-object families' bytes-per-call meaningless.  Bytes counted are the
// range bytes actually delivered to (or committed from) the caller's buffers,
// which is what the caller sees and what the sglang tree connector reports on
// its side -- note that a REMOTE ranged get still moves the whole object over
// the wire into the scratch arena, so its wire traffic exceeds what this
// histogram credits (mori_umbp_client_*bound_get_bytes_total covers that).

#define MORI_UMBP_METRIC_CLIENT_BATCH_PUT_RANGES_BANDWIDTH \
  "mori_umbp_client_batch_put_ranges_bandwidth_gibps"
#define MORI_UMBP_METRIC_CLIENT_BATCH_PUT_RANGES_BANDWIDTH_HELP                                  \
  "BatchPutRanges e2e call bandwidth in GiB/s (committed range bytes only, split by client and " \
  "local/remote traffic)"

#define MORI_UMBP_METRIC_CLIENT_BATCH_GET_RANGES_BANDWIDTH \
  "mori_umbp_client_batch_get_ranges_bandwidth_gibps"
#define MORI_UMBP_METRIC_CLIENT_BATCH_GET_RANGES_BANDWIDTH_HELP                                  \
  "BatchGetRanges e2e call bandwidth in GiB/s (delivered range bytes only, split by client and " \
  "local/remote traffic)"

// --- MasterClient -> MasterServer RPC latency (client-perceived) -----------
// Histogram of round-trip latency for every RPC method on the
// MasterClient channel, reported by clients via ReportMetrics.  Labels
// added by the binary: rpc=<MethodName>, status=ok|error.  Master then
// injects node=<node_id> as a third label.

#define MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY "mori_umbp_master_client_rpc_latency_seconds"
#define MORI_UMBP_METRIC_MASTER_CLIENT_RPC_LATENCY_HELP \
  "Latency of MasterClient RPC calls (client-perceived, includes network)"

#define MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL "mori_umbp_master_client_rpc_errors_total"
#define MORI_UMBP_METRIC_MASTER_CLIENT_RPC_ERRORS_TOTAL_HELP \
  "Number of MasterClient RPC calls returning a non-OK gRPC status"

#define MORI_UMBP_METRIC_MASTER_CLIENT_METRICS_DROPPED_TOTAL \
  "mori_umbp_master_client_metrics_dropped_total"
#define MORI_UMBP_METRIC_MASTER_CLIENT_METRICS_DROPPED_TOTAL_HELP                                \
  "Number of histogram observations dropped client-side because the pending buffer hit its cap " \
  "(see kMasterClientMaxPendingHistograms in master_client.h)"

// --- Storage-backend and transfer-layer metrics -----------------------------
//
// NOT here.  Everything a medium or a transport reports now lives in
// umbp/distributed/metrics/component_metrics.h under names that carry the
// medium in a LABEL rather than in the identifier, because the previous
// arrangement — mori_umbp_ssd_* beside an implicit DRAM set — is what forced a
// separate dashboard per medium and left an SSD panel wired to counters that
// no longer had a publisher after the backend-agnostic refactor.
//
// This header keeps what the MASTER itself measures.  A peer-side component's
// metrics arrive through ReportMetrics and are named by the component.

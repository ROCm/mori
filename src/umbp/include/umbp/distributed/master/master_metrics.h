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

// --- Alive client count (gauge) --------------------------------------------

#define MORI_UMBP_METRIC_CLIENT_COUNT "mori_umbp_client_count"
#define MORI_UMBP_METRIC_CLIENT_COUNT_HELP "Number of alive clients registered with the master"

// --- Per-client tier capacity gauges ---------------------------------------
// Full name: prefix + sanitized_node_id + "_" + tier (hbm|dram|ssd)

#define MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL_PREFIX "mori_umbp_client_capacity_total_bytes_"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_TOTAL_HELP_PREFIX "Total capacity bytes for client "

#define MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL_PREFIX "mori_umbp_client_capacity_available_bytes_"
#define MORI_UMBP_METRIC_CLIENT_CAPACITY_AVAIL_HELP_PREFIX "Available capacity bytes for client "

// --- Per-client RPC call counters ------------------------------------------
// Full name: prefix + sanitized_node_id

#define MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_PREFIX "mori_umbp_client_route_put_total_"
#define MORI_UMBP_METRIC_CLIENT_ROUTE_PUT_HELP_PREFIX "Total RoutePut calls targeting client "

#define MORI_UMBP_METRIC_CLIENT_ROUTE_GET_PREFIX "mori_umbp_client_route_get_total_"
#define MORI_UMBP_METRIC_CLIENT_ROUTE_GET_HELP_PREFIX "Total RouteGet hits served by client "

#define MORI_UMBP_METRIC_CLIENT_LOOKUP_PREFIX "mori_umbp_client_lookup_total_"
#define MORI_UMBP_METRIC_CLIENT_LOOKUP_HELP_PREFIX "Total Lookup (exists) hits for keys on client "

// --- Per-client batch RPC call counters ------------------------------------
// Full name: prefix + sanitized_node_id

#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_PREFIX "mori_umbp_client_batch_route_put_total_"
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_PUT_HELP_PREFIX \
  "Total BatchRoutePut entries targeting client "

#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_PREFIX "mori_umbp_client_batch_route_get_total_"
#define MORI_UMBP_METRIC_CLIENT_BATCH_ROUTE_GET_HELP_PREFIX \
  "Total BatchRouteGet hits served by client "

// --- Core block management API call counters --------------------------------

#define MORI_UMBP_METRIC_REGISTER_TOTAL "mori_umbp_register_total"
#define MORI_UMBP_METRIC_REGISTER_TOTAL_HELP \
  "Total number of Register API calls received by the master"

#define MORI_UMBP_METRIC_UNREGISTER_TOTAL "mori_umbp_unregister_total"
#define MORI_UMBP_METRIC_UNREGISTER_TOTAL_HELP \
  "Total number of Unregister API calls received by the master"

#define MORI_UMBP_METRIC_FINALIZE_ALLOCATION_TOTAL "mori_umbp_finalize_allocation_total"
#define MORI_UMBP_METRIC_FINALIZE_ALLOCATION_TOTAL_HELP \
  "Total number of FinalizeAllocation API calls received by the master"

#define MORI_UMBP_METRIC_PUBLISH_LOCAL_BLOCK_TOTAL "mori_umbp_publish_local_block_total"
#define MORI_UMBP_METRIC_PUBLISH_LOCAL_BLOCK_TOTAL_HELP \
  "Total number of PublishLocalBlock API calls received by the master"

#define MORI_UMBP_METRIC_ABORT_ALLOCATION_TOTAL "mori_umbp_abort_allocation_total"
#define MORI_UMBP_METRIC_ABORT_ALLOCATION_TOTAL_HELP \
  "Total number of AbortAllocation API calls received by the master"

// --- Batch API call counters ------------------------------------------------

#define MORI_UMBP_METRIC_BATCH_LOOKUP_TOTAL "mori_umbp_batch_lookup_total"
#define MORI_UMBP_METRIC_BATCH_LOOKUP_TOTAL_HELP \
  "Total number of BatchLookup API calls received by the master"

#define MORI_UMBP_METRIC_BATCH_LOOKUP_KEYS_TOTAL "mori_umbp_batch_lookup_keys_total"
#define MORI_UMBP_METRIC_BATCH_LOOKUP_KEYS_TOTAL_HELP \
  "Total keys queried across all BatchLookup calls"

#define MORI_UMBP_METRIC_BATCH_LOOKUP_FOUND_TOTAL "mori_umbp_batch_lookup_found_total"
#define MORI_UMBP_METRIC_BATCH_LOOKUP_FOUND_TOTAL_HELP \
  "Total keys found across all BatchLookup calls"

#define MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_TOTAL "mori_umbp_batch_finalize_allocation_total"
#define MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_TOTAL_HELP \
  "Total number of BatchFinalizeAllocation API calls received by the master"

#define MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_KEYS_TOTAL \
  "mori_umbp_batch_finalize_allocation_keys_total"
#define MORI_UMBP_METRIC_BATCH_FINALIZE_ALLOCATION_KEYS_TOTAL_HELP \
  "Total keys processed across all BatchFinalizeAllocation calls"

#define MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_TOTAL "mori_umbp_batch_abort_allocation_total"
#define MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_TOTAL_HELP \
  "Total number of BatchAbortAllocation API calls received by the master"

#define MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_ENTRIES_TOTAL \
  "mori_umbp_batch_abort_allocation_entries_total"
#define MORI_UMBP_METRIC_BATCH_ABORT_ALLOCATION_ENTRIES_TOTAL_HELP \
  "Total entries processed across all BatchAbortAllocation calls"

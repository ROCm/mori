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

// Server-side Lua for every multi-key atomic mutation in the Redis metadata
// backend. Each script runs atomically on the store; all keys it touches share
// the deployment hash tag (passed as ARGV[1]) so it stays single-slot and
// cross-key atomic on Redis Cluster / Dragonfly / Valkey too.
//
// Dragonfly multithreading (per-script "--!df" flag, NOT the global launch flag):
//   Dragonfly parallelizes scripts across proactor threads ONLY when a script
//   declares every key it touches (lock-ahead mode: each shard's script runs on
//   its slot's thread, concurrently with others). A script that reaches keys not
//   in KEYS[] must run as a GLOBAL transaction, which takes a store-wide lock and
//   serializes ALL shards — so launching Dragonfly with
//   `--default_lua_flags=allow-undeclared-keys` forces EVERY script (including the
//   read hot path) global and kills the sharding win.
//   Fix: the two read hot-path scripts (route_get_batch / exists_batch) already
//   pass every key via KEYS[], so they need no flag and run per-shard in parallel.
//   The write/control scripts below DO derive auxiliary same-slot keys from the
//   hash tag, so each carries a first-line "--!df flags=allow-undeclared-keys"
//   directive that lets Dragonfly run just that script global. Redis treats the
//   line as a plain Lua comment (it only honours a "#!" shebang), so this is a
//   no-op there and the single-node / Cluster behaviour is byte-for-byte
//   unchanged. Deploy Dragonfly WITHOUT the global flag to get the parallelism.
//
// Determinism: scripts never read the server clock or randomkey; every
// timestamp is passed in by the caller as epoch-milliseconds (hazard #7 in
// master_metadata_store.h), so they are replication-safe.
//
// Phase 1 scope: the eviction LRU ZSET is NOT maintained here (eviction /
// EnumerateEvictionCandidates is Phase 2). The block hash still carries the
// _lease / _lacc / _acnt meta so RouteGet semantics match the in-memory impl.

#pragma once

#include <string>

namespace mori::umbp::redis {

// Each script is a single, stable global object (not a per-use temporary): the
// SHA cache in RespClient / RespClusterClient keys on the script object's ADDRESS
// (`&script`), not its ~1.5KB text, so the read hot path never rehashes/compares
// the whole Lua body under the cache lock on every EVALSHA. That only works if a
// call site passes a reference to THESE objects (an `inline const std::string`,
// one instance per literal across the program) rather than constructing a fresh
// std::string from a `const char*` each call. Do NOT copy a script into a local
// std::string before Eval() — that would defeat the pointer-identity cache.

// register_client:
//   KEYS[1] = node key
//   ARGV = [tag, node_id, now_ms, stale_after_ms, addr, peer, caps, engine, tags]
//   Returns 1 if registered/revived, 0 if rejected (ALIVE and not stale).
inline const std::string kRegisterClientLua = R"LUA(--!df flags=allow-undeclared-keys
local nodeKey = KEYS[1]
local tag = ARGV[1]
local nodeId = ARGV[2]
local now = tonumber(ARGV[3])
local stale = tonumber(ARGV[4])
if redis.call('EXISTS', nodeKey) == 1 then
  local status = tonumber(redis.call('HGET', nodeKey, 'status') or '0')
  local lastHb = tonumber(redis.call('HGET', nodeKey, 'last_hb') or '0')
  local isStale = ((now - lastHb) > stale) or (status == 2)
  if status == 1 and (not isStale) then
    return 0
  end
end
redis.call('DEL', nodeKey)
redis.call('HSET', nodeKey,
  'status', 1, 'last_hb', now, 'reg_at', now, 'seq', 0,
  'addr', ARGV[5], 'peer', ARGV[6], 'caps', ARGV[7],
  'engine', ARGV[8], 'tags', ARGV[9])
redis.call('SADD', tag .. ':nodes:alive', nodeId)
redis.call('HSET', tag .. ':alive_peers', nodeId, ARGV[6])
return 1
)LUA";

// apply_heartbeat:
//   KEYS[1] = node key
//   ARGV = [tag, node_id, seq, now_ms, is_full_sync, caps_blob, n_events,
//           then per event: kind, block_key, tier, size]
//   `block_key` is the FULL (already sharded) block key composed by the caller
//   (KeySchema::Block), so this script never has to know the shard layout. The
//   node's reverse-index set stores these full block keys as members.
//   Returns { status_string, acked_seq_string }
//   status_string in { "UNKNOWN", "SEQ_GAP", "APPLIED" }.
inline const std::string kApplyHeartbeatLua = R"LUA(--!df flags=allow-undeclared-keys
local nodeKey = KEYS[1]
local tag = ARGV[1]
local nodeId = ARGV[2]
local seq = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local full = tonumber(ARGV[5])
local caps = ARGV[6]
local nev = tonumber(ARGV[7])

if redis.call('EXISTS', nodeKey) == 0 then
  return { 'UNKNOWN', '0' }
end
local last = tonumber(redis.call('HGET', nodeKey, 'seq') or '0')
local aliveSet = tag .. ':nodes:alive'
local peers = tag .. ':alive_peers'
local peer = redis.call('HGET', nodeKey, 'peer') or ''

if full == 0 and seq ~= last + 1 then
  redis.call('HSET', nodeKey, 'last_hb', now, 'status', 1)
  redis.call('SADD', aliveSet, nodeId)
  redis.call('HSET', peers, nodeId, peer)
  return { 'SEQ_GAP', tostring(last) }
end

redis.call('HSET', nodeKey, 'last_hb', now, 'status', 1, 'seq', seq, 'caps', caps)
redis.call('SADD', aliveSet, nodeId)
redis.call('HSET', peers, nodeId, peer)

local blocksSet = tag .. ':node:' .. nodeId .. ':blocks'
local nodePfx = 'l|' .. nodeId .. '|'

local function cleanupEmpty(bk)
  local flds = redis.call('HKEYS', bk)
  local anyLoc = false
  for _, f in ipairs(flds) do
    if string.sub(f, 1, 2) == 'l|' then anyLoc = true break end
  end
  if not anyLoc then redis.call('DEL', bk) end
end

local function addLoc(blockKey, tier, size)
  if redis.call('EXISTS', blockKey) == 0 then
    redis.call('HSET', blockKey, '_created', now, '_lacc', now, '_acnt', 0)
  end
  local field = 'l|' .. nodeId .. '|' .. tier
  if redis.call('HEXISTS', blockKey, field) == 0 then
    redis.call('HSET', blockKey, field, size)
    redis.call('SADD', blocksSet, blockKey)
  end
end

local function removeLoc(blockKey, tier)
  local field = 'l|' .. nodeId .. '|' .. tier
  if redis.call('HDEL', blockKey, field) == 1 then
    local flds = redis.call('HKEYS', blockKey)
    local nodeStill = false
    for _, f in ipairs(flds) do
      if string.sub(f, 1, string.len(nodePfx)) == nodePfx then nodeStill = true break end
    end
    if not nodeStill then redis.call('SREM', blocksSet, blockKey) end
    cleanupEmpty(blockKey)
  end
end

if full == 1 then
  local members = redis.call('SMEMBERS', blocksSet)
  for _, bk in ipairs(members) do
    local flds = redis.call('HKEYS', bk)
    for _, f in ipairs(flds) do
      if string.sub(f, 1, string.len(nodePfx)) == nodePfx then
        redis.call('HDEL', bk, f)
      end
    end
    cleanupEmpty(bk)
  end
  redis.call('DEL', blocksSet)
  for i = 0, nev - 1 do
    local base = 8 + i * 4
    if tonumber(ARGV[base]) == 0 then
      addLoc(ARGV[base + 1], ARGV[base + 2], ARGV[base + 3])
    end
  end
else
  for i = 0, nev - 1 do
    local base = 8 + i * 4
    if tonumber(ARGV[base]) == 0 then
      addLoc(ARGV[base + 1], ARGV[base + 2], ARGV[base + 3])
    else
      removeLoc(ARGV[base + 1], ARGV[base + 2])
    end
  end
end
return { 'APPLIED', tostring(seq) }
)LUA";

// route_get_batch:
//   KEYS[1..n] = block keys (all in one shard: the caller groups a batch by
//     shard and runs one invocation per shard via RespClient::EvalPipeline, so
//     KEYS stays single-slot on Redis Cluster / one proactor on Dragonfly).
//   ARGV = [now_ms, lease_ms, n_exclude, exclude_node_1..exclude_node_k]
//   Returns an array of n elements; element i is a flat array
//   [node, size, tier, node, size, tier, ...] of the surviving locations.
//   Only keys with >=1 surviving location get a lease + access bump.
//
//   The per-key HGETALL already returns _acnt, so the lease/access bump folds
//   into a SINGLE HSET (_lease/_lacc/_acnt) instead of HSET + a separate
//   HINCRBY. This is semantics-preserving (_acnt still ends at old+1) and drops
//   the per-touched-key write commands from 2 to 1 — the single-slot server is
//   redis.call()-count bound, so fewer calls per script is a direct win.
inline const std::string kRouteGetBatchLua = R"LUA(
local now = tonumber(ARGV[1])
local lease = tonumber(ARGV[2])
local ne = tonumber(ARGV[3])
local excl = {}
for i = 1, ne do excl[ARGV[3 + i]] = true end
local out = {}
for i = 1, #KEYS do
  local flds = redis.call('HGETALL', KEYS[i])
  local locs = {}
  local touched = false
  local acnt = 0
  local j = 1
  while j <= #flds do
    local f = flds[j]
    local v = flds[j + 1]
    j = j + 2
    if string.sub(f, 1, 2) == 'l|' then
      local rest = string.sub(f, 3)
      local sep = string.find(rest, '|', 1, true)
      if sep ~= nil then
        local node = string.sub(rest, 1, sep - 1)
        local tier = string.sub(rest, sep + 1)
        if not excl[node] then
          locs[#locs + 1] = node
          locs[#locs + 1] = v
          locs[#locs + 1] = tier
          touched = true
        end
      end
    elseif f == '_acnt' then
      acnt = tonumber(v) or 0
    end
  end
  if touched then
    redis.call('HSET', KEYS[i], '_lease', now + lease, '_lacc', now, '_acnt', acnt + 1)
  end
  out[i] = locs
end
return out
)LUA";

// exists_batch:
//   KEYS[1..n] = block keys (single-slot per invocation, grouped by shard by
//     the caller, same as route_get_batch).
//   Returns an array of n integers (1 if the key has >=1 location, else 0).
inline const std::string kExistsBatchLua = R"LUA(
local out = {}
for i = 1, #KEYS do
  local flds = redis.call('HKEYS', KEYS[i])
  local has = 0
  for _, f in ipairs(flds) do
    if string.sub(f, 1, 2) == 'l|' then has = 1 break end
  end
  out[i] = has
end
return out
)LUA";

// list_alive:
//   ARGV = [tag]
//   Returns an array; each element is { node_id, flat_hgetall_of_node_hash }
//   for every ALIVE node.
inline const std::string kListAliveLua = R"LUA(--!df flags=allow-undeclared-keys
local tag = ARGV[1]
local members = redis.call('SMEMBERS', tag .. ':nodes:alive')
local out = {}
for _, id in ipairs(members) do
  local nk = tag .. ':node:' .. id
  local status = tonumber(redis.call('HGET', nk, 'status') or '0')
  if status == 1 then
    out[#out + 1] = { id, redis.call('HGETALL', nk) }
  end
end
return out
)LUA";

// unregister_client:
//   KEYS[1] = node key
//   ARGV = [tag, node_id]
//   Returns 1 if the client existed, 0 otherwise.
inline const std::string kUnregisterClientLua = R"LUA(--!df flags=allow-undeclared-keys
local nodeKey = KEYS[1]
local tag = ARGV[1]
local nodeId = ARGV[2]
if redis.call('EXISTS', nodeKey) == 0 then return 0 end
redis.call('DEL', nodeKey)
redis.call('SREM', tag .. ':nodes:alive', nodeId)
redis.call('HDEL', tag .. ':alive_peers', nodeId)
local blocksSet = tag .. ':node:' .. nodeId .. ':blocks'
local nodePfx = 'l|' .. nodeId .. '|'
local members = redis.call('SMEMBERS', blocksSet)
for _, bk in ipairs(members) do
  local flds = redis.call('HKEYS', bk)
  for _, f in ipairs(flds) do
    if string.sub(f, 1, string.len(nodePfx)) == nodePfx then redis.call('HDEL', bk, f) end
  end
  local flds2 = redis.call('HKEYS', bk)
  local anyLoc = false
  for _, f in ipairs(flds2) do
    if string.sub(f, 1, 2) == 'l|' then anyLoc = true break end
  end
  if not anyLoc then redis.call('DEL', bk) end
end
redis.call('DEL', blocksSet)
redis.call('DEL', tag .. ':extkv:node:' .. nodeId)
return 1
)LUA";

// expire_stale:
//   ARGV = [tag, cutoff_ms]
//   Flips ALIVE->EXPIRED for nodes whose last_hb < cutoff (keeping the row),
//   drops their block locations + external-kv, and returns the dead node ids.
inline const std::string kExpireStaleLua = R"LUA(--!df flags=allow-undeclared-keys
local tag = ARGV[1]
local cutoff = tonumber(ARGV[2])
local members = redis.call('SMEMBERS', tag .. ':nodes:alive')
local dead = {}
for _, id in ipairs(members) do
  local nk = tag .. ':node:' .. id
  local status = tonumber(redis.call('HGET', nk, 'status') or '0')
  local lastHb = tonumber(redis.call('HGET', nk, 'last_hb') or '0')
  if status == 1 and lastHb < cutoff then
    redis.call('HSET', nk, 'status', 2)
    redis.call('SREM', tag .. ':nodes:alive', id)
    redis.call('HDEL', tag .. ':alive_peers', id)
    local blocksSet = tag .. ':node:' .. id .. ':blocks'
    local nodePfx = 'l|' .. id .. '|'
    local ms = redis.call('SMEMBERS', blocksSet)
    for _, bk in ipairs(ms) do
      local flds = redis.call('HKEYS', bk)
      for _, f in ipairs(flds) do
        if string.sub(f, 1, string.len(nodePfx)) == nodePfx then redis.call('HDEL', bk, f) end
      end
      local flds2 = redis.call('HKEYS', bk)
      local anyLoc = false
      for _, f in ipairs(flds2) do
        if string.sub(f, 1, 2) == 'l|' then anyLoc = true break end
      end
      if not anyLoc then redis.call('DEL', bk) end
    end
    redis.call('DEL', blocksSet)
    redis.call('DEL', tag .. ':extkv:node:' .. id)
    dead[#dead + 1] = id
  end
end
return dead
)LUA";

// =====================================================================
// Multi-endpoint (sharded across Redis instances) scripts.
//
// When block shards live on DIFFERENT Redis instances than the control keys, no
// single Lua script can span them. The cross-store writes are therefore split
// into a control script (runs on the control instance) plus per-shard block
// scripts (run on each shard's own instance, single-slot). The store runs the
// control step first, then the block step(s); block ADD/REMOVE and full-sync
// clear+replay are idempotent, so a failed/retried block step is safe and a
// partial failure is healed by the peer's next SEQ_GAP -> full_sync. The
// single-endpoint path above (M==1) is unchanged and still fully atomic.
// =====================================================================

// apply_heartbeat_control (control instance):
//   KEYS[1] = node key
//   ARGV = [tag, node_id, seq, now_ms, is_full_sync, caps_blob]
//   seq-CAS + record + nodes:alive/alive_peers ONLY (no block work).
//   Returns { status_string, acked_seq_string } like apply_heartbeat.
inline const std::string kApplyHeartbeatControlLua = R"LUA(--!df flags=allow-undeclared-keys
local nodeKey = KEYS[1]
local tag = ARGV[1]
local nodeId = ARGV[2]
local seq = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local full = tonumber(ARGV[5])
local caps = ARGV[6]
if redis.call('EXISTS', nodeKey) == 0 then
  return { 'UNKNOWN', '0' }
end
local last = tonumber(redis.call('HGET', nodeKey, 'seq') or '0')
local aliveSet = tag .. ':nodes:alive'
local peers = tag .. ':alive_peers'
local peer = redis.call('HGET', nodeKey, 'peer') or ''
if full == 0 and seq ~= last + 1 then
  redis.call('HSET', nodeKey, 'last_hb', now, 'status', 1)
  redis.call('SADD', aliveSet, nodeId)
  redis.call('HSET', peers, nodeId, peer)
  return { 'SEQ_GAP', tostring(last) }
end
redis.call('HSET', nodeKey, 'last_hb', now, 'status', 1, 'seq', seq, 'caps', caps)
redis.call('SADD', aliveSet, nodeId)
redis.call('HSET', peers, nodeId, peer)
return { 'APPLIED', tostring(seq) }
)LUA";

// apply_block_events (one shard's instance):
//   ARGV = [revidx_key, node_prefix, is_full_sync, now_ms, n_events,
//           then per event: kind, block_key, tier, size]
//   `revidx_key` is this (node, shard) reverse-index set; `node_prefix` is
//   'l|<node>|'. All block keys passed in ARGV are on this instance/slot.
//   Idempotent: ADD overwrites, REMOVE of a missing loc is a no-op, full_sync
//   clears the node's blocks here then replays the ADDs. Returns 'OK'.
inline const std::string kApplyBlockEventsLua = R"LUA(--!df flags=allow-undeclared-keys
local revidx = ARGV[1]
local nodePfx = ARGV[2]
local full = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local nev = tonumber(ARGV[5])
local pfxLen = string.len(nodePfx)

local function cleanupEmpty(bk)
  local flds = redis.call('HKEYS', bk)
  for _, f in ipairs(flds) do
    if string.sub(f, 1, 2) == 'l|' then return end
  end
  redis.call('DEL', bk)
end

local function addLoc(bk, tier, size)
  if redis.call('EXISTS', bk) == 0 then
    redis.call('HSET', bk, '_created', now, '_lacc', now, '_acnt', 0)
  end
  local field = nodePfx .. tier
  if redis.call('HEXISTS', bk, field) == 0 then
    redis.call('HSET', bk, field, size)
    redis.call('SADD', revidx, bk)
  end
end

local function removeLoc(bk, tier)
  local field = nodePfx .. tier
  if redis.call('HDEL', bk, field) == 1 then
    local flds = redis.call('HKEYS', bk)
    local still = false
    for _, f in ipairs(flds) do
      if string.sub(f, 1, pfxLen) == nodePfx then still = true break end
    end
    if not still then redis.call('SREM', revidx, bk) end
    cleanupEmpty(bk)
  end
end

if full == 1 then
  local members = redis.call('SMEMBERS', revidx)
  for _, bk in ipairs(members) do
    local flds = redis.call('HKEYS', bk)
    for _, f in ipairs(flds) do
      if string.sub(f, 1, pfxLen) == nodePfx then redis.call('HDEL', bk, f) end
    end
    cleanupEmpty(bk)
  end
  redis.call('DEL', revidx)
  for i = 0, nev - 1 do
    local base = 6 + i * 4
    if tonumber(ARGV[base]) == 0 then addLoc(ARGV[base + 1], ARGV[base + 2], ARGV[base + 3]) end
  end
else
  for i = 0, nev - 1 do
    local base = 6 + i * 4
    if tonumber(ARGV[base]) == 0 then
      addLoc(ARGV[base + 1], ARGV[base + 2], ARGV[base + 3])
    else
      removeLoc(ARGV[base + 1], ARGV[base + 2])
    end
  end
end
return 'OK'
)LUA";

// wipe_node_blocks (one shard's instance):
//   ARGV = [revidx_key, node_prefix]
//   Drains the (node, shard) reverse index, deletes the node's location fields
//   from each block, drops now-empty blocks, and deletes the reverse index.
//   Idempotent. Returns 1.
inline const std::string kWipeNodeBlocksLua = R"LUA(--!df flags=allow-undeclared-keys
local revidx = ARGV[1]
local nodePfx = ARGV[2]
local pfxLen = string.len(nodePfx)
local members = redis.call('SMEMBERS', revidx)
for _, bk in ipairs(members) do
  local flds = redis.call('HKEYS', bk)
  for _, f in ipairs(flds) do
    if string.sub(f, 1, pfxLen) == nodePfx then redis.call('HDEL', bk, f) end
  end
  local flds2 = redis.call('HKEYS', bk)
  local anyLoc = false
  for _, f in ipairs(flds2) do
    if string.sub(f, 1, 2) == 'l|' then anyLoc = true break end
  end
  if not anyLoc then redis.call('DEL', bk) end
end
redis.call('DEL', revidx)
return 1
)LUA";

// unregister_control (control instance):
//   KEYS[1] = node key; ARGV = [tag, node_id]
//   Drops the client record + nodes:alive + alive_peers + extkv reverse index.
//   Block locations are wiped separately per shard. Returns 1 if it existed.
inline const std::string kUnregisterControlLua = R"LUA(--!df flags=allow-undeclared-keys
local nodeKey = KEYS[1]
local tag = ARGV[1]
local nodeId = ARGV[2]
if redis.call('EXISTS', nodeKey) == 0 then return 0 end
redis.call('DEL', nodeKey)
redis.call('SREM', tag .. ':nodes:alive', nodeId)
redis.call('HDEL', tag .. ':alive_peers', nodeId)
redis.call('DEL', tag .. ':extkv:node:' .. nodeId)
return 1
)LUA";

// expire_control (control instance):
//   ARGV = [tag, cutoff_ms]
//   Flips ALIVE->EXPIRED for nodes whose last_hb < cutoff, drops them from
//   nodes:alive/alive_peers + extkv, and returns the dead node ids. Block
//   locations are wiped separately per shard by the caller.
inline const std::string kExpireControlLua = R"LUA(--!df flags=allow-undeclared-keys
local tag = ARGV[1]
local cutoff = tonumber(ARGV[2])
local members = redis.call('SMEMBERS', tag .. ':nodes:alive')
local dead = {}
for _, id in ipairs(members) do
  local nk = tag .. ':node:' .. id
  local status = tonumber(redis.call('HGET', nk, 'status') or '0')
  local lastHb = tonumber(redis.call('HGET', nk, 'last_hb') or '0')
  if status == 1 and lastHb < cutoff then
    redis.call('HSET', nk, 'status', 2)
    redis.call('SREM', tag .. ':nodes:alive', id)
    redis.call('HDEL', tag .. ':alive_peers', id)
    redis.call('DEL', tag .. ':extkv:node:' .. id)
    dead[#dead + 1] = id
  end
end
return dead
)LUA";

}  // namespace mori::umbp::redis

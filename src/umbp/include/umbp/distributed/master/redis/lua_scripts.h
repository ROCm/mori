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
// Determinism: scripts never read the server clock or randomkey; every
// timestamp is passed in by the caller as epoch-milliseconds (hazard #7 in
// master_metadata_store.h), so they are replication-safe.
//
// Phase 1 scope: the eviction LRU ZSET is NOT maintained here (eviction /
// EnumerateEvictionCandidates is Phase 2). The block hash still carries the
// _lease / _lacc / _acnt meta so RouteGet semantics match the in-memory impl.

#pragma once

namespace mori::umbp::redis {

// register_client:
//   KEYS[1] = node key
//   ARGV = [tag, node_id, now_ms, stale_after_ms, addr, peer, caps, engine, tags]
//   Returns 1 if registered/revived, 0 if rejected (ALIVE and not stale).
inline constexpr const char* kRegisterClientLua = R"LUA(
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
//           then per event: kind, key, tier, size]
//   Returns { status_string, acked_seq_string }
//   status_string in { "UNKNOWN", "SEQ_GAP", "APPLIED" }.
inline constexpr const char* kApplyHeartbeatLua = R"LUA(
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

local function addLoc(userkey, tier, size)
  local bk = tag .. ':block:' .. userkey
  if redis.call('EXISTS', bk) == 0 then
    redis.call('HSET', bk, '_created', now, '_lacc', now, '_acnt', 0)
  end
  local field = 'l|' .. nodeId .. '|' .. tier
  if redis.call('HEXISTS', bk, field) == 0 then
    redis.call('HSET', bk, field, size)
    redis.call('SADD', blocksSet, userkey)
  end
end

local function removeLoc(userkey, tier)
  local bk = tag .. ':block:' .. userkey
  local field = 'l|' .. nodeId .. '|' .. tier
  if redis.call('HDEL', bk, field) == 1 then
    local flds = redis.call('HKEYS', bk)
    local nodeStill = false
    for _, f in ipairs(flds) do
      if string.sub(f, 1, string.len(nodePfx)) == nodePfx then nodeStill = true break end
    end
    if not nodeStill then redis.call('SREM', blocksSet, userkey) end
    cleanupEmpty(bk)
  end
end

if full == 1 then
  local members = redis.call('SMEMBERS', blocksSet)
  for _, userkey in ipairs(members) do
    local bk = tag .. ':block:' .. userkey
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
//   KEYS[1..n] = block keys
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
inline constexpr const char* kRouteGetBatchLua = R"LUA(
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
//   KEYS[1..n] = block keys
//   Returns an array of n integers (1 if the key has >=1 location, else 0).
inline constexpr const char* kExistsBatchLua = R"LUA(
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
inline constexpr const char* kListAliveLua = R"LUA(
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
inline constexpr const char* kUnregisterClientLua = R"LUA(
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
for _, userkey in ipairs(members) do
  local bk = tag .. ':block:' .. userkey
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
inline constexpr const char* kExpireStaleLua = R"LUA(
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
    for _, userkey in ipairs(ms) do
      local bk = tag .. ':block:' .. userkey
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

}  // namespace mori::umbp::redis

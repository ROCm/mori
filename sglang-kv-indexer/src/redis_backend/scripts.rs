//! Lua scripts. Every script declares all keys it touches in `KEYS[]` and all
//! keys share one hash tag, so each script runs in a single cluster slot (legal
//! and atomic on Redis Cluster) and, on Dragonfly, runs per-slot in parallel
//! without needing `allow-undeclared-keys`. Bitmasks are manipulated with plain
//! arithmetic (no `bit`/bitop library) for cross-engine portability.

use std::sync::LazyLock;

use redis::Script;

/// Refreshes a worker's liveness, records its address/incarnation in the durable
/// meta hash, and reports whether a restart reset is owed.
///
/// KEYS: `[worker_meta_key, worker_live_key, worker_retired_incarnations_key]`
/// ARGV: `[now_ms, ttl_ms, addr, incarnation]`
///
/// The durable meta hash (`seq`, `incarnation`, `addr`, `reset_pending`) never
/// expires; liveness lives in the separate `worker_live_key`, which is (re)armed
/// with a `ttl_ms` TTL only when `ttl_ms > 0`. Writes `addr` only when non-empty.
/// `PERSIST`s the meta key so that a meta left with a TTL by an older build (which
/// used to expire the whole hash) is made durable on first contact — otherwise it
/// could still expire and lose `incarnation`, resurrecting a restarted worker's
/// stale placements. When `incarnation` is non-empty and *differs* from the
/// stored one (a restart), the new value is stored and a durable `reset_pending`
/// flag is set so the caller wipes the previous incarnation's state. The old
/// token is added to the durable retired set; a later request carrying a retired
/// token returns status `-1` and cannot roll the current incarnation backwards.
/// Returns `{status, generation}` where status is `-1` for a retired token, `1`
/// when reset is pending, and `0` otherwise. Generation starts at zero and is
/// incremented on every accepted incarnation change.
pub static TOUCH_META: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
redis.call('PERSIST', KEYS[1])
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation')) or 0
if ARGV[4] ~= '' then
  local prev = redis.call('HGET', KEYS[1], 'incarnation')
  if not prev then
    redis.call('HSET', KEYS[1], 'incarnation', ARGV[4], 'generation', generation)
  elseif prev ~= ARGV[4] then
    if redis.call('SISMEMBER', KEYS[3], ARGV[4]) == 1 then
      return {-1, generation}
    end
    redis.call('SADD', KEYS[3], prev)
    generation = redis.call('HINCRBY', KEYS[1], 'generation', 1)
    redis.call('HSET', KEYS[1], 'incarnation', ARGV[4])
    redis.call('HSET', KEYS[1], 'reset_pending', '1')
  end
end
if ARGV[3] ~= '' then
  redis.call('HSET', KEYS[1], 'addr', ARGV[3])
end
if tonumber(ARGV[2]) > 0 then
  redis.call('SET', KEYS[2], ARGV[1], 'PX', tonumber(ARGV[2]))
end
if redis.call('HGET', KEYS[1], 'reset_pending') == '1' then
  return {1, generation}
end
return {0, generation}
"#,
    )
});

/// Reads a worker's routing address and liveness for `match`.
///
/// KEYS: `[worker_meta_key, worker_live_key]`
/// ARGV: `[ttl_enabled ("0"|"1")]`
/// Returns `{addr, alive, generation}`.
/// `alive` is `0` when a restart reset is still pending (`reset_pending=1`) — the
/// worker's placement is being wiped and must not be routed to, regardless of the
/// TTL setting — or, when liveness is enabled, when the `worker_live_key` has
/// expired (the worker stopped applying/heartbeating). Otherwise `1`.
pub static WORKER_VIEW: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local addr = redis.call('HGET', KEYS[1], 'addr')
if not addr then addr = '' end
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation')) or 0
if redis.call('HGET', KEYS[1], 'reset_pending') == '1' then
  return {addr, 0, generation}
end
local alive = 1
if ARGV[1] == '1' and redis.call('EXISTS', KEYS[2]) == 0 then
  alive = 0
end
return {addr, alive, generation}
"#,
    )
});

/// Removes a worker from a placement hash only when the stored generation is
/// older than the current reset generation.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[worker_id, current_generation]`
/// Deletes the co-located hit key when the placement hash becomes empty, mirror-
/// ing [`PLACEMENT_CLEAR`] so an evicted block cannot leak its `:h` key.
pub static PLACEMENT_CLEAR_WORKER: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local raw = redis.call('HGET', KEYS[1], ARGV[1])
if raw then
  local stored_generation = 0
  local sep = string.find(raw, ':', 1, true)
  if sep then
    stored_generation = tonumber(string.sub(raw, 1, sep - 1)) or 0
  end
  if stored_generation < tonumber(ARGV[2]) then
    redis.call('HDEL', KEYS[1], ARGV[1])
  end
end
if redis.call('HLEN', KEYS[1]) == 0 then
  redis.call('DEL', KEYS[1])
  redis.call('DEL', KEYS[2])
end
return 1
"#,
    )
});

/// Idempotency gate for a worker's apply batch, keyed on the monotonic per-worker
/// seq stored in the worker meta hash (`{w:worker}:meta` field `seq`).
///
/// KEYS: `[worker_meta_key]`
/// ARGV: `[seq, expected_generation]`
/// Returns `{proceed, last}`: `proceed=-1` if the worker generation changed
/// after TOUCH_META, `0` for a duplicate, and `1` otherwise.
pub static SEQ_CHECK: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local last = redis.call('HGET', KEYS[1], 'seq')
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation')) or 0
if generation ~= tonumber(ARGV[2]) then
  return {-1, last or '-1'}
end
if last ~= false and tonumber(ARGV[1]) <= tonumber(last) then
  return {0, last}
end
if last == false then
  return {1, '-1'}
end
return {1, last}
"#,
    )
});

/// Monotonically advances a worker's stored seq after its batch is applied.
///
/// KEYS: `[worker_meta_key]`
/// ARGV: `[seq, expected_generation]`
/// Returns `{status, last}` where status is `-1` if generation changed during
/// the batch and `1` otherwise. Sets `seq` to `max(stored, seq)`. Run only on
/// the apply path (never for a duplicate), and only after the batch mutations
/// have committed, so a crash between mutations and this call simply leaves the
/// stored seq behind and the next (idempotent) replay repairs it.
pub static SEQ_COMMIT: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local last = redis.call('HGET', KEYS[1], 'seq')
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation')) or 0
if generation ~= tonumber(ARGV[2]) then
  return {-1, last or '-1'}
end
if last == false or tonumber(ARGV[1]) > tonumber(last) then
  redis.call('HSET', KEYS[1], 'seq', ARGV[1])
  return {1, ARGV[1]}
end
return {1, last}
"#,
    )
});

/// Finishes a generation reset exactly once. Concurrent resetters may clean the
/// same old entries, but only the first one that still observes reset_pending
/// removes the old seq and opens the generation for applies.
///
/// KEYS: `[worker_meta_key]`
/// ARGV: `[expected_generation]`
pub static RESET_FINISH: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local generation = tonumber(redis.call('HGET', KEYS[1], 'generation')) or 0
if generation ~= tonumber(ARGV[1]) then return -1 end
if redis.call('HGET', KEYS[1], 'reset_pending') ~= '1' then return 0 end
redis.call('HDEL', KEYS[1], 'seq')
redis.call('HDEL', KEYS[1], 'reset_pending')
return 1
"#,
    )
});

/// Sets a tier bit for a worker in a placement hash.
///
/// KEYS: `[placement_key]`
/// ARGV: `[worker_id, bit, generation]`
/// Returns `-1` if a newer generation owns the field, `1` if the placement
/// changed, else `0`. The caller always performs
/// the idempotent reverse-index `SADD`, even when this script returns `0`, so a
/// replay repairs a prior failure between the two index updates.
pub static PLACEMENT_SET: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local raw = redis.call('HGET', KEYS[1], ARGV[1])
local cur = 0
local stored_generation = 0
if raw then
  local sep = string.find(raw, ':', 1, true)
  if sep then
    stored_generation = tonumber(string.sub(raw, 1, sep - 1)) or 0
    cur = tonumber(string.sub(raw, sep + 1)) or 0
  else
    cur = tonumber(raw) or 0
  end
end
local generation = tonumber(ARGV[3])
if stored_generation > generation then return -1 end
if stored_generation < generation then cur = 0 end
local bit = tonumber(ARGV[2])
if math.floor(cur / bit) % 2 == 1 then
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], ARGV[3] .. ':' .. (cur + bit))
return 1
"#,
    )
});

/// Clears a tier bit for a worker in a placement hash.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[worker_id, bit, generation]`
/// Returns `{worker_gone, key_empty}`; `worker_gone=-1` fences an older writer,
/// and `worker_gone=1` when the worker no longer
/// holds the hash at any tier (caller should remove it from the reverse index);
/// `key_empty=1` when the placement hash became empty and was deleted.
///
/// When the placement hash becomes empty the block no longer exists anywhere, so
/// the co-located hit key (same `{hash}` slot) is deleted in the same script.
/// Otherwise a matched-then-evicted block would leak its `:h` key forever, since
/// the hit key is created lazily on match and nothing else ever removes it.
pub static PLACEMENT_CLEAR: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local raw = redis.call('HGET', KEYS[1], ARGV[1])
local cur = nil
local stored_generation = 0
if raw then
  local sep = string.find(raw, ':', 1, true)
  if sep then
    stored_generation = tonumber(string.sub(raw, 1, sep - 1)) or 0
    cur = tonumber(string.sub(raw, sep + 1))
  else
    cur = tonumber(raw)
  end
end
local generation = tonumber(ARGV[3])
local bit = tonumber(ARGV[2])
local worker_gone = 0
if cur == nil then
  -- Placement already reached the target state, but a previous SREM may have
  -- failed. Ask the caller to retry the idempotent reverse-index removal.
  worker_gone = 1
elseif stored_generation ~= generation then
  if stored_generation > generation then
    return {-1, 0}
  else
    -- A stale orphan from an older generation is never part of the current
    -- worker state. Remove it while repairing the reverse-index postcondition.
    redis.call('HDEL', KEYS[1], ARGV[1])
    worker_gone = 1
  end
else
  local new = cur
  if math.floor(cur / bit) % 2 == 1 then new = cur - bit end
  if new == 0 then
    redis.call('HDEL', KEYS[1], ARGV[1])
    worker_gone = 1
  else
    redis.call('HSET', KEYS[1], ARGV[1], ARGV[3] .. ':' .. new)
  end
end
local empty = 0
if redis.call('HLEN', KEYS[1]) == 0 then
  redis.call('DEL', KEYS[1])
  redis.call('DEL', KEYS[2])
  empty = 1
end
return {worker_gone, empty}
"#,
    )
});

/// Reads a placement hash. Hit counting happens only after Rust filters
/// liveness and generation, so stale placement cannot count as a returned hit.
///
/// KEYS: `[placement_key]`
/// Returns `[worker, "generation:mask", ...]`. Legacy numeric masks are
/// interpreted by the caller as generation zero.
pub static MATCH_HASH: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
return redis.call('HGETALL', KEYS[1])
"#,
    )
});

/// Increments a hit only after at least one routable current-generation worker
/// was included in the final match response.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[now_ms, worker_id, generation, ...]`
pub static HIT_BUMP: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
for i = 2, #ARGV, 2 do
  local raw = redis.call('HGET', KEYS[1], ARGV[i])
  if raw then
    local stored_generation = 0
    local sep = string.find(raw, ':', 1, true)
    if sep then
      stored_generation = tonumber(string.sub(raw, 1, sep - 1)) or 0
    end
    if stored_generation == tonumber(ARGV[i + 1]) then
      redis.call('HINCRBY', KEYS[2], 'c', 1)
      redis.call('HSET', KEYS[2], 'ls', ARGV[1])
      return 1
    end
  end
end
return 0
"#,
    )
});

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
/// KEYS: `[worker_meta_key, worker_live_key]`
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
/// flag is set so the caller wipes the previous incarnation's state. Returns `1`
/// whenever `reset_pending` is set — including on later calls after a partial
/// reset — so the caller keeps retrying the (idempotent) reset until it clears
/// the flag. A first-ever incarnation (no prior value) does not set the flag.
pub static TOUCH_META: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
redis.call('PERSIST', KEYS[1])
if ARGV[3] ~= '' then
  redis.call('HSET', KEYS[1], 'addr', ARGV[3])
end
if tonumber(ARGV[2]) > 0 then
  redis.call('SET', KEYS[2], ARGV[1], 'PX', tonumber(ARGV[2]))
end
if ARGV[4] ~= '' then
  local prev = redis.call('HGET', KEYS[1], 'incarnation')
  if not prev then
    redis.call('HSET', KEYS[1], 'incarnation', ARGV[4])
  elseif prev ~= ARGV[4] then
    redis.call('HSET', KEYS[1], 'incarnation', ARGV[4])
    redis.call('HSET', KEYS[1], 'reset_pending', '1')
  end
end
if redis.call('HGET', KEYS[1], 'reset_pending') == '1' then
  return 1
end
return 0
"#,
    )
});

/// Reads a worker's routing address and liveness for `match`.
///
/// KEYS: `[worker_meta_key, worker_live_key]`
/// ARGV: `[ttl_enabled ("0"|"1")]`
/// Returns `{addr, alive}`: `addr` is the stored address (empty when unset).
/// `alive` is `0` when a restart reset is still pending (`reset_pending=1`) — the
/// worker's placement is being wiped and must not be routed to, regardless of the
/// TTL setting — or, when liveness is enabled, when the `worker_live_key` has
/// expired (the worker stopped applying/heartbeating). Otherwise `1`.
pub static WORKER_VIEW: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local addr = redis.call('HGET', KEYS[1], 'addr')
if not addr then addr = '' end
if redis.call('HGET', KEYS[1], 'reset_pending') == '1' then
  return {addr, 0}
end
local alive = 1
if ARGV[1] == '1' and redis.call('EXISTS', KEYS[2]) == 0 then
  alive = 0
end
return {addr, alive}
"#,
    )
});

/// Removes a worker from a placement hash entirely, regardless of which tier
/// bits it held (used when wiping a restarted worker's previous incarnation).
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[worker_id]`
/// Deletes the co-located hit key when the placement hash becomes empty, mirror-
/// ing [`PLACEMENT_CLEAR`] so an evicted block cannot leak its `:h` key.
pub static PLACEMENT_CLEAR_WORKER: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
redis.call('HDEL', KEYS[1], ARGV[1])
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
/// ARGV: `[seq]`
/// Returns `{proceed, last}`: `proceed=0` when `seq <= last` (a duplicate that
/// the caller must skip); `proceed=1` otherwise. `last` is the currently stored
/// seq, or `-1` when the worker has no stored seq yet. This only reads state; the
/// caller advances the stored seq with [`SEQ_COMMIT`] after the mutations succeed
/// so the durable position never moves ahead of applied data.
pub static SEQ_CHECK: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local last = redis.call('HGET', KEYS[1], 'seq')
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
/// ARGV: `[seq]`
/// Sets `seq` to `max(stored, seq)` and returns the resulting value. Run only on
/// the apply path (never for a duplicate), and only after the batch mutations
/// have committed, so a crash between mutations and this call simply leaves the
/// stored seq behind and the next (idempotent) replay repairs it.
pub static SEQ_COMMIT: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local last = redis.call('HGET', KEYS[1], 'seq')
if last == false or tonumber(ARGV[1]) > tonumber(last) then
  redis.call('HSET', KEYS[1], 'seq', ARGV[1])
  return ARGV[1]
end
return last
"#,
    )
});

/// Sets a tier bit for a worker in a placement hash.
///
/// KEYS: `[placement_key]`
/// ARGV: `[worker_id, bit]`
/// Returns `1` if the placement changed, else `0`. The caller always performs
/// the idempotent reverse-index `SADD`, even when this script returns `0`, so a
/// replay repairs a prior failure between the two index updates.
pub static PLACEMENT_SET: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local cur = tonumber(redis.call('HGET', KEYS[1], ARGV[1])) or 0
local bit = tonumber(ARGV[2])
if math.floor(cur / bit) % 2 == 1 then
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], cur + bit)
return 1
"#,
    )
});

/// Clears a tier bit for a worker in a placement hash.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[worker_id, bit]`
/// Returns `{worker_gone, key_empty}`: `worker_gone=1` when the worker no longer
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
local cur = tonumber(redis.call('HGET', KEYS[1], ARGV[1]))
local bit = tonumber(ARGV[2])
local worker_gone = 0
if cur == nil then
  -- Placement already reached the target state, but a previous SREM may have
  -- failed. Ask the caller to retry the idempotent reverse-index removal.
  worker_gone = 1
else
  local new = cur
  if math.floor(cur / bit) % 2 == 1 then new = cur - bit end
  if new == 0 then
    redis.call('HDEL', KEYS[1], ARGV[1])
    worker_gone = 1
  else
    redis.call('HSET', KEYS[1], ARGV[1], new)
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

/// Reads a placement hash and, only if it is non-empty and `count_as_hit` is set,
/// bumps the hit counter. Placement and hit keys share the block-hash tag, so
/// this is a single-slot atomic operation.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[count_as_hit ("0"|"1"), now_ms]`
/// Returns the flat `HGETALL` of the placement hash: `[worker, mask, ...]`.
pub static MATCH_HASH: LazyLock<Script> = LazyLock::new(|| {
    Script::new(
        r#"
local flat = redis.call('HGETALL', KEYS[1])
if #flat > 0 and ARGV[1] == '1' then
  redis.call('HINCRBY', KEYS[2], 'c', 1)
  redis.call('HSET', KEYS[2], 'ls', ARGV[2])
end
return flat
"#,
    )
});

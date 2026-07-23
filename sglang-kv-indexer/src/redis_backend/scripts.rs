//! Lua scripts. Every script declares all keys it touches in `KEYS[]` and all
//! keys share one hash tag, so each script runs in a single cluster slot (legal
//! and atomic on Redis Cluster) and, on Dragonfly, runs per-slot in parallel
//! without needing `allow-undeclared-keys`. Bitmasks are manipulated with plain
//! arithmetic (no `bit`/bitop library) for cross-engine portability.

use std::sync::LazyLock;

use redis::Script;

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

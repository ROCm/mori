# Pool rules: per-pool management policy

UMBP is a byte store. It does not know what a key holds. Pool rules let the
caller say which keys belong together and how each group should be managed,
without telling UMBP what the bytes mean:

- A **pool** is a name the caller chooses for a group of keys, for example one
  per component it stores.
- A **tag** is a number the caller attaches to each key inside a pool. It is an
  opaque class label; UMBP only uses it to look up a rule.
- Each pool has a **table** with one **rule** per tag. A rule answers the four
  management decisions for objects of that class.

| Decision | Rule field | Meaning | Default |
|---|---|---|---|
| put | `admit` | store the object, or decline the put | `true` |
| evict | `weight` | the object with the largest *idle time × weight* leaves first; above 1 ages faster, below 1 slower | `1` |
| move | `demote` | on leaving a tier, write it to the next tier (`true`) or drop it (`false`) | `true` |
| move | `retain_s` | idle seconds after which it is dropped rather than written to the next tier; `0` = no limit | `0` |
| get | `promote` | on a read from a lower tier: `"tier"` follows the tier's `promote_trigger`, `"never"` stays, `"on_read"` moves up | `"tier"` |

A key that was never given a class is managed exactly as before. A store that
never uses pool rules pays one atomic load per operation.

A table is plain data. It can be loaded at start-up and replaced at any time
while the store runs. Replacing a table does not touch stored bytes; the next
decision uses the new table.

## Using it

Three calls, on `IUMBPClient` (C++) and `mori.umbp.UMBPClient` (Python). They
work the same in-process and through `umbp_standalone_server`.

| C++ | Python | Purpose |
|---|---|---|
| `BatchSetKeyClass(pool, keys, tags)` | `batch_set_key_class(pool, keys, tags)` | give keys their pool and tag |
| `SetPoolRules(pool, table_json, &error)` | `set_pool_rules(pool, table_json)` | replace a pool's table at run time |
| `GetPoolRulesState()` | `get_pool_rules_state()` | tables and per-class counters, as JSON |

### 1. Give keys their class

Call `BatchSetKeyClass` **before** the put, so the put decision can use it.
`tags` holds one tag per key, or a single tag for all keys. It returns the
number of keys recorded (0 from a server built without pool rules).

```python
import json
from mori.umbp import UMBPClient, UMBPConfig

client = UMBPClient(UMBPConfig())

KV, STATE = "kv", "state"
BLOCK, WINDOW = 1, 2                         # tags: any numbers the caller likes

client.batch_set_key_class(KV, kv_keys, [0])                 # one tag for all
client.batch_set_key_class(STATE, state_keys, state_tags)    # one tag per key
results = client.batch_put_from_ptr(kv_keys + state_keys, ptrs, sizes)
```

A put that the rule declines returns `False` for that key, like any put that
did not store. A class set for a key that is not stored yet is remembered until
the put arrives (up to about 260 thousand pending keys; older ones are
forgotten and those keys are then unclassified).

A key that is already stored changes class immediately and keeps its recency,
so a caller can re-tag objects as their role changes.

### 2. Write a table

```json
{
  "policy": "hand-tuned",
  "default": {"admit": true, "weight": 1, "demote": true, "retain_s": 0, "promote": "tier"},
  "rules": {
    "2": {"weight": 8, "demote": false},
    "9": {"admit": false}
  }
}
```

- `policy` is free text naming who or what produced the table. It is reported
  back in the state.
- `default` is the rule for tags without an entry in `rules`.
- Every field is optional. A rule inherits the fields it omits from `default`.
- Unknown fields, a weight that is not above 0 and a `rules` key that is not a
  number are errors; the table is rejected whole and the one in force stays.

This example reads: tag 2 ages eight times as fast as unclassified data and is
dropped instead of being written to the next tier; tag 9 is not stored at all;
every other tag behaves as it did without rules.

### 3. Load tables at start-up

Set `UMBP_POOL_RULES` to a file with one table per pool:

```json
{"pools": {
  "kv":    {},
  "state": {"policy": "hand-tuned", "rules": {"2": {"weight": 8, "demote": false}}}
}}
```

The process that owns the storage reads it: `umbp_standalone_server` in
standalone mode, the engine process when UMBP is embedded. A file that cannot
be read or parsed stops the start with the reason in the log; the file is taken
whole or not at all.

### 4. Replace a table while the store runs

```python
client.set_pool_rules("state", json.dumps({
    "policy": "tuned-2026-11-19",
    "rules": {"2": {"admit": False}},
}))
```

Python raises `ValueError` with the reason when the table is rejected; C++
returns `false` and fills `*error`. Each accepted table gets the pool's next
`version` number.

Any program that can reach the store can do this: an operator's script, a
controller that computes tables from the counters below, or the engine itself.

### 5. Read the state

`get_pool_rules_state()` returns one JSON document:

```json
{
  "idle_bucket_upper_s": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
  "pools": {
    "state": {
      "table": {"policy": "hand-tuned", "version": 1, "default": {...}, "rules": {"2": {...}}},
      "classes": {
        "2": {
          "declared": 5120, "admitted": 5120, "admitted_bytes": 1546188226560,
          "declined": 0, "declined_bytes": 0,
          "reads": 498, "touches": 0,
          "reuse_by_idle": [0, 0, 0, 3, 12, 80, 201, 150, 40, 10, 2, 0, 0, 0],
          "demoted": 0, "demoted_bytes": 0, "promoted": 0, "promoted_bytes": 0,
          "dropped": 4410, "dropped_bytes": 1331439206400, "dropped_unread": 3982,
          "mean_residency_s": 96.4,
          "live": {"DRAM": {"keys": 710, "bytes": 214748364800}}
        }
      }
    }
  }
}
```

| Field | Meaning |
|---|---|
| `declared` | keys given this class |
| `admitted`, `admitted_bytes` | puts stored |
| `declined`, `declined_bytes` | puts refused by `admit: false` |
| `reads`, `touches` | reads served and `BatchTouch` hints received |
| `reuse_by_idle` | reads and touches by how long the key had been idle; bucket *i* ends at `idle_bucket_upper_s[i]`, the last bucket is open |
| `demoted*`, `promoted*` | moves to a lower / higher tier |
| `dropped`, `dropped_bytes` | keys that left the store |
| `dropped_unread` | of those, keys never read while stored |
| `mean_residency_s` | mean time from put to leaving the store |
| `live` | keys and bytes held now, per tier |

Bytes are counted as the medium charges them (whole pages), so they add up
with the tier's capacity figures.

These numbers are what a policy needs in order to choose a table. For
example: a class with `dropped_unread` close to `dropped` is a candidate for
`admit: false`; a class whose `reuse_by_idle` mass sits in the first buckets
can take a high `weight`; a class that is rarely read after it has been
demoted can take `demote: false` to spare the lower tier's writes.

## How the decisions are applied

- **Put.** `PeerPool::BatchAllocate` looks up the key's class. If its rule has
  `admit: false`, nothing is allocated and the put reports failure.
- **Evict.** When a tier is over its high watermark, local eviction takes the
  medium's coldest keys and reorders them by idle time × weight. Keys of
  classes with a weight above 1 are also taken from deeper than the medium's
  coldest window, so a young key of a fast-ageing class can leave before an
  older unclassified one.
- **Move.** When a key leaves a tier that has a lower tier, it is written there
  unless its rule has `demote: false` or it has been idle for `retain_s` or
  longer; then it is dropped. Watermark-triggered offload skips such keys too.
- **Get.** A read from a lower tier follows the key's `promote` rule; `"tier"`
  keeps the tier's own `promote_trigger`.

## Limits in this version

- **One node.** Classes and tables live in the pool of the node that received
  the call. A key that the master routes to another node is unclassified
  there, and the master's own ranking (`UMBP_EVICTION_MODE=master`,
  `master_ranked`) does not use weights: a medium whose candidates come from
  the master is evicted in the master's order.
- **Paged media only for weights.** Weights apply on DRAM and HBM. The file
  SSD tier frees whole segments in its own order and ignores them; `admit`,
  `demote`, `retain_s` and `promote` apply everywhere.
- **Memory.** Each classified key costs one more map entry in the pool
  (the key string plus about 100 bytes). Unclassified keys cost nothing.
- **Not persistent.** Classes and counters are in memory. After a restart,
  keys recovered from SSD are unclassified until the caller sets their class
  again; tables come back from `UMBP_POOL_RULES`.
- **Counters do not decay.** They are totals since start; a consumer that
  wants rates takes differences between two reads.

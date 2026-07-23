//! Redis storage backend for the KV indexer.
//!
//! Data model (see [`schema`]): placement is a per-block-hash HASH of
//! `worker -> tier bitmask`; a per-worker SET is the reverse index; a per-worker
//! durable HASH holds the registry (address, seq, incarnation, reset_pending)
//! while a separate per-worker key carries the liveness TTL; hit counts live in a per-hash
//! HASH co-located with placement (and are deleted together with the placement
//! when a block is fully revoked, so they never outlive the block). All writes
//! flow through
//! [`RedisKvIndexerBackend::apply`], which is naturally idempotent (bit set/clear,
//! SADD/SREM), so a verbatim batch replay (same `seq`) is a no-op and never
//! double-counts hits (hits are only bumped on match).
//!
//! Ordering / idempotency is anchored by a durable per-worker seq stored in the
//! worker meta hash (`{w:worker}:meta` field `seq`). Each apply first runs a
//! seq gate: a batch whose seq is `<= last_applied` is skipped as a duplicate;
//! otherwise the batch is applied and the stored seq is advanced to
//! `max(last, seq)` *after* the mutations commit. The response carries
//! `last_applied_seq` so the bridge can resume from the durable position after a
//! restart. Crash between mutations and the seq advance is safe: the stored seq
//! stays behind, so the next (idempotent) replay re-applies rather than skips.
//!
//! On Cluster an apply batch spans many slots and is therefore not globally
//! atomic; each block-hash mutation is atomic on its own slot and the batch is
//! idempotent, so a partial failure is corrected by the bridge replaying the
//! whole batch under the same seq.

mod conn;
mod schema;
mod scripts;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use futures::future::try_join_all;
use redis::FromRedisValue;
use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, ExternalKvActionType,
    ExternalKvNodeMatch, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    HitCountEntry, MatchExternalKvRequest, MatchExternalKvResponse, TierHashes,
};
use crate::service::KvIndexerBackend;

use conn::{ClusterConn, RedisConn, SingleConn};
use schema::{
    hit_key, placement_key, tier_bit, tiers_from_mask, worker_blocks_key, worker_live_key,
    worker_meta_key, worker_retired_incarnations_key,
};
use scripts::{
    HIT_BUMP, MATCH_HASH, PLACEMENT_CLEAR, PLACEMENT_CLEAR_WORKER, PLACEMENT_SET, RESET_FINISH,
    SEQ_CHECK, SEQ_COMMIT, TOUCH_META, WORKER_VIEW,
};

/// Boxed error for construction paths (env parsing / connect / ping).
type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_NAMESPACE: &str = "kvidx";
/// Per-request bound on concurrent Redis operations. Requests may contain more
/// hashes (up to the service-layer protocol limit), but every read/write path
/// processes them in sequential chunks so one request cannot create unbounded
/// in-flight work against Redis.
const REDIS_FANOUT_CHUNK: usize = 256;

/// Resolved connection target parsed from the environment.
enum Target {
    Single(String),
    Cluster(Vec<String>),
}

struct WorkerTouch {
    reset_needed: bool,
    generation: i64,
}

pub struct RedisKvIndexerBackend {
    conn: Arc<dyn RedisConn>,
    ns: String,
    worker_locks: Mutex<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
    /// When set, every apply refreshes a separate per-worker liveness key with
    /// this TTL, and `match` ignores workers whose liveness key has expired.
    /// The durable meta key never expires. `None` disables liveness filtering.
    worker_ttl: Option<Duration>,
}

impl RedisKvIndexerBackend {
    /// Connects to a single Redis/Dragonfly instance.
    pub async fn connect_single(url: &str, ns: impl Into<String>) -> Result<Self, BoxError> {
        let conn = SingleConn::connect(url).await?;
        Ok(Self {
            conn: Arc::new(conn),
            ns: ns.into(),
            worker_locks: Mutex::new(HashMap::new()),
            worker_ttl: None,
        })
    }

    /// Connects to a Redis Cluster from a list of seed node URLs.
    pub async fn connect_cluster(
        nodes: Vec<String>,
        ns: impl Into<String>,
    ) -> Result<Self, BoxError> {
        let conn = ClusterConn::connect(nodes).await?;
        Ok(Self {
            conn: Arc::new(conn),
            ns: ns.into(),
            worker_locks: Mutex::new(HashMap::new()),
            worker_ttl: None,
        })
    }

    /// Builds a single-instance backend without connecting: the connection is
    /// established lazily on first use. Used for degraded startup so the server
    /// can come up before Redis is reachable and self-heal once it is.
    pub fn connect_single_deferred(url: &str, ns: impl Into<String>) -> Self {
        Self {
            conn: Arc::new(SingleConn::deferred(url)),
            ns: ns.into(),
            worker_locks: Mutex::new(HashMap::new()),
            worker_ttl: None,
        }
    }

    /// Cluster counterpart to [`connect_single_deferred`].
    pub fn connect_cluster_deferred(nodes: Vec<String>, ns: impl Into<String>) -> Self {
        Self {
            conn: Arc::new(ClusterConn::deferred(nodes)),
            ns: ns.into(),
            worker_locks: Mutex::new(HashMap::new()),
            worker_ttl: None,
        }
    }

    /// Sets the per-worker liveness TTL (0 / `None` disables it).
    pub fn with_worker_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.worker_ttl = ttl;
        self
    }

    /// Builds the backend from the environment:
    ///   * `KV_INDEXER_REDIS_NAMESPACE` (default `kvidx`)
    ///   * `KV_INDEXER_REDIS_CLUSTER_NODES` (comma-separated) → Cluster, else
    ///   * `KV_INDEXER_REDIS_URL` → single instance (required)
    ///   * `KV_INDEXER_REDIS_REQUIRED` (default `1`): connect + PING on startup
    ///     and fail fast (bounded, with a clear error) if Redis is unreachable;
    ///     set `0` to start degraded — the server comes up immediately and the
    ///     connection is established lazily / retried on demand.
    ///   * `KV_INDEXER_WORKER_TTL_SECS` (default `120`): per-worker liveness TTL;
    ///     a worker that stops applying/heartbeating for this long is dropped
    ///     from `match`. `0` disables liveness (keys never expire).
    pub async fn from_env() -> Result<Self, BoxError> {
        let ns = std::env::var("KV_INDEXER_REDIS_NAMESPACE")
            .unwrap_or_else(|_| DEFAULT_NAMESPACE.into());
        let required = std::env::var("KV_INDEXER_REDIS_REQUIRED")
            .map(|v| v != "0")
            .unwrap_or(true);
        let worker_ttl = parse_worker_ttl()?;

        let target = if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
            let nodes: Vec<String> = nodes
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect();
            if nodes.is_empty() {
                return Err("KV_INDEXER_REDIS_CLUSTER_NODES is empty".into());
            }
            Target::Cluster(nodes)
        } else {
            let url = std::env::var("KV_INDEXER_REDIS_URL").map_err(|_| {
                "KV_INDEXER_REDIS_URL (or KV_INDEXER_REDIS_CLUSTER_NODES) is required for the redis backend"
            })?;
            Target::Single(url)
        };

        if required {
            // Fast, loud startup failure: the connect is bounded by a timeout /
            // limited retries (see conn.rs), then we PING to confirm readiness.
            let backend = match target {
                Target::Cluster(nodes) => Self::connect_cluster(nodes, ns)
                    .await
                    .map_err(|e| format!("redis connect failed: {e}"))?,
                Target::Single(url) => Self::connect_single(&url, ns)
                    .await
                    .map_err(|e| format!("redis connect failed: {e}"))?,
            };
            backend
                .ping()
                .await
                .map_err(|e| format!("redis readiness probe (PING) failed: {e}"))?;
            Ok(backend.with_worker_ttl(worker_ttl))
        } else {
            // Degraded: do not verify Redis now; start serving immediately and
            // connect lazily on first use (requests fail with Unavailable until
            // Redis is reachable, then the manager reconnects automatically).
            tracing::warn!(
                "KV_INDEXER_REDIS_REQUIRED=0: starting degraded; Redis not verified at startup"
            );
            Ok(match target {
                Target::Cluster(nodes) => Self::connect_cluster_deferred(nodes, ns),
                Target::Single(url) => Self::connect_single_deferred(&url, ns),
            }
            .with_worker_ttl(worker_ttl))
        }
    }

    async fn ping(&self) -> redis::RedisResult<()> {
        let _: redis::Value = self.conn.query(redis::cmd("PING").clone()).await?;
        Ok(())
    }

    fn worker_lock(&self, worker: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.worker_locks.lock().unwrap();
        if let Some(lock) = locks.get(worker).and_then(Weak::upgrade) {
            return lock;
        }
        if locks.len() >= 1024 {
            locks.retain(|_, lock| lock.strong_count() > 0);
        }
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(worker.to_string(), Arc::downgrade(&lock));
        lock
    }

    // --- write path ---------------------------------------------------------

    async fn apply(
        &self,
        req: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        let worker = req.worker_id.as_str();
        // The bridge is sequential, but request timeout/retry overlap and other
        // callers can still reach one server concurrently. Serialize a worker's
        // gate/mutations/commit within this backend process.
        let worker_lock = self.worker_lock(worker);
        let _worker_guard = worker_lock.lock().await;
        let meta_key = worker_meta_key(&self.ns, worker);

        // Every apply is proof of life: refresh the worker's liveness (a separate
        // TTL key) before anything else, so even a duplicate or an empty-actions
        // heartbeat keeps the worker routable. This also tracks the worker's
        // incarnation in the durable meta and reports whether a restart reset is
        // owed (either just detected, or a prior reset that did not finish).
        let touch = self
            .touch_meta(&meta_key, worker, &req.worker_address, &req.incarnation)
            .await?;
        if touch.reset_needed {
            // The worker restarted: its cache is empty and its sequence counter
            // reset. Wipe the stale placement/reverse entries from the previous
            // incarnation and drop the durable seq so the fresh sequence is
            // accepted instead of being skipped as a duplicate. The reset is
            // idempotent and only clears the durable `reset_pending` flag once it
            // fully succeeds, so a mid-reset failure is retried on the next apply.
            self.reset_worker(worker, &meta_key, touch.generation)
                .await?;
        }

        // An empty-actions batch is a pure heartbeat: liveness was just
        // refreshed; report the current durable seq and return.
        if req.actions.is_empty() {
            let stored = self.read_worker_seq(&meta_key).await?;
            return Ok(ApplyExternalKvBatchResponse {
                last_applied_seq: stored.max(0) as u64,
                duplicate: false,
            });
        }

        // Durable idempotency gate: skip a batch whose seq was already applied.
        // `last == -1` means the worker has no stored seq yet (accept any start).
        let gate = self
            .conn
            .invoke(
                &SEQ_CHECK,
                vec![meta_key.clone()],
                vec![req.seq.to_string(), touch.generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let gate: Vec<i64> = Vec::<i64>::from_redis_value(&gate).map_err(to_status)?;
        let gate_status = gate.first().copied().unwrap_or(1);
        if gate_status < 0 {
            return Err(Status::failed_precondition(
                "worker incarnation changed while applying batch",
            ));
        }
        let proceed = gate_status == 1;
        let last = gate.get(1).copied().unwrap_or(-1);
        if !proceed {
            return Ok(ApplyExternalKvBatchResponse {
                last_applied_seq: last.max(0) as u64,
                duplicate: true,
            });
        }

        for action in &req.actions {
            let bit = tier_bit(action.tier);
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    for chunk in action.hashes.chunks(REDIS_FANOUT_CHUNK) {
                        try_join_all(
                            chunk
                                .iter()
                                .map(|h| self.report_one(worker, h, bit, touch.generation)),
                        )
                        .await?;
                    }
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    for chunk in action.hashes.chunks(REDIS_FANOUT_CHUNK) {
                        try_join_all(
                            chunk
                                .iter()
                                .map(|h| self.revoke_one(worker, h, bit, touch.generation)),
                        )
                        .await?;
                    }
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    self.clear_all_at_tier(worker, bit, touch.generation)
                        .await?;
                }
                _ => return Err(Status::invalid_argument("unsupported action type")),
            }
        }

        // Advance the durable position only after the mutations committed, so a
        // crash mid-batch leaves seq behind and the next replay (idempotent)
        // repairs it rather than being skipped as a duplicate.
        let committed = self
            .conn
            .invoke(
                &SEQ_COMMIT,
                vec![meta_key],
                vec![req.seq.to_string(), touch.generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let committed: Vec<i64> = Vec::<i64>::from_redis_value(&committed).map_err(to_status)?;
        if committed.first().copied().unwrap_or(1) < 0 {
            return Err(Status::failed_precondition(
                "worker incarnation changed while applying batch",
            ));
        }
        let committed = committed.get(1).copied().unwrap_or(-1);
        Ok(ApplyExternalKvBatchResponse {
            last_applied_seq: committed.max(0) as u64,
            duplicate: false,
        })
    }

    /// Refreshes a worker's liveness key (with TTL, when enabled) and records its
    /// address/incarnation in the durable meta. Runs on every apply — including
    /// heartbeats and duplicates — so an active worker never expires out of
    /// `match`. Returns `true` when a restart reset is owed: the incarnation just
    /// changed, or a previously flagged reset has not finished yet.
    async fn touch_meta(
        &self,
        meta_key: &str,
        worker: &str,
        addr: &str,
        incarnation: &str,
    ) -> Result<WorkerTouch, Status> {
        let ttl_ms = self.worker_ttl.map(|d| d.as_millis() as u64).unwrap_or(0);
        // Empty incarnation is retained for wire compatibility but participates
        // in fencing as one stable legacy token.
        let incarnation = if incarnation.is_empty() {
            "__legacy__"
        } else {
            incarnation
        };
        let v = self
            .conn
            .invoke(
                &TOUCH_META,
                vec![
                    meta_key.to_string(),
                    worker_live_key(&self.ns, worker),
                    worker_retired_incarnations_key(&self.ns, worker),
                ],
                vec![
                    now_ms().to_string(),
                    ttl_ms.to_string(),
                    addr.to_string(),
                    incarnation.to_string(),
                ],
            )
            .await
            .map_err(to_status)?;
        let result: Vec<i64> = Vec::<i64>::from_redis_value(&v).map_err(to_status)?;
        let status = result.first().copied().unwrap_or(0);
        let generation = result.get(1).copied().unwrap_or(0);
        if status < 0 {
            return Err(Status::failed_precondition(format!(
                "worker incarnation has been retired: {incarnation}"
            )));
        }
        Ok(WorkerTouch {
            reset_needed: status == 1,
            generation,
        })
    }

    /// Wipes all placement/reverse-index state for a worker and resets its
    /// durable seq, then clears the `reset_pending` flag. Used when a worker's
    /// incarnation changes (a restart): the previous incarnation's cache is gone,
    /// so its index entries are stale. Every step is idempotent and the flag is
    /// cleared only after everything else succeeds, so a failure partway through
    /// leaves `reset_pending` set and the next apply retries the whole reset.
    async fn reset_worker(
        &self,
        worker: &str,
        meta_key: &str,
        generation: i64,
    ) -> Result<(), Status> {
        let mut cmd = redis::cmd("SMEMBERS");
        cmd.arg(worker_blocks_key(&self.ns, worker));
        let v = self.conn.query(cmd).await.map_err(to_status)?;
        let members: Vec<String> = Vec::<String>::from_redis_value(&v).map_err(to_status)?;
        let stale: Vec<(String, String)> = members
            .into_iter()
            .filter_map(|member| {
                let (member_generation, hash) = parse_reverse_member(&member);
                if member_generation < generation {
                    let hash = hash.to_string();
                    Some((member, hash))
                } else {
                    None
                }
            })
            .collect();
        for chunk in stale.chunks(REDIS_FANOUT_CHUNK) {
            try_join_all(chunk.iter().map(|(member, hash)| {
                self.clear_old_reverse_member(worker, member, hash, generation)
            }))
            .await?;
        }

        // Finish exactly once. A concurrent resetter arriving after another
        // request has already opened the generation must not delete its seq.
        let v = self
            .conn
            .invoke(
                &RESET_FINISH,
                vec![meta_key.to_string()],
                vec![generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        if i64::from_redis_value(&v).map_err(to_status)? < 0 {
            return Err(Status::failed_precondition(
                "worker generation changed while resetting",
            ));
        }
        Ok(())
    }

    async fn clear_old_reverse_member(
        &self,
        worker: &str,
        member: &str,
        hash: &str,
        generation: i64,
    ) -> Result<(), Status> {
        let v = self
            .conn
            .invoke(
                &PLACEMENT_CLEAR_WORKER,
                vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                vec![worker.to_string(), generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let _: i64 = i64::from_redis_value(&v).map_err(to_status)?;
        let mut srem = redis::cmd("SREM");
        srem.arg(worker_blocks_key(&self.ns, worker)).arg(member);
        self.conn.query(srem).await.map_err(to_status)?;
        Ok(())
    }

    /// Reads the worker's durable seq (`-1` when none has been stored yet).
    async fn read_worker_seq(&self, meta_key: &str) -> Result<i64, Status> {
        let mut cmd = redis::cmd("HGET");
        cmd.arg(meta_key).arg("seq");
        let v = self.conn.query(cmd).await.map_err(to_status)?;
        Ok(Option::<i64>::from_redis_value(&v)
            .map_err(to_status)?
            .unwrap_or(-1))
    }

    async fn report_one(
        &self,
        worker: &str,
        hash: &str,
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        let v = self
            .conn
            .invoke(
                &PLACEMENT_SET,
                vec![placement_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string(), generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        if i64::from_redis_value(&v).map_err(to_status)? < 0 {
            return Err(Status::failed_precondition(
                "newer worker generation already owns placement",
            ));
        }

        // Always enforce the reverse-index postcondition. SADD is idempotent,
        // so replay repairs a prior failure after PLACEMENT_SET even when the
        // placement bit was already set by the first attempt.
        let mut cmd = redis::cmd("SADD");
        cmd.arg(worker_blocks_key(&self.ns, worker))
            .arg(reverse_member(generation, hash));
        self.conn.query(cmd).await.map_err(to_status)?;
        Ok(())
    }

    async fn revoke_one(
        &self,
        worker: &str,
        hash: &str,
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        // Pass both the placement and the co-located hit key: when the placement
        // empties, PLACEMENT_CLEAR drops the hit key too (same `{hash}` slot) so a
        // matched-then-evicted block cannot leak its `:h` key. Keys share the hash
        // tag, so this stays a single-slot atomic op on Cluster.
        let v = self
            .conn
            .invoke(
                &PLACEMENT_CLEAR,
                vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string(), generation.to_string()],
            )
            .await
            .map_err(to_status)?;
        let flags: Vec<i64> = Vec::<i64>::from_redis_value(&v).map_err(to_status)?;
        let status = flags.first().copied().unwrap_or(0);
        if status < 0 {
            return Err(Status::failed_precondition(
                "newer worker generation already owns placement",
            ));
        }
        let worker_gone = status == 1;
        if worker_gone {
            let mut cmd = redis::cmd("SREM");
            cmd.arg(worker_blocks_key(&self.ns, worker))
                .arg(reverse_member(generation, hash));
            if generation == 0 {
                cmd.arg(hash);
            }
            self.conn.query(cmd).await.map_err(to_status)?;
        }
        Ok(())
    }

    async fn clear_all_at_tier(
        &self,
        worker: &str,
        bit: i64,
        generation: i64,
    ) -> Result<(), Status> {
        let mut cmd = redis::cmd("SMEMBERS");
        cmd.arg(worker_blocks_key(&self.ns, worker));
        let v = self.conn.query(cmd).await.map_err(to_status)?;
        let members: Vec<String> = Vec::<String>::from_redis_value(&v).map_err(to_status)?;
        let hashes: Vec<String> = members
            .iter()
            .filter_map(|member| {
                let (member_generation, hash) = parse_reverse_member(member);
                (member_generation == generation).then(|| hash.to_string())
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            try_join_all(
                chunk
                    .iter()
                    .map(|h| self.revoke_one(worker, h, bit, generation)),
            )
            .await?;
        }
        Ok(())
    }

    // --- read path ----------------------------------------------------------

    async fn do_match(
        &self,
        req: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);

        // Per-hash placement read, preserving hash order. Hit counting happens
        // after worker liveness and generation have been validated.
        let mut per_hash = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|hash| async move {
                let v = self
                    .conn
                    .invoke(&MATCH_HASH, vec![placement_key(&self.ns, hash)], Vec::new())
                    .await?;
                let flat: Vec<String> = Vec::<String>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>(flat)
            }))
            .await
            .map_err(to_status)?;
            per_hash.extend(values);
        }

        // Keep the placement generation alongside each candidate. Legacy
        // numeric values are generation zero.
        let mut worker_order: Vec<String> = Vec::new();
        let mut by_worker: HashMap<String, Vec<(String, i64, i64)>> = HashMap::new();
        for (hash, flat) in hashes.iter().zip(per_hash) {
            for pair in flat.chunks(2) {
                if pair.len() != 2 {
                    continue;
                }
                let worker = &pair[0];
                let (generation, mask) = parse_placement_value(&pair[1]);
                let entry = by_worker.entry(worker.clone()).or_insert_with(|| {
                    worker_order.push(worker.clone());
                    Vec::new()
                });
                entry.push((hash.clone(), generation, mask));
            }
        }

        // Fetch each matched worker's `addr` (for routing) and liveness. When a
        // TTL is configured, liveness is the presence of the separate
        // `worker_live_key`, which expires once a worker stops applying /
        // heartbeating; such stale workers are dropped so `match` never routes to
        // a dead node while its placement/reverse entries linger. The durable
        // meta itself never expires, so a revived worker keeps its seq/incarnation.
        let ttl_flag = if self.worker_ttl.is_some() { "1" } else { "0" };
        let mut metas = Vec::with_capacity(worker_order.len());
        for chunk in worker_order.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|worker| async move {
                let v = self
                    .conn
                    .invoke(
                        &WORKER_VIEW,
                        vec![
                            worker_meta_key(&self.ns, worker),
                            worker_live_key(&self.ns, worker),
                        ],
                        vec![ttl_flag.to_string()],
                    )
                    .await?;
                let (addr, alive, generation): (String, i64, i64) =
                    <(String, i64, i64)>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>((addr, alive == 1, generation))
            }))
            .await
            .map_err(to_status)?;
            metas.extend(values);
        }

        let mut matches = Vec::new();
        let mut matched_hashes: HashMap<String, Vec<(String, i64)>> = HashMap::new();
        for (worker, (address, alive, current_generation)) in worker_order.into_iter().zip(metas) {
            if !alive {
                continue;
            }
            let mut by_tier: BTreeMap<i32, Vec<String>> = BTreeMap::new();
            for (hash, generation, mask) in by_worker.remove(&worker).unwrap_or_default() {
                if generation != current_generation {
                    continue;
                }
                for tier in tiers_from_mask(mask) {
                    by_tier.entry(tier).or_default().push(hash.clone());
                }
                matched_hashes
                    .entry(hash)
                    .or_default()
                    .push((worker.clone(), current_generation));
            }
            if by_tier.is_empty() {
                continue;
            }
            matches.push(ExternalKvNodeMatch {
                worker_id: worker,
                address,
                hashes_by_tier: by_tier
                    .into_iter()
                    .map(|(tier, hashes)| TierHashes { tier, hashes })
                    .collect(),
            });
        }

        if req.count_as_hit {
            let matched_hashes: Vec<(String, Vec<(String, i64)>)> =
                matched_hashes.into_iter().collect();
            let now = now_ms().to_string();
            for chunk in matched_hashes.chunks(REDIS_FANOUT_CHUNK) {
                try_join_all(chunk.iter().map(|(hash, owners)| {
                    let mut args = Vec::with_capacity(1 + owners.len() * 2);
                    args.push(now.clone());
                    for (worker, generation) in owners {
                        args.push(worker.clone());
                        args.push(generation.to_string());
                    }
                    self.conn.invoke(
                        &HIT_BUMP,
                        vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                        args,
                    )
                }))
                .await
                .map_err(to_status)?;
            }
        }

        Ok(MatchExternalKvResponse { matches })
    }

    async fn do_hit_counts(
        &self,
        req: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let mut counts = Vec::with_capacity(hashes.len());
        for chunk in hashes.chunks(REDIS_FANOUT_CHUNK) {
            let values = try_join_all(chunk.iter().map(|hash| async move {
                let mut cmd = redis::cmd("HGET");
                cmd.arg(hit_key(&self.ns, hash)).arg("c");
                let v = self.conn.query(cmd).await?;
                let count: Option<i64> = Option::<i64>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>(count)
            }))
            .await
            .map_err(to_status)?;
            counts.extend(values);
        }

        let entries = hashes
            .into_iter()
            .zip(counts)
            .filter_map(|(hash, count)| {
                count.map(|c| HitCountEntry {
                    hash,
                    hit_count_total: c.max(0) as u64,
                })
            })
            .collect();
        Ok(GetExternalKvHitCountsResponse { entries })
    }
}

#[tonic::async_trait]
impl KvIndexerBackend for RedisKvIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.apply(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        self.do_match(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        self.do_hit_counts(request).await
    }

    /// Ready only when Redis answers a PING (covers degraded/lazy-connect starts
    /// and transient store outages).
    async fn health(&self) -> bool {
        self.ping().await.is_ok()
    }
}

fn dedup_preserve_order(hashes: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    hashes
        .iter()
        .filter(|h| seen.insert(h.as_str().to_string()))
        .cloned()
        .collect()
}

fn parse_placement_value(value: &str) -> (i64, i64) {
    match value.split_once(':') {
        Some((generation, mask)) => (generation.parse().unwrap_or(0), mask.parse().unwrap_or(0)),
        None => (0, value.parse().unwrap_or(0)),
    }
}

fn reverse_member(generation: i64, hash: &str) -> String {
    format!("{generation}:{hash}")
}

fn parse_reverse_member(member: &str) -> (i64, &str) {
    match member.split_once(':') {
        Some((generation, hash)) => match generation.parse() {
            Ok(generation) => (generation, hash),
            Err(_) => (0, member),
        },
        None => (0, member),
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn to_status(err: redis::RedisError) -> Status {
    Status::unavailable(format!("redis backend error: {err}"))
}

/// Parses `KV_INDEXER_WORKER_TTL_SECS` (default `120`; `0` disables liveness).
fn parse_worker_ttl() -> Result<Option<Duration>, BoxError> {
    const DEFAULT_WORKER_TTL_SECS: u64 = 120;
    let secs = match std::env::var("KV_INDEXER_WORKER_TTL_SECS") {
        Ok(v) => v.trim().parse::<u64>().map_err(|_| {
            format!("KV_INDEXER_WORKER_TTL_SECS must be a non-negative integer, got {v:?}")
        })?,
        Err(_) => DEFAULT_WORKER_TTL_SECS,
    };
    Ok((secs > 0).then(|| Duration::from_secs(secs)))
}

#[cfg(test)]
mod fault_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_preserves_first_seen_order() {
        let input = vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "c".to_string(),
            "b".to_string(),
        ];
        assert_eq!(dedup_preserve_order(&input), vec!["a", "b", "c"]);
    }

    #[test]
    fn placement_value_parser_is_backward_compatible() {
        assert_eq!(parse_placement_value("6"), (0, 6));
        assert_eq!(parse_placement_value("3:10"), (3, 10));
        assert_eq!(parse_placement_value("bad"), (0, 0));
    }

    #[test]
    fn reverse_member_parser_is_backward_compatible() {
        assert_eq!(parse_reverse_member("hash-a"), (0, "hash-a"));
        assert_eq!(parse_reverse_member("legacy:hash"), (0, "legacy:hash"));
        assert_eq!(parse_reverse_member("3:hash-a"), (3, "hash-a"));
        assert_eq!(reverse_member(3, "hash-a"), "3:hash-a");
    }

    #[test]
    fn worker_lock_is_shared_while_active_and_prunes_inactive_entries() {
        let backend = RedisKvIndexerBackend::connect_single_deferred("redis://127.0.0.1:1", "test");
        let first = backend.worker_lock("worker-a");
        let same = backend.worker_lock("worker-a");
        assert!(Arc::ptr_eq(&first, &same));
        drop(first);
        drop(same);

        for index in 0..1100 {
            drop(backend.worker_lock(&format!("worker-{index}")));
        }
        assert!(backend.worker_locks.lock().unwrap().len() < 1024);
    }
}

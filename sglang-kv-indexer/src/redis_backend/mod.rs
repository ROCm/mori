//! Redis storage backend for the KV indexer.
//!
//! Data model (see [`schema`]): placement is a per-block-hash HASH of
//! `worker -> tier bitmask`; a per-worker SET is the reverse index; a per-worker
//! HASH holds the registry (address + last_seen); hit counts live in a per-hash
//! HASH co-located with placement (and are deleted together with the placement
//! when a block is fully revoked, so they never outlive the block). All writes
//! flow through
//! [`RedisKvIndexerBackend::apply`], which is naturally idempotent (bit set/clear,
//! SADD/SREM), so a verbatim batch replay (same `seq`) is a no-op and never
//! double-counts hits (hits are only bumped on match).
//!
//! On Cluster an apply batch spans many slots and is therefore not globally
//! atomic; each block-hash mutation is atomic on its own slot and the batch is
//! idempotent, so a partial failure is corrected by the bridge replaying the
//! whole batch under the same seq.

mod conn;
mod schema;
mod scripts;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::try_join_all;
use redis::FromRedisValue;
use tonic::Status;

use crate::pb::{
    ApplyExternalKvBatchRequest, ExternalKvActionType, ExternalKvNodeMatch,
    GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, HitCountEntry,
    MatchExternalKvRequest, MatchExternalKvResponse, TierHashes,
};
use crate::service::KvIndexerBackend;

use conn::{ClusterConn, RedisConn, SingleConn};
use schema::{
    hit_key, placement_key, tier_bit, tiers_from_mask, worker_blocks_key, worker_meta_key,
};
use scripts::{MATCH_HASH, PLACEMENT_CLEAR, PLACEMENT_SET};

/// Boxed error for construction paths (env parsing / connect / ping).
type BoxError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_NAMESPACE: &str = "kvidx";
/// Bound on how many hashes a single `CLEAR_ALL_AT_TIER` fans out concurrently.
const CLEAR_CHUNK: usize = 256;

/// Resolved connection target parsed from the environment.
enum Target {
    Single(String),
    Cluster(Vec<String>),
}

pub struct RedisKvIndexerBackend {
    conn: Arc<dyn RedisConn>,
    ns: String,
}

impl RedisKvIndexerBackend {
    /// Connects to a single Redis/Dragonfly instance.
    pub async fn connect_single(url: &str, ns: impl Into<String>) -> Result<Self, BoxError> {
        let conn = SingleConn::connect(url).await?;
        Ok(Self {
            conn: Arc::new(conn),
            ns: ns.into(),
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
        })
    }

    /// Builds a single-instance backend without connecting: the connection is
    /// established lazily on first use. Used for degraded startup so the server
    /// can come up before Redis is reachable and self-heal once it is.
    pub fn connect_single_deferred(url: &str, ns: impl Into<String>) -> Self {
        Self {
            conn: Arc::new(SingleConn::deferred(url)),
            ns: ns.into(),
        }
    }

    /// Cluster counterpart to [`connect_single_deferred`].
    pub fn connect_cluster_deferred(nodes: Vec<String>, ns: impl Into<String>) -> Self {
        Self {
            conn: Arc::new(ClusterConn::deferred(nodes)),
            ns: ns.into(),
        }
    }

    /// Builds the backend from the environment:
    ///   * `KV_INDEXER_REDIS_NAMESPACE` (default `kvidx`)
    ///   * `KV_INDEXER_REDIS_CLUSTER_NODES` (comma-separated) → Cluster, else
    ///   * `KV_INDEXER_REDIS_URL` → single instance (required)
    ///   * `KV_INDEXER_REDIS_REQUIRED` (default `1`): connect + PING on startup
    ///     and fail fast (bounded, with a clear error) if Redis is unreachable;
    ///     set `0` to start degraded — the server comes up immediately and the
    ///     connection is established lazily / retried on demand.
    pub async fn from_env() -> Result<Self, BoxError> {
        let ns = std::env::var("KV_INDEXER_REDIS_NAMESPACE")
            .unwrap_or_else(|_| DEFAULT_NAMESPACE.into());
        let required = std::env::var("KV_INDEXER_REDIS_REQUIRED")
            .map(|v| v != "0")
            .unwrap_or(true);

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
            Ok(backend)
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
            })
        }
    }

    async fn ping(&self) -> redis::RedisResult<()> {
        let _: redis::Value = self.conn.query(redis::cmd("PING").clone()).await?;
        Ok(())
    }

    // --- write path ---------------------------------------------------------

    async fn apply(&self, req: ApplyExternalKvBatchRequest) -> Result<(), Status> {
        let worker = req.worker_id.as_str();

        if !req.worker_address.is_empty() {
            let mut cmd = redis::cmd("HSET");
            cmd.arg(worker_meta_key(&self.ns, worker))
                .arg("addr")
                .arg(&req.worker_address)
                .arg("last_seen")
                .arg(now_ms());
            self.conn.query(cmd).await.map_err(to_status)?;
        }

        for action in &req.actions {
            let bit = tier_bit(action.tier);
            match ExternalKvActionType::try_from(action.r#type) {
                Ok(ExternalKvActionType::ActionReport) => {
                    try_join_all(
                        action
                            .hashes
                            .iter()
                            .map(|h| self.report_one(worker, h, bit)),
                    )
                    .await?;
                }
                Ok(ExternalKvActionType::ActionRevoke) => {
                    try_join_all(
                        action
                            .hashes
                            .iter()
                            .map(|h| self.revoke_one(worker, h, bit)),
                    )
                    .await?;
                }
                Ok(ExternalKvActionType::ActionClearAllAtTier) => {
                    self.clear_all_at_tier(worker, bit).await?;
                }
                _ => return Err(Status::invalid_argument("unsupported action type")),
            }
        }
        Ok(())
    }

    async fn report_one(&self, worker: &str, hash: &str, bit: i64) -> Result<(), Status> {
        self.conn
            .invoke(
                &PLACEMENT_SET,
                vec![placement_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string()],
            )
            .await
            .map_err(to_status)?;

        // Always enforce the reverse-index postcondition. SADD is idempotent,
        // so replay repairs a prior failure after PLACEMENT_SET even when the
        // placement bit was already set by the first attempt.
        let mut cmd = redis::cmd("SADD");
        cmd.arg(worker_blocks_key(&self.ns, worker)).arg(hash);
        self.conn.query(cmd).await.map_err(to_status)?;
        Ok(())
    }

    async fn revoke_one(&self, worker: &str, hash: &str, bit: i64) -> Result<(), Status> {
        // Pass both the placement and the co-located hit key: when the placement
        // empties, PLACEMENT_CLEAR drops the hit key too (same `{hash}` slot) so a
        // matched-then-evicted block cannot leak its `:h` key. Keys share the hash
        // tag, so this stays a single-slot atomic op on Cluster.
        let v = self
            .conn
            .invoke(
                &PLACEMENT_CLEAR,
                vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                vec![worker.to_string(), bit.to_string()],
            )
            .await
            .map_err(to_status)?;
        let flags: Vec<i64> = Vec::<i64>::from_redis_value(&v).map_err(to_status)?;
        let worker_gone = flags.first().copied().unwrap_or(0) == 1;
        if worker_gone {
            let mut cmd = redis::cmd("SREM");
            cmd.arg(worker_blocks_key(&self.ns, worker)).arg(hash);
            self.conn.query(cmd).await.map_err(to_status)?;
        }
        Ok(())
    }

    async fn clear_all_at_tier(&self, worker: &str, bit: i64) -> Result<(), Status> {
        let mut cmd = redis::cmd("SMEMBERS");
        cmd.arg(worker_blocks_key(&self.ns, worker));
        let v = self.conn.query(cmd).await.map_err(to_status)?;
        let hashes: Vec<String> = Vec::<String>::from_redis_value(&v).map_err(to_status)?;
        for chunk in hashes.chunks(CLEAR_CHUNK) {
            try_join_all(chunk.iter().map(|h| self.revoke_one(worker, h, bit))).await?;
        }
        Ok(())
    }

    // --- read path ----------------------------------------------------------

    async fn do_match(
        &self,
        req: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let count_flag = if req.count_as_hit { "1" } else { "0" };
        let now = now_ms().to_string();

        // Per-hash placement read (+ optional hit bump), preserving hash order.
        let per_hash = try_join_all(hashes.iter().map(|hash| {
            let now = now.clone();
            async move {
                let v = self
                    .conn
                    .invoke(
                        &MATCH_HASH,
                        vec![placement_key(&self.ns, hash), hit_key(&self.ns, hash)],
                        vec![count_flag.to_string(), now],
                    )
                    .await?;
                let flat: Vec<String> = Vec::<String>::from_redis_value(&v)?;
                Ok::<_, redis::RedisError>(flat)
            }
        }))
        .await
        .map_err(to_status)?;

        // Aggregate into worker -> tier -> [hash], preserving worker first-seen order.
        let mut worker_order: Vec<String> = Vec::new();
        let mut by_worker: HashMap<String, BTreeMap<i32, Vec<String>>> = HashMap::new();
        for (hash, flat) in hashes.iter().zip(per_hash) {
            for pair in flat.chunks(2) {
                if pair.len() != 2 {
                    continue;
                }
                let worker = &pair[0];
                let mask: i64 = pair[1].parse().unwrap_or(0);
                let entry = by_worker.entry(worker.clone()).or_insert_with(|| {
                    worker_order.push(worker.clone());
                    BTreeMap::new()
                });
                for tier in tiers_from_mask(mask) {
                    entry.entry(tier).or_default().push(hash.clone());
                }
            }
        }

        // Fill each matched worker's address from its registry.
        let addrs = try_join_all(worker_order.iter().map(|worker| async move {
            let mut cmd = redis::cmd("HGET");
            cmd.arg(worker_meta_key(&self.ns, worker)).arg("addr");
            let v = self.conn.query(cmd).await?;
            let addr: Option<String> = Option::<String>::from_redis_value(&v)?;
            Ok::<_, redis::RedisError>(addr.unwrap_or_default())
        }))
        .await
        .map_err(to_status)?;

        let matches = worker_order
            .into_iter()
            .zip(addrs)
            .map(|(worker, address)| {
                let hashes_by_tier = by_worker
                    .remove(&worker)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(tier, hashes)| TierHashes { tier, hashes })
                    .collect();
                ExternalKvNodeMatch {
                    worker_id: worker,
                    address,
                    hashes_by_tier,
                }
            })
            .collect();

        Ok(MatchExternalKvResponse { matches })
    }

    async fn do_hit_counts(
        &self,
        req: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        let hashes = dedup_preserve_order(&req.hashes);
        let counts = try_join_all(hashes.iter().map(|hash| async move {
            let mut cmd = redis::cmd("HGET");
            cmd.arg(hit_key(&self.ns, hash)).arg("c");
            let v = self.conn.query(cmd).await?;
            let count: Option<i64> = Option::<i64>::from_redis_value(&v)?;
            Ok::<_, redis::RedisError>(count)
        }))
        .await
        .map_err(to_status)?;

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
    ) -> Result<(), Status> {
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
}

fn dedup_preserve_order(hashes: &[String]) -> Vec<String> {
    let mut seen = HashSet::new();
    hashes
        .iter()
        .filter(|h| seen.insert(h.as_str().to_string()))
        .cloned()
        .collect()
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
}

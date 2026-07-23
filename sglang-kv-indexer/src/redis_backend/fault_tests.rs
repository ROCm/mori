//! Failure-boundary tests for placement/reverse-index convergence.

use std::sync::atomic::{AtomicBool, Ordering};

use super::*;

/// Delegates to a real Redis connection but fails the first selected
/// reverse-index command, after the placement Lua script has committed.
struct FailOnceConn {
    inner: SingleConn,
    command: &'static [u8],
    failed: AtomicBool,
}

#[tonic::async_trait]
impl RedisConn for FailOnceConn {
    async fn query(&self, cmd: redis::Cmd) -> redis::RedisResult<redis::Value> {
        let is_target = {
            let mut args = cmd.args_iter();
            matches!(
                args.next(),
                Some(redis::Arg::Simple(name)) if name.eq_ignore_ascii_case(self.command)
            )
        };
        if is_target && !self.failed.swap(true, Ordering::SeqCst) {
            return Err(redis::RedisError::from((
                redis::ErrorKind::IoError,
                "injected reverse-index failure",
            )));
        }
        self.inner.query(cmd).await
    }

    async fn invoke(
        &self,
        script: &redis::Script,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> redis::RedisResult<redis::Value> {
        self.inner.invoke(script, keys, args).await
    }
}

async fn backend(test: &str, command: &'static [u8]) -> Option<RedisKvIndexerBackend> {
    let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") else {
        eprintln!("skipping {test}: set KV_INDEXER_REDIS_URL");
        return None;
    };
    let inner = SingleConn::connect(&url).await.expect("connect redis");
    Some(RedisKvIndexerBackend {
        conn: Arc::new(FailOnceConn {
            inner,
            command,
            failed: AtomicBool::new(false),
        }),
        ns: format!("fault:{test}:{}", now_ms()),
        worker_ttl: None,
    })
}

fn batch(worker: &str, seq: u64, kind: ExternalKvActionType) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions: vec![crate::pb::ExternalKvAction {
            r#type: kind as i32,
            tier: crate::pb::TierType::TierHbm as i32,
            hashes: vec!["hash-a".to_string()],
        }],
        worker_address: String::new(),
        incarnation: String::new(),
    }
}

fn batch_inc(
    worker: &str,
    seq: u64,
    incarnation: &str,
    kind: ExternalKvActionType,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        incarnation: incarnation.to_string(),
        ..batch(worker, seq, kind)
    }
}

async fn reverse_hashes(backend: &RedisKvIndexerBackend, worker: &str) -> Vec<String> {
    let mut cmd = redis::cmd("SMEMBERS");
    cmd.arg(worker_blocks_key(&backend.ns, worker));
    let value = backend.conn.query(cmd).await.expect("read reverse index");
    Vec::<String>::from_redis_value(&value).expect("decode reverse index")
}

async fn meta_field(backend: &RedisKvIndexerBackend, worker: &str, field: &str) -> Option<String> {
    let mut cmd = redis::cmd("HGET");
    cmd.arg(worker_meta_key(&backend.ns, worker)).arg(field);
    let value = backend.conn.query(cmd).await.expect("read meta field");
    Option::<String>::from_redis_value(&value).expect("decode meta field")
}

#[tokio::test]
async fn report_replay_repairs_missing_reverse_index_after_sadd_failure() {
    let Some(backend) = backend("report_repair", b"SADD").await else {
        return;
    };
    let request = batch("worker-0", 1, ExternalKvActionType::ActionReport);

    // PLACEMENT_SET commits, then SADD fails.
    assert!(backend.apply(request.clone()).await.is_err());
    assert!(reverse_hashes(&backend, "worker-0").await.is_empty());

    // Replay must run SADD although placement is already set.
    backend.apply(request).await.expect("report replay");
    assert_eq!(
        reverse_hashes(&backend, "worker-0").await,
        vec!["hash-a".to_string()]
    );

    backend
        .apply(batch("worker-0", 2, ExternalKvActionType::ActionRevoke))
        .await
        .expect("cleanup");
}

#[tokio::test]
async fn revoke_replay_repairs_stale_reverse_index_after_srem_failure() {
    let Some(backend) = backend("revoke_repair", b"SREM").await else {
        return;
    };
    backend
        .apply(batch("worker-0", 1, ExternalKvActionType::ActionReport))
        .await
        .expect("initial report");

    let request = batch("worker-0", 2, ExternalKvActionType::ActionRevoke);

    // PLACEMENT_CLEAR commits, then SREM fails.
    assert!(backend.apply(request.clone()).await.is_err());
    assert_eq!(
        reverse_hashes(&backend, "worker-0").await,
        vec!["hash-a".to_string()]
    );

    // Replay sees placement already absent but must retry SREM.
    backend.apply(request).await.expect("revoke replay");
    assert!(reverse_hashes(&backend, "worker-0").await.is_empty());
}

#[tokio::test]
async fn restart_reset_retried_after_mid_reset_failure() {
    // P1 regression: a worker restart bumps the durable incarnation and sets a
    // `reset_pending` flag BEFORE the (non-atomic) reset runs. If the reset fails
    // partway, the flag must survive so the next apply retries and finishes it —
    // otherwise the incarnation already looks current and the stale state sticks.
    let Some(backend) = backend("reset_retry", b"DEL").await else {
        return;
    };

    // Incarnation "A" reports hash-a.
    backend
        .apply(batch_inc("worker-0", 1, "A", ExternalKvActionType::ActionReport))
        .await
        .expect("initial report");
    assert_eq!(reverse_hashes(&backend, "worker-0").await, vec!["hash-a".to_string()]);

    // Incarnation "B" (a restart): reset_pending is set, but the reverse-index
    // DEL inside reset_worker fails once, aborting the reset mid-flight.
    let restart = batch_inc("worker-0", 1, "B", ExternalKvActionType::ActionReport);
    assert!(backend.apply(restart.clone()).await.is_err());
    assert_eq!(
        meta_field(&backend, "worker-0", "reset_pending").await.as_deref(),
        Some("1"),
        "reset_pending must persist after a failed reset so it can be retried",
    );

    // Retry: incarnation is already "B" (unchanged), yet the lingering
    // reset_pending must still drive an idempotent reset to completion.
    backend.apply(restart).await.expect("restart reset retry");
    assert_eq!(
        meta_field(&backend, "worker-0", "reset_pending").await,
        None,
        "reset_pending must be cleared once the reset fully succeeds",
    );
    assert_eq!(
        reverse_hashes(&backend, "worker-0").await,
        vec!["hash-a".to_string()],
        "restarted incarnation's fresh report must be indexed after the reset",
    );
}

#[tokio::test]
async fn reset_window_hides_worker_from_match_until_reset_completes() {
    // P1: while a restart reset is pending, the worker's still-present placement
    // must not be routed to. Fail the reset's very first step (SMEMBERS) so the
    // stale placement/reverse entries survive with reset_pending=1, then assert
    // `match` drops the worker (even with liveness TTL disabled) until a retry
    // finishes the reset.
    let Some(backend) = backend("reset_window", b"SMEMBERS").await else {
        return;
    };

    // Incarnation "A" reports hash-a: matchable while healthy.
    backend
        .apply(batch_inc("worker-0", 1, "A", ExternalKvActionType::ActionReport))
        .await
        .expect("initial report");
    let hit = backend
        .do_match(MatchExternalKvRequest {
            hashes: vec!["hash-a".to_string()],
            count_as_hit: false,
        })
        .await
        .expect("match healthy");
    assert!(hit.matches.iter().any(|m| m.worker_id == "worker-0"));

    // Incarnation "B": reset_pending is set, but SMEMBERS fails so the reset never
    // clears the stale placement. The worker must now be hidden from match.
    let restart = batch_inc("worker-0", 1, "B", ExternalKvActionType::ActionReport);
    assert!(backend.apply(restart.clone()).await.is_err());
    assert_eq!(
        meta_field(&backend, "worker-0", "reset_pending").await.as_deref(),
        Some("1"),
    );
    let during = backend
        .do_match(MatchExternalKvRequest {
            hashes: vec!["hash-a".to_string()],
            count_as_hit: false,
        })
        .await
        .expect("match during pending");
    assert!(
        during.matches.is_empty(),
        "worker with reset_pending must be dropped from match, got {:?}",
        during.matches
    );

    // Retry completes the reset and re-indexes the fresh report; routable again.
    backend.apply(restart).await.expect("reset retry");
    assert_eq!(meta_field(&backend, "worker-0", "reset_pending").await, None);
    let after = backend
        .do_match(MatchExternalKvRequest {
            hashes: vec!["hash-a".to_string()],
            count_as_hit: false,
        })
        .await
        .expect("match after reset");
    assert!(after.matches.iter().any(|m| m.worker_id == "worker-0"));
}

#[tokio::test]
async fn touch_meta_persists_meta_left_with_ttl_by_old_build() {
    // P1 (rolling upgrade): an older build set a TTL on the whole meta hash. The
    // new build must PERSIST it on first contact so `incarnation`/`seq` can no
    // longer expire (which would silently resurrect stale placements).
    let Some(backend) = backend("ttl_persist", b"UNUSED-CMD").await else {
        return;
    };
    let worker = "worker-0";
    let meta = worker_meta_key(&backend.ns, worker);

    // Simulate the legacy meta hash: incarnation + seq, with a TTL on the key.
    let mut hset = redis::cmd("HSET");
    hset.arg(&meta).arg("incarnation").arg("A").arg("seq").arg("3");
    backend.conn.query(hset).await.expect("seed legacy meta");
    let mut pexpire = redis::cmd("PEXPIRE");
    pexpire.arg(&meta).arg(60_000);
    backend.conn.query(pexpire).await.expect("seed ttl");

    // Any apply from the same incarnation (no reset) must persist the key.
    backend
        .apply(batch_inc(worker, 4, "A", ExternalKvActionType::ActionReport))
        .await
        .expect("apply");

    let mut pttl = redis::cmd("PTTL");
    pttl.arg(&meta);
    let v = backend.conn.query(pttl).await.expect("pttl");
    let ttl = i64::from_redis_value(&v).expect("decode pttl");
    assert_eq!(ttl, -1, "meta key must be persisted (no TTL) after touch_meta");
    assert_eq!(
        meta_field(&backend, worker, "incarnation").await.as_deref(),
        Some("A"),
        "persisting must not disturb the durable incarnation",
    );
}

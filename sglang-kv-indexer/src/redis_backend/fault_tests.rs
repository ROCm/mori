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
    }
}

async fn reverse_hashes(backend: &RedisKvIndexerBackend, worker: &str) -> Vec<String> {
    let mut cmd = redis::cmd("SMEMBERS");
    cmd.arg(worker_blocks_key(&backend.ns, worker));
    let value = backend.conn.query(cmd).await.expect("read reverse index");
    Vec::<String>::from_redis_value(&value).expect("decode reverse index")
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

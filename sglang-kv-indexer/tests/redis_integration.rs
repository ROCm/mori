//! Integration tests for the Redis backend.
//!
//! These require a live store and are opt-in via environment:
//!   * `KV_INDEXER_REDIS_URL`            → single Redis/Dragonfly, or
//!   * `KV_INDEXER_REDIS_CLUSTER_NODES`  → Redis Cluster (comma-separated seeds)
//!
//! When neither is set every test skips (prints a note and returns), so the
//! default `cargo test --features redis-backend` run stays green without a store.
//!
//! Each test uses a unique namespace so a shared store never causes collisions.
#![cfg(feature = "redis-backend")]

use std::time::{SystemTime, UNIX_EPOCH};

use sglang_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    GetExternalKvHitCountsRequest, MatchExternalKvRequest, MatchExternalKvResponse, TierType,
};
use sglang_kv_indexer::{KvIndexerBackend, RedisKvIndexerBackend};

fn hbm() -> i32 {
    TierType::TierHbm as i32
}
fn dram() -> i32 {
    TierType::TierDram as i32
}

/// Builds a backend against the configured store with a unique namespace, or
/// returns `None` (skip) when no store env is set.
async fn backend(test: &str) -> Option<RedisKvIndexerBackend> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let ns = format!("itest:{test}:{nanos}");
    if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
        let nodes: Vec<String> = nodes
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        Some(
            RedisKvIndexerBackend::connect_cluster(nodes, ns)
                .await
                .expect("connect cluster"),
        )
    } else if let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") {
        Some(
            RedisKvIndexerBackend::connect_single(&url, ns)
                .await
                .expect("connect single"),
        )
    } else {
        eprintln!("skipping {test}: set KV_INDEXER_REDIS_URL or KV_INDEXER_REDIS_CLUSTER_NODES");
        None
    }
}

fn hashes(hs: &[&str]) -> Vec<String> {
    hs.iter().map(|h| h.to_string()).collect()
}

fn action(kind: ExternalKvActionType, tier: i32, hs: &[&str]) -> ExternalKvAction {
    ExternalKvAction {
        r#type: kind as i32,
        tier,
        hashes: hashes(hs),
    }
}

fn apply_req(
    worker: &str,
    addr: &str,
    seq: u64,
    actions: Vec<ExternalKvAction>,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        actions,
        worker_address: addr.to_string(),
    }
}

fn match_req(hs: &[&str], count_as_hit: bool) -> MatchExternalKvRequest {
    MatchExternalKvRequest {
        hashes: hashes(hs),
        count_as_hit,
    }
}

/// Returns the tiers a worker holds a hash at, per the match response.
fn tiers_for(resp: &MatchExternalKvResponse, worker: &str, hash: &str) -> Vec<i32> {
    let mut tiers = Vec::new();
    for m in &resp.matches {
        if m.worker_id != worker {
            continue;
        }
        for th in &m.hashes_by_tier {
            if th.hashes.iter().any(|h| h == hash) {
                tiers.push(th.tier);
            }
        }
    }
    tiers.sort_unstable();
    tiers
}

macro_rules! itest {
    ($name:ident, $b:ident, $body:block) => {
        #[tokio::test]
        async fn $name() {
            let Some($b) = backend(stringify!($name)).await else {
                return;
            };
            $body
        }
    };
}

itest!(report_then_match_returns_worker_and_address, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "10.0.0.1:9000",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &["1", "2"],
        )],
    ))
    .await
    .unwrap();

    let resp = b
        .match_external_kv(match_req(&["1", "2", "3"], false))
        .await
        .unwrap();
    assert_eq!(resp.matches.len(), 1);
    let m = &resp.matches[0];
    assert_eq!(m.worker_id, "w1");
    assert_eq!(m.address, "10.0.0.1:9000");
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
    assert_eq!(tiers_for(&resp, "w1", "2"), vec![hbm()]);
    assert!(tiers_for(&resp, "w1", "3").is_empty());
});

itest!(duplicate_report_is_idempotent, b, {
    for _ in 0..3 {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    }
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
});

itest!(same_seq_batch_replay_is_idempotent, b, {
    // A batch that stores then removes then stores again the same hash; the net
    // state is "stored". Replaying the identical batch (same seq) must not change it.
    let batch = apply_req(
        "w1",
        "a",
        7,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["9"]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &["9"]),
            action(ExternalKvActionType::ActionReport, hbm(), &["9"]),
        ],
    );
    b.apply_external_kv_batch(batch.clone()).await.unwrap();
    let first = b.match_external_kv(match_req(&["9"], false)).await.unwrap();
    b.apply_external_kv_batch(batch).await.unwrap();
    let second = b.match_external_kv(match_req(&["9"], false)).await.unwrap();
    assert_eq!(tiers_for(&first, "w1", "9"), vec![hbm()]);
    assert_eq!(tiers_for(&second, "w1", "9"), vec![hbm()]);
});

itest!(revoke_partial_tier_keeps_other_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["1"]),
            action(ExternalKvActionType::ActionReport, dram(), &["1"]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &["1"]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![dram()]);
});

itest!(revoke_missing_hash_is_idempotent, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["404"])],
    ))
    .await
    .unwrap();
    let resp = b
        .match_external_kv(match_req(&["404"], false))
        .await
        .unwrap();
    assert!(resp.matches.is_empty());
});

itest!(multi_worker_multi_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a1",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w2",
        "a2",
        1,
        vec![action(ExternalKvActionType::ActionReport, dram(), &["1"])],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(resp.matches.len(), 2);
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
    assert_eq!(tiers_for(&resp, "w2", "1"), vec![dram()]);
});

itest!(clear_all_at_tier_removes_only_that_tier, b, {
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["1", "2", "3"]),
            action(ExternalKvActionType::ActionReport, dram(), &["1"]),
        ],
    ))
    .await
    .unwrap();
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(
            ExternalKvActionType::ActionClearAllAtTier,
            hbm(),
            &[],
        )],
    ))
    .await
    .unwrap();
    let resp = b
        .match_external_kv(match_req(&["1", "2", "3"], false))
        .await
        .unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![dram()]);
    assert!(tiers_for(&resp, "w1", "2").is_empty());
    assert!(tiers_for(&resp, "w1", "3").is_empty());
});

itest!(
    count_as_hit_only_counts_matched_and_replay_does_not_double,
    b,
    {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();

        // Diagnostic match (count_as_hit=false) must not count.
        b.match_external_kv(match_req(&["1", "2"], false))
            .await
            .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1", "2"]),
            })
            .await
            .unwrap();
        assert!(counts.entries.is_empty());

        // Counting match: only the matched hash "1" is counted, "2" (a miss) is not.
        b.match_external_kv(match_req(&["1", "2"], true))
            .await
            .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1", "2"]),
            })
            .await
            .unwrap();
        assert_eq!(counts.entries.len(), 1);
        assert_eq!(counts.entries[0].hash, "1");
        assert_eq!(counts.entries[0].hit_count_total, 1);

        // Replaying the apply batch must not touch hit counts.
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
        let counts = b
            .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
                hashes: hashes(&["1"]),
            })
            .await
            .unwrap();
        assert_eq!(counts.entries[0].hit_count_total, 1);
    }
);

itest!(full_revoke_drops_hit_key, b, {
    // Report a block, count a hit (creates the co-located :h key), then fully
    // revoke it. The hit key must be removed together with placement; otherwise a
    // matched-then-evicted block leaks its :h key forever (slow Redis memory growth).
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
    ))
    .await
    .unwrap();

    // Counting match creates the hit key with c=1.
    b.match_external_kv(match_req(&["1"], true)).await.unwrap();
    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: hashes(&["1"]),
        })
        .await
        .unwrap();
    assert_eq!(counts.entries.len(), 1);
    assert_eq!(counts.entries[0].hit_count_total, 1);

    // Fully revoke the block: placement empties, so the hit key must go too.
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
    ))
    .await
    .unwrap();

    // Placement is gone.
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert!(resp.matches.is_empty());

    // Hit key is gone too: a leaked :h would still report a count here.
    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: hashes(&["1"]),
        })
        .await
        .unwrap();
    assert!(
        counts.entries.is_empty(),
        "hit key leaked after full revoke: {:?}",
        counts.entries
    );
});

itest!(partial_revoke_keeps_hit_key, b, {
    // Block present at two tiers; count a hit, then revoke only one tier. Placement
    // is still non-empty, so the hit key must survive (guard against over-deletion).
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["1"]),
            action(ExternalKvActionType::ActionReport, dram(), &["1"]),
        ],
    ))
    .await
    .unwrap();

    b.match_external_kv(match_req(&["1"], true)).await.unwrap();

    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
    ))
    .await
    .unwrap();

    let counts = b
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: hashes(&["1"]),
        })
        .await
        .unwrap();
    assert_eq!(counts.entries.len(), 1);
    assert_eq!(counts.entries[0].hit_count_total, 1);
});

itest!(batch_action_order_is_preserved, b, {
    // revoke-then-report on the same hash within one batch must net to "stored".
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        1,
        vec![
            action(ExternalKvActionType::ActionRevoke, hbm(), &["5"]),
            action(ExternalKvActionType::ActionReport, hbm(), &["5"]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["5"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "5"), vec![hbm()]);

    // report-then-revoke on the same hash must net to "absent".
    b.apply_external_kv_batch(apply_req(
        "w1",
        "a",
        2,
        vec![
            action(ExternalKvActionType::ActionReport, hbm(), &["6"]),
            action(ExternalKvActionType::ActionRevoke, hbm(), &["6"]),
        ],
    ))
    .await
    .unwrap();
    let resp = b.match_external_kv(match_req(&["6"], false)).await.unwrap();
    assert!(tiers_for(&resp, "w1", "6").is_empty());
});

// --- server-side seq gate (durable idempotency) -----------------------------

itest!(duplicate_seq_batch_is_skipped_not_reapplied, b, {
    // Apply a report at seq=1, then re-send seq=1 carrying a *different*
    // mutation (a revoke). Because seq=1 was already applied, the whole batch
    // is a duplicate and must be skipped, so the revoke has no effect.
    let applied = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert_eq!(applied.last_applied_seq, 1);
    assert!(!applied.duplicate);

    let dup = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert!(dup.duplicate, "re-sent seq must be reported as a duplicate");
    assert_eq!(dup.last_applied_seq, 1);

    // The revoke was skipped: the hash is still present.
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
});

itest!(lower_seq_batch_is_skipped_as_stale, b, {
    // Advance to seq=5, then a late/out-of-order seq=3 arrives. It must be
    // treated as stale (<= last) and skipped, leaving the ack at 5.
    for seq in [1u64, 5] {
        b.apply_external_kv_batch(apply_req(
            "w1",
            "a",
            seq,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    }
    let stale = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            3,
            vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert!(stale.duplicate);
    assert_eq!(stale.last_applied_seq, 5);
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(tiers_for(&resp, "w1", "1"), vec![hbm()]);
});

itest!(seq_is_per_worker_independent, b, {
    // seq counters are independent per worker: the same seq value applied to a
    // different worker is not a duplicate.
    let a = b
        .apply_external_kv_batch(apply_req(
            "w1",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    let c = b
        .apply_external_kv_batch(apply_req(
            "w2",
            "a",
            1,
            vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        ))
        .await
        .unwrap();
    assert!(!a.duplicate && !c.duplicate);
    let resp = b.match_external_kv(match_req(&["1"], false)).await.unwrap();
    assert_eq!(resp.matches.len(), 2);
});

// --- T5 regression: degraded-start / lazy-connect semantics -----------------

/// A deferred backend (KV_INDEXER_REDIS_REQUIRED=0 path) does not connect at
/// construction; it connects lazily on first use and serves correctly once the
/// store is reachable. Requires a live store, so it skips when unset.
#[tokio::test]
async fn deferred_backend_connects_lazily_and_serves() {
    let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") else {
        eprintln!("skipping deferred_backend_connects_lazily_and_serves: set KV_INDEXER_REDIS_URL");
        return;
    };
    let ns = format!(
        "itest:deferred_serves:{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    // Construction never touches the network.
    let b = RedisKvIndexerBackend::connect_single_deferred(&url, ns);
    // First use lazily establishes the connection and succeeds.
    b.apply_external_kv_batch(apply_req(
        "wd",
        "10.9.9.9:9000",
        1,
        vec![action(
            ExternalKvActionType::ActionReport,
            hbm(),
            &["100", "101"],
        )],
    ))
    .await
    .expect("lazy connect + apply should succeed against a live store");
    let resp = b
        .match_external_kv(match_req(&["100", "101"], false))
        .await
        .unwrap();
    assert!(
        resp.matches.iter().any(|m| m.worker_id == "wd"),
        "reported hashes must be matchable after a lazy connect"
    );
}

/// A deferred backend pointed at an unreachable Redis must fail requests with an
/// error within a bounded time (connect timeout), never hang. No store needed.
#[tokio::test]
async fn deferred_backend_unreachable_errors_within_bound() {
    let b = RedisKvIndexerBackend::connect_single_deferred(
        "redis://127.0.0.1:6399",
        "itest:deferred_dead",
    );
    let started = std::time::Instant::now();
    let res = b.match_external_kv(match_req(&["x"], false)).await;
    let elapsed = started.elapsed();
    assert!(
        res.is_err(),
        "match against an unreachable redis must error, got {res:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(15),
        "request must be bounded by the connect timeout, took {elapsed:?}"
    );
}

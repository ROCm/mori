//! gRPC contract tests: exercise all three RPCs of the `KVIndexer` service
//! over the wire (real tonic server + client), not just the backend trait.
//!
//! Like the backend integration tests these require a live store and are
//! opt-in via `KV_INDEXER_REDIS_URL` (or `KV_INDEXER_REDIS_CLUSTER_NODES`);
//! when neither is set every test skips. Each test uses a unique namespace and
//! unique worker/hash ids so a shared store never causes collisions.
#![cfg(feature = "redis-backend")]

use std::net::SocketAddr;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tonic::transport::Server;
use tonic::Code;

use sglang_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sglang_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sglang_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType,
    GetExternalKvHitCountsRequest, MatchExternalKvRequest, TierType,
};
use sglang_kv_indexer::{KvIndexerService, RedisKvIndexerBackend};

fn hbm() -> i32 {
    TierType::TierHbm as i32
}
fn dram() -> i32 {
    TierType::TierDram as i32
}
fn report() -> i32 {
    ExternalKvActionType::ActionReport as i32
}
fn revoke() -> i32 {
    ExternalKvActionType::ActionRevoke as i32
}
fn clear_all_at_tier() -> i32 {
    ExternalKvActionType::ActionClearAllAtTier as i32
}

fn nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
}

/// Reserves an ephemeral loopback port. The listener is dropped immediately;
/// the tiny window before the server rebinds is covered by the client's
/// connect-retry loop below.
fn free_addr() -> SocketAddr {
    let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    l.local_addr().unwrap()
}

/// Starts a real gRPC server backed by Redis on a unique namespace and returns
/// a connected client, or `None` (skip) when no store env is configured.
async fn start(test: &str) -> Option<KvIndexerClient<tonic::transport::Channel>> {
    let url = match std::env::var("KV_INDEXER_REDIS_URL") {
        Ok(u) => u,
        Err(_) => {
            eprintln!("skipping {test}: set KV_INDEXER_REDIS_URL");
            return None;
        }
    };
    let ns = format!("grpc:{test}:{}", nanos());
    let backend = RedisKvIndexerBackend::connect_single(&url, ns)
        .await
        .expect("connect redis");
    let svc = KvIndexerServer::new(KvIndexerService::new(backend));
    let addr = free_addr();
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve(addr)
            .await
            .expect("server serve");
    });

    let endpoint = format!("http://{addr}");
    for _ in 0..50 {
        if let Ok(c) = KvIndexerClient::connect(endpoint.clone()).await {
            return Some(c);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("client failed to connect to {endpoint}");
}

fn apply(
    worker: &str,
    addr: &str,
    seq: u64,
    action_type: i32,
    tier: i32,
    hashes: &[&str],
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id: worker.to_string(),
        seq,
        worker_address: addr.to_string(),
        actions: vec![ExternalKvAction {
            r#type: action_type,
            tier,
            hashes: hashes.iter().map(|s| s.to_string()).collect(),
        }],
    }
}

fn apply_report(
    worker: &str,
    addr: &str,
    seq: u64,
    tier: i32,
    hashes: &[&str],
) -> ApplyExternalKvBatchRequest {
    apply(worker, addr, seq, report(), tier, hashes)
}

#[tokio::test]
async fn apply_match_and_hit_counts_over_grpc() {
    let Some(mut c) = start("apply_match").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let (h1, h2, miss) = ("am-h1", "am-h2", "am-miss");

    c.apply_external_kv_batch(apply_report(&w, "10.0.0.1:9000", 1, hbm(), &[h1, h2]))
        .await
        .expect("apply ok");

    let resp = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h1.into(), h2.into(), miss.into()],
            count_as_hit: true,
        })
        .await
        .expect("match ok")
        .into_inner();

    let m = resp
        .matches
        .iter()
        .find(|m| m.worker_id == w)
        .expect("worker present in matches");
    assert_eq!(m.address, "10.0.0.1:9000");
    let tier = m
        .hashes_by_tier
        .iter()
        .find(|t| t.tier == hbm())
        .expect("HBM tier present");
    let mut got: Vec<&String> = tier.hashes.iter().collect();
    got.sort();
    assert_eq!(got, vec![&h1.to_string(), &h2.to_string()]);

    let hc = c
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: vec![h1.into(), h2.into(), miss.into()],
        })
        .await
        .expect("hit counts ok")
        .into_inner();
    let count = |h: &str| {
        hc.entries
            .iter()
            .find(|e| e.hash == h)
            .map(|e| e.hit_count_total)
            .unwrap_or(0)
    };
    assert!(count(h1) >= 1, "h1 should have a hit");
    assert!(count(h2) >= 1, "h2 should have a hit");
    assert_eq!(count(miss), 0, "unmatched hash must not be counted");
}

#[tokio::test]
async fn diagnostic_match_does_not_count_hits_over_grpc() {
    let Some(mut c) = start("diag_match").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let h = "diag-h1";
    c.apply_external_kv_batch(apply_report(&w, "10.0.0.2:9000", 1, hbm(), &[h]))
        .await
        .expect("apply ok");

    // count_as_hit=false must not bump counters
    c.match_external_kv(MatchExternalKvRequest {
        hashes: vec![h.into()],
        count_as_hit: false,
    })
    .await
    .expect("match ok");

    let hc = c
        .get_external_kv_hit_counts(GetExternalKvHitCountsRequest {
            hashes: vec![h.into()],
        })
        .await
        .expect("hit counts ok")
        .into_inner();
    let count = hc
        .entries
        .iter()
        .find(|e| e.hash == h)
        .map(|e| e.hit_count_total)
        .unwrap_or(0);
    assert_eq!(count, 0, "diagnostic match must not increase hit count");
}

#[tokio::test]
async fn apply_report_then_revoke_over_grpc() {
    let Some(mut c) = start("apply_rr").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let h = "apply-rr-h1";

    c.apply_external_kv_batch(apply_report(&w, "10.0.0.3:9000", 1, hbm(), &[h]))
        .await
        .expect("report apply ok");

    let before = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h.into()],
            count_as_hit: false,
        })
        .await
        .expect("match ok")
        .into_inner();
    assert!(
        before.matches.iter().any(|m| m.worker_id == w),
        "reported hash should match"
    );

    c.apply_external_kv_batch(apply(&w, "10.0.0.3:9000", 2, revoke(), hbm(), &[h]))
        .await
        .expect("revoke apply ok");

    let after = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h.into()],
            count_as_hit: false,
        })
        .await
        .expect("match ok")
        .into_inner();
    assert!(
        !after.matches.iter().any(|m| m.worker_id == w),
        "revoked hash must not match"
    );
}

#[tokio::test]
async fn revoke_all_at_tier_over_grpc() {
    let Some(mut c) = start("revoke_all").await else {
        return;
    };
    let w = format!("w-{}", nanos());
    let h = "ra-h1";

    // same hash present in both HBM and DRAM
    c.apply_external_kv_batch(apply_report(&w, "10.0.0.3:9000", 1, hbm(), &[h]))
        .await
        .expect("apply hbm");
    c.apply_external_kv_batch(apply_report(&w, "10.0.0.3:9000", 2, dram(), &[h]))
        .await
        .expect("apply dram");

    c.apply_external_kv_batch(apply(
        &w,
        "10.0.0.3:9000",
        3,
        clear_all_at_tier(),
        hbm(),
        &[],
    ))
    .await
    .expect("clear-all apply ok");

    let resp = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![h.into()],
            count_as_hit: false,
        })
        .await
        .expect("match ok")
        .into_inner();
    let m = resp
        .matches
        .iter()
        .find(|m| m.worker_id == w)
        .expect("worker still present via DRAM");
    let tiers: Vec<i32> = m.hashes_by_tier.iter().map(|t| t.tier).collect();
    assert!(tiers.contains(&dram()), "DRAM tier must remain");
    assert!(!tiers.contains(&hbm()), "HBM tier must be cleared");
}

#[tokio::test]
async fn match_miss_returns_empty_over_grpc() {
    let Some(mut c) = start("match_miss").await else {
        return;
    };
    let resp = c
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec![format!("never-reported-{}", nanos())],
            count_as_hit: true,
        })
        .await
        .expect("match ok")
        .into_inner();
    assert!(resp.matches.is_empty(), "unknown hash yields no matches");
}

#[tokio::test]
async fn validation_errors_map_to_invalid_argument_over_grpc() {
    let Some(mut c) = start("validation").await else {
        return;
    };

    // empty worker_id
    let err = c
        .apply_external_kv_batch(apply_report("", "addr", 1, hbm(), &["h"]))
        .await
        .expect_err("empty worker_id must be rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "empty worker_id -> InvalidArgument"
    );

    // REPORT action with no hashes
    let err = c
        .apply_external_kv_batch(apply("w", "addr", 1, report(), hbm(), &[]))
        .await
        .expect_err("empty hashes must be rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "empty hashes -> InvalidArgument"
    );

    // unknown action type
    let bad = ApplyExternalKvBatchRequest {
        worker_id: "w".into(),
        seq: 1,
        worker_address: String::new(),
        actions: vec![ExternalKvAction {
            r#type: 999,
            tier: hbm(),
            hashes: vec!["h".into()],
        }],
    };
    let err = c
        .apply_external_kv_batch(bad)
        .await
        .expect_err("unknown action type rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "unknown action type -> InvalidArgument"
    );

    // invalid tier in an ApplyBatch action
    let err = c
        .apply_external_kv_batch(apply("w", "addr", 1, report(), 999, &["h"]))
        .await
        .expect_err("bad tier rejected");
    assert_eq!(
        err.code(),
        Code::InvalidArgument,
        "bad tier -> InvalidArgument"
    );
}

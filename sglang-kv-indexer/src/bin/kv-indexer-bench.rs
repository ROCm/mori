//! Load generator for the Redis backend, modelled on SGLang usage: a single
//! node runs several worker processes (default 8), each of which forwards its
//! own KV events via `ApplyExternalKvBatch`; a router calls `MatchExternalKv` on
//! the hot read path. This bench drives both and reports throughput + p50/p99.
//!
//! Store selection matches the integration tests / server:
//!   * `KV_INDEXER_REDIS_URL`            → single Redis/Dragonfly, or
//!   * `KV_INDEXER_REDIS_CLUSTER_NODES`  → Redis Cluster (comma-separated seeds)
//!
//! Tunables (env):
//!   KV_BENCH_WORKERS            (default 8)     apply workers (SGLang processes)
//!   KV_BENCH_HASHES_PER_WORKER  (default 20000) keyspace populated per worker
//!   KV_BENCH_APPLY_BATCH        (default 32)    hashes per apply batch
//!   KV_BENCH_MATCH_CONCURRENCY  (default 8)     concurrent router match callers
//!   KV_BENCH_MATCH_BATCH        (default 16)    hashes per match query
//!   KV_BENCH_DURATION_SECS      (default 5)     measurement window per phase
//!
//! Run (inside the container, with a store up):
//!   KV_INDEXER_REDIS_URL=redis://127.0.0.1:6379 \
//!     cargo run --release --features redis-backend --bin kv-indexer-bench

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sglang_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, MatchExternalKvRequest,
    TierType,
};
use sglang_kv_indexer::{KvIndexerBackend, RedisKvIndexerBackend};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

struct Config {
    workers: usize,
    hashes_per_worker: u64,
    apply_batch: usize,
    match_concurrency: usize,
    match_batch: usize,
    duration: Duration,
    ns: String,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

async fn connect(ns: &str) -> Result<RedisKvIndexerBackend, BoxError> {
    if let Ok(nodes) = std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES") {
        let nodes: Vec<String> = nodes
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect();
        RedisKvIndexerBackend::connect_cluster(nodes, ns.to_string()).await
    } else if let Ok(url) = std::env::var("KV_INDEXER_REDIS_URL") {
        RedisKvIndexerBackend::connect_single(&url, ns.to_string()).await
    } else {
        Err("set KV_INDEXER_REDIS_URL or KV_INDEXER_REDIS_CLUSTER_NODES".into())
    }
}

/// Small deterministic PRNG so the bench needs no rand dependency.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn report_action(tier: i32, hashes: Vec<String>) -> ExternalKvAction {
    ExternalKvAction {
        r#type: ExternalKvActionType::ActionReport as i32,
        tier,
        hashes,
    }
}

/// Percentile from a sorted slice of microsecond latencies.
fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 * p).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

struct Stats {
    label: String,
    ops: u64,
    secs: f64,
    p50: u64,
    p99: u64,
    mean: u64,
}

fn summarize(label: &str, mut lat: Vec<u64>, secs: f64) -> Stats {
    lat.sort_unstable();
    let ops = lat.len() as u64;
    let mean = lat.iter().sum::<u64>().checked_div(ops).unwrap_or(0);
    Stats {
        label: label.to_string(),
        ops,
        secs,
        p50: pct(&lat, 0.50),
        p99: pct(&lat, 0.99),
        mean,
    }
}

fn print_stats(s: &Stats) {
    let ops_s = if s.secs > 0.0 {
        s.ops as f64 / s.secs
    } else {
        0.0
    };
    println!(
        "{:<8} ops={:<8} {:>10.0} ops/s   p50={:>6}us  p99={:>7}us  mean={:>6}us",
        s.label, s.ops, ops_s, s.p50, s.p99, s.mean
    );
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let cfg = Config {
        workers: env_usize("KV_BENCH_WORKERS", 8),
        hashes_per_worker: env_u64("KV_BENCH_HASHES_PER_WORKER", 20_000),
        apply_batch: env_usize("KV_BENCH_APPLY_BATCH", 32),
        match_concurrency: env_usize("KV_BENCH_MATCH_CONCURRENCY", 8),
        match_batch: env_usize("KV_BENCH_MATCH_BATCH", 16),
        duration: Duration::from_secs(env_u64("KV_BENCH_DURATION_SECS", 5)),
        ns: format!("kvbench:{nanos}"),
    };

    let topology = if std::env::var("KV_INDEXER_REDIS_CLUSTER_NODES").is_ok() {
        "cluster"
    } else {
        "single"
    };
    println!(
        "store={topology} ns={} workers={} hashes/worker={} apply_batch={} match_conc={} match_batch={} window={}s",
        cfg.ns,
        cfg.workers,
        cfg.hashes_per_worker,
        cfg.apply_batch,
        cfg.match_concurrency,
        cfg.match_batch,
        cfg.duration.as_secs(),
    );

    let hbm = TierType::TierHbm as i32;
    let total_hashes = cfg.workers as u64 * cfg.hashes_per_worker;

    // --- Populate: each worker reports its whole keyspace once (also warms EVALSHA). ---
    let populate_start = Instant::now();
    let mut tasks = Vec::new();
    for w in 0..cfg.workers {
        let backend = Arc::new(connect(&cfg.ns).await?);
        let base = w as u64 * cfg.hashes_per_worker;
        let count = cfg.hashes_per_worker;
        let batch = cfg.apply_batch;
        let worker_id = format!("worker-{w}");
        let addr = format!("10.0.0.{w}:9000");
        tasks.push(tokio::spawn(async move {
            let mut next = base;
            let end = base + count;
            let mut seq = 0u64;
            while next < end {
                let hi = (next + batch as u64).min(end);
                let hashes: Vec<String> = (next..hi).map(|h| h.to_string()).collect();
                next = hi;
                seq += 1;
                let req = ApplyExternalKvBatchRequest {
                    worker_id: worker_id.clone(),
                    seq,
                    actions: vec![report_action(hbm, hashes)],
                    worker_address: addr.clone(),
                    incarnation: String::new(),
                };
                backend.apply_external_kv_batch(req).await.unwrap();
            }
        }));
    }
    for t in tasks {
        t.await?;
    }
    println!(
        "populate: {} hashes in {:.2}s",
        total_hashes,
        populate_start.elapsed().as_secs_f64()
    );

    // --- Apply throughput: N workers apply REPORT batches concurrently for the window. ---
    let mut tasks = Vec::new();
    for w in 0..cfg.workers {
        let backend = Arc::new(connect(&cfg.ns).await?);
        let base = w as u64 * cfg.hashes_per_worker;
        let count = cfg.hashes_per_worker;
        let batch = cfg.apply_batch;
        let dur = cfg.duration;
        let worker_id = format!("worker-{w}");
        let addr = format!("10.0.0.{w}:9000");
        tasks.push(tokio::spawn(async move {
            let mut lat = Vec::new();
            let mut rng = base.wrapping_add(0x9E3779B97F4A7C15) | 1;
            let deadline = Instant::now() + dur;
            let mut seq = 1_000_000u64;
            while Instant::now() < deadline {
                let start_off = xorshift(&mut rng) % count.max(1);
                let hashes: Vec<String> = (0..batch as u64)
                    .map(|i| (base + (start_off + i) % count.max(1)).to_string())
                    .collect();
                seq += 1;
                let req = ApplyExternalKvBatchRequest {
                    worker_id: worker_id.clone(),
                    seq,
                    actions: vec![report_action(hbm, hashes)],
                    worker_address: addr.clone(),
                    incarnation: String::new(),
                };
                let t0 = Instant::now();
                backend.apply_external_kv_batch(req).await.unwrap();
                lat.push(t0.elapsed().as_micros() as u64);
            }
            lat
        }));
    }
    let mut apply_lat = Vec::new();
    for t in tasks {
        apply_lat.extend(t.await?);
    }
    print_stats(&summarize("apply", apply_lat, cfg.duration.as_secs_f64()));

    // --- Match throughput: router callers query batches of existing hashes. ---
    let mut tasks = Vec::new();
    for c in 0..cfg.match_concurrency {
        let backend = Arc::new(connect(&cfg.ns).await?);
        let mbatch = cfg.match_batch;
        let dur = cfg.duration;
        tasks.push(tokio::spawn(async move {
            let mut lat = Vec::new();
            let mut rng = (c as u64).wrapping_mul(0xD1B54A32D192ED03) | 1;
            let deadline = Instant::now() + dur;
            while Instant::now() < deadline {
                let hashes: Vec<String> = (0..mbatch)
                    .map(|_| (xorshift(&mut rng) % total_hashes.max(1)).to_string())
                    .collect();
                let req = MatchExternalKvRequest {
                    hashes,
                    count_as_hit: true,
                };
                let t0 = Instant::now();
                backend.match_external_kv(req).await.unwrap();
                lat.push(t0.elapsed().as_micros() as u64);
            }
            lat
        }));
    }
    let mut match_lat = Vec::new();
    for t in tasks {
        match_lat.extend(t.await?);
    }
    print_stats(&summarize("match", match_lat, cfg.duration.as_secs_f64()));

    Ok(())
}

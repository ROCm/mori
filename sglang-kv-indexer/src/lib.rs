//! SGLang KV Indexer: a gRPC service that tracks externally-managed KV cache
//! block placements (as reported by inference engines such as SGLang HiCache)
//! and answers placement-match queries for KV-aware routing.

pub mod bridge;

pub mod pb {
    tonic::include_proto!("kv_indexer.v1");
}

mod service;

#[cfg(feature = "redis-backend")]
pub mod redis_backend;

pub use service::{KvIndexerBackend, KvIndexerService};

#[cfg(feature = "redis-backend")]
pub use redis_backend::RedisKvIndexerBackend;

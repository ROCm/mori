//! SGLang KV Indexer: a gRPC service that tracks externally-managed KV cache
//! block placements (as reported by inference engines such as SGLang HiCache)
//! and answers placement-match queries for KV-aware routing.

pub mod bridge;

pub mod pb {
    tonic::include_proto!("kv_indexer.v1");
}

mod service;

pub use service::{KvIndexerBackend, KvIndexerService, NoopKvIndexerBackend};

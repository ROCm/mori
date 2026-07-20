use std::net::SocketAddr;
use std::sync::Arc;

use sglang_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sglang_kv_indexer::{KvIndexerBackend, KvIndexerService, NoopKvIndexerBackend};
use tonic::transport::Server;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr = std::env::var("KV_INDEXER_LISTEN_ADDR")
        .unwrap_or_else(|_| "[::1]:50051".to_string())
        .parse::<SocketAddr>()?;

    let backend = select_backend().await?;
    let service = KvIndexerServer::new(KvIndexerService::new(backend));

    info!(%addr, "starting SGLang KV Indexer gRPC server");
    Server::builder()
        .add_service(service)
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    Ok(())
}

/// Selects the storage backend from `KV_INDEXER_BACKEND` (`noop` | `redis`,
/// default `noop`). The Redis backend lives behind the `redis-backend` cargo
/// feature so the default build stays light; requesting `redis` without that
/// feature is a loud startup error rather than a silent fallback.
async fn select_backend(
) -> Result<Arc<dyn KvIndexerBackend>, Box<dyn std::error::Error + Send + Sync>> {
    let backend = std::env::var("KV_INDEXER_BACKEND").unwrap_or_else(|_| "noop".to_string());
    match backend.as_str() {
        "noop" => {
            info!("using noop backend");
            Ok(Arc::new(NoopKvIndexerBackend))
        }
        "redis" => {
            #[cfg(feature = "redis-backend")]
            {
                info!("using redis backend");
                let backend = sglang_kv_indexer::RedisKvIndexerBackend::from_env().await?;
                Ok(Arc::new(backend))
            }
            #[cfg(not(feature = "redis-backend"))]
            {
                Err(
                    "KV_INDEXER_BACKEND=redis requires building with --features redis-backend"
                        .into(),
                )
            }
        }
        other => Err(format!("unknown KV_INDEXER_BACKEND: {other}").into()),
    }
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "failed to install ctrl-c handler");
    }
}

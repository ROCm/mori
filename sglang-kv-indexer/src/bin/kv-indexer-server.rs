use std::net::SocketAddr;

use sglang_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sglang_kv_indexer::{KvIndexerService, NoopKvIndexerBackend};
use tonic::transport::Server;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr = std::env::var("KV_INDEXER_LISTEN_ADDR")
        .unwrap_or_else(|_| "[::1]:50051".to_string())
        .parse::<SocketAddr>()?;

    let service = KvIndexerServer::new(KvIndexerService::new(NoopKvIndexerBackend));

    info!(%addr, "starting SGLang KV Indexer gRPC server");
    Server::builder()
        .add_service(service)
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    if let Err(error) = tokio::signal::ctrl_c().await {
        tracing::warn!(%error, "failed to install ctrl-c handler");
    }
}

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Mutex;

use sglang_kv_indexer::pb::{
    GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, MatchExternalKvRequest,
    MatchExternalKvResponse, ReportExternalKvBlocksRequest, RevokeAllExternalKvBlocksAtTierRequest,
    RevokeExternalKvBlocksRequest,
};
use sglang_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sglang_kv_indexer::{KvIndexerBackend, KvIndexerService};
use tonic::transport::Server;
use tonic::Status;
use tracing::info;

/// A small stateful backend for joint debugging: it keeps the live set of
/// (tier, hash) blocks in memory and logs running totals on every RPC so the
/// indexer side of the SGLang -> bridge -> indexer chain is observable.
#[derive(Default)]
struct LoggingKvIndexerBackend {
    live: Mutex<HashSet<(i32, String)>>,
}

impl LoggingKvIndexerBackend {
    fn total(&self) -> usize {
        self.live.lock().unwrap().len()
    }
}

impl KvIndexerBackend for LoggingKvIndexerBackend {
    fn report_external_kv_blocks(
        &self,
        request: ReportExternalKvBlocksRequest,
    ) -> Result<(), Status> {
        {
            let mut live = self.live.lock().unwrap();
            for hash in &request.hashes {
                live.insert((request.tier, hash.clone()));
            }
        }
        info!(
            worker = %request.worker_id,
            tier = request.tier,
            added = request.hashes.len(),
            live_total = self.total(),
            "REPORT external kv blocks"
        );
        Ok(())
    }

    fn revoke_external_kv_blocks(
        &self,
        request: RevokeExternalKvBlocksRequest,
    ) -> Result<(), Status> {
        {
            let mut live = self.live.lock().unwrap();
            for hash in &request.hashes {
                live.remove(&(request.tier, hash.clone()));
            }
        }
        info!(
            worker = %request.worker_id,
            tier = request.tier,
            removed = request.hashes.len(),
            live_total = self.total(),
            "REVOKE external kv blocks"
        );
        Ok(())
    }

    fn revoke_all_external_kv_blocks_at_tier(
        &self,
        request: RevokeAllExternalKvBlocksAtTierRequest,
    ) -> Result<(), Status> {
        let removed = {
            let mut live = self.live.lock().unwrap();
            let before = live.len();
            live.retain(|(tier, _)| *tier != request.tier);
            before - live.len()
        };
        info!(
            worker = %request.worker_id,
            tier = request.tier,
            removed,
            live_total = self.total(),
            "REVOKE_ALL external kv blocks at tier"
        );
        Ok(())
    }

    fn match_external_kv(
        &self,
        _request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        Ok(MatchExternalKvResponse { matches: vec![] })
    }

    fn get_external_kv_hit_counts(
        &self,
        _request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        Ok(GetExternalKvHitCountsResponse { entries: vec![] })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr = std::env::var("KV_INDEXER_LISTEN_ADDR")
        .unwrap_or_else(|_| "[::1]:50051".to_string())
        .parse::<SocketAddr>()?;

    let service = KvIndexerServer::new(KvIndexerService::new(LoggingKvIndexerBackend::default()));

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

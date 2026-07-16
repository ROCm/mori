use tonic::{Request, Response, Status};

use crate::pb::kv_indexer_server::KvIndexer;
use crate::pb::{
    Empty, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse, MatchExternalKvRequest,
    MatchExternalKvResponse, ReportExternalKvBlocksRequest, RevokeAllExternalKvBlocksAtTierRequest,
    RevokeExternalKvBlocksRequest,
};

pub trait KvIndexerBackend: Send + Sync + 'static {
    fn report_external_kv_blocks(
        &self,
        request: ReportExternalKvBlocksRequest,
    ) -> Result<(), Status>;

    fn revoke_external_kv_blocks(
        &self,
        request: RevokeExternalKvBlocksRequest,
    ) -> Result<(), Status>;

    fn revoke_all_external_kv_blocks_at_tier(
        &self,
        request: RevokeAllExternalKvBlocksAtTierRequest,
    ) -> Result<(), Status>;

    fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status>;

    fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status>;
}

#[derive(Debug, Default)]
pub struct NoopKvIndexerBackend;

impl KvIndexerBackend for NoopKvIndexerBackend {
    fn report_external_kv_blocks(
        &self,
        _request: ReportExternalKvBlocksRequest,
    ) -> Result<(), Status> {
        Ok(())
    }

    fn revoke_external_kv_blocks(
        &self,
        _request: RevokeExternalKvBlocksRequest,
    ) -> Result<(), Status> {
        Ok(())
    }

    fn revoke_all_external_kv_blocks_at_tier(
        &self,
        _request: RevokeAllExternalKvBlocksAtTierRequest,
    ) -> Result<(), Status> {
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

#[derive(Debug)]
pub struct KvIndexerService<B> {
    backend: B,
}

impl<B> KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    pub fn new(backend: B) -> Self {
        Self { backend }
    }
}

#[tonic::async_trait]
impl<B> KvIndexer for KvIndexerService<B>
where
    B: KvIndexerBackend,
{
    async fn report_external_kv_blocks(
        &self,
        request: Request<ReportExternalKvBlocksRequest>,
    ) -> Result<Response<Empty>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_hashes(&request.hashes)?;
        validate_tier(request.tier)?;
        self.backend.report_external_kv_blocks(request)?;
        Ok(Response::new(Empty {}))
    }

    async fn revoke_external_kv_blocks(
        &self,
        request: Request<RevokeExternalKvBlocksRequest>,
    ) -> Result<Response<Empty>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_hashes(&request.hashes)?;
        validate_tier(request.tier)?;
        self.backend.revoke_external_kv_blocks(request)?;
        Ok(Response::new(Empty {}))
    }

    async fn revoke_all_external_kv_blocks_at_tier(
        &self,
        request: Request<RevokeAllExternalKvBlocksAtTierRequest>,
    ) -> Result<Response<Empty>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_tier(request.tier)?;
        self.backend
            .revoke_all_external_kv_blocks_at_tier(request)?;
        Ok(Response::new(Empty {}))
    }

    async fn match_external_kv(
        &self,
        request: Request<MatchExternalKvRequest>,
    ) -> Result<Response<MatchExternalKvResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.match_external_kv(request)?;
        Ok(Response::new(response))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: Request<GetExternalKvHitCountsRequest>,
    ) -> Result<Response<GetExternalKvHitCountsResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.get_external_kv_hit_counts(request)?;
        Ok(Response::new(response))
    }
}

fn validate_worker_id(worker_id: &str) -> Result<(), Status> {
    if worker_id.is_empty() {
        return Err(Status::invalid_argument("worker_id must not be empty"));
    }
    Ok(())
}

fn validate_hashes(hashes: &[String]) -> Result<(), Status> {
    if hashes.is_empty() {
        return Err(Status::invalid_argument("hashes must not be empty"));
    }
    if hashes.iter().any(|hash| hash.is_empty()) {
        return Err(Status::invalid_argument("hashes must not contain empty values"));
    }
    Ok(())
}

fn validate_tier(tier: i32) -> Result<(), Status> {
    match tier {
        1..=3 => Ok(()),
        0 => Err(Status::invalid_argument("tier must not be TIER_UNKNOWN")),
        _ => Err(Status::invalid_argument("tier is not supported")),
    }
}

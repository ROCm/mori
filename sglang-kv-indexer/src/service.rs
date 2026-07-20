use tonic::{Request, Response, Status};

use crate::pb::kv_indexer_server::KvIndexer;
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, Empty, ExternalKvAction,
    ExternalKvActionType, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, ReportExternalKvBlocksRequest,
    RevokeAllExternalKvBlocksAtTierRequest, RevokeExternalKvBlocksRequest,
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

    /// Applies a whole SGLang KVEventBatch. The actions are pre-validated and
    /// must be applied in order. `seq` is metadata: implementations may use
    /// `(worker_id, seq)` for idempotency but must not gate ordering on it.
    fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<(), Status>;
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

    fn apply_external_kv_batch(
        &self,
        _request: ApplyExternalKvBatchRequest,
    ) -> Result<(), Status> {
        Ok(())
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

    async fn apply_external_kv_batch(
        &self,
        request: Request<ApplyExternalKvBatchRequest>,
    ) -> Result<Response<ApplyExternalKvBatchResponse>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_actions(&request.actions)?;
        self.backend.apply_external_kv_batch(request)?;
        Ok(Response::new(ApplyExternalKvBatchResponse {}))
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

fn validate_actions(actions: &[ExternalKvAction]) -> Result<(), Status> {
    if actions.is_empty() {
        return Err(Status::invalid_argument("actions must not be empty"));
    }
    for action in actions {
        validate_tier(action.tier)?;
        match ExternalKvActionType::try_from(action.r#type) {
            Ok(ExternalKvActionType::ActionReport)
            | Ok(ExternalKvActionType::ActionRevoke) => {
                validate_hashes(&action.hashes)?;
            }
            // CLEAR_ALL_AT_TIER carries only a tier; hashes are ignored.
            Ok(ExternalKvActionType::ActionClearAllAtTier) => {}
            Ok(ExternalKvActionType::ActionUnknown) | Err(_) => {
                return Err(Status::invalid_argument("action type is not supported"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hbm() -> i32 {
        crate::pb::TierType::TierHbm as i32
    }

    fn action(r#type: ExternalKvActionType, tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: r#type as i32,
            tier,
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
        }
    }

    fn service() -> KvIndexerService<NoopKvIndexerBackend> {
        KvIndexerService::new(NoopKvIndexerBackend)
    }

    #[test]
    fn validate_actions_rejects_empty() {
        assert!(validate_actions(&[]).is_err());
    }

    #[test]
    fn validate_actions_rejects_unknown_type() {
        let actions = [action(ExternalKvActionType::ActionUnknown, hbm(), &["1"])];
        assert!(validate_actions(&actions).is_err());
    }

    #[test]
    fn validate_actions_rejects_bad_tier() {
        let actions = [action(ExternalKvActionType::ActionReport, 0, &["1"])];
        assert!(validate_actions(&actions).is_err());
    }

    #[test]
    fn validate_actions_requires_hashes_for_report_and_revoke() {
        assert!(validate_actions(&[action(ExternalKvActionType::ActionReport, hbm(), &[])]).is_err());
        assert!(validate_actions(&[action(ExternalKvActionType::ActionRevoke, hbm(), &[])]).is_err());
    }

    #[test]
    fn validate_actions_allows_empty_hashes_for_clear_all_at_tier() {
        let actions = [action(ExternalKvActionType::ActionClearAllAtTier, hbm(), &[])];
        assert!(validate_actions(&actions).is_ok());
    }

    #[tokio::test]
    async fn apply_batch_accepts_valid_request() {
        let request = Request::new(ApplyExternalKvBatchRequest {
            worker_id: "worker-1".to_string(),
            seq: 3,
            actions: vec![
                action(ExternalKvActionType::ActionReport, hbm(), &["1", "2"]),
                action(ExternalKvActionType::ActionClearAllAtTier, hbm(), &[]),
            ],
        });
        assert!(service().apply_external_kv_batch(request).await.is_ok());
    }

    #[tokio::test]
    async fn apply_batch_rejects_empty_worker_id() {
        let request = Request::new(ApplyExternalKvBatchRequest {
            worker_id: String::new(),
            seq: 0,
            actions: vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
        });
        assert!(service().apply_external_kv_batch(request).await.is_err());
    }
}

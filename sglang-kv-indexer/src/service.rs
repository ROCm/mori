use tonic::{Request, Response, Status};

use crate::pb::kv_indexer_server::KvIndexer;
use crate::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, Empty, ExternalKvAction,
    ExternalKvActionType, GetExternalKvHitCountsRequest, GetExternalKvHitCountsResponse,
    MatchExternalKvRequest, MatchExternalKvResponse, ReportExternalKvBlocksRequest,
    RevokeAllExternalKvBlocksAtTierRequest, RevokeExternalKvBlocksRequest,
};

/// Storage backend for the indexer. Deliberately narrow: every write flows
/// through `apply_external_kv_batch` so there is a single write path. The three
/// legacy single-mutation RPCs are translated (in the service layer) into
/// single-action apply batches, so backends never implement them directly.
///
/// Async because real backends (e.g. Redis) do network IO; the trait is made
/// dyn-safe via `#[tonic::async_trait]` so the server can select a backend at
/// runtime and hold it as `Arc<dyn KvIndexerBackend>`.
#[tonic::async_trait]
pub trait KvIndexerBackend: Send + Sync + 'static {
    /// Applies a whole SGLang KVEventBatch. The actions are pre-validated and
    /// must be applied in order. `seq` is metadata: the batch is expected to be
    /// naturally idempotent so a verbatim replay (same `seq`) is a no-op; the
    /// backend must not gate ordering on `seq`.
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<(), Status>;

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status>;

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status>;
}

/// Blanket impl so the server can hold the selected backend as
/// `Arc<dyn KvIndexerBackend>` and still satisfy `KvIndexerService<B>`.
#[tonic::async_trait]
impl KvIndexerBackend for std::sync::Arc<dyn KvIndexerBackend> {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<(), Status> {
        (**self).apply_external_kv_batch(request).await
    }

    async fn match_external_kv(
        &self,
        request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        (**self).match_external_kv(request).await
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: GetExternalKvHitCountsRequest,
    ) -> Result<GetExternalKvHitCountsResponse, Status> {
        (**self).get_external_kv_hit_counts(request).await
    }
}

#[derive(Debug, Default)]
pub struct NoopKvIndexerBackend;

#[tonic::async_trait]
impl KvIndexerBackend for NoopKvIndexerBackend {
    async fn apply_external_kv_batch(
        &self,
        _request: ApplyExternalKvBatchRequest,
    ) -> Result<(), Status> {
        Ok(())
    }

    async fn match_external_kv(
        &self,
        _request: MatchExternalKvRequest,
    ) -> Result<MatchExternalKvResponse, Status> {
        Ok(MatchExternalKvResponse { matches: vec![] })
    }

    async fn get_external_kv_hit_counts(
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
        let batch = single_action_batch(
            request.worker_id,
            ExternalKvActionType::ActionReport,
            request.tier,
            request.hashes,
        );
        self.backend.apply_external_kv_batch(batch).await?;
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
        let batch = single_action_batch(
            request.worker_id,
            ExternalKvActionType::ActionRevoke,
            request.tier,
            request.hashes,
        );
        self.backend.apply_external_kv_batch(batch).await?;
        Ok(Response::new(Empty {}))
    }

    async fn revoke_all_external_kv_blocks_at_tier(
        &self,
        request: Request<RevokeAllExternalKvBlocksAtTierRequest>,
    ) -> Result<Response<Empty>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_tier(request.tier)?;
        let batch = single_action_batch(
            request.worker_id,
            ExternalKvActionType::ActionClearAllAtTier,
            request.tier,
            Vec::new(),
        );
        self.backend.apply_external_kv_batch(batch).await?;
        Ok(Response::new(Empty {}))
    }

    async fn match_external_kv(
        &self,
        request: Request<MatchExternalKvRequest>,
    ) -> Result<Response<MatchExternalKvResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.match_external_kv(request).await?;
        Ok(Response::new(response))
    }

    async fn get_external_kv_hit_counts(
        &self,
        request: Request<GetExternalKvHitCountsRequest>,
    ) -> Result<Response<GetExternalKvHitCountsResponse>, Status> {
        let request = request.into_inner();
        validate_hashes(&request.hashes)?;
        let response = self.backend.get_external_kv_hit_counts(request).await?;
        Ok(Response::new(response))
    }

    async fn apply_external_kv_batch(
        &self,
        request: Request<ApplyExternalKvBatchRequest>,
    ) -> Result<Response<ApplyExternalKvBatchResponse>, Status> {
        let request = request.into_inner();
        validate_worker_id(&request.worker_id)?;
        validate_actions(&request.actions)?;
        self.backend.apply_external_kv_batch(request).await?;
        Ok(Response::new(ApplyExternalKvBatchResponse {}))
    }
}

/// Wraps a single legacy mutation as an `ApplyExternalKvBatchRequest` so every
/// write goes through the one backend entry point. Legacy callers carry no
/// address (`worker_address` is empty) and no batch sequence (`seq` is 0); the
/// apply path is idempotent so a synthetic seq is harmless.
fn single_action_batch(
    worker_id: String,
    action_type: ExternalKvActionType,
    tier: i32,
    hashes: Vec<String>,
) -> ApplyExternalKvBatchRequest {
    ApplyExternalKvBatchRequest {
        worker_id,
        seq: 0,
        actions: vec![ExternalKvAction {
            r#type: action_type as i32,
            tier,
            hashes,
        }],
        worker_address: String::new(),
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
        return Err(Status::invalid_argument(
            "hashes must not contain empty values",
        ));
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
            Ok(ExternalKvActionType::ActionReport) | Ok(ExternalKvActionType::ActionRevoke) => {
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

    /// Backend that records every apply batch it receives, so tests can assert
    /// how the legacy RPCs are translated into apply batches.
    #[derive(Clone, Default)]
    struct RecordingBackend {
        applied: std::sync::Arc<std::sync::Mutex<Vec<ApplyExternalKvBatchRequest>>>,
    }

    #[tonic::async_trait]
    impl KvIndexerBackend for RecordingBackend {
        async fn apply_external_kv_batch(
            &self,
            request: ApplyExternalKvBatchRequest,
        ) -> Result<(), Status> {
            self.applied.lock().unwrap().push(request);
            Ok(())
        }

        async fn match_external_kv(
            &self,
            _request: MatchExternalKvRequest,
        ) -> Result<MatchExternalKvResponse, Status> {
            Ok(MatchExternalKvResponse { matches: vec![] })
        }

        async fn get_external_kv_hit_counts(
            &self,
            _request: GetExternalKvHitCountsRequest,
        ) -> Result<GetExternalKvHitCountsResponse, Status> {
            Ok(GetExternalKvHitCountsResponse { entries: vec![] })
        }
    }

    #[tokio::test]
    async fn legacy_report_translates_to_single_report_action() {
        let backend = RecordingBackend::default();
        let applied = backend.applied.clone();
        let svc = KvIndexerService::new(backend);
        svc.report_external_kv_blocks(Request::new(ReportExternalKvBlocksRequest {
            worker_id: "w1".to_string(),
            hashes: vec!["1".to_string(), "2".to_string()],
            tier: hbm(),
        }))
        .await
        .unwrap();
        let applied = applied.lock().unwrap();
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0].worker_id, "w1");
        assert!(applied[0].worker_address.is_empty());
        assert_eq!(
            applied[0].actions,
            vec![action(
                ExternalKvActionType::ActionReport,
                hbm(),
                &["1", "2"]
            )]
        );
    }

    #[tokio::test]
    async fn legacy_revoke_translates_to_single_revoke_action() {
        let backend = RecordingBackend::default();
        let applied = backend.applied.clone();
        let svc = KvIndexerService::new(backend);
        svc.revoke_external_kv_blocks(Request::new(RevokeExternalKvBlocksRequest {
            worker_id: "w1".to_string(),
            hashes: vec!["7".to_string()],
            tier: hbm(),
        }))
        .await
        .unwrap();
        let applied = applied.lock().unwrap();
        assert_eq!(
            applied[0].actions,
            vec![action(ExternalKvActionType::ActionRevoke, hbm(), &["7"])]
        );
    }

    #[tokio::test]
    async fn legacy_clear_all_translates_to_clear_action_with_no_hashes() {
        let backend = RecordingBackend::default();
        let applied = backend.applied.clone();
        let svc = KvIndexerService::new(backend);
        svc.revoke_all_external_kv_blocks_at_tier(Request::new(
            RevokeAllExternalKvBlocksAtTierRequest {
                worker_id: "w1".to_string(),
                tier: hbm(),
            },
        ))
        .await
        .unwrap();
        let applied = applied.lock().unwrap();
        assert_eq!(
            applied[0].actions,
            vec![action(
                ExternalKvActionType::ActionClearAllAtTier,
                hbm(),
                &[]
            )]
        );
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
        assert!(
            validate_actions(&[action(ExternalKvActionType::ActionReport, hbm(), &[])]).is_err()
        );
        assert!(
            validate_actions(&[action(ExternalKvActionType::ActionRevoke, hbm(), &[])]).is_err()
        );
    }

    #[test]
    fn validate_actions_allows_empty_hashes_for_clear_all_at_tier() {
        let actions = [action(
            ExternalKvActionType::ActionClearAllAtTier,
            hbm(),
            &[],
        )];
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
            worker_address: "127.0.0.1:9000".to_string(),
        });
        assert!(service().apply_external_kv_batch(request).await.is_ok());
    }

    #[tokio::test]
    async fn apply_batch_rejects_empty_worker_id() {
        let request = Request::new(ApplyExternalKvBatchRequest {
            worker_id: String::new(),
            seq: 0,
            actions: vec![action(ExternalKvActionType::ActionReport, hbm(), &["1"])],
            worker_address: String::new(),
        });
        assert!(service().apply_external_kv_batch(request).await.is_err());
    }
}

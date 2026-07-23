//! End-to-end reliability test for the bridge's in-session sequence-gap
//! recovery. A fake SGLang publisher (ZMQ PUB for live events + ROUTER for the
//! replay buffer) emits batches with a deliberate hole in the `seq` stream; a
//! capturing in-memory gRPC indexer records the `seq` of every applied batch.
//! We assert the bridge detects the gap, pulls the missing batches from the
//! replay endpoint (DEALER -> ROUTER), and applies everything exactly once in
//! monotonic order.
//!
//! No Redis required: the capturing backend implements `KvIndexerBackend`
//! directly, so this runs in the default `cargo test`.

#[path = "common/net.rs"]
mod test_net;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use tonic::transport::Server;
use tonic::Status;
use zeromq::{PubSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

use sglang_kv_indexer::bridge::{run_bridge, BridgeConfig};
use sglang_kv_indexer::pb::kv_indexer_server::KvIndexerServer;
use sglang_kv_indexer::pb::{
    ApplyExternalKvBatchRequest, ApplyExternalKvBatchResponse, GetExternalKvHitCountsRequest,
    GetExternalKvHitCountsResponse, MatchExternalKvRequest, MatchExternalKvResponse,
};
use sglang_kv_indexer::{KvIndexerBackend, KvIndexerService};
use test_net::free_addr;

/// gRPC backend that just records the seq of every applied batch, in order.
#[derive(Clone, Default)]
struct CapturingBackend {
    seqs: Arc<Mutex<Vec<u64>>>,
}

#[tonic::async_trait]
impl KvIndexerBackend for CapturingBackend {
    async fn apply_external_kv_batch(
        &self,
        request: ApplyExternalKvBatchRequest,
    ) -> Result<ApplyExternalKvBatchResponse, Status> {
        self.seqs.lock().unwrap().push(request.seq);
        Ok(ApplyExternalKvBatchResponse {
            last_applied_seq: request.seq,
            duplicate: false,
        })
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

/// Encodes a minimal SGLang `KVEventBatch` = [ts, [events]] carrying one
/// `BlockStored` (7 fields; [1]=block_hashes, [6]=medium) so the bridge decodes
/// it into a single REPORT action.
fn stored_payload(hash: i64, medium: &str) -> Vec<u8> {
    use rmpv::Value;
    let event = Value::Array(vec![
        Value::from("BlockStored"),
        Value::Array(vec![Value::from(hash)]),
        Value::Nil,
        Value::Nil,
        Value::Nil,
        Value::Nil,
        Value::from(medium),
    ]);
    let batch = Value::Array(vec![Value::from(0u64), Value::Array(vec![event])]);
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &batch).expect("encode msgpack");
    buf
}

/// One live PUB frame: [seq(8B BE u64), payload]. Empty topic => the bridge's
/// empty subscription receives it.
fn pub_frame(seq: u64, payload: Vec<u8>) -> ZmqMessage {
    let mut m = ZmqMessage::from(Bytes::copy_from_slice(&seq.to_be_bytes()));
    m.push_back(Bytes::from(payload));
    m
}

/// One ROUTER reply routed to `peer`: [peer, b"", seq(8B BE i64), payload].
/// After ROUTER pops `peer`, the bridge's DEALER sees [b"", seq, payload].
fn reply_frame(peer: Bytes, seq: i64, payload: Vec<u8>) -> ZmqMessage {
    let mut m = ZmqMessage::from(peer);
    m.push_back(Bytes::new());
    m.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    m.push_back(Bytes::from(payload));
    m
}

#[tokio::test]
async fn bridge_recovers_seq_gap_via_replay() {
    // --- capturing gRPC indexer ---
    let backend = CapturingBackend::default();
    let seqs = backend.seqs.clone();
    let grpc_addr = free_addr();
    let svc = KvIndexerServer::new(KvIndexerService::new(backend));
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve(grpc_addr)
            .await
            .expect("grpc serve");
    });

    // --- fake SGLang: PUB (live) + ROUTER (replay buffer) ---
    let mut publisher = PubSocket::new();
    let pub_ep = publisher.bind("tcp://127.0.0.1:0").await.expect("bind pub");
    let mut router = RouterSocket::new();
    let router_ep = router.bind("tcp://127.0.0.1:0").await.expect("bind router");

    // ROUTER responder: on a replay request [peer, b"", start_seq], stream the
    // buffered missing batches (2, 3) then a negative-seq terminator.
    tokio::spawn(async move {
        let req = router.recv().await.expect("router recv").into_vec();
        let peer = req[0].clone();
        for seq in [2i64, 3i64] {
            router
                .send(reply_frame(
                    peer.clone(),
                    seq,
                    stored_payload(1000 + seq, "GPU"),
                ))
                .await
                .expect("router send batch");
        }
        // terminator: seq < 0 tells the bridge the replay stream is done
        router
            .send(reply_frame(peer, -1, Vec::new()))
            .await
            .expect("router send terminator");
    });

    // --- bridge under test ---
    let config = BridgeConfig {
        worker_id: "worker-test".to_string(),
        worker_address: String::new(),
        event_endpoint: pub_ep.to_string(),
        event_replay_endpoint: Some(router_ep.to_string()),
        event_topic: String::new(),
        indexer_endpoint: format!("http://{grpc_addr}"),
        clear_tiers: vec![],
        heartbeat_interval: None,
        incarnation: "replay-test".to_string(),
    };
    tokio::spawn(async move {
        let _ = run_bridge(config).await;
    });

    // Let the bridge's SUB connect/subscribe and gRPC client connect before we
    // publish (PUB/SUB has no handshake; early sends would be dropped).
    tokio::time::sleep(Duration::from_millis(1200)).await;

    // Live stream with a hole: 0, 1, then jump to 4 (2 and 3 are "missed").
    publisher
        .send(pub_frame(0, stored_payload(1000, "GPU")))
        .await
        .expect("pub 0");
    tokio::time::sleep(Duration::from_millis(100)).await;
    publisher
        .send(pub_frame(1, stored_payload(1001, "GPU")))
        .await
        .expect("pub 1");
    tokio::time::sleep(Duration::from_millis(100)).await;
    publisher
        .send(pub_frame(4, stored_payload(1004, "GPU")))
        .await
        .expect("pub 4");

    // Wait until all five seqs are applied (or time out).
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        {
            let got = seqs.lock().unwrap();
            if got.len() >= 5 {
                break;
            }
        }
        if std::time::Instant::now() > deadline {
            panic!(
                "timed out; applied seqs so far: {:?}",
                *seqs.lock().unwrap()
            );
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    let got = seqs.lock().unwrap().clone();
    assert_eq!(
        got,
        vec![0, 1, 2, 3, 4],
        "bridge must apply every batch exactly once in monotonic order (gap 2,3 recovered via replay)"
    );
}

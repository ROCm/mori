use std::io::Cursor;
use std::time::Duration;

use rmpv::decode::value::read_value;
use rmpv::Value;
use tonic::transport::Channel;
use tracing::{debug, info, warn};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{
    ApplyExternalKvBatchRequest, ExternalKvAction, ExternalKvActionType, MatchExternalKvRequest,
    TierType,
};

/// Backoff bounds for the reconnect supervisor loop.
const RECONNECT_MIN_DELAY: Duration = Duration::from_millis(500);
const RECONNECT_MAX_DELAY: Duration = Duration::from_secs(10);
const REPLAY_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub worker_id: String,
    /// The worker's KV-transfer address, forwarded on every apply batch so the
    /// indexer can answer MatchExternalKv with an address. Empty if unset.
    pub worker_address: String,
    pub event_endpoint: String,
    pub event_replay_endpoint: Option<String>,
    pub event_topic: String,
    pub indexer_endpoint: String,
    pub clear_tiers: Vec<i32>,
}

impl BridgeConfig {
    pub fn from_env() -> Result<Self, BridgeError> {
        let worker_id = std::env::var("KV_INDEXER_WORKER_ID")
            .map_err(|_| BridgeError::Config("KV_INDEXER_WORKER_ID is required".to_string()))?;
        let worker_address = std::env::var("KV_INDEXER_WORKER_ADDRESS").unwrap_or_default();
        let event_endpoint = std::env::var("SGLANG_KV_EVENT_ENDPOINT")
            .unwrap_or_else(|_| "tcp://127.0.0.1:5557".to_string());
        let event_replay_endpoint = std::env::var("SGLANG_KV_EVENT_REPLAY_ENDPOINT").ok();
        let event_topic =
            std::env::var("SGLANG_KV_EVENT_TOPIC").unwrap_or_else(|_| "kv-events".to_string());
        let indexer_endpoint = std::env::var("KV_INDEXER_ENDPOINT")
            .unwrap_or_else(|_| "http://[::1]:50051".to_string());
        let clear_tiers = parse_clear_tiers(
            &std::env::var("KV_INDEXER_CLEAR_TIERS").unwrap_or_else(|_| "HBM,DRAM,SSD".to_string()),
        )?;

        Ok(Self {
            worker_id,
            worker_address,
            event_endpoint,
            event_replay_endpoint,
            event_topic,
            indexer_endpoint,
            clear_tiers,
        })
    }
}

#[derive(Debug)]
pub enum BridgeError {
    Config(String),
    Decode(String),
    Rpc(tonic::Status),
    Timeout(String),
    Transport(tonic::transport::Error),
    Zmq(zeromq::ZmqError),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Config(message) => write!(f, "bridge config error: {message}"),
            BridgeError::Decode(message) => write!(f, "bridge decode error: {message}"),
            BridgeError::Rpc(status) => write!(f, "indexer rpc error: {status}"),
            BridgeError::Timeout(message) => write!(f, "bridge timeout: {message}"),
            BridgeError::Transport(error) => write!(f, "indexer transport error: {error}"),
            BridgeError::Zmq(error) => write!(f, "zmq error: {error}"),
        }
    }
}

impl std::error::Error for BridgeError {}

impl From<zeromq::ZmqError> for BridgeError {
    fn from(error: zeromq::ZmqError) -> Self {
        BridgeError::Zmq(error)
    }
}

impl From<tonic::transport::Error> for BridgeError {
    fn from(error: tonic::transport::Error) -> Self {
        BridgeError::Transport(error)
    }
}

impl From<tonic::Status> for BridgeError {
    fn from(status: tonic::Status) -> Self {
        BridgeError::Rpc(status)
    }
}

/// A single indexer mutation, kept in the exact order it appeared in the event
/// batch so that store/remove/clear operations on the same hash are never
/// reordered relative to each other.
#[derive(Debug, PartialEq, Eq)]
enum Action {
    Report { tier: i32, hashes: Vec<String> },
    Revoke { tier: i32, hashes: Vec<String> },
    ClearAll,
}

#[derive(Debug, Default)]
struct EventActions {
    actions: Vec<Action>,
}

#[derive(Debug, Clone)]
struct RawBatch {
    seq: u64,
    payload: Vec<u8>,
}

impl EventActions {
    /// Append a store, coalescing only with an immediately-preceding store to
    /// the same tier. Coalescing never crosses a revoke/clear, so ordering (and
    /// therefore the final per-hash state) is preserved.
    fn report(&mut self, tier: i32, hashes: Vec<String>) {
        if hashes.is_empty() {
            return;
        }
        if let Some(Action::Report {
            tier: last_tier,
            hashes: last,
        }) = self.actions.last_mut()
        {
            if *last_tier == tier {
                last.extend(hashes);
                return;
            }
        }
        self.actions.push(Action::Report { tier, hashes });
    }

    fn revoke(&mut self, tier: i32, hashes: Vec<String>) {
        if hashes.is_empty() {
            return;
        }
        if let Some(Action::Revoke {
            tier: last_tier,
            hashes: last,
        }) = self.actions.last_mut()
        {
            if *last_tier == tier {
                last.extend(hashes);
                return;
            }
        }
        self.actions.push(Action::Revoke { tier, hashes });
    }

    fn clear_all(&mut self) {
        self.actions.push(Action::ClearAll);
    }
}

pub async fn run_bridge(config: BridgeConfig) -> Result<(), BridgeError> {
    info!(
        worker_id = %config.worker_id,
        event_endpoint = %config.event_endpoint,
        event_replay_endpoint = ?config.event_replay_endpoint,
        event_topic = %config.event_topic,
        indexer_endpoint = %config.indexer_endpoint,
        "starting SGLang KV event bridge"
    );

    // Supervisor loop: (re)connect to both the indexer and the ZMQ publisher,
    // run until a connection-level error, then back off and retry. Decode-level
    // problems are handled inside the session and never tear down the bridge.
    let mut delay = RECONNECT_MIN_DELAY;
    let mut next_seq: Option<u64> = None;
    let mut pending_batch: Option<RawBatch> = None;
    loop {
        match connect(&config).await {
            Ok((client, subscriber)) => {
                delay = RECONNECT_MIN_DELAY;
                match run_session(
                    &config,
                    client,
                    subscriber,
                    &mut next_seq,
                    &mut pending_batch,
                )
                .await
                {
                    Ok(()) => {
                        info!("bridge shut down cleanly");
                        return Ok(());
                    }
                    Err(error) => {
                        warn!(%error, retry_in = ?delay, "bridge session lost; reconnecting");
                    }
                }
            }
            Err(error) => {
                warn!(%error, retry_in = ?delay, "bridge connect failed; retrying");
            }
        }

        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(RECONNECT_MAX_DELAY);
    }
}

async fn connect(
    config: &BridgeConfig,
) -> Result<(KvIndexerClient<Channel>, SubSocket), BridgeError> {
    let client = KvIndexerClient::connect(config.indexer_endpoint.clone()).await?;
    let mut subscriber = SubSocket::new();
    subscriber.subscribe(&config.event_topic).await?;
    subscriber.connect(&config.event_endpoint).await?;
    info!("bridge session established");
    Ok((client, subscriber))
}

/// Forwards one raw batch, marking it pending for the duration so an
/// interrupted forward is re-driven verbatim after a reconnect.
async fn commit_batch(
    config: &BridgeConfig,
    client: &mut KvIndexerClient<Channel>,
    batch: &RawBatch,
    pending_batch: &mut Option<RawBatch>,
) -> Result<(), BridgeError> {
    *pending_batch = Some(batch.clone());
    forward_raw_batch(config, client, batch).await?;
    *pending_batch = None;
    Ok(())
}

/// Runs a single connected session. Returns `Ok(())` only on a clean shutdown
/// (ctrl-c); any connection-level error is propagated so the supervisor can
/// reconnect. Malformed frames / undecodable batches are logged and skipped.
async fn run_session(
    config: &BridgeConfig,
    mut client: KvIndexerClient<Channel>,
    mut subscriber: SubSocket,
    next_seq: &mut Option<u64>,
    pending_batch: &mut Option<RawBatch>,
) -> Result<(), BridgeError> {
    // A batch left pending by the previous disconnect is re-driven once before
    // consuming new events. It can only be set at session entry: every in-loop
    // commit either clears it or returns an error that ends the session.
    if let Some(batch) = pending_batch.clone() {
        commit_batch(config, &mut client, &batch, pending_batch).await?;
        *next_seq = batch.seq.checked_add(1);
    }

    loop {
        let message = tokio::select! {
            result = subscriber.recv() => result?,
            _ = tokio::signal::ctrl_c() => {
                info!("received ctrl-c; shutting down bridge");
                return Ok(());
            }
        };

        let (seq, payload) = match parse_zmq_frames(&message.into_vec()) {
            Ok((seq, payload)) => (seq, payload.to_vec()),
            Err(error) => {
                warn!(%error, "skipping malformed ZMQ message");
                continue;
            }
        };

        if let Some(expected) = *next_seq {
            if seq < expected {
                warn!(
                    expected,
                    actual = seq,
                    "skipping stale SGLang KV event batch"
                );
                continue;
            }
            if seq > expected {
                warn!(expected, actual = seq, "SGLang KV event sequence gap");
                replay_missing_batches(config, &mut client, expected, seq, pending_batch).await?;
            }
        }

        commit_batch(
            config,
            &mut client,
            &RawBatch { seq, payload },
            pending_batch,
        )
        .await?;
        *next_seq = seq.checked_add(1);
    }
}

async fn replay_missing_batches(
    config: &BridgeConfig,
    client: &mut KvIndexerClient<Channel>,
    start_seq: u64,
    stop_before_seq: u64,
    pending_batch: &mut Option<RawBatch>,
) -> Result<(), BridgeError> {
    let Some(endpoint) = &config.event_replay_endpoint else {
        warn!(
            start_seq,
            stop_before_seq, "cannot replay missing SGLang KV events without replay endpoint"
        );
        return Ok(());
    };

    let mut socket = DealerSocket::new();
    socket.connect(endpoint).await?;
    // DEALER -> ROUTER: prepend an empty delimiter frame so the publisher's
    // ROUTER sees [identity, b"", start_seq]. Unlike REQ, DEALER neither adds
    // this delimiter nor enforces strict send/recv alternation, so we can read
    // the many per-batch replies the ROUTER streams back for one request.
    let mut request = ZmqMessage::from(bytes::Bytes::copy_from_slice(&start_seq.to_be_bytes()));
    request.push_front(bytes::Bytes::new());
    tokio::time::timeout(REPLAY_REQUEST_TIMEOUT, socket.send(request))
        .await
        .map_err(|_| BridgeError::Timeout("SGLang replay request send timed out".to_string()))??;

    let mut replay_expected = start_seq;
    let mut recovered = 0_u64;
    loop {
        let frames = tokio::time::timeout(REPLAY_REQUEST_TIMEOUT, socket.recv())
            .await
            .map_err(|_| BridgeError::Timeout("SGLang replay response timed out".to_string()))??
            .into_vec();
        let (seq, payload) = parse_replay_frames(&frames)?;
        if seq < 0 {
            break;
        }

        let seq = seq as u64;
        if seq < start_seq {
            continue;
        }
        if seq != replay_expected {
            warn!(
                expected = replay_expected,
                actual = seq,
                "SGLang replay buffer did not return a contiguous batch"
            );
            break;
        }
        if seq >= stop_before_seq {
            break;
        }

        let batch = RawBatch {
            seq,
            payload: payload.to_vec(),
        };
        commit_batch(config, client, &batch, pending_batch).await?;
        replay_expected = seq.checked_add(1).unwrap_or(seq);
        recovered += 1;
    }

    if replay_expected < stop_before_seq {
        warn!(
            from_seq = replay_expected,
            stop_before_seq, "SGLang KV event gap remains after replay attempt"
        );
    } else {
        info!(
            start_seq,
            stop_before_seq, recovered, "replayed missing SGLang KV event batches"
        );
    }

    Ok(())
}

fn parse_replay_frames(frames: &[bytes::Bytes]) -> Result<(i64, &[u8]), BridgeError> {
    // A DEALER connected to the publisher's ROUTER receives replies with the
    // leading empty delimiter intact: [b"", seq, payload]. Tolerate a bare
    // [seq, payload] variant too.
    match frames.len() {
        3 => Ok((decode_signed_seq(&frames[1])?, frames[2].as_ref())),
        2 => Ok((decode_signed_seq(&frames[0])?, frames[1].as_ref())),
        n => Err(BridgeError::Decode(format!(
            "expected 2 or 3 replay frames, got {n}"
        ))),
    }
}

fn decode_signed_seq(bytes: &[u8]) -> Result<i64, BridgeError> {
    let seq_bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| BridgeError::Decode("sequence frame must be 8 bytes".to_string()))?;
    Ok(i64::from_be_bytes(seq_bytes))
}

fn parse_zmq_frames(frames: &[bytes::Bytes]) -> Result<(u64, &[u8]), BridgeError> {
    match frames.len() {
        2 => Ok((decode_seq(&frames[0])?, frames[1].as_ref())),
        3 => Ok((decode_seq(&frames[1])?, frames[2].as_ref())),
        n => Err(BridgeError::Decode(format!(
            "expected 2 or 3 ZMQ frames, got {n}"
        ))),
    }
}

fn decode_seq(bytes: &[u8]) -> Result<u64, BridgeError> {
    let seq_bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| BridgeError::Decode("sequence frame must be 8 bytes".to_string()))?;
    Ok(u64::from_be_bytes(seq_bytes))
}

async fn forward_raw_batch(
    config: &BridgeConfig,
    client: &mut KvIndexerClient<Channel>,
    batch: &RawBatch,
) -> Result<(), BridgeError> {
    let actions = match decode_event_batch(&batch.payload) {
        Ok(actions) => actions,
        Err(error) => {
            warn!(seq = batch.seq, %error, "skipping undecodable event batch");
            return Ok(());
        }
    };

    let request = build_apply_request(config, batch.seq, actions);
    // Nothing decodable to a supported mutation (e.g. only ignored event tags):
    // preserve the previous no-op behaviour and skip the RPC entirely.
    if request.actions.is_empty() {
        return Ok(());
    }
    client.apply_external_kv_batch(request).await?;
    Ok(())
}

/// Maps a decoded `EventActions` into a single `ApplyExternalKvBatchRequest`,
/// preserving the exact per-action order. A `ClearAll` is expanded, in place,
/// into one `CLEAR_ALL_AT_TIER` action per configured clear tier so the batch
/// carries the same semantics as the legacy per-tier revoke-all RPCs.
fn build_apply_request(
    config: &BridgeConfig,
    seq: u64,
    events: EventActions,
) -> ApplyExternalKvBatchRequest {
    let mut actions = Vec::with_capacity(events.actions.len());
    for action in events.actions {
        match action {
            Action::Report { tier, hashes } => actions.push(ExternalKvAction {
                r#type: ExternalKvActionType::ActionReport as i32,
                tier,
                hashes,
            }),
            Action::Revoke { tier, hashes } => actions.push(ExternalKvAction {
                r#type: ExternalKvActionType::ActionRevoke as i32,
                tier,
                hashes,
            }),
            Action::ClearAll => {
                for tier in &config.clear_tiers {
                    actions.push(ExternalKvAction {
                        r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
                        tier: *tier,
                        hashes: Vec::new(),
                    });
                }
            }
        }
    }

    ApplyExternalKvBatchRequest {
        worker_id: config.worker_id.clone(),
        seq,
        actions,
        worker_address: config.worker_address.clone(),
    }
}

fn decode_event_batch(payload: &[u8]) -> Result<EventActions, BridgeError> {
    let mut cursor = Cursor::new(payload);
    let value = read_value(&mut cursor).map_err(|error| BridgeError::Decode(error.to_string()))?;
    let batch = expect_array(&value, "KVEventBatch")?;
    if batch.len() < 2 {
        return Err(BridgeError::Decode(
            "KVEventBatch must contain timestamp and events".to_string(),
        ));
    }

    let events = expect_array(&batch[1], "KVEventBatch.events")?;
    let mut actions = EventActions::default();
    for event in events {
        decode_event(event, &mut actions)?;
    }
    Ok(actions)
}

fn decode_event(event: &Value, actions: &mut EventActions) -> Result<(), BridgeError> {
    let event = expect_array(event, "KV event")?;
    let event_type = expect_str(
        event
            .first()
            .ok_or_else(|| BridgeError::Decode("KV event is empty".to_string()))?,
        "KV event tag",
    )?;

    match event_type {
        "BlockStored" => {
            if event.len() < 7 {
                return Err(BridgeError::Decode(
                    "BlockStored must have 7 array fields".to_string(),
                ));
            }
            let tier = medium_to_tier(expect_optional_str(&event[6], "BlockStored.medium")?)?;
            actions.report(tier, decode_hashes(&event[1])?);
        }
        "BlockRemoved" => {
            if event.len() < 3 {
                return Err(BridgeError::Decode(
                    "BlockRemoved must have 3 array fields".to_string(),
                ));
            }
            let tier = medium_to_tier(expect_optional_str(&event[2], "BlockRemoved.medium")?)?;
            actions.revoke(tier, decode_hashes(&event[1])?);
        }
        "AllBlocksCleared" => {
            actions.clear_all();
        }
        other => {
            debug!(event_type = other, "ignoring unsupported SGLang KV event");
        }
    }
    Ok(())
}

fn decode_hashes(value: &Value) -> Result<Vec<String>, BridgeError> {
    expect_array(value, "block_hashes")?
        .iter()
        .map(|value| {
            if let Some(value) = value.as_i64() {
                return Ok(value.to_string());
            }
            if let Some(value) = value.as_u64() {
                return Ok(value.to_string());
            }
            Err(BridgeError::Decode(
                "block hash must be an integer".to_string(),
            ))
        })
        .collect()
}

fn medium_to_tier(medium: Option<&str>) -> Result<i32, BridgeError> {
    match medium {
        Some("GPU") => Ok(TierType::TierHbm as i32),
        Some("CPU_PINNED") => Ok(TierType::TierDram as i32),
        Some("DISK") => Ok(TierType::TierSsd as i32),
        Some("EXTERNAL") => Err(BridgeError::Decode(
            "EXTERNAL medium does not map to a local indexer tier".to_string(),
        )),
        Some(other) => Err(BridgeError::Decode(format!(
            "unsupported SGLang storage medium: {other}"
        ))),
        None => Err(BridgeError::Decode(
            "SGLang storage medium is missing".to_string(),
        )),
    }
}

fn parse_clear_tiers(value: &str) -> Result<Vec<i32>, BridgeError> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| match part {
            "HBM" | "GPU" => Ok(TierType::TierHbm as i32),
            "DRAM" | "CPU" | "CPU_PINNED" => Ok(TierType::TierDram as i32),
            "SSD" | "DISK" => Ok(TierType::TierSsd as i32),
            other => Err(BridgeError::Config(format!(
                "unsupported clear tier: {other}"
            ))),
        })
        .collect()
}

fn expect_array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value], BridgeError> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| BridgeError::Decode(format!("{field} must be an array")))
}

fn expect_str<'a>(value: &'a Value, field: &str) -> Result<&'a str, BridgeError> {
    value
        .as_str()
        .ok_or_else(|| BridgeError::Decode(format!("{field} must be a string")))
}

fn expect_optional_str<'a>(value: &'a Value, field: &str) -> Result<Option<&'a str>, BridgeError> {
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    expect_str(value, field).map(Some)
}

#[allow(dead_code)]
async fn _probe_indexer(client: &mut KvIndexerClient<Channel>) -> Result<(), BridgeError> {
    client
        .match_external_kv(MatchExternalKvRequest {
            hashes: vec!["__bridge_probe__".to_string()],
            count_as_hit: false,
        })
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use rmpv::Value;

    fn hbm() -> i32 {
        TierType::TierHbm as i32
    }
    fn dram() -> i32 {
        TierType::TierDram as i32
    }
    fn ssd() -> i32 {
        TierType::TierSsd as i32
    }

    fn encode(value: &Value) -> Vec<u8> {
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, value).unwrap();
        buf
    }

    fn ints(values: &[i64]) -> Value {
        Value::Array(values.iter().map(|v| Value::from(*v)).collect())
    }

    fn stored(hashes: &[i64], medium: &str) -> Value {
        Value::Array(vec![
            Value::String("BlockStored".into()),
            ints(hashes),
            Value::Nil,         // parent_block_hash
            ints(&[1]),         // token_ids
            Value::from(1_i64), // block_size
            Value::Nil,         // lora_id
            Value::String(medium.into()),
        ])
    }

    fn removed(hashes: &[i64], medium: &str) -> Value {
        Value::Array(vec![
            Value::String("BlockRemoved".into()),
            ints(hashes),
            Value::String(medium.into()),
        ])
    }

    fn cleared() -> Value {
        Value::Array(vec![Value::String("AllBlocksCleared".into())])
    }

    /// Wrap events in a 3-element batch [ts, events, attn_dp_rank].
    fn batch(events: Vec<Value>) -> Vec<u8> {
        encode(&Value::Array(vec![
            Value::from(1.0_f64),
            Value::Array(events),
            Value::from(0_i64),
        ]))
    }

    fn actions_of(events: Vec<Value>) -> Vec<Action> {
        decode_event_batch(&batch(events)).unwrap().actions
    }

    fn test_config(clear_tiers: Vec<i32>) -> BridgeConfig {
        BridgeConfig {
            worker_id: "worker-1".to_string(),
            worker_address: "127.0.0.1:9000".to_string(),
            event_endpoint: "tcp://127.0.0.1:5557".to_string(),
            event_replay_endpoint: None,
            event_topic: "kv-events".to_string(),
            indexer_endpoint: "http://[::1]:50051".to_string(),
            clear_tiers,
        }
    }

    /// Build the apply-batch request the bridge would send for a set of events.
    fn request_of(
        config: &BridgeConfig,
        seq: u64,
        events: Vec<Value>,
    ) -> ApplyExternalKvBatchRequest {
        build_apply_request(config, seq, decode_event_batch(&batch(events)).unwrap())
    }

    fn report(tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionReport as i32,
            tier,
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
        }
    }

    fn revoke(tier: i32, hashes: &[&str]) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionRevoke as i32,
            tier,
            hashes: hashes.iter().map(|h| h.to_string()).collect(),
        }
    }

    fn clear_at(tier: i32) -> ExternalKvAction {
        ExternalKvAction {
            r#type: ExternalKvActionType::ActionClearAllAtTier as i32,
            tier,
            hashes: Vec::new(),
        }
    }

    #[test]
    fn request_carries_worker_id_and_seq() {
        let config = test_config(vec![hbm()]);
        let request = request_of(&config, 42, vec![stored(&[1], "GPU")]);
        assert_eq!(request.worker_id, "worker-1");
        assert_eq!(request.seq, 42);
    }

    #[test]
    fn request_carries_worker_address() {
        let config = test_config(vec![hbm()]);
        let request = request_of(&config, 0, vec![stored(&[1], "GPU")]);
        assert_eq!(request.worker_address, "127.0.0.1:9000");
    }

    #[test]
    fn report_and_revoke_map_to_actions_in_order() {
        let config = test_config(vec![hbm()]);
        let request = request_of(
            &config,
            0,
            vec![removed(&[9], "GPU"), stored(&[9], "CPU_PINNED")],
        );
        assert_eq!(
            request.actions,
            vec![revoke(hbm(), &["9"]), report(dram(), &["9"])]
        );
    }

    #[test]
    fn clear_all_expands_to_one_action_per_clear_tier_in_place() {
        let config = test_config(vec![hbm(), dram(), ssd()]);
        let request = request_of(
            &config,
            7,
            vec![stored(&[1], "GPU"), cleared(), stored(&[2], "GPU")],
        );
        assert_eq!(
            request.actions,
            vec![
                report(hbm(), &["1"]),
                clear_at(hbm()),
                clear_at(dram()),
                clear_at(ssd()),
                report(hbm(), &["2"]),
            ]
        );
    }

    #[test]
    fn batch_with_only_ignored_events_has_no_actions() {
        let config = test_config(vec![hbm()]);
        let events = vec![Value::Array(vec![Value::String("BlockUpdated".into())])];
        assert!(request_of(&config, 0, events).actions.is_empty());
    }

    #[test]
    fn block_stored_maps_to_report_on_tier() {
        assert_eq!(
            actions_of(vec![stored(&[123], "GPU")]),
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["123".to_string()],
            }]
        );
    }

    #[test]
    fn mediums_map_to_expected_tiers() {
        assert_eq!(
            actions_of(vec![stored(&[1], "CPU_PINNED")]),
            vec![Action::Report {
                tier: dram(),
                hashes: vec!["1".to_string()],
            }]
        );
        assert_eq!(
            actions_of(vec![removed(&[2], "DISK")]),
            vec![Action::Revoke {
                tier: ssd(),
                hashes: vec!["2".to_string()],
            }]
        );
    }

    // --- ordering regressions (the whole point of the in-order rewrite) ---

    #[test]
    fn remove_then_store_same_hash_keeps_order() {
        // Net state must be "stored"; reordering to report-then-revoke would drop it.
        assert_eq!(
            actions_of(vec![removed(&[9], "GPU"), stored(&[9], "GPU")]),
            vec![
                Action::Revoke {
                    tier: hbm(),
                    hashes: vec!["9".to_string()],
                },
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["9".to_string()],
                },
            ]
        );
    }

    #[test]
    fn clear_then_store_keeps_order() {
        assert_eq!(
            actions_of(vec![cleared(), stored(&[7], "GPU")]),
            vec![
                Action::ClearAll,
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["7".to_string()],
                },
            ]
        );
    }

    #[test]
    fn store_then_clear_keeps_order() {
        assert_eq!(
            actions_of(vec![stored(&[7], "GPU"), cleared()]),
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["7".to_string()],
                },
                Action::ClearAll,
            ]
        );
    }

    // --- coalescing rules ---

    #[test]
    fn adjacent_same_tier_stores_coalesce() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), stored(&[2], "GPU")]),
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["1".to_string(), "2".to_string()],
            }]
        );
    }

    #[test]
    fn different_tier_stores_do_not_coalesce() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), stored(&[2], "CPU_PINNED")]),
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
                Action::Report {
                    tier: dram(),
                    hashes: vec!["2".to_string()],
                },
            ]
        );
    }

    #[test]
    fn store_then_remove_same_tier_do_not_merge() {
        assert_eq!(
            actions_of(vec![stored(&[1], "GPU"), removed(&[1], "GPU")]),
            vec![
                Action::Report {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
                Action::Revoke {
                    tier: hbm(),
                    hashes: vec!["1".to_string()],
                },
            ]
        );
    }

    #[test]
    fn unknown_event_tag_is_ignored() {
        let events = vec![Value::Array(vec![Value::String("BlockUpdated".into())])];
        assert!(actions_of(events).is_empty());
    }

    #[test]
    fn two_element_batch_without_dp_rank_decodes() {
        let payload = encode(&Value::Array(vec![
            Value::from(1.0_f64),
            Value::Array(vec![stored(&[5], "GPU")]),
        ]));
        assert_eq!(
            decode_event_batch(&payload).unwrap().actions,
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["5".to_string()],
            }]
        );
    }

    #[test]
    fn negative_hashes_are_stringified() {
        assert_eq!(
            actions_of(vec![stored(&[-1905904552702706914], "GPU")]),
            vec![Action::Report {
                tier: hbm(),
                hashes: vec!["-1905904552702706914".to_string()],
            }]
        );
    }

    // --- error / mapping units ---

    #[test]
    fn external_medium_is_rejected() {
        assert!(decode_event_batch(&batch(vec![stored(&[1], "EXTERNAL")])).is_err());
    }

    #[test]
    fn unknown_medium_is_rejected() {
        assert!(decode_event_batch(&batch(vec![stored(&[1], "TAPE")])).is_err());
    }

    #[test]
    fn medium_to_tier_mapping() {
        assert_eq!(medium_to_tier(Some("GPU")).unwrap(), hbm());
        assert_eq!(medium_to_tier(Some("CPU_PINNED")).unwrap(), dram());
        assert_eq!(medium_to_tier(Some("DISK")).unwrap(), ssd());
        assert!(medium_to_tier(Some("EXTERNAL")).is_err());
        assert!(medium_to_tier(None).is_err());
    }

    #[test]
    fn parse_clear_tiers_defaults_and_aliases() {
        assert_eq!(
            parse_clear_tiers("HBM,DRAM,SSD").unwrap(),
            vec![hbm(), dram(), ssd()]
        );
        assert_eq!(
            parse_clear_tiers(" GPU , CPU_PINNED , DISK ").unwrap(),
            vec![hbm(), dram(), ssd()]
        );
        assert!(parse_clear_tiers("HBM,NVME").is_err());
    }

    #[test]
    fn decode_hashes_accepts_signed_and_unsigned() {
        let value = Value::Array(vec![
            Value::from(1_i64),
            Value::from(-2_i64),
            Value::from(u64::MAX),
        ]);
        assert_eq!(
            decode_hashes(&value).unwrap(),
            vec!["1".to_string(), "-2".to_string(), u64::MAX.to_string()]
        );
        assert!(decode_hashes(&Value::Array(vec![Value::String("x".into())])).is_err());
    }

    // --- frame / sequence parsing ---

    #[test]
    fn parse_zmq_frames_two_and_three() {
        let seq = 42_u64;
        let two = [
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from_static(b"p"),
        ];
        assert_eq!(parse_zmq_frames(&two).unwrap().0, seq);
        let three = [
            Bytes::from_static(b"kv-events"),
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from_static(b"p"),
        ];
        assert_eq!(parse_zmq_frames(&three).unwrap().0, seq);
        let one = [Bytes::from_static(b"p")];
        assert!(parse_zmq_frames(&one).is_err());
    }

    #[test]
    fn parse_replay_frames_dealer_three_and_bare_two() {
        let seq = 7_i64;
        // DEALER keeps the empty delimiter: [b"", seq, payload]
        let three = [
            Bytes::new(),
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from_static(b"payload"),
        ];
        let (parsed, payload) = parse_replay_frames(&three).unwrap();
        assert_eq!(parsed, seq);
        assert_eq!(payload, b"payload");
        // bare 2-frame form
        let two = [
            Bytes::copy_from_slice(&seq.to_be_bytes()),
            Bytes::from_static(b"p"),
        ];
        assert_eq!(parse_replay_frames(&two).unwrap().0, seq);
        assert!(parse_replay_frames(&[Bytes::new()]).is_err());
    }

    #[test]
    fn seq_decoders_are_big_endian() {
        assert_eq!(decode_seq(&5_u64.to_be_bytes()).unwrap(), 5);
        // END_SEQ sentinel: -1 as 8-byte big-endian signed
        assert_eq!(decode_signed_seq(&(-1_i64).to_be_bytes()).unwrap(), -1);
        assert!(decode_seq(&[0_u8; 4]).is_err());
    }
}

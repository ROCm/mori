use std::io::Cursor;
use std::time::Duration;

use rmpv::decode::value::read_value;
use rmpv::Value;
use tonic::transport::Channel;
use tracing::{debug, info, warn};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, SubSocket, ZmqMessage};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{
    MatchExternalKvRequest, ReportExternalKvBlocksRequest, RevokeAllExternalKvBlocksAtTierRequest,
    RevokeExternalKvBlocksRequest, TierType,
};

/// Backoff bounds for the reconnect supervisor loop.
const RECONNECT_MIN_DELAY: Duration = Duration::from_millis(500);
const RECONNECT_MAX_DELAY: Duration = Duration::from_secs(10);
const REPLAY_REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub worker_id: String,
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
        let event_endpoint = std::env::var("SGLANG_KV_EVENT_ENDPOINT")
            .unwrap_or_else(|_| "tcp://127.0.0.1:5557".to_string());
        let event_replay_endpoint = std::env::var("SGLANG_KV_EVENT_REPLAY_ENDPOINT").ok();
        let event_topic =
            std::env::var("SGLANG_KV_EVENT_TOPIC").unwrap_or_else(|_| "kv-events".to_string());
        let indexer_endpoint = std::env::var("KV_INDEXER_ENDPOINT")
            .unwrap_or_else(|_| "http://[::1]:50051".to_string());
        let clear_tiers = parse_clear_tiers(
            &std::env::var("KV_INDEXER_CLEAR_TIERS")
                .unwrap_or_else(|_| "HBM,DRAM,SSD".to_string()),
        )?;

        Ok(Self {
            worker_id,
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
#[derive(Debug)]
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
    loop {
        if let Some(batch) = pending_batch.clone() {
            forward_raw_batch(config, &mut client, &batch).await?;
            *pending_batch = None;
            *next_seq = batch.seq.checked_add(1);
        }

        let message = tokio::select! {
            result = subscriber.recv() => result?,
            _ = tokio::signal::ctrl_c() => {
                info!("received ctrl-c; shutting down bridge");
                return Ok(());
            }
        };

        let frames = message.into_vec();
        let (seq, payload) = match parse_zmq_frames(&frames) {
            Ok((seq, payload)) => (seq, payload.to_vec()),
            Err(error) => {
                warn!(%error, "skipping malformed ZMQ message");
                continue;
            }
        };

        if let Some(expected) = *next_seq {
            if seq < expected {
                warn!(expected, actual = seq, "skipping stale SGLang KV event batch");
                continue;
            }
            if seq > expected {
                warn!(expected, actual = seq, "SGLang KV event sequence gap");
                replay_missing_batches(config, &mut client, expected, seq, pending_batch).await?;
            }
        }

        let batch = RawBatch { seq, payload };
        *pending_batch = Some(batch.clone());
        forward_raw_batch(config, &mut client, &batch).await?;
        *pending_batch = None;
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
    let mut request =
        ZmqMessage::from(bytes::Bytes::copy_from_slice(&start_seq.to_be_bytes()));
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
        *pending_batch = Some(batch.clone());
        forward_raw_batch(config, client, &batch).await?;
        *pending_batch = None;
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
    forward_actions(config, client, actions).await
}

fn decode_event_batch(payload: &[u8]) -> Result<EventActions, BridgeError> {
    let mut cursor = Cursor::new(payload);
    let value =
        read_value(&mut cursor).map_err(|error| BridgeError::Decode(error.to_string()))?;
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

async fn forward_actions(
    config: &BridgeConfig,
    client: &mut KvIndexerClient<Channel>,
    actions: EventActions,
) -> Result<(), BridgeError> {
    for action in actions.actions {
        match action {
            Action::Report { tier, hashes } => {
                client
                    .report_external_kv_blocks(ReportExternalKvBlocksRequest {
                        worker_id: config.worker_id.clone(),
                        hashes,
                        tier,
                    })
                    .await?;
            }
            Action::Revoke { tier, hashes } => {
                client
                    .revoke_external_kv_blocks(RevokeExternalKvBlocksRequest {
                        worker_id: config.worker_id.clone(),
                        hashes,
                        tier,
                    })
                    .await?;
            }
            Action::ClearAll => {
                for tier in &config.clear_tiers {
                    client
                        .revoke_all_external_kv_blocks_at_tier(
                            RevokeAllExternalKvBlocksAtTierRequest {
                                worker_id: config.worker_id.clone(),
                                tier: *tier,
                            },
                        )
                        .await?;
                }
            }
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

fn expect_optional_str<'a>(
    value: &'a Value,
    field: &str,
) -> Result<Option<&'a str>, BridgeError> {
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

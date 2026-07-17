use std::collections::BTreeMap;
use std::io::Cursor;

use rmpv::decode::value::read_value;
use rmpv::Value;
use tonic::transport::Channel;
use tracing::{debug, info, warn};
use zeromq::{Socket, SocketRecv, SubSocket};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{
    MatchExternalKvRequest, ReportExternalKvBlocksRequest, RevokeAllExternalKvBlocksAtTierRequest,
    RevokeExternalKvBlocksRequest, TierType,
};

#[derive(Debug, Clone)]
pub struct BridgeConfig {
    pub worker_id: String,
    pub event_endpoint: String,
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
    Transport(tonic::transport::Error),
    Zmq(zeromq::ZmqError),
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BridgeError::Config(message) => write!(f, "bridge config error: {message}"),
            BridgeError::Decode(message) => write!(f, "bridge decode error: {message}"),
            BridgeError::Rpc(status) => write!(f, "indexer rpc error: {status}"),
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

#[derive(Debug, Default)]
struct EventActions {
    reports: BTreeMap<i32, Vec<String>>,
    revokes: BTreeMap<i32, Vec<String>>,
    clear_all: bool,
}

pub async fn run_bridge(config: BridgeConfig) -> Result<(), BridgeError> {
    let mut client = KvIndexerClient::connect(config.indexer_endpoint.clone()).await?;
    let mut subscriber = SubSocket::new();
    subscriber.subscribe(&config.event_topic).await?;
    subscriber.connect(&config.event_endpoint).await?;

    info!(
        worker_id = %config.worker_id,
        event_endpoint = %config.event_endpoint,
        event_topic = %config.event_topic,
        indexer_endpoint = %config.indexer_endpoint,
        "starting SGLang KV event bridge"
    );

    let mut expected_seq: Option<u64> = None;
    loop {
        let message = subscriber.recv().await?;
        let frames = message.into_vec();
        let (seq, payload) = parse_zmq_frames(&frames)?;

        if let Some(expected) = expected_seq {
            if seq != expected {
                warn!(expected, actual = seq, "SGLang KV event sequence gap");
            }
        }
        expected_seq = seq.checked_add(1);

        let actions = decode_event_batch(payload)?;
        forward_actions(&config, &mut client, actions).await?;
    }
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
            actions
                .reports
                .entry(tier)
                .or_default()
                .extend(decode_hashes(&event[1])?);
        }
        "BlockRemoved" => {
            if event.len() < 3 {
                return Err(BridgeError::Decode(
                    "BlockRemoved must have 3 array fields".to_string(),
                ));
            }
            let tier = medium_to_tier(expect_optional_str(&event[2], "BlockRemoved.medium")?)?;
            actions
                .revokes
                .entry(tier)
                .or_default()
                .extend(decode_hashes(&event[1])?);
        }
        "AllBlocksCleared" => {
            actions.clear_all = true;
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
    for (tier, hashes) in actions.reports {
        if hashes.is_empty() {
            continue;
        }
        client
            .report_external_kv_blocks(ReportExternalKvBlocksRequest {
                worker_id: config.worker_id.clone(),
                hashes,
                tier,
            })
            .await?;
    }

    for (tier, hashes) in actions.revokes {
        if hashes.is_empty() {
            continue;
        }
        client
            .revoke_external_kv_blocks(RevokeExternalKvBlocksRequest {
                worker_id: config.worker_id.clone(),
                hashes,
                tier,
            })
            .await?;
    }

    if actions.clear_all {
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

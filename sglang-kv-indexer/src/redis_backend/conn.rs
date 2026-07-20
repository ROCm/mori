//! Topology-agnostic connection seam. The backend logic issues commands and
//! script invocations through `RedisConn`, so the same apply/match code runs on
//! a single Redis/Dragonfly instance or a Redis Cluster. Connections are cheaply
//! cloneable and multiplexed, so per-hash operations are issued concurrently
//! (one connection, many in-flight commands) rather than serialized.

use redis::{Cmd, RedisResult, Script, Value};

#[tonic::async_trait]
pub(crate) trait RedisConn: Send + Sync + 'static {
    /// Runs a single command, routed by its key on Cluster.
    async fn query(&self, cmd: Cmd) -> RedisResult<Value>;

    /// Runs a Lua script (EVALSHA with automatic NOSCRIPT fallback), routed by
    /// `keys[0]` on Cluster. All `keys` must share a hash tag.
    async fn invoke(
        &self,
        script: &Script,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> RedisResult<Value>;
}

/// Single Redis/Dragonfly instance via an auto-reconnecting multiplexed manager.
pub(crate) struct SingleConn {
    conn: redis::aio::ConnectionManager,
}

impl SingleConn {
    pub(crate) async fn connect(url: &str) -> RedisResult<Self> {
        let client = redis::Client::open(url)?;
        let conn = redis::aio::ConnectionManager::new(client).await?;
        Ok(Self { conn })
    }
}

#[tonic::async_trait]
impl RedisConn for SingleConn {
    async fn query(&self, cmd: Cmd) -> RedisResult<Value> {
        let mut c = self.conn.clone();
        cmd.query_async(&mut c).await
    }

    async fn invoke(
        &self,
        script: &Script,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> RedisResult<Value> {
        let mut c = self.conn.clone();
        let mut inv = script.prepare_invoke();
        for k in &keys {
            inv.key(k.as_str());
        }
        for a in &args {
            inv.arg(a.as_str());
        }
        let v: Value = inv.invoke_async(&mut c).await?;
        Ok(v)
    }
}

/// Redis Cluster via the async cluster connection. MOVED/ASK redirects and slot
/// map refresh are handled by the client; each command/script is routed by key.
pub(crate) struct ClusterConn {
    conn: redis::cluster_async::ClusterConnection,
}

impl ClusterConn {
    pub(crate) async fn connect(nodes: Vec<String>) -> RedisResult<Self> {
        let client = redis::cluster::ClusterClient::new(nodes)?;
        let conn = client.get_async_connection().await?;
        Ok(Self { conn })
    }
}

#[tonic::async_trait]
impl RedisConn for ClusterConn {
    async fn query(&self, cmd: Cmd) -> RedisResult<Value> {
        let mut c = self.conn.clone();
        cmd.query_async(&mut c).await
    }

    async fn invoke(
        &self,
        script: &Script,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> RedisResult<Value> {
        let mut c = self.conn.clone();
        let mut inv = script.prepare_invoke();
        for k in &keys {
            inv.key(k.as_str());
        }
        for a in &args {
            inv.arg(a.as_str());
        }
        let v: Value = inv.invoke_async(&mut c).await?;
        Ok(v)
    }
}

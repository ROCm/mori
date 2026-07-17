use sglang_kv_indexer::bridge::{run_bridge, BridgeConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = BridgeConfig::from_env()?;
    run_bridge(config).await?;
    Ok(())
}

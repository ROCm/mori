#!/usr/bin/env bash
# Start VictoriaMetrics + Grafana for xgmi-netmon (host network, 127.0.0.1 only;
# Grafana allows anonymous admin, do not expose it).
#   ./up.sh DATA_DIR          DATA_DIR on local disk (VictoriaMetrics does not support NFS)
#   XGMI_VM_PORT=28428 XGMI_GRAFANA_PORT=23000 ./up.sh DATA_DIR
set -euo pipefail
cd "$(dirname "$0")"
data="$(realpath -m "${1:?usage: ./up.sh DATA_DIR}")"
vm_port="${XGMI_VM_PORT:-28428}"
gf_port="${XGMI_GRAFANA_PORT:-23000}"
p="xgmi-netmon-$(id -un)"
mkdir -p "$data/vm" "$data/grafana"
# the docker daemon may not see this directory (NFS home with root squash): use a copy on DATA_DIR
rm -rf "$data/conf" && cp -r grafana "$data/conf"
docker run -d --name "$p-vm" --network host --restart unless-stopped --user "$(id -u):$(id -g)" \
  -v "$data/vm:/storage" victoriametrics/victoria-metrics:v1.156.0 \
  -storageDataPath=/storage -httpListenAddr="127.0.0.1:$vm_port" -retentionPeriod=100y \
  -search.latencyOffset=0s >/dev/null
docker run -d --name "$p-grafana" --network host --restart unless-stopped --user "$(id -u):$(id -g)" \
  -e GF_SERVER_HTTP_ADDR=127.0.0.1 -e GF_SERVER_HTTP_PORT="$gf_port" \
  -e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin -e GF_AUTH_DISABLE_LOGIN_FORM=true \
  -e GF_DASHBOARDS_MIN_REFRESH_INTERVAL=1s \
  -e GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/xgmi/xgmi.json \
  -e XGMI_VM_PORT="$vm_port" \
  -v "$data/grafana:/var/lib/grafana" \
  -v "$data/conf/provisioning/datasources:/etc/grafana/provisioning/datasources:ro" \
  -v "$data/conf/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro" \
  -v "$data/conf/dashboards:/var/lib/grafana/dashboards/xgmi:ro" \
  grafana/grafana:13.3.1 >/dev/null
echo "VictoriaMetrics: http://127.0.0.1:$vm_port   Grafana: http://127.0.0.1:$gf_port"

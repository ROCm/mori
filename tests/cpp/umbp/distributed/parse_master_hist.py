#!/usr/bin/env python3
"""Fetch a UMBP master's Prometheus /metrics and print per-RPC p50/p95/p99.

Aggregates mori_umbp_master_client_rpc_latency_seconds_bucket across ALL node
labels (the histogram is per-client), then linear-interpolates the quantile
within the matched bucket (standard Prometheus histogram_quantile).
"""
import sys
import re
import urllib.request

METRIC = "mori_umbp_master_client_rpc_latency_seconds_bucket"
COUNT = "mori_umbp_master_client_rpc_latency_seconds_count"

BUCKET_RE = re.compile(r'^%s\{([^}]*)\}\s+([0-9.eE+]+)\s*$' % re.escape(METRIC))
COUNT_RE = re.compile(r'^%s\{([^}]*)\}\s+([0-9.eE+]+)\s*$' % re.escape(COUNT))


def labels(s):
    d = {}
    for m in re.finditer(r'(\w+)="([^"]*)"', s):
        d[m.group(1)] = m.group(2)
    return d


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9091/metrics"
    body = urllib.request.urlopen(url, timeout=15).read().decode("utf-8", "replace")

    # rpc -> {le(float) -> summed cumulative count}; rpc -> total count
    buckets = {}
    counts = {}
    for line in body.splitlines():
        m = BUCKET_RE.match(line)
        if m:
            lab = labels(m.group(1))
            rpc = lab.get("rpc", "?")
            le = lab.get("le")
            le = float("inf") if le == "+Inf" else float(le)
            buckets.setdefault(rpc, {})
            buckets[rpc][le] = buckets[rpc].get(le, 0.0) + float(m.group(2))
            continue
        m = COUNT_RE.match(line)
        if m:
            lab = labels(m.group(1))
            rpc = lab.get("rpc", "?")
            counts[rpc] = counts.get(rpc, 0.0) + float(m.group(2))


    def quantile(edges, q):
        total = edges[-1][1]  # +Inf cumulative = total
        if total <= 0:
            return 0.0
        rank = q * total
        prev_edge, prev_cum = 0.0, 0.0
        for le, cum in edges:
            if cum >= rank:
                if le == float("inf"):
                    return prev_edge * 1000.0  # cannot interpolate the open bucket
                # linear interpolation within (prev_edge, le]
                span = cum - prev_cum
                frac = (rank - prev_cum) / span if span > 0 else 0.0
                return (prev_edge + (le - prev_edge) * frac) * 1000.0
            prev_edge, prev_cum = le, cum
        return prev_edge * 1000.0

    print("rpc,count,p50_ms,p95_ms,p99_ms")
    for rpc in sorted(buckets):
        edges = sorted(buckets[rpc].items())
        cnt = counts.get(rpc, edges[-1][1])
        print("%s,%.0f,%.3f,%.3f,%.3f" % (
            rpc, cnt, quantile(edges, 0.50), quantile(edges, 0.95), quantile(edges, 0.99)))


if __name__ == "__main__":
    main()

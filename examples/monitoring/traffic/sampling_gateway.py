#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""One address for the metrics sampling of every running MORI process.

Grafana reaches it as the data source `mori-sampling`; GET or POST /sampling is passed on to
every process Prometheus scrapes. Whoever can reach it can change what every job counts.

It also answers a Prometheus instant query (/api/v1/query, data source `mori-sampling-now`)
with what the processes are set to at this moment, whatever time the query names: the
dashboard's switch shows the setting in effect even when the dashboard looks at the past.
  mori_sampling_period{module="ep"}             the period (the largest among the processes)
  mori_sampling_is{module="ep",period="8"}      1 if every process is at that period, else 0

With --prometheus-config it also sets how often Prometheus reads the processes: POST
/scrape?interval=2s rewrites scrape_interval (and scrape_timeout) in that file and asks
Prometheus to reload it (Prometheus needs --web.enable-lifecycle). Queries:
  mori_scrape_is{interval="2s"}                 1 if Prometheus reads every 2 s, else 0
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

SETTINGS = ("ep", "io", "interval_ms")
# Scrape intervals the switch offers. Longer than 10 s and the dashboards' 20 s rate window
# (the data source's $__rate_interval) holds fewer than two samples.
SCRAPE_INTERVALS = ("1s", "2s", "5s", "10s")
JOB = "mori_traffic"
TIMEOUT_S = 10


def active_targets(args):
    """host:port of every process to talk to."""
    if args.targets:
        return [t for t in args.targets.split(",") if t]
    url = args.prometheus.rstrip("/") + "/api/v1/targets?state=active"
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        targets = json.load(r)["data"]["activeTargets"]
    return sorted(
        t["labels"]["instance"]
        for t in targets
        if t.get("health") == "up" and t["labels"].get("job") == JOB
    )


def ask(target, query):
    """One process: its settings, or the error as a string."""
    url = "http://%s/sampling" % target
    data = None
    if query:
        url += "?" + query
        data = b""  # a POST
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, data=data), timeout=TIMEOUT_S
        ) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return "HTTP %d: %s" % (e.code, e.read().decode(errors="replace").strip()[:200])
    except Exception as e:  # refused, timed out, not JSON: the process did not take it
        return "%s: %s" % (type(e).__name__, e)


def fan_out(args, query):
    targets = active_targets(args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        answers = list(pool.map(lambda t: ask(t, query), targets))
    ok = [a for a in answers if isinstance(a, dict)]
    out = {
        "processes": len(targets),
        "answered": len(ok),
        "failed": {t: a for t, a in zip(targets, answers) if not isinstance(a, dict)},
    }
    for key in SETTINGS:
        out[key] = sorted({a[key] for a in ok if key in a})
    return out


# What the processes are set to, asked at most once in STATE_AGE_S however many panels query it
# (a refresh of the switch is ten queries); dropped when a setting is changed.
STATE_AGE_S = 2.0
_state = {"at": 0.0, "value": None}
_state_lock = threading.Lock()


def current_state(args):
    with _state_lock:
        if _state["value"] is None or time.time() - _state["at"] > STATE_AGE_S:
            _state["value"] = fan_out(args, "")
            _state["at"] = time.time()
        return _state["value"]


def forget_state():
    with _state_lock:
        _state["value"] = None


def instant_vector(args, expr, at):
    """The answer to one of the two queries in the module comment; empty for anything else, and
    when no process answers (the switch then shows no setting)."""
    m = re.fullmatch(
        r"\s*(mori_sampling_period|mori_sampling_is|mori_scrape_is)\s*\{(.*)\}\s*",
        expr or "",
    )
    if not m:
        return []
    labels = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(2)))
    if m.group(1) == "mori_scrape_is":
        value = 1 if scrape_interval(args) == labels.get("interval") else 0
        return [
            {"metric": dict(labels, __name__=m.group(1)), "value": [at, str(value)]}
        ]
    module = labels.get("module")
    if module not in ("ep", "io"):
        return []
    values = current_state(args).get(module) or []
    if not values:
        return []
    if m.group(1) == "mori_sampling_period":
        value = max(values)
    else:
        value = 1 if values == [int(labels.get("period", "-1"))] else 0
    return [{"metric": dict(labels, __name__=m.group(1)), "value": [at, str(value)]}]


def scrape_interval(args):
    """What Prometheus reads at: the global scrape_interval of the configuration it runs."""
    url = args.prometheus.rstrip("/") + "/api/v1/status/config"
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
        yaml = json.load(r)["data"]["yaml"]
    m = re.search(r"^\s*scrape_interval:\s*(\S+)", yaml, re.M)
    return m.group(1) if m else None


def set_scrape_interval(args, interval):
    """Rewrites the global scrape_interval / scrape_timeout of --prometheus-config in place (the
    file is bind-mounted into the Prometheus container: a new file would not be seen there) and
    has Prometheus reload it."""
    seconds = int(interval[:-1])
    timeout = "%dms" % min(
        4000, seconds * 800
    )  # under the interval, as Prometheus requires
    with open(args.prometheus_config, "r+") as f:
        text = f.read()
        head, sep, rest = text.partition("scrape_configs:")
        head = re.sub(
            r"(?m)^(\s*scrape_interval:\s*)\S+", r"\g<1>" + interval, head, count=1
        )
        head = re.sub(
            r"(?m)^(\s*scrape_timeout:\s*)\S+", r"\g<1>" + timeout, head, count=1
        )
        f.seek(0)
        f.write(head + sep + rest)
        f.truncate()
    url = args.prometheus.rstrip("/") + "/-/reload"
    urllib.request.urlopen(
        urllib.request.Request(url, data=b""), timeout=TIMEOUT_S
    ).read()
    return scrape_interval(args)


class Server(
    ThreadingMixIn, HTTPServer
):  # http.server has its own only from Python 3.7 on
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    args = None

    def reply(self, status, body):
        data = (json.dumps(body) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the client gave up: a dashboard cancels the queries of the refresh before

    def handle_scrape(self, change):
        url = urllib.parse.urlsplit(self.path)
        try:
            if not change:
                return self.reply(200, {"scrape_interval": scrape_interval(self.args)})
            if not self.args.prometheus_config:
                return self.reply(404, {"error": "started without --prometheus-config"})
            interval = dict(urllib.parse.parse_qsl(url.query)).get("interval")
            if interval not in SCRAPE_INTERVALS:
                return self.reply(
                    400,
                    {"error": "interval is one of %s" % ", ".join(SCRAPE_INTERVALS)},
                )
            return self.reply(
                200, {"scrape_interval": set_scrape_interval(self.args, interval)}
            )
        except Exception as e:
            return self.reply(502, {"error": "%s: %s" % (type(e).__name__, e)})

    def handle_sampling(self, change):
        url = urllib.parse.urlsplit(self.path)
        if url.path.rstrip("/") != "/sampling":
            return self.reply(404, {"error": "try /sampling"})
        query = ""
        if change:
            # Exactly what a process accepts, and nothing else goes through.
            pairs = urllib.parse.parse_qsl(url.query, keep_blank_values=True)
            bad = [
                k
                for k, v in pairs
                if k not in SETTINGS or not re.fullmatch(r"\d{1,9}", v)
            ]
            if bad or not pairs:
                return self.reply(
                    400,
                    {
                        "error": "settings are ep=<n>, io=<n>, interval_ms=<n>",
                        "bad": bad,
                    },
                )
            query = urllib.parse.urlencode(pairs)
        try:
            out = fan_out(self.args, query)
        except Exception as e:
            return self.reply(502, {"error": "cannot list the processes: %s" % e})
        if change:
            forget_state()
        if out["processes"] == 0:
            return self.reply(
                404 if change else 200, dict(out, error="no running MORI process")
            )
        self.reply(502 if change and out["failed"] else 200, out)

    def handle_prometheus(self, params):
        """The part of the Prometheus HTTP API a Grafana data source needs."""
        path = urllib.parse.urlsplit(self.path).path
        if path != "/api/v1/query":
            # Labels, metadata, build information: nothing to say.
            return self.reply(
                200, {"status": "success", "data": {} if "status" in path else []}
            )
        try:
            at = float(params.get("time") or time.time())
            result = instant_vector(self.args, params.get("query"), at)
        except Exception as e:
            return self.reply(
                200, {"status": "error", "errorType": "execution", "error": str(e)}
            )
        self.reply(
            200,
            {"status": "success", "data": {"resultType": "vector", "result": result}},
        )

    def do_GET(self):
        url = urllib.parse.urlsplit(self.path)
        if url.path.startswith("/api/v1/"):
            return self.handle_prometheus(dict(urllib.parse.parse_qsl(url.query)))
        if url.path.rstrip("/") == "/scrape":
            return self.handle_scrape(change=False)
        self.handle_sampling(change=False)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(min(length, 1 << 20)) if length else b""
        url = urllib.parse.urlsplit(self.path)
        if url.path.startswith("/api/v1/"):
            # A query, sent as a form: it changes nothing.
            params = dict(urllib.parse.parse_qsl(url.query))
            params.update(urllib.parse.parse_qsl(body.decode(errors="replace")))
            return self.handle_prometheus(params)
        # A button sends a body with its request; the settings are in the query.
        if url.path.rstrip("/") == "/scrape":
            return self.handle_scrape(change=True)
        self.handle_sampling(change=True)

    def log_message(self, fmt, *a):
        if self.command == "POST" and self.path.startswith(("/sampling", "/scrape")):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % a))
            sys.stderr.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--listen", default=os.environ.get("TRAFFIC_GATEWAY_LISTEN", "127.0.0.1:9390")
    )
    p.add_argument(
        "--prometheus",
        default=os.environ.get("TRAFFIC_PROM_URL", "http://127.0.0.1:9090"),
    )
    p.add_argument("--targets", default=os.environ.get("TRAFFIC_TARGETS", ""))
    p.add_argument(
        "--prometheus-config",
        default=os.environ.get("TRAFFIC_PROM_CONFIG", ""),
        help="the Prometheus configuration file, writable: enables POST /scrape",
    )
    args = p.parse_args()
    host, _, port = args.listen.rpartition(":")
    Handler.args = args
    server = Server((host or "127.0.0.1", int(port)), Handler)
    print(
        "sampling gateway on %s: processes from %s"
        % (args.listen, args.targets or args.prometheus),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

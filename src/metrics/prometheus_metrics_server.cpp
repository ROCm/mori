// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include "mori/metrics/prometheus_metrics_server.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "mori/utils/mori_log.hpp"

namespace mori {
namespace metrics {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

MetricsServer::MetricsServer(int port) : port_(port) {
  ModuleLogger::GetInstance().InitModule(modules::METRICS);

  server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd_ < 0) {
    throw std::runtime_error("[metrics] socket() failed: " + std::string(std::strerror(errno)));
  }

  int opt = 1;
  ::setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(port_));

  if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    ::close(server_fd_);
    server_fd_ = -1;
    throw std::runtime_error("[metrics] bind() on port " + std::to_string(port_) +
                             " failed: " + std::strerror(errno));
  }

  if (::listen(server_fd_, 16) < 0) {
    ::close(server_fd_);
    server_fd_ = -1;
    throw std::runtime_error("[metrics] listen() failed: " + std::string(std::strerror(errno)));
  }

  running_.store(true, std::memory_order_relaxed);
  accept_thread_ = std::thread(&MetricsServer::acceptLoop, this);

  MORI_INFO(modules::METRICS, "Prometheus metrics server listening on http://0.0.0.0:{}/metrics",
            port_);
}

MetricsServer::~MetricsServer() {
  running_.store(false, std::memory_order_relaxed);
  // shutdown() reliably unblocks accept() on Linux; close() alone does not.
  if (server_fd_ >= 0) {
    ::shutdown(server_fd_, SHUT_RDWR);
    ::close(server_fd_);
    server_fd_ = -1;
  }
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
  MORI_INFO(modules::METRICS, "Prometheus metrics server stopped");
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

std::string MetricsServer::SanitizeName(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      c = '_';
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public metric update API
// ---------------------------------------------------------------------------

void MetricsServer::setGauge(std::string_view name, std::string_view help, double value) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto& entry = gauges_[std::string(name)];
  entry.help = help;
  entry.value = value;
}

void MetricsServer::addCounter(std::string_view name, std::string_view help, uint64_t delta) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto& entry = counters_[std::string(name)];
  entry.help = help;
  entry.value += delta;
}

void MetricsServer::observe(std::string_view name, std::string_view help,
                            const std::vector<double>& bounds, double value) {
  std::lock_guard<std::mutex> lk(mutex_);
  auto key = std::string(name);
  auto it = histograms_.find(key);

  if (it == histograms_.end()) {
    // First observation: initialise the histogram layout.
    HistogramEntry entry;
    entry.help = help;
    entry.bounds = bounds;
    entry.bucket_counts.assign(bounds.size(), 0u);
    histograms_.emplace(key, std::move(entry));
    it = histograms_.find(key);
  }

  HistogramEntry& h = it->second;

  // Increment all cumulative buckets whose upper bound >= value.
  for (std::size_t i = 0; i < h.bounds.size(); ++i) {
    if (value <= h.bounds[i]) {
      h.bucket_counts[i]++;
    }
  }
  h.count++;
  h.sum += value;
}

// ---------------------------------------------------------------------------
// Labeled metric API
// ---------------------------------------------------------------------------

static std::string FormatLabels(const mori::metrics::MetricsServer::Labels& labels) {
  if (labels.empty()) return "";
  std::string s = "{";
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (i > 0) s += ",";
    s += labels[i].first + "=\"" + labels[i].second + "\"";
  }
  s += "}";
  return s;
}

void MetricsServer::addCounter(std::string_view name, std::string_view help, const Labels& labels,
                               uint64_t delta) {
  auto label_str = FormatLabels(labels);
  std::lock_guard<std::mutex> lk(mutex_);
  auto& family = labeled_counters_[std::string(name)];
  family.help = help;
  family.series[label_str] += delta;
}

void MetricsServer::setGauge(std::string_view name, std::string_view help, const Labels& labels,
                             double value) {
  auto label_str = FormatLabels(labels);
  std::lock_guard<std::mutex> lk(mutex_);
  auto& family = labeled_gauges_[std::string(name)];
  family.help = help;
  family.series[label_str] = value;
}

// ---------------------------------------------------------------------------
// Serialisation (Prometheus text format 0.0.4)
// ---------------------------------------------------------------------------

std::string MetricsServer::serializeLocked() const {
  std::ostringstream out;

  // Gauges
  for (const auto& [name, g] : gauges_) {
    out << "# HELP " << name << " " << g.help << "\n";
    out << "# TYPE " << name << " gauge\n";
    out << name << " " << g.value << "\n\n";
  }

  // Counters
  for (const auto& [name, c] : counters_) {
    out << "# HELP " << name << " " << c.help << "\n";
    out << "# TYPE " << name << " counter\n";
    out << name << " " << c.value << "\n\n";
  }

  // Labeled gauges
  for (const auto& [name, family] : labeled_gauges_) {
    out << "# HELP " << name << " " << family.help << "\n";
    out << "# TYPE " << name << " gauge\n";
    for (const auto& [label_str, value] : family.series) {
      out << name << label_str << " " << value << "\n";
    }
    out << "\n";
  }

  // Labeled counters
  for (const auto& [name, family] : labeled_counters_) {
    out << "# HELP " << name << " " << family.help << "\n";
    out << "# TYPE " << name << " counter\n";
    for (const auto& [label_str, value] : family.series) {
      out << name << label_str << " " << value << "\n";
    }
    out << "\n";
  }

  // Histograms
  for (const auto& [name, h] : histograms_) {
    out << "# HELP " << name << " " << h.help << "\n";
    out << "# TYPE " << name << " histogram\n";

    // Explicit upper-bound buckets (already cumulative in bucket_counts).
    for (std::size_t i = 0; i < h.bounds.size(); ++i) {
      out << name << "_bucket{le=\"" << h.bounds[i] << "\"} " << h.bucket_counts[i] << "\n";
    }
    // +Inf bucket == total observation count.
    out << name << "_bucket{le=\"+Inf\"} " << h.count << "\n";
    out << name << "_sum " << h.sum << "\n";
    out << name << "_count " << h.count << "\n\n";
  }

  return out.str();
}

// ---------------------------------------------------------------------------
// HTTP server internals
// ---------------------------------------------------------------------------

void MetricsServer::acceptLoop() {
  while (running_.load(std::memory_order_relaxed)) {
    int client_fd = ::accept(server_fd_, nullptr, nullptr);
    if (client_fd < 0) {
      // Either the server fd was closed (shutdown) or a transient error.
      if (!running_.load(std::memory_order_relaxed)) break;
      MORI_WARN(modules::METRICS, "accept() returned error: {}", std::strerror(errno));
      continue;
    }
    handleClient(client_fd);
  }
}

void MetricsServer::handleClient(int client_fd) {
  char buf[4096] = {};
  // Read the request (best-effort; we only care about the first line).
  ::read(client_fd, buf, sizeof(buf) - 1);

  std::string body;
  std::string status;

  if (std::strncmp(buf, "GET /metrics", 12) == 0) {
    std::lock_guard<std::mutex> lk(mutex_);
    body = serializeLocked();
    status = "200 OK";
  } else if (std::strncmp(buf, "GET /", 5) == 0) {
    body = "Try GET /metrics\n";
    status = "404 Not Found";
  } else {
    body = "Bad Request\n";
    status = "400 Bad Request";
  }

  std::string response = "HTTP/1.1 " + status +
                         "\r\n"
                         "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
                         "Content-Length: " +
                         std::to_string(body.size()) +
                         "\r\n"
                         "Connection: close\r\n"
                         "\r\n" +
                         body;

  ::write(client_fd, response.data(), response.size());
  ::close(client_fd);
}

}  // namespace metrics
}  // namespace mori

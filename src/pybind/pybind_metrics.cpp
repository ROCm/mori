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

// What Python sees of the traffic metrics: the process's metric registry (mori.metrics) and the
// per-peer token counts of an EP op (op.get_traffic_stats()).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>

#include "mori/metrics/metrics.hpp"
#include "mori/ops/ops.hpp"
#include "src/pybind/mori.hpp"

namespace py = pybind11;

namespace {

py::dict EpStatsToDict(const mori::moe::EpTrafficStats& stats) {
  const mori::moe::EpTrafficStats::Snapshot snap = stats.Read();
  py::dict d;
  d["dispatch_sent"] = snap.sent[mori::moe::kEpStatsDispatch];
  d["dispatch_recv"] = snap.recv[mori::moe::kEpStatsDispatch];
  d["combine_sent"] = snap.sent[mori::moe::kEpStatsCombine];
  d["combine_recv"] = snap.recv[mori::moe::kEpStatsCombine];
  d["dispatch_calls"] = snap.calls[mori::moe::kEpStatsDispatch];
  d["combine_calls"] = snap.calls[mori::moe::kEpStatsCombine];
  const double tps = stats.TicksPerSecond();
  d["dispatch_kernel_seconds"] = tps > 0 ? snap.ticks[mori::moe::kEpStatsDispatch] / tps : 0.0;
  d["combine_kernel_seconds"] = tps > 0 ? snap.ticks[mori::moe::kEpStatsCombine] / tps : 0.0;
  return d;
}

// For the v2 ops. `peers` is the communicator's identity table (Comm.pe_identities()).
std::shared_ptr<mori::moe::EpTrafficStats> MakeEpTrafficStats(int rank, int worldSize,
                                                              int gpuPerNode,
                                                              const std::string& impl,
                                                              const std::string& kernel,
                                                              const py::list& peers) {
  mori::moe::EpTrafficStatsConfig cfg;
  cfg.rank = rank;
  cfg.worldSize = worldSize;
  cfg.gpuPerNode = gpuPerNode;
  cfg.impl = impl;
  cfg.kernel = kernel;
  for (const py::handle& item : peers) {
    const py::dict d = py::reinterpret_borrow<py::dict>(item);
    mori::metrics::PeIdentity id;
    id.hostname = d["hostname"].cast<std::string>();
    id.gpuIndex = d["gpu_index"].cast<int>();
    id.gpuPci = d["gpu_pci"].cast<std::string>();
    id.ibdev = d["ibdev"].cast<std::string>();
    id.netdev = d["netdev"].cast<std::string>();
    cfg.peers.push_back(std::move(id));
  }
  return std::make_shared<mori::moe::EpTrafficStats>(std::move(cfg));
}

void RegisterEpTrafficStats(py::module_& m) {
  using mori::moe::EpDispatchCombineHandle;
  using mori::moe::EpTrafficStats;

  py::class_<EpTrafficStats, std::shared_ptr<EpTrafficStats>>(m, "EpTrafficStats")
      .def(py::init(&MakeEpTrafficStats), py::arg("rank"), py::arg("world_size"),
           py::arg("gpu_per_node"), py::arg("impl"), py::arg("kernel"),
           py::arg("peers") = py::list())
      // Address a kernel is given; 0 when the block could not be set up.
      .def_property_readonly(
          "device_ptr",
          [](const EpTrafficStats& s) { return reinterpret_cast<intptr_t>(s.DevicePtr()); })
      .def("set_bytes_per_token", &EpTrafficStats::SetBytesPerToken, py::arg("op"),
           py::arg("bytes"))
      .def("read", &EpStatsToDict);

  // The same for a v1 op, whose handle owns the statistics.
  m.def(
      "ep_stats_enabled",
      [](const EpDispatchCombineHandle& handle) {
        return handle.trafficStatsOwner != nullptr && handle.trafficStats != nullptr;
      },
      py::arg("handle"));
  m.def(
      "ep_stats_note_launch",
      [](EpDispatchCombineHandle& handle, int op, int hidden_dim) {
        handle.NoteLaunchForStats(op, hidden_dim);
      },
      py::arg("handle"), py::arg("op"), py::arg("hidden_dim") = -1);
  m.def(
      "ep_stats_read",
      [](const EpDispatchCombineHandle& handle) -> py::object {
        if (!handle.trafficStatsOwner) return py::none();
        return EpStatsToDict(*handle.trafficStatsOwner);
      },
      py::arg("handle"));
}

}  // namespace

namespace mori {

void RegisterMoriMetrics(py::module_& m) {
  RegisterEpTrafficStats(m);
#ifdef MORI_ENABLE_METRICS
  m.def(
      "metrics_snapshot",
      []() {
        std::vector<metrics::Registry::Series> series;
        {
          // Sampling takes the registry lock, which a tick may hold, and needs no GIL.
          py::gil_scoped_release release;
          series = metrics::Registry::Instance().Snapshot();
        }
        py::list out;
        for (const auto& s : series) {
          py::dict d;
          d["name"] = s.name;
          d["help"] = s.help;
          d["kind"] = s.kind == metrics::MetricKind::kCounter ? "counter"
                      : s.kind == metrics::MetricKind::kGauge ? "gauge"
                                                              : "histogram";
          py::dict labels;
          for (const auto& [k, v] : s.labels) labels[py::str(k)] = v;
          d["labels"] = labels;
          d["value"] = s.value;
          if (s.kind == metrics::MetricKind::kHistogram) {
            d["bounds"] = s.bounds;
            d["buckets"] = s.buckets;
            d["count"] = s.count;
          }
          out.append(d);
        }
        return out;
      },
      "Sample every registered source and return all series as a list of dicts.");
  m.def(
      "metrics_port", []() { return metrics::Registry::Instance().port(); },
      "The port /metrics is served on, or -1.");
  m.def(
      "metrics_enabled", []() { return metrics::Enabled(); },
      "Whether MORI_METRICS_ENABLE / MORI_METRICS_PORT turned metrics on for this process.");
  // Reads the sampling settings (every argument negative) or changes them; returns those in effect.
  m.def(
      "metrics_sampling",
      [](long long ep, long long io, long long interval_ms) {
        std::string error;
        metrics::Sampling now;
        {
          py::gil_scoped_release release;
          now = metrics::Registry::Instance().SetSampling(ep, io, interval_ms, &error);
        }
        if (!error.empty()) throw py::value_error(error);
        py::dict d;
        d["ep"] = now.ep;
        d["io"] = now.io;
        d["interval_ms"] = now.interval_ms;
        return d;
      },
      py::arg("ep") = -1, py::arg("io") = -1, py::arg("interval_ms") = -1);
#else
  m.def("metrics_snapshot", []() { return py::list(); });
  m.def("metrics_port", []() { return -1; });
  m.def("metrics_enabled", []() { return false; });
  m.def(
      "metrics_sampling",
      [](long long, long long, long long) {
        py::dict d;
        d["ep"] = 0;
        d["io"] = 0;
        d["interval_ms"] = 0;
        return d;
      },
      py::arg("ep") = -1, py::arg("io") = -1, py::arg("interval_ms") = -1);
#endif
}

}  // namespace mori

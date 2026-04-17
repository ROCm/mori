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
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/io/io.hpp"
#include "src/pybind/mori.hpp"

namespace py = pybind11;

namespace mori {
void RegisterMoriIo(pybind11::module_& m) {
  m.def("set_log_level", &mori::io::SetLogLevel);

  py::enum_<mori::io::BackendType>(m, "BackendType")
      .value("Unknown", mori::io::BackendType::Unknown)
      .value("XGMI", mori::io::BackendType::XGMI)
      .value("RDMA", mori::io::BackendType::RDMA)
      .value("TCP", mori::io::BackendType::TCP)
      .export_values();

  py::enum_<mori::io::MemoryLocationType>(m, "MemoryLocationType")
      .value("Unknown", mori::io::MemoryLocationType::Unknown)
      .value("CPU", mori::io::MemoryLocationType::CPU)
      .value("GPU", mori::io::MemoryLocationType::GPU)
      .export_values();

  py::enum_<mori::io::StatusCode>(m, "StatusCode")
      .value("SUCCESS", mori::io::StatusCode::SUCCESS)
      .value("INIT", mori::io::StatusCode::INIT)
      .value("IN_PROGRESS", mori::io::StatusCode::IN_PROGRESS)
      .value("ERR_INVALID_ARGS", mori::io::StatusCode::ERR_INVALID_ARGS)
      .value("ERR_NOT_FOUND", mori::io::StatusCode::ERR_NOT_FOUND)
      .value("ERR_RDMA_OP", mori::io::StatusCode::ERR_RDMA_OP)
      .value("ERR_BAD_STATE", mori::io::StatusCode::ERR_BAD_STATE)
      .value("ERR_GPU_OP", mori::io::StatusCode::ERR_GPU_OP)
      .export_values();

  py::enum_<mori::io::PollCqMode>(m, "PollCqMode")
      .value("POLLING", mori::io::PollCqMode::POLLING)
      .value("EVENT", mori::io::PollCqMode::EVENT);

  py::class_<mori::io::BackendConfig>(m, "BackendConfig");

  py::class_<mori::io::RdmaBackendConfig, mori::io::BackendConfig>(m, "RdmaBackendConfig")
      .def(py::init<int, int, int, mori::io::PollCqMode, bool, uint32_t>(),
           py::arg("qp_per_transfer") = 1, py::arg("post_batch_size") = -1,
           py::arg("num_worker_threads") = -1,
           py::arg("poll_cq_mode") = mori::io::PollCqMode::POLLING,
           py::arg("enable_notification") = true, py::arg("notif_per_qp") = 1024)
      .def_readwrite("qp_per_transfer", &mori::io::RdmaBackendConfig::qpPerTransfer)
      .def_readwrite("post_batch_size", &mori::io::RdmaBackendConfig::postBatchSize)
      .def_readwrite("num_worker_threads", &mori::io::RdmaBackendConfig::numWorkerThreads)
      .def_readwrite("poll_cq_mode", &mori::io::RdmaBackendConfig::pollCqMode)
      .def_readwrite("enable_notification", &mori::io::RdmaBackendConfig::enableNotification)
      .def_readwrite("notif_per_qp", &mori::io::RdmaBackendConfig::notifPerQp)
      .def_readwrite("max_send_wr", &mori::io::RdmaBackendConfig::maxSendWr)
      .def_readwrite("max_cqe_num", &mori::io::RdmaBackendConfig::maxCqeNum)
      .def_readwrite("max_msg_sge", &mori::io::RdmaBackendConfig::maxMsgSge);

  py::class_<mori::io::XgmiBackendConfig, mori::io::BackendConfig>(m, "XgmiBackendConfig")
      .def(py::init<int, int>(), py::arg("num_streams") = 64, py::arg("num_events") = 64)
      .def_readwrite("num_streams", &mori::io::XgmiBackendConfig::numStreams)
      .def_readwrite("num_events", &mori::io::XgmiBackendConfig::numEvents);

  py::class_<mori::io::IOEngineConfig>(m, "IOEngineConfig")
      .def(py::init<std::string, uint16_t>(), py::arg("host") = "", py::arg("port") = 0)
      .def_readwrite("host", &mori::io::IOEngineConfig::host)
      .def_readwrite("port", &mori::io::IOEngineConfig::port);

  py::class_<mori::io::TransferStatus>(m, "TransferStatus")
      .def(py::init<>())
      .def("Code", &mori::io::TransferStatus::Code)
      .def("Message", &mori::io::TransferStatus::Message)
      .def("Update", &mori::io::TransferStatus::Update)
      .def("Init", &mori::io::TransferStatus::Init)
      .def("InProgress", &mori::io::TransferStatus::InProgress)
      .def("Succeeded", &mori::io::TransferStatus::Succeeded)
      .def("Failed", &mori::io::TransferStatus::Failed)
      .def("SetCode", &mori::io::TransferStatus::SetCode)
      .def("SetMessage", &mori::io::TransferStatus::SetMessage)
      .def("Wait", &mori::io ::TransferStatus::Wait);

  py::class_<mori::io::EngineDesc>(m, "EngineDesc")
      .def_readonly("key", &mori::io::EngineDesc::key)
      .def_readonly("node_id", &mori::io::EngineDesc::nodeId)
      .def_readonly("hostname", &mori::io::EngineDesc::hostname)
      .def_readonly("host", &mori::io::EngineDesc::host)
      .def_readonly("port", &mori::io::EngineDesc::port)
      .def_readonly("pid", &mori::io::EngineDesc::pid)
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::EngineDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::EngineDesc>();
      });

  py::class_<mori::io::MemoryDesc>(m, "MemoryDesc")
      .def(py::init<>())
      .def_readonly("engine_key", &mori::io::MemoryDesc::engineKey)
      .def_readonly("id", &mori::io::MemoryDesc::id)
      .def_readonly("device_id", &mori::io::MemoryDesc::deviceId)
      .def_readonly("device_bus_id", &mori::io::MemoryDesc::deviceBusId)
      .def_property_readonly("data",
                             [](const mori::io::MemoryDesc& desc) -> uintptr_t {
                               return reinterpret_cast<uintptr_t>(desc.data);
                             })
      .def_readonly("size", &mori::io::MemoryDesc::size)
      .def_readonly("loc", &mori::io::MemoryDesc::loc)
      .def_property_readonly("ipc_handle",
                             [](const mori::io::MemoryDesc& desc) {
                               return py::bytes(desc.ipcHandle.data(), desc.ipcHandle.size());
                             })
      .def(pybind11::self == pybind11::self)
      .def("pack",
           [](const mori::io::MemoryDesc& d) {
             msgpack::sbuffer buf;
             msgpack::pack(buf, d);
             return py::bytes(buf.data(), buf.size());
           })
      .def_static("unpack", [](const py::bytes& b) {
        Py_ssize_t len = PyBytes_Size(b.ptr());
        const char* data = PyBytes_AsString(b.ptr());
        auto out = msgpack::unpack(data, len);
        return out.get().as<mori::io::MemoryDesc>();
      });

  py::class_<mori::io::IOEngineSession>(m, "IOEngineSession")
      .def("AllocateTransferUniqueId", &mori::io::IOEngineSession::AllocateTransferUniqueId,
           py::call_guard<py::gil_scoped_release>())
      .def("Read", &mori::io::IOEngineSession::Read, py::call_guard<py::gil_scoped_release>())
      .def("BatchRead", &mori::io::IOEngineSession::BatchRead,
           py::call_guard<py::gil_scoped_release>())
      .def("Write", &mori::io::IOEngineSession::Write, py::call_guard<py::gil_scoped_release>())
      .def("BatchWrite", &mori::io::IOEngineSession::BatchWrite,
           py::call_guard<py::gil_scoped_release>())
      .def("Alive", &mori::io::IOEngineSession::Alive);

  py::class_<mori::io::IOEngine>(m, "IOEngine")
      .def(py::init<const mori::io::EngineKey&, const mori::io::IOEngineConfig&>())
      .def("GetEngineDesc", &mori::io::IOEngine::GetEngineDesc)
      .def("CreateBackend", &mori::io::IOEngine::CreateBackend)
      .def("RemoveBackend", &mori::io::IOEngine::RemoveBackend)
      .def("RegisterRemoteEngine", &mori::io::IOEngine::RegisterRemoteEngine)
      .def("DeregisterRemoteEngine", &mori::io::IOEngine::DeregisterRemoteEngine)
      .def("RegisterMemory", &mori::io::IOEngine::RegisterMemory)
      .def("DeregisterMemory", &mori::io::IOEngine::DeregisterMemory)
      .def("AllocateTransferUniqueId", &mori::io::IOEngine::AllocateTransferUniqueId,
           py::call_guard<py::gil_scoped_release>())
      .def("Read", &mori::io::IOEngine::Read, py::call_guard<py::gil_scoped_release>())
      .def("BatchRead", &mori::io::IOEngine::BatchRead, py::call_guard<py::gil_scoped_release>())
      .def("Write", &mori::io::IOEngine::Write, py::call_guard<py::gil_scoped_release>())
      .def("BatchWrite", &mori::io::IOEngine::BatchWrite, py::call_guard<py::gil_scoped_release>())
      .def("CreateSession", &mori::io::IOEngine::CreateSession)
      .def("PopInboundTransferStatus", &mori::io::IOEngine::PopInboundTransferStatus,
           py::call_guard<py::gil_scoped_release>())
      .def("LoadScatterGatherModule", &mori::io::IOEngine::LoadScatterGatherModule)
      .def(
          "GetTelemetrySnapshot",
          [](const mori::io::IOEngine& self, mori::io::BackendType type) -> py::dict {
            auto snap = self.GetTelemetrySnapshot(type);
            auto verbs_to_dict = [](const mori::io::TelemetrySnapshot::VerbsStats& v) -> py::dict {
              py::dict d;
              d["bytes_posted"] = v.bytes_posted;
              d["bytes_completed"] = v.bytes_completed;
              d["ops_posted"] = v.ops_posted;
              d["ops_completed"] = v.ops_completed;
              d["cqe_errors"] = v.cqe_errors;
              d["post_send_errors"] = v.post_send_errors;
              d["sq_full_events"] = v.sq_full_events;
              d["cq_polls"] = v.cq_polls;
              d["cq_empty_polls"] = v.cq_empty_polls;
              d["cq_utilization"] = v.cq_utilization;
              d["poll_mode"] = v.poll_mode;
              d["sq_depth"] = v.sq_depth;
              d["sq_depth_hwm"] = v.sq_depth_hwm;
              d["sq_depth_max"] = v.sq_depth_max;
              d["rdma_latency_ewma_us"] = v.rdma_latency_ewma_us;
              d["rdma_latency_min_us"] = v.rdma_latency_min_us;
              d["rdma_latency_max_us"] = v.rdma_latency_max_us;
              d["rdma_latency_sample_count"] = v.rdma_latency_sample_count;
              return d;
            };

            py::dict result;

            // API stats
            py::dict api;
            api["read_calls"] = snap.api.read_calls;
            api["write_calls"] = snap.api.write_calls;
            api["batch_read_calls"] = snap.api.batch_read_calls;
            api["batch_write_calls"] = snap.api.batch_write_calls;
            api["rejected_calls"] = snap.api.rejected_calls;
            api["total_bytes_read"] = snap.api.total_bytes_read;
            api["total_bytes_written"] = snap.api.total_bytes_written;
            api["jct_ewma_us"] = snap.api.jct_ewma_us;
            api["jct_min_us"] = snap.api.jct_min_us;
            api["jct_max_us"] = snap.api.jct_max_us;
            api["jct_sample_count"] = snap.api.jct_sample_count;
            result["api"] = api;

            // Verbs stats
            result["verbs"] = verbs_to_dict(snap.verbs);

            // Breakdown
            py::dict breakdown;
            breakdown["api_total_ewma_us"] = snap.breakdown.api_total_ewma_us;
            breakdown["rdma_wall_ewma_us"] = snap.breakdown.rdma_wall_ewma_us;
            breakdown["sw_overhead_ewma_us"] = snap.breakdown.sw_overhead_ewma_us;
            result["breakdown"] = breakdown;

            // Per-EP
            py::list per_ep;
            for (const auto& ep : snap.per_ep) {
              py::dict ep_dict;
              ep_dict["ep_id"] = ep.ep_id;
              ep_dict["verbs"] = verbs_to_dict(ep.verbs);
              per_ep.append(ep_dict);
            }
            result["per_ep"] = per_ep;

            result["snapshot_tsc"] = snap.snapshot_tsc;
            result["tsc_freq_ghz"] = snap.tsc_freq_ghz;

            return result;
          },
          py::arg("backend_type") = mori::io::BackendType::RDMA);
}

}  // namespace mori

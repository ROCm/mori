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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "src/pybind/mori.hpp"
#include "umbp/common/config.h"
#include "umbp/distributed/config.h"
#include "umbp/distributed/distributed_client.h"
#include "umbp/distributed/master/master_client.h"
#include "umbp/distributed/types.h"
#include "umbp/umbp_client.h"

namespace py = pybind11;

namespace mori {
using namespace umbp;
void RegisterMoriUmbp(py::module_& m) {
  py::enum_<TierType>(m, "UMBPTierType")
      .value("Unknown", TierType::UNKNOWN)
      .value("HBM", TierType::HBM)
      .value("DRAM", TierType::DRAM)
      .value("SSD", TierType::SSD)
      .export_values();

  py::class_<IUMBPClient::ExternalKvMatch>(m, "UMBPExternalKvMatch")
      .def(py::init<>())
      .def_readwrite("node_id", &IUMBPClient::ExternalKvMatch::node_id)
      .def_readwrite("peer_address", &IUMBPClient::ExternalKvMatch::peer_address)
      .def_readwrite("matched_hashes", &IUMBPClient::ExternalKvMatch::matched_hashes)
      .def_readwrite("tier", &IUMBPClient::ExternalKvMatch::tier)
      .def("__repr__", [](const IUMBPClient::ExternalKvMatch& m) {
        return "<UMBPExternalKvMatch node_id='" + m.node_id + "' matched=" +
               std::to_string(m.matched_hashes.size()) + ">";
      });

  py::enum_<UMBPRole>(m, "UMBPRole")
      .value("Standalone", UMBPRole::Standalone)
      .value("SharedSSDLeader", UMBPRole::SharedSSDLeader)
      .value("SharedSSDFollower", UMBPRole::SharedSSDFollower)
      .export_values();

  py::enum_<UMBPSsdLayoutMode>(m, "UMBPSsdLayoutMode")
      .value("SegmentedLog", UMBPSsdLayoutMode::SegmentedLog)
      .export_values();

  py::enum_<UMBPIoBackend>(m, "UMBPIoBackend")
      .value("PThread", UMBPIoBackend::PThread)
      .value("IoUring", UMBPIoBackend::IoUring)
      .export_values();

  py::enum_<UMBPDurabilityMode>(m, "UMBPDurabilityMode")
      .value("Strict", UMBPDurabilityMode::Strict)
      .value("Relaxed", UMBPDurabilityMode::Relaxed)
      .export_values();

  py::class_<UMBPDramConfig>(m, "UMBPDramConfig")
      .def(py::init<>())
      .def_readwrite("capacity_bytes", &UMBPDramConfig::capacity_bytes)
      .def_readwrite("use_shared_memory", &UMBPDramConfig::use_shared_memory)
      .def_readwrite("shm_name", &UMBPDramConfig::shm_name)
      .def_readwrite("high_watermark", &UMBPDramConfig::high_watermark)
      .def_readwrite("low_watermark", &UMBPDramConfig::low_watermark);

  py::class_<UMBPIoConfig>(m, "UMBPIoConfig")
      .def(py::init<>())
      .def_readwrite("backend", &UMBPIoConfig::backend)
      .def_readwrite("queue_depth", &UMBPIoConfig::queue_depth);

  py::class_<UMBPDurabilityConfig>(m, "UMBPDurabilityConfig")
      .def(py::init<>())
      .def_readwrite("mode", &UMBPDurabilityConfig::mode)
      .def_readwrite("enable_background_gc", &UMBPDurabilityConfig::enable_background_gc);

  py::class_<UMBPSsdConfig>(m, "UMBPSsdConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &UMBPSsdConfig::enabled)
      .def_readwrite("storage_dir", &UMBPSsdConfig::storage_dir)
      .def_readwrite("capacity_bytes", &UMBPSsdConfig::capacity_bytes)
      .def_readwrite("layout_mode", &UMBPSsdConfig::layout_mode)
      .def_readwrite("segment_size_bytes", &UMBPSsdConfig::segment_size_bytes)
      .def_readwrite("io", &UMBPSsdConfig::io)
      .def_readwrite("durability", &UMBPSsdConfig::durability);

  py::class_<UMBPEvictionConfig>(m, "UMBPEvictionConfig")
      .def(py::init<>())
      .def_readwrite("policy", &UMBPEvictionConfig::policy)
      .def_readwrite("candidate_window", &UMBPEvictionConfig::candidate_window)
      .def_readwrite("auto_promote_on_read", &UMBPEvictionConfig::auto_promote_on_read);

  py::class_<UMBPCopyPipelineConfig>(m, "UMBPCopyPipelineConfig")
      .def(py::init<>())
      .def_readwrite("async_enabled", &UMBPCopyPipelineConfig::async_enabled)
      .def_readwrite("queue_depth", &UMBPCopyPipelineConfig::queue_depth)
      .def_readwrite("worker_threads", &UMBPCopyPipelineConfig::worker_threads)
      .def_readwrite("batch_max_ops", &UMBPCopyPipelineConfig::batch_max_ops);

  py::class_<UMBPMasterClientConfig>(m, "UMBPMasterClientConfig")
      .def(py::init<>())
      .def_readwrite("master_address", &UMBPMasterClientConfig::master_address)
      .def_readwrite("node_id", &UMBPMasterClientConfig::node_id)
      .def_readwrite("node_address", &UMBPMasterClientConfig::node_address)
      .def_readwrite("auto_heartbeat", &UMBPMasterClientConfig::auto_heartbeat);

  py::class_<UMBPIoEngineConfig>(m, "UMBPIoEngineConfig")
      .def(py::init<>())
      .def_readwrite("host", &UMBPIoEngineConfig::host)
      .def_readwrite("port", &UMBPIoEngineConfig::port);

  py::class_<UMBPDistributedConfig>(m, "UMBPDistributedConfig")
      .def(py::init<>())
      .def_readwrite("master_config", &UMBPDistributedConfig::master_config)
      .def_readwrite("io_engine", &UMBPDistributedConfig::io_engine)
      .def_readwrite("staging_buffer_size", &UMBPDistributedConfig::staging_buffer_size)
      .def_readwrite("peer_service_port", &UMBPDistributedConfig::peer_service_port)
      .def_readwrite("cache_remote_fetches", &UMBPDistributedConfig::cache_remote_fetches)
      .def_readwrite("dram_page_size", &UMBPDistributedConfig::dram_page_size);

  py::class_<UMBPConfig>(m, "UMBPConfig")
      .def(py::init<>())
      .def_static("from_environment", &UMBPConfig::FromEnvironment)
      .def_readwrite("dram", &UMBPConfig::dram)
      .def_readwrite("ssd", &UMBPConfig::ssd)
      .def_readwrite("eviction", &UMBPConfig::eviction)
      .def_readwrite("copy_pipeline", &UMBPConfig::copy_pipeline)
      .def_readwrite("role", &UMBPConfig::role)
      .def_readwrite("follower_mode", &UMBPConfig::follower_mode)
      .def_readwrite("force_ssd_copy_on_write", &UMBPConfig::force_ssd_copy_on_write)
      .def_readwrite("ssd_backend", &UMBPConfig::ssd_backend)
      .def_readwrite("spdk_nvme_pci_addr", &UMBPConfig::spdk_nvme_pci_addr)
      .def_readwrite("spdk_proxy_shm_name", &UMBPConfig::spdk_proxy_shm_name)
      .def_readwrite("spdk_proxy_tenant_id", &UMBPConfig::spdk_proxy_tenant_id)
      .def_readwrite("spdk_proxy_tenant_quota_bytes", &UMBPConfig::spdk_proxy_tenant_quota_bytes)
      .def_readwrite("spdk_proxy_max_channels", &UMBPConfig::spdk_proxy_max_channels)
      .def_readwrite("spdk_proxy_data_per_channel_mb", &UMBPConfig::spdk_proxy_data_per_channel_mb)
      .def_readwrite("spdk_proxy_startup_timeout_ms", &UMBPConfig::spdk_proxy_startup_timeout_ms)
      .def_readwrite("spdk_proxy_auto_start", &UMBPConfig::spdk_proxy_auto_start)
      .def_readwrite("spdk_proxy_idle_exit_timeout_ms",
                     &UMBPConfig::spdk_proxy_idle_exit_timeout_ms)
      .def_readwrite("spdk_proxy_allow_borrow", &UMBPConfig::spdk_proxy_allow_borrow)
      .def_readwrite("spdk_proxy_reserved_shared_bytes",
                     &UMBPConfig::spdk_proxy_reserved_shared_bytes)
      .def_readwrite("distributed", &UMBPConfig::distributed);

  py::class_<IUMBPClient, std::unique_ptr<IUMBPClient>>(m, "UMBPClient")
      .def(py::init([](const UMBPConfig& cfg) { return CreateUMBPClient(cfg); }),
           py::arg("config") = UMBPConfig{})
      .def("put_from_ptr", &IUMBPClient::Put, py::arg("key"), py::arg("src"), py::arg("size"))
      .def("get_into_ptr", &IUMBPClient::Get, py::arg("key"), py::arg("dst"), py::arg("size"))
      .def("exists", &IUMBPClient::Exists, py::arg("key"))
      .def("batch_put_from_ptr", &IUMBPClient::BatchPut, py::arg("keys"), py::arg("ptrs"),
           py::arg("sizes"))
      .def("batch_put_from_ptr_with_depth", &IUMBPClient::BatchPutWithDepth, py::arg("keys"),
           py::arg("ptrs"), py::arg("sizes"), py::arg("depths"))
      .def("batch_get_into_ptr", &IUMBPClient::BatchGet, py::arg("keys"), py::arg("ptrs"),
           py::arg("sizes"))
      .def("batch_exists", &IUMBPClient::BatchExists, py::arg("keys"))
      .def("batch_exists_consecutive", &IUMBPClient::BatchExistsConsecutive, py::arg("keys"))
      .def("clear", &IUMBPClient::Clear)
      .def("flush", &IUMBPClient::Flush)
      .def("is_distributed", &IUMBPClient::IsDistributed)
      .def("register_memory", &IUMBPClient::RegisterMemory, py::arg("ptr"), py::arg("size"))
      .def("deregister_memory", &IUMBPClient::DeregisterMemory, py::arg("ptr"))
      .def("report_external_kv_blocks", &IUMBPClient::ReportExternalKvBlocks,
           py::arg("hashes"), py::arg("tier"))
      .def("revoke_external_kv_blocks", &IUMBPClient::RevokeExternalKvBlocks,
           py::arg("hashes"))
      .def("match_external_kv", &IUMBPClient::MatchExternalKv,
           py::arg("hashes"));

  // UMBPMasterClient is a read-only query client for the UMBP master.
  // It is intended solely for information lookup (e.g. matching external KV
  // blocks) and does not register with the master, send heartbeats, or mutate
  // any master state.
  py::class_<MasterClient::ExternalKvNodeMatch>(m, "UMBPExternalKvNodeMatch")
      .def(py::init<>())
      .def_readwrite("node_id", &MasterClient::ExternalKvNodeMatch::node_id)
      .def_readwrite("peer_address", &MasterClient::ExternalKvNodeMatch::peer_address)
      .def_readwrite("matched_hashes", &MasterClient::ExternalKvNodeMatch::matched_hashes)
      .def_readwrite("tier", &MasterClient::ExternalKvNodeMatch::tier)
      .def("__repr__", [](const MasterClient::ExternalKvNodeMatch& m) {
        return "<UMBPExternalKvNodeMatch node_id='" + m.node_id + "' matched=" +
               std::to_string(m.matched_hashes.size()) + ">";
      });

  py::class_<MasterClient>(m, "UMBPMasterClient")
      .def(py::init([](const std::string& master_address, const std::string& node_id,
                       const std::string& node_address) {
             MasterClientConfig cfg;
             cfg.master_address = master_address;
             cfg.node_id = node_id;
             cfg.node_address = node_address;
             cfg.auto_heartbeat = false;
             return std::make_unique<MasterClient>(cfg);
           }),
           py::arg("master_address"), py::arg("node_id") = std::string{},
           py::arg("node_address") = std::string{})
      .def(
          "register_self",
          [](MasterClient& self,
             const std::map<TierType, std::pair<uint64_t, uint64_t>>& tier_capacities) {
            std::map<TierType, TierCapacity> caps;
            for (const auto& [tier, total_avail] : tier_capacities) {
              caps[tier] = {total_avail.first, total_avail.second};
            }
            auto status = self.RegisterSelf(caps);
            if (!status.ok())
              throw std::runtime_error("RegisterSelf failed: " + status.error_message());
          },
          py::arg("tier_capacities") = std::map<TierType, std::pair<uint64_t, uint64_t>>{})
      .def("unregister_self",
           [](MasterClient& self) {
             auto status = self.UnregisterSelf();
             if (!status.ok())
               throw std::runtime_error("UnregisterSelf failed: " + status.error_message());
           })
      .def("is_registered", &MasterClient::IsRegistered)
      .def(
          "report_external_kv_blocks",
          [](MasterClient& self, const std::string& node_id,
             const std::vector<std::string>& hashes, TierType tier) {
            auto status = self.ReportExternalKvBlocks(node_id, hashes, tier);
            if (!status.ok())
              throw std::runtime_error("ReportExternalKvBlocks failed: " +
                                       status.error_message());
          },
          py::arg("node_id"), py::arg("hashes"), py::arg("tier"))
      .def(
          "revoke_external_kv_blocks",
          [](MasterClient& self, const std::string& node_id,
             const std::vector<std::string>& hashes) {
            auto status = self.RevokeExternalKvBlocks(node_id, hashes);
            if (!status.ok())
              throw std::runtime_error("RevokeExternalKvBlocks failed: " +
                                       status.error_message());
          },
          py::arg("node_id"), py::arg("hashes"))
      .def(
          "match_external_kv",
          [](MasterClient& self, const std::vector<std::string>& hashes) {
            std::vector<MasterClient::ExternalKvNodeMatch> matches;
            auto status = self.MatchExternalKv(hashes, &matches);
            if (!status.ok())
              throw std::runtime_error("MatchExternalKv failed: " +
                                       status.error_message());
            return matches;
          },
          py::arg("hashes"));
}

}  // namespace mori

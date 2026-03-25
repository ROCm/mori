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
#include "umbp/local/umbp_client.h"

namespace py = pybind11;

namespace mori {
void RegisterMoriUmbp(py::module_& m) {
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
                     &UMBPConfig::spdk_proxy_reserved_shared_bytes);

  py::class_<UMBPClient>(m, "UMBPClient")
      .def(py::init<const UMBPConfig&>(), py::arg("config") = UMBPConfig{})
      .def("put_from_ptr", &UMBPClient::PutFromPtr, py::arg("key"), py::arg("src"), py::arg("size"))
      .def("get_into_ptr", &UMBPClient::GetIntoPtr, py::arg("key"), py::arg("dst"), py::arg("size"))
      .def("exists", &UMBPClient::Exists, py::arg("key"))
      .def("remove", &UMBPClient::Remove, py::arg("key"))
      .def("batch_put_from_ptr", &UMBPClient::BatchPutFromPtr, py::arg("keys"), py::arg("ptrs"),
           py::arg("sizes"))
      .def("batch_put_from_ptr_with_depth", &UMBPClient::BatchPutFromPtrWithDepth, py::arg("keys"),
           py::arg("ptrs"), py::arg("sizes"), py::arg("depths"))
      .def("batch_get_into_ptr", &UMBPClient::BatchGetIntoPtr, py::arg("keys"), py::arg("ptrs"),
           py::arg("sizes"))
      .def("batch_exists", &UMBPClient::BatchExists, py::arg("keys"))
      .def("batch_exists_consecutive", &UMBPClient::BatchExistsConsecutive, py::arg("keys"))
      .def("clear", &UMBPClient::Clear);
}

}  // namespace mori

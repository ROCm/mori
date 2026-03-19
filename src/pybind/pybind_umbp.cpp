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
#include "umbp/umbp_client.h"

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

  py::class_<UMBPConfig>(m, "UMBPConfig")
      .def(py::init<>())
      .def_readwrite("dram_capacity_bytes", &UMBPConfig::dram_capacity_bytes)
      .def_readwrite("ssd_enabled", &UMBPConfig::ssd_enabled)
      .def_readwrite("ssd_storage_dir", &UMBPConfig::ssd_storage_dir)
      .def_readwrite("ssd_capacity_bytes", &UMBPConfig::ssd_capacity_bytes)
      .def_readwrite("ssd_io_backend", &UMBPConfig::ssd_io_backend)
      .def_readwrite("ssd_durability_mode", &UMBPConfig::ssd_durability_mode)
      .def_readwrite("ssd_segment_size_bytes", &UMBPConfig::ssd_segment_size_bytes)
      .def_readwrite("ssd_batch_max_ops", &UMBPConfig::ssd_batch_max_ops)
      .def_readwrite("ssd_queue_depth", &UMBPConfig::ssd_queue_depth)
      .def_readwrite("ssd_writer_threads", &UMBPConfig::ssd_writer_threads)
      .def_readwrite("ssd_enable_background_gc", &UMBPConfig::ssd_enable_background_gc)
      .def_readwrite("eviction_policy", &UMBPConfig::eviction_policy)
      .def_readwrite("auto_promote_on_read", &UMBPConfig::auto_promote_on_read)
      .def_readwrite("use_shared_memory", &UMBPConfig::use_shared_memory)
      .def_readwrite("shm_name", &UMBPConfig::shm_name)
      .def_readwrite("dram_high_watermark", &UMBPConfig::dram_high_watermark)
      .def_readwrite("dram_low_watermark", &UMBPConfig::dram_low_watermark)
      .def_readwrite("role", &UMBPConfig::role)
      .def_readwrite("follower_mode", &UMBPConfig::follower_mode)
      .def_readwrite("force_ssd_copy_on_write", &UMBPConfig::force_ssd_copy_on_write)
      .def_readwrite("copy_to_ssd_async", &UMBPConfig::copy_to_ssd_async)
      .def_readwrite("copy_to_ssd_queue_depth", &UMBPConfig::copy_to_ssd_queue_depth)
      .def_readwrite("eviction_candidate_window", &UMBPConfig::eviction_candidate_window);

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

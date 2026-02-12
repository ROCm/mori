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

// Framework-agnostic pybind bindings for MoRI shmem and IO APIs.
// No Torch dependency -- only pybind11, HIP, and mori core libraries.

#include "src/pybind/mori_core.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/io/io.hpp"
#include "mori/shmem/shmem_host_api.hpp"

namespace py = pybind11;

/* ---------------------------------------------------------------------------------------------- */
/*                                    Shmem wrapper functions                                     */
/* ---------------------------------------------------------------------------------------------- */
namespace {

int64_t ShmemFinalize() { return mori::shmem::ShmemFinalize(); }

int64_t ShmemModuleInit(uint64_t hipModule) {
  return mori::shmem::ShmemModuleInit(reinterpret_cast<void*>(hipModule));
}

int64_t ShmemMyPe() { return mori::shmem::ShmemMyPe(); }

int64_t ShmemNPes() { return mori::shmem::ShmemNPes(); }

// UniqueId-based initialization APIs
py::bytes ShmemGetUniqueId() {
  mori::shmem::mori_shmem_uniqueid_t uid;
  mori::shmem::ShmemGetUniqueId(&uid);
  return py::bytes(reinterpret_cast<const char*>(uid.data()), uid.size());
}

int64_t ShmemInitAttr(unsigned int flags, int32_t rank, int32_t nranks,
                      const py::bytes& uid_bytes) {
  mori::shmem::mori_shmem_init_attr_t attr;
  mori::shmem::mori_shmem_uniqueid_t uid;

  // Convert Python bytes to uniqueid
  Py_ssize_t len = PyBytes_Size(uid_bytes.ptr());
  const char* data = PyBytes_AsString(uid_bytes.ptr());
  if (len != MORI_SHMEM_UNIQUE_ID_BYTES) {
    throw std::runtime_error("Invalid unique ID size");
  }
  std::memcpy(uid.data(), data, MORI_SHMEM_UNIQUE_ID_BYTES);

  // Set attributes
  mori::shmem::ShmemSetAttrUniqueIdArgs(rank, nranks, &uid, &attr);

  return mori::shmem::ShmemInitAttr(flags, &attr);
}

void ShmemBarrierAll() { mori::shmem::ShmemBarrierAll(); }

// Symmetric memory APIs
uintptr_t ShmemMalloc(size_t size) {
  void* ptr = mori::shmem::ShmemMalloc(size);
  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t ShmemMallocAlign(size_t alignment, size_t size) {
  void* ptr = mori::shmem::ShmemMallocAlign(alignment, size);
  return reinterpret_cast<uintptr_t>(ptr);
}

uintptr_t ShmemExtMallocWithFlags(size_t size, unsigned int flags) {
  void* ptr = mori::shmem::ShmemExtMallocWithFlags(size, flags);
  return reinterpret_cast<uintptr_t>(ptr);
}

void ShmemFree(uintptr_t ptr) { mori::shmem::ShmemFree(reinterpret_cast<void*>(ptr)); }

int64_t ShmemBufferRegister(uintptr_t ptr, size_t size) {
  return mori::shmem::ShmemBufferRegister(reinterpret_cast<void*>(ptr), size);
}

int64_t ShmemBufferDeregister(uintptr_t ptr, size_t size) {
  return mori::shmem::ShmemBufferDeregister(reinterpret_cast<void*>(ptr), size);
}

// P2P address translation
uint64_t ShmemPtrP2p(uint64_t destPtr, int myPe, int destPe) {
  return mori::shmem::ShmemPtrP2p(destPtr, myPe, destPe);
}

int64_t ShmemNumQpPerPe() { return mori::shmem::ShmemNumQpPerPe(); }

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                               Registration functions                                           */
/* ---------------------------------------------------------------------------------------------- */

namespace mori {

void RegisterMoriCoreShmem(py::module_& m) {
  // Initialization flags
  m.attr("MORI_SHMEM_INIT_WITH_MPI_COMM") = mori::shmem::MORI_SHMEM_INIT_WITH_MPI_COMM;
  m.attr("MORI_SHMEM_INIT_WITH_UNIQUEID") = mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;

  // UniqueId-based initialization APIs (nvshmem/rocshmem compatible)
  m.def("shmem_get_unique_id", &ShmemGetUniqueId,
        "Get a unique ID for shmem initialization (returns bytes)");

  m.def("shmem_init_attr", &ShmemInitAttr, py::arg("flags"), py::arg("rank"), py::arg("nranks"),
        py::arg("unique_id"),
        "Initialize shmem with attributes (unique_id should be bytes from shmem_get_unique_id)");

  m.def("shmem_finalize", &ShmemFinalize, "Finalize shmem");

  //  Module-specific initialization (for Triton kernels)
  m.def("shmem_module_init", &ShmemModuleInit, py::arg("hip_module"),
        "Initialize globalGpuStates in a specific HIP module (for Triton kernels)");

  // Query APIs
  m.def("shmem_mype", &ShmemMyPe, "Get my PE (process element) ID");
  m.def("shmem_npes", &ShmemNPes, "Get number of PEs");

  // Collective operations
  m.def("shmem_barrier_all", &ShmemBarrierAll, "Global barrier synchronization");

  // Symmetric memory management
  m.def("shmem_malloc", &ShmemMalloc, py::arg("size"),
        "Allocate symmetric memory (returns address as int)");

  m.def("shmem_malloc_align", &ShmemMallocAlign, py::arg("alignment"), py::arg("size"),
        "Allocate aligned symmetric memory (returns address as int)");

  m.def("shmem_ext_malloc_with_flags", &ShmemExtMallocWithFlags, py::arg("size"), py::arg("flags"),
        "Allocate symmetric memory with flags (returns address as int)");

  m.def("shmem_free", &ShmemFree, py::arg("ptr"),
        "Free symmetric memory (ptr should be int address)");

  // Buffer registration
  m.def("shmem_buffer_register", &ShmemBufferRegister, py::arg("ptr"), py::arg("size"),
        "Register an existing buffer for RDMA (ptr should be int address)");

  m.def("shmem_buffer_deregister", &ShmemBufferDeregister, py::arg("ptr"), py::arg("size"),
        "Deregister a buffer from RDMA (ptr should be int address)");

  // P2P address translation
  m.def("shmem_ptr_p2p", &ShmemPtrP2p, py::arg("dest_ptr"), py::arg("my_pe"), py::arg("dest_pe"),
        "Convert local symmetric memory pointer to remote P2P address. "
        "Returns 0 if connection uses RDMA or if pointer is invalid. "
        "Returns P2P accessible address if connection uses P2P transport.");

  m.def("shmem_num_qp_per_pe", &ShmemNumQpPerPe);
}

void RegisterMoriCoreIo(pybind11::module_& m) {
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
      .def(py::init<int, int, int, mori::io::PollCqMode, bool>(), py::arg("qp_per_transfer") = 1,
           py::arg("post_batch_size") = -1, py::arg("num_worker_threads") = -1,
           py::arg("poll_cq_mode") = mori::io::PollCqMode::POLLING,
           py::arg("enable_notification") = true)
      .def_readwrite("qp_per_transfer", &mori::io::RdmaBackendConfig::qpPerTransfer)
      .def_readwrite("post_batch_size", &mori::io::RdmaBackendConfig::postBatchSize)
      .def_readwrite("num_worker_threads", &mori::io::RdmaBackendConfig::numWorkerThreads)
      .def_readwrite("poll_cq_mode", &mori::io::RdmaBackendConfig::pollCqMode)
      .def_readwrite("enable_notification", &mori::io::RdmaBackendConfig::enableNotification);

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
      .def("Wait", &mori::io::TransferStatus::Wait);

  py::class_<mori::io::EngineDesc>(m, "EngineDesc")
      .def_readonly("key", &mori::io::EngineDesc::key)
      .def_readonly("hostname", &mori::io::EngineDesc::hostname)
      .def_readonly("host", &mori::io::EngineDesc::host)
      .def_readonly("port", &mori::io::EngineDesc::port)
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
      .def("AllocateTransferUniqueId", &mori::io::IOEngineSession::AllocateTransferUniqueId)
      .def("Read", &mori::io::IOEngineSession::Read)
      .def("BatchRead", &mori::io::IOEngineSession::BatchRead)
      .def("Write", &mori::io::IOEngineSession::Write)
      .def("BatchWrite", &mori::io::IOEngineSession::BatchWrite)
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
      .def("AllocateTransferUniqueId", &mori::io::IOEngine::AllocateTransferUniqueId)
      .def("Read", &mori::io::IOEngine::Read)
      .def("BatchRead", &mori::io::IOEngine::BatchRead)
      .def("Write", &mori::io::IOEngine::Write)
      .def("BatchWrite", &mori::io::IOEngine::BatchWrite)
      .def("CreateSession", &mori::io::IOEngine::CreateSession)
      .def("PopInboundTransferStatus", &mori::io::IOEngine::PopInboundTransferStatus);
}

}  // namespace mori

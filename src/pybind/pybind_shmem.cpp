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
#include "src/pybind/mori.hpp"

#include <pybind11/pybind11.h>

#include "mori/shmem/shmem.hpp"

namespace py = pybind11;

/* ---------------------------------------------------------------------------------------------- */
/*                                           Shmem APIs                                           */
/* ---------------------------------------------------------------------------------------------- */
namespace {
int64_t ShmemTorchProcessGroupInit(const std::string& groupName) {
  return mori::shmem::ShmemTorchProcessGroupInit(groupName);
}

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
void ShmemBarrierOnStream(hipStream_t stream) { mori::shmem::ShmemBarrierOnStream(stream); }

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


namespace mori {
void RegisterMoriShmem(py::module_& m) {
  // Initialization flags
  m.attr("MORI_SHMEM_INIT_WITH_MPI_COMM") = mori::shmem::MORI_SHMEM_INIT_WITH_MPI_COMM;
  m.attr("MORI_SHMEM_INIT_WITH_UNIQUEID") = mori::shmem::MORI_SHMEM_INIT_WITH_UNIQUEID;

  // Traditional initialization APIs
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit, py::arg("group_name"),
        "Initialize shmem from PyTorch process group");

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

  m.def(
      "shmem_barrier_on_stream",
      [](int64_t stream) { ShmemBarrierOnStream(reinterpret_cast<hipStream_t>(stream)); },
      py::arg("stream"), "Launch device barrier on a HIP stream");

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
  m.def("shmem_torch_process_group_init", &ShmemTorchProcessGroupInit);
  m.def("shmem_finalize", &ShmemFinalize);
  m.def("shmem_mype", &ShmemMyPe);
  m.def("shmem_npes", &ShmemNPes);
  m.def("shmem_num_qp_per_pe", &ShmemNumQpPerPe);
}


}  // namespace mori

// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>

#include "src/pybind/mori.hpp"
#include "umbp/engine/kv_allocator.h"

namespace py = pybind11;

namespace mori {

void RegisterMoriUmbpEngine(py::module_& parent) {
  auto m = parent.def_submodule("engine", "UMBP engine-owned KV memory primitives");
  using namespace umbp::engine;

  py::enum_<VmmHandleKind>(m, "VmmHandleKind")
      .value("PosixFd", VmmHandleKind::kPosixFd)
      .value("Fabric", VmmHandleKind::kFabric);

  m.def("allocator_stats", [] {
    auto& allocator = GlobalKvVmmAllocator();
    py::dict out;
    out["allocations"] = allocator.AllocationCount();
    out["quarantined"] = allocator.QuarantinedCount();
    out["bytes"] = allocator.AllocatedBytes();
    out["failures"] = allocator.FailureCount();
    return out;
  });
  m.def(
      "lookup_allocation",
      [](uintptr_t ptr, size_t length) -> py::object {
        auto allocation = GlobalKvVmmAllocator().Lookup(reinterpret_cast<void*>(ptr), length);
        if (!allocation.has_value()) return py::none();
        py::dict out;
        out["base"] = reinterpret_cast<uintptr_t>(allocation->base);
        out["requested_size"] = allocation->requested_size;
        out["mapped_size"] = allocation->mapped_size;
        out["device"] = allocation->device;
        out["handle_kind"] = allocation->handle_kind;
        return std::move(out);
      },
      py::arg("ptr"), py::arg("length") = 1);
  m.def(
      "export_allocation",
      [](uintptr_t ptr) -> py::object {
        auto exported = GlobalKvVmmAllocator().Export(reinterpret_cast<void*>(ptr));
        if (!exported.has_value()) return py::none();
        py::dict out;
        out["handle_kind"] = exported->kind;
        out["fd"] = exported->fd;
        out["fabric"] = py::bytes(reinterpret_cast<const char*>(exported->fabric.data()),
                                  exported->fabric.size());
        out["mapped_size"] = exported->mapped_size;
        return std::move(out);
      },
      py::arg("ptr"));
}

}  // namespace mori

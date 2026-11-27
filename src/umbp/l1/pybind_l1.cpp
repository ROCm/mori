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

// Python client for UMBP L1 block pools. Its own module, with only the L1
// proto in it, so an engine process loads nothing else of UMBP.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "umbp/l1/l1_fabric.h"
#include "umbp/l1/l1_service.h"

namespace py = pybind11;
using namespace mori::umbp::l1;

namespace {

class FetchInDoubtError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void Check(const Status& status) {
  if (!status.ok) throw std::runtime_error(status.error);
}

py::list BytesList(const std::vector<std::string>& values) {
  py::list out;
  for (const std::string& value : values) out.append(py::bytes(value));
  return out;
}

}  // namespace

PYBIND11_MODULE(mori_umbp_l1, m) {
  m.doc() = "Client for UMBP-owned L1 (GPU) KV block pools";
  m.attr("OP_HOLD") = static_cast<uint32_t>(OpKind::kHold);
  m.attr("OP_RELEASE") = static_cast<uint32_t>(OpKind::kRelease);
  m.attr("OP_PUBLISH") = static_cast<uint32_t>(OpKind::kPublish);
  m.attr("NATIVE_L1_FABRIC_VERSION") = 2;
  py::register_exception<FetchInDoubtError>(m, "FetchInDoubtError", PyExc_RuntimeError);

  py::class_<L1Client>(m, "L1Client")
      .def(py::init<const std::string&>(), py::arg("address"))
      .def(
          "attach",
          [](L1Client& client, const std::string& pool_id, const std::string& group, int device,
             uint64_t bytes, const py::bytes& external_handle, uint64_t external_offset,
             const std::string& external_kind) {
            ExternalMemory external{std::string(external_handle), external_offset, external_kind};
            AttachResult result;
            {
              py::gil_scoped_release release;
              result = client.Attach(pool_id, group, device, bytes, external);
            }
            Check(result.status);
            return py::make_tuple(result.memory.kind, py::bytes(result.memory.handle), result.bytes,
                                  result.created);
          },
          py::arg("pool_id"), py::arg("group"), py::arg("device"), py::arg("bytes"),
          py::arg("external_handle") = py::bytes(), py::arg("external_offset") = 0,
          py::arg("external_kind") = "",
          "Create or re-attach a pool. Returns (handle_kind, handle, bytes, created). With "
          "`external_handle` (a hip_ipc handle of the caller's allocation; on the host test "
          "server, its 8-byte address) the pool adopts that memory at `external_offset` "
          "instead of allocating: the server neither zeroes nor frees it. FABRIC callers "
          "pass external_kind='hip_fabric' and an export_fabric_memory descriptor.")
      .def(
          "bind_layout",
          [](L1Client& client, const std::string& pool_id, uint32_t num_slots,
             const std::vector<std::pair<uint64_t, uint64_t>>& regions) {
            Layout layout;
            layout.num_slots = num_slots;
            for (const auto& region : regions) {
              layout.regions.push_back(Region{region.first, region.second});
            }
            Status status;
            {
              py::gil_scoped_release release;
              status = client.BindLayout(pool_id, layout);
            }
            Check(status);
          },
          py::arg("pool_id"), py::arg("num_slots"), py::arg("regions"),
          "Declare the slot layout as (offset, unit_bytes) regions.")
      .def(
          "open",
          [](L1Client& client, const std::string& pool_id) {
            OpenResult result;
            {
              py::gil_scoped_release release;
              result = client.Open(pool_id);
            }
            Check(result.status);
            return result.num_slots;
          },
          py::arg("pool_id"), "Take over slot management; every slot becomes vacant.")
      .def(
          "allocate",
          [](L1Client& client, const std::string& pool_id, uint32_t count, uint32_t min_count) {
            AllocateResult result;
            {
              py::gil_scoped_release release;
              result = client.Allocate(pool_id, count, min_count);
            }
            Check(result.status);
            return py::make_tuple(result.slots, BytesList(result.evicted_keys));
          },
          py::arg("pool_id"), py::arg("count"), py::arg("min_count"),
          "Up to count slots, evicting cached content only to reach min_count. "
          "Returns (slots, evicted_keys).")
      .def(
          "apply",
          [](L1Client& client, const std::string& pool_id, const std::vector<uint32_t>& kinds,
             const std::vector<uint32_t>& slots, const std::vector<std::string>& keys,
             uint64_t stamp) {
            if (kinds.size() != slots.size()) {
              throw std::invalid_argument("kinds and slots differ in length");
            }
            std::vector<Op> ops;
            ops.reserve(kinds.size());
            size_t next_key = 0;
            for (size_t i = 0; i < kinds.size(); ++i) {
              Op op;
              op.kind = static_cast<OpKind>(kinds[i]);
              op.slot = slots[i];
              if (op.kind == OpKind::kPublish) {
                if (next_key >= keys.size()) throw std::invalid_argument("a publish has no key");
                op.key = keys[next_key++];
              }
              ops.push_back(std::move(op));
            }
            Status status;
            {
              py::gil_scoped_release release;
              status = client.Apply(pool_id, ops, stamp);
            }
            Check(status);
          },
          py::arg("pool_id"), py::arg("kinds"), py::arg("slots"), py::arg("keys"),
          py::arg("stamp") = 0,
          "Apply hold / release / publish ops in order; keys has one entry per publish. "
          "Slots published or released stay unsettled until commit(stamp or later).")
      .def(
          "commit",
          [](L1Client& client, const std::string& pool_id, uint64_t committed,
             uint32_t timeout_ms) {
            Status status;
            {
              py::gil_scoped_release release;
              status = client.Commit(pool_id, committed, timeout_ms);
            }
            Check(status);
          },
          py::arg("pool_id"), py::arg("committed"), py::arg("timeout_ms") = 120000,
          "The owner's device has finished every write before `committed`. "
          "timeout_ms bounds RPC waiting; a timeout does not establish cancellation or drain.")
      .def(
          "fetch",
          [](L1Client& client, const std::string& pool_id, const std::vector<std::string>& keys,
             const std::vector<uint32_t>& dst_slots, bool probe_only) {
            FetchOptions options;
            options.dst_slots = dst_slots;
            options.probe_only = probe_only;
            FetchResult result;
            {
              py::gil_scoped_release release;
              result = client.Fetch(pool_id, keys, options);
            }
            if (result.completion_unknown) throw FetchInDoubtError(result.status.error);
            Check(result.status);
            return py::make_tuple(result.slots, BytesList(result.evicted_keys), result.copied_slots,
                                  result.copied_bytes, result.copy_micros, result.available);
          },
          py::arg("pool_id"), py::arg("keys"), py::arg("dst_slots") = std::vector<uint32_t>{},
          py::arg("probe_only") = false,
          "Returns (slots, evicted_keys, copied_slots, copied_bytes, copy_micros, available). "
          "With `dst_slots` keys land in those held slots and nothing is evicted or "
          "indexed; with `probe_only` nothing is copied and `available` counts the "
          "leading keys the group holds.")
      .def(
          "clear",
          [](L1Client& client, const std::string& pool_id) {
            Status status;
            {
              py::gil_scoped_release release;
              status = client.Clear(pool_id);
            }
            Check(status);
          },
          py::arg("pool_id"))
      .def(
          "detach",
          [](L1Client& client, const std::string& pool_id) {
            Status status;
            {
              py::gil_scoped_release release;
              status = client.Detach(pool_id);
            }
            Check(status);
          },
          py::arg("pool_id"))
      .def("stats", [](L1Client& client) {
        std::vector<PoolStats> pools;
        Status status;
        {
          py::gil_scoped_release release;
          status = client.Stats(&pools);
        }
        Check(status);
        py::list out;
        for (const PoolStats& stats : pools) {
          py::dict d;
          d["pool_id"] = stats.pool_id;
          d["group"] = stats.group;
          d["device"] = stats.device;
          d["bytes"] = stats.bytes;
          d["num_slots"] = stats.num_slots;
          d["held"] = stats.held;
          d["cached"] = stats.cached;
          d["vacant"] = stats.vacant;
          d["evicted"] = stats.evicted;
          d["fetched_in_slots"] = stats.fetched_in_slots;
          d["fetched_in_bytes"] = stats.fetched_in_bytes;
          d["served_out_slots"] = stats.served_out_slots;
          d["copy_micros"] = stats.copy_micros;
          out.append(std::move(d));
        }
        return out;
      });

  // A server over host memory, for tests that exercise a client without a GPU.
  py::class_<L1Server>(m, "HostTestServer")
      .def(py::init([](const std::string& address) {
             return std::make_unique<L1Server>(address, MakeHostMemoryApi());
           }),
           py::arg("address"))
      .def("ok", &L1Server::ok);

  m.def(
      "export_fabric_memory",
      [](uintptr_t ptr, uint64_t bytes, int device) {
        ExportedMemory exported;
        std::string error;
        bool ok;
        {
          py::gil_scoped_release release;
          ok = ExportFabricMemory(device, reinterpret_cast<void*>(ptr), bytes, &exported, &error);
        }
        if (!ok) throw std::runtime_error(error);
        return py::bytes(exported.handle);
      },
      py::arg("ptr"), py::arg("bytes"), py::arg("device"),
      "Export native HBM allocated with mori.io.make_fabric_mem_pool; caller retains ownership.");

  m.def(
      "start_fabric_server",
      [](const std::string& address, const std::vector<std::string>& peers,
         const std::string& kernel_path) {
        py::gil_scoped_release release;
        auto server = std::make_unique<L1Server>(address, MakeFabricMemoryApi(kernel_path), peers);
        if (!server->ok()) throw std::runtime_error("could not start native L1 FABRIC server");
        return server;
      },
      py::arg("address"), py::arg("peers") = std::vector<std::string>{},
      py::arg("kernel_path") = "",
      "Start the L1-only FABRIC service; retain the returned server for its lifetime.");
}

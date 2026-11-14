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

#include "umbp/l1/l1_service.h"

namespace py = pybind11;
using namespace mori::umbp::l1;

namespace {

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

  py::class_<L1Client>(m, "L1Client")
      .def(py::init<const std::string&>(), py::arg("address"))
      .def(
          "attach",
          [](L1Client& client, const std::string& pool_id, const std::string& group, int device,
             uint64_t bytes) {
            AttachResult result;
            {
              py::gil_scoped_release release;
              result = client.Attach(pool_id, group, device, bytes);
            }
            Check(result.status);
            return py::make_tuple(result.memory.kind, py::bytes(result.memory.handle), result.bytes,
                                  result.created);
          },
          py::arg("pool_id"), py::arg("group"), py::arg("device"), py::arg("bytes"),
          "Create or re-attach a pool. Returns (handle_kind, handle, bytes, created).")
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
          [](L1Client& client, const std::string& pool_id, uint64_t committed) {
            Status status;
            {
              py::gil_scoped_release release;
              status = client.Commit(pool_id, committed);
            }
            Check(status);
          },
          py::arg("pool_id"), py::arg("committed"),
          "The owner's device has finished every write issued before `committed`.")
      .def(
          "fetch",
          [](L1Client& client, const std::string& pool_id, const std::vector<std::string>& keys) {
            FetchResult result;
            {
              py::gil_scoped_release release;
              result = client.Fetch(pool_id, keys);
            }
            Check(result.status);
            return py::make_tuple(result.slots, BytesList(result.evicted_keys), result.copied_slots,
                                  result.copied_bytes, result.copy_micros);
          },
          py::arg("pool_id"), py::arg("keys"),
          "Returns (slots, evicted_keys, copied_slots, copied_bytes, copy_micros).")
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
}

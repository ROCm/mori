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
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ep_op.hpp"
#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace py = pybind11;
using mori::kiwi::ep::ep_op_config_t;
using mori::kiwi::ep::ep_op_t;

namespace {

template <typename T>
T* as_ptr(uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

hipStream_t as_stream(uintptr_t value) { return reinterpret_cast<hipStream_t>(value); }

// BootstrapNetwork over a Python `allgather(payload: bytes) -> bytes`, which
// must return the rank-ordered concatenation of every rank's payload. It is
// called only from the constructor, which holds the GIL.
class callback_bootstrap_t : public mori::application::BootstrapNetwork {
 public:
  callback_bootstrap_t(int rank, int world_size, py::function allgather)
      : allgather_(std::move(allgather)) {
    localRank = rank;
    worldSize = world_size;
  }

  void Initialize() override {}
  void Finalize() override {}

  void Allgather(void* send, void* recv, size_t bytes) override {
    py::gil_scoped_acquire gil;
    py::bytes payload(static_cast<const char*>(send), bytes);
    std::string data = allgather_(payload).cast<py::bytes>();
    if (data.size() != bytes * static_cast<size_t>(worldSize)) {
      throw std::runtime_error("KIWI EP bootstrap allgather returned " +
                               std::to_string(data.size()) + " bytes, expected " +
                               std::to_string(bytes * static_cast<size_t>(worldSize)));
    }
    std::memcpy(recv, data.data(), data.size());
  }

  // Row r of the gathered matrix is rank r's send buffer; keep column localRank.
  void AllToAll(void* send, void* recv, size_t bytes) override {
    const size_t n = static_cast<size_t>(worldSize);
    std::vector<char> all(bytes * n * n);
    Allgather(send, all.data(), bytes * n);
    for (size_t r = 0; r < n; ++r) {
      std::memcpy(static_cast<char*>(recv) + r * bytes,
                  all.data() + (r * n + static_cast<size_t>(localRank)) * bytes, bytes);
    }
  }

  void Barrier() override {
    char token = 0;
    std::vector<char> tokens(static_cast<size_t>(worldSize));
    Allgather(&token, tokens.data(), sizeof(token));
  }

 private:
  py::function allgather_;
};

}  // namespace

PYBIND11_MODULE(libmori_kiwi_ep_pybinds, m) {
  m.doc() = "KIWI EP dispatch/combine kernels with an RDMA CPU proxy";

  py::class_<ep_op_config_t>(m, "KiwiEpOpConfig")
      .def(py::init<>())
      .def_readwrite("rank", &ep_op_config_t::rank)
      .def_readwrite("world_size", &ep_op_config_t::world_size)
      .def_readwrite("hidden_dim", &ep_op_config_t::hidden_dim)
      .def_readwrite("topk", &ep_op_config_t::topk)
      .def_readwrite("num_experts_per_rank", &ep_op_config_t::num_experts_per_rank)
      .def_readwrite("max_num_inp_token_per_rank", &ep_op_config_t::max_num_inp_token_per_rank)
      .def_readwrite("dispatch_elem_size", &ep_op_config_t::dispatch_elem_size)
      .def_readwrite("combine_elem_size", &ep_op_config_t::combine_elem_size)
      .def_readwrite("dispatch_quant_format", &ep_op_config_t::dispatch_quant_format)
      .def_readwrite("combine_quant_format", &ep_op_config_t::combine_quant_format)
      .def_readwrite("dispatch_scale_dim", &ep_op_config_t::dispatch_scale_dim)
      .def_readwrite("dispatch_scale_type_size", &ep_op_config_t::dispatch_scale_type_size)
      .def_readwrite("num_blocks", &ep_op_config_t::num_blocks)
      .def_readwrite("num_proxy_threads", &ep_op_config_t::num_proxy_threads)
      .def_readwrite("queue_capacity", &ep_op_config_t::queue_capacity)
      .def_readwrite("rcomp_backend", &ep_op_config_t::rcomp_backend)
      .def_readwrite("device_name", &ep_op_config_t::device_name);

  py::class_<ep_op_t>(m, "KiwiEpOpHandle")
      .def(py::init([](const ep_op_config_t& config, py::function allgather) {
             auto bootstrap = std::make_shared<callback_bootstrap_t>(config.rank, config.world_size,
                                                                     std::move(allgather));
             return std::make_unique<ep_op_t>(config, std::move(bootstrap));
           }),
           py::arg("config"), py::arg("allgather"))
      .def(
          "dispatch",
          [](ep_op_t& self, uintptr_t input, uintptr_t weights, uintptr_t scales, uintptr_t indices,
             int num_tokens, uintptr_t stream) {
            self.dispatch(as_ptr<const void>(input), as_ptr<const float>(weights),
                          as_ptr<const void>(scales), as_ptr<const int32_t>(indices), num_tokens,
                          as_stream(stream));
          },
          py::arg("input"), py::arg("weights"), py::arg("scales"), py::arg("indices"),
          py::arg("num_tokens"), py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def(
          "combine",
          [](ep_op_t& self, uintptr_t input, uintptr_t weights, uintptr_t indices, uintptr_t output,
             uintptr_t output_weights, int num_tokens, uintptr_t stream) {
            self.combine(as_ptr<const void>(input), as_ptr<const float>(weights),
                         as_ptr<const int32_t>(indices), as_ptr<void>(output),
                         as_ptr<float>(output_weights), num_tokens, as_stream(stream));
          },
          py::arg("input"), py::arg("weights"), py::arg("indices"), py::arg("output"),
          py::arg("output_weights"), py::arg("num_tokens"), py::arg("stream"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reset", [](ep_op_t& self, uintptr_t stream) { self.reset(as_stream(stream)); },
          py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def("recv_hidden_ptr",
           [](const ep_op_t& self) { return reinterpret_cast<uintptr_t>(self.recv_hidden()); })
      .def("recv_weights_ptr",
           [](const ep_op_t& self) { return reinterpret_cast<uintptr_t>(self.recv_weights()); })
      .def("recv_indices_ptr",
           [](const ep_op_t& self) { return reinterpret_cast<uintptr_t>(self.recv_indices()); })
      .def("recv_scales_ptr",
           [](const ep_op_t& self) { return reinterpret_cast<uintptr_t>(self.recv_scales()); })
      .def("recv_count_ptr",
           [](const ep_op_t& self) { return reinterpret_cast<uintptr_t>(self.recv_count()); })
      .def("recv_src_token_pos_ptr",
           [](const ep_op_t& self) {
             return reinterpret_cast<uintptr_t>(self.recv_src_token_pos());
           })
      .def(
          "copy_recv_count_to_host",
          [](const ep_op_t& self, uintptr_t stream) {
            return self.copy_recv_count_to_host(as_stream(stream));
          },
          py::arg("stream"), py::call_guard<py::gil_scoped_release>())
      .def("max_slots", &ep_op_t::max_slots)
      .def("dispatch_hidden_elems", &ep_op_t::dispatch_hidden_elems)
      .def("combine_hidden_elems", &ep_op_t::combine_hidden_elems)
      .def("max_num_tokens_to_recv", &ep_op_t::max_num_tokens_to_recv)
      .def("max_num_tokens_to_recv_per_rank", &ep_op_t::max_num_tokens_to_recv_per_rank)
      .def("max_num_tokens_to_send", &ep_op_t::max_num_tokens_to_send)
      .def("max_num_tokens_to_send_per_rank", &ep_op_t::max_num_tokens_to_send_per_rank)
      .def("num_input_tokens", &ep_op_t::num_input_tokens)
      .def("rank", &ep_op_t::rank)
      .def("world_size", &ep_op_t::world_size);
}

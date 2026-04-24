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

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mori/collective/all2all/oneshot_all2all_sdma_class.hpp"
#include "mori/collective/allgather/oneshot_allgather_sdma_class.hpp"
#include "mori/collective/allreduce/twoshot_allreduce_sdma_class.hpp"
#include "src/pybind/mori.hpp"

namespace py = pybind11;

namespace {
template <typename T>
void BindAllreduceHandle(py::module_& m, const char* python_name) {
  using Handle = mori::collective::AllreduceSdma<T>;

  py::class_<Handle>(m, python_name)
      .def(py::init<int, int, size_t, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(py::init<int, int, size_t, bool, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true, py::arg("use_graph_mode") = false)
      .def(
          "__call__",
          [](Handle& self, uintptr_t input_ptr, uintptr_t output_ptr, size_t count,
             int64_t stream) -> bool {
            return self(reinterpret_cast<T*>(input_ptr), reinterpret_cast<T*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0)
      .def(
          "allreduce_inplace",
          [](Handle& self, uintptr_t data_ptr, size_t count, int64_t stream) -> bool {
            return self.allreduce_inplace(reinterpret_cast<T*>(data_ptr), count,
                                          reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("data_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute in-place AllReduce SDMA operation")
      .def(
          "start_async",
          [](Handle& self, uintptr_t input_ptr, uintptr_t output_ptr, size_t count,
             int64_t stream) -> bool {
            return self.start_async(reinterpret_cast<T*>(input_ptr),
                                    reinterpret_cast<T*>(output_ptr), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0)
      .def(
          "wait_async",
          [](Handle& self, int64_t stream) -> double {
            return self.wait_async(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream") = 0)
      .def("is_async_in_progress", &Handle::is_async_in_progress)
      .def("cancel_async", &Handle::cancel_async)
      .def("reset_flags", &Handle::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](Handle& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer");
}
}  // namespace

namespace mori {
void RegisterMoriCcl(pybind11::module_& m) {
  // =========================================================================
  // All2allSdma (uint32_t)
  // =========================================================================
  py::class_<mori::collective::All2allSdma<uint32_t>>(m, "All2allSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "__call__",
          [](mori::collective::All2allSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> double {
            return self(reinterpret_cast<uint32_t*>(input_ptr),
                        reinterpret_cast<uint32_t*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute All2all SDMA operation (raw GPU pointers)")
      .def(
          "start_async",
          [](mori::collective::All2allSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self.start_async(reinterpret_cast<uint32_t*>(input_ptr),
                                    reinterpret_cast<uint32_t*>(output_ptr), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Start asynchronous All2all SDMA operation (PUT phase)")
      .def(
          "wait_async",
          [](mori::collective::All2allSdma<uint32_t>& self, int64_t stream) -> double {
            return self.wait_async(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream") = 0,
          "Wait for asynchronous All2all SDMA operation to complete (WAIT phase)")
      .def("is_async_in_progress", &mori::collective::All2allSdma<uint32_t>::is_async_in_progress)
      .def("cancel_async", &mori::collective::All2allSdma<uint32_t>::cancel_async)
      .def("reset_flags", &mori::collective::All2allSdma<uint32_t>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::All2allSdma<uint32_t>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer");

  // =========================================================================
  // AllgatherSdma (uint32_t)
  // =========================================================================
  py::class_<mori::collective::AllgatherSdma<uint32_t>>(m, "AllgatherSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "__call__",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self(reinterpret_cast<uint32_t*>(input_ptr),
                        reinterpret_cast<uint32_t*>(output_ptr), count,
                        reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Execute Allgather SDMA operation (raw GPU pointers)")
      .def(
          "start_async",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t input_ptr,
             uintptr_t output_ptr, size_t count, int64_t stream) -> bool {
            return self.start_async(reinterpret_cast<uint32_t*>(input_ptr),
                                    reinterpret_cast<uint32_t*>(output_ptr), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream") = 0,
          "Start asynchronous Allgather SDMA operation (PUT phase)")
      .def(
          "wait_async",
          [](mori::collective::AllgatherSdma<uint32_t>& self, int64_t stream) -> double {
            return self.wait_async(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream") = 0, "Wait for asynchronous Allgather SDMA operation to complete")
      .def("is_async_in_progress", &mori::collective::AllgatherSdma<uint32_t>::is_async_in_progress)
      .def("cancel_async", &mori::collective::AllgatherSdma<uint32_t>::cancel_async)
      .def("reset_flags", &mori::collective::AllgatherSdma<uint32_t>::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](mori::collective::AllgatherSdma<uint32_t>& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer")
      .def(
          "register_output_buffer",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t ptr, size_t size) {
            self.register_output_buffer(reinterpret_cast<void*>(ptr), size);
          },
          py::arg("ptr"), py::arg("size"),
          "Register a GPU buffer as direct SDMA output target (collective)")
      .def(
          "deregister_output_buffer",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t ptr) {
            self.deregister_output_buffer(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Deregister a previously registered output buffer (collective)")
      .def(
          "is_output_registered",
          [](mori::collective::AllgatherSdma<uint32_t>& self, uintptr_t ptr) -> bool {
            return self.is_output_registered(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Check whether an output buffer is registered for direct SDMA writes");

  BindAllreduceHandle<uint32_t>(m, "AllreduceSdmaHandle");
  BindAllreduceHandle<int32_t>(m, "AllreduceSdmaHandleInt32");
  BindAllreduceHandle<float>(m, "AllreduceSdmaHandleFp32");
  BindAllreduceHandle<half>(m, "AllreduceSdmaHandleFp16");
  BindAllreduceHandle<hip_bfloat16>(m, "AllreduceSdmaHandleBf16");
}
}  // namespace mori

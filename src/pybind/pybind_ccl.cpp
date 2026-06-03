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
#include "mori/collective/allgather/allgather_into_tensor.hpp"
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
      // JIT sync path (two-shot: reduce-scatter kernel, then allgather kernel)
      .def(
          "prepare_reduce_scatter",
          [](Handle& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_reduce_scatter(reinterpret_cast<const T*>(input),
                                               reinterpret_cast<T*>(output), count,
                                               reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "get_reduce_scatter_grid",
          [](Handle& self, size_t count) -> py::tuple {
            auto [blocks, threads] = self.get_reduce_scatter_grid(count);
            return py::make_tuple(blocks, threads);
          },
          py::arg("count"))
      .def(
          "prepare_allgather",
          [](Handle& self, size_t count, int64_t stream) -> int64_t {
            return self.prepare_allgather(count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](Handle& self, uintptr_t output, size_t count, int64_t stream,
             bool force_copy_output_to_user) -> double {
            return self.finish_sync(reinterpret_cast<T*>(output), count,
                                    reinterpret_cast<hipStream_t>(stream),
                                    force_copy_output_to_user);
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"),
          py::arg("force_copy_output_to_user") = false)
      // JIT async path
      .def(
          "prepare_async_reduce_scatter",
          [](Handle& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_async_reduce_scatter(reinterpret_cast<const T*>(input),
                                                     reinterpret_cast<T*>(output), count,
                                                     reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_allgather_put",
          [](Handle& self, size_t count, int64_t stream) -> int64_t {
            return self.prepare_async_allgather_put(count, reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("count"), py::arg("stream"))
      .def("after_async_start", &Handle::after_async_start)
      .def(
          "prepare_async_wait",
          [](Handle& self, int64_t stream) -> int64_t {
            return self.prepare_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def(
          "finish_async_wait",
          [](Handle& self, int64_t stream) -> double {
            return self.finish_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
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
          "Return (ptr, size_bytes) of the output transit buffer")
      .def("max_blocks", &Handle::max_blocks)
      .def("npes", &Handle::npes);
}
}  // namespace

namespace mori {
void RegisterMoriCcl(pybind11::module_& m) {
  // =========================================================================
  // All2allSdma (uint32_t) — JIT launch path
  // =========================================================================
  using All2allU32 = mori::collective::All2allSdma<uint32_t>;
  py::class_<All2allU32>(m, "All2allSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "prepare_sync",
          [](All2allU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_sync(reinterpret_cast<uint32_t*>(input),
                                     reinterpret_cast<uint32_t*>(output), count,
                                     reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](All2allU32& self, uintptr_t output, size_t count, int64_t stream) -> double {
            return self.finish_sync(reinterpret_cast<uint32_t*>(output), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_start",
          [](All2allU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_async_start(reinterpret_cast<uint32_t*>(input),
                                            reinterpret_cast<uint32_t*>(output), count,
                                            reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def("after_async_start", &All2allU32::after_async_start)
      .def(
          "prepare_async_wait",
          [](All2allU32& self, int64_t stream) -> int64_t {
            return self.prepare_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def(
          "finish_async_wait",
          [](All2allU32& self, int64_t stream) -> double {
            return self.finish_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def("is_async_in_progress", &All2allU32::is_async_in_progress)
      .def("cancel_async", &All2allU32::cancel_async)
      .def("reset_flags", &All2allU32::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](All2allU32& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer");

  // =========================================================================
  // AllgatherSdma (uint32_t) — JIT launch path
  // =========================================================================
  using AllgatherU32 = mori::collective::AllgatherSdma<uint32_t>;
  py::class_<AllgatherU32>(m, "AllgatherSdmaHandle")
      .def(py::init<int, int, size_t, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("input_buffer_size"), py::arg("output_buffer_size"),
           py::arg("copy_output_to_user") = true)
      .def(py::init<int, int, size_t, bool>(), py::arg("my_pe"), py::arg("npes"),
           py::arg("transit_buffer_size") = 512 * 1024 * 1024,
           py::arg("copy_output_to_user") = true)
      .def(
          "prepare_sync",
          [](AllgatherU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_sync(reinterpret_cast<uint32_t*>(input),
                                     reinterpret_cast<uint32_t*>(output), count,
                                     reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "finish_sync",
          [](AllgatherU32& self, uintptr_t output, size_t count, int64_t stream) -> double {
            return self.finish_sync(reinterpret_cast<uint32_t*>(output), count,
                                    reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def(
          "prepare_async_start",
          [](AllgatherU32& self, uintptr_t input, uintptr_t output, size_t count,
             int64_t stream) -> int64_t {
            return self.prepare_async_start(reinterpret_cast<uint32_t*>(input),
                                            reinterpret_cast<uint32_t*>(output), count,
                                            reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("input_ptr"), py::arg("output_ptr"), py::arg("count"), py::arg("stream"))
      .def("after_async_start", &AllgatherU32::after_async_start)
      .def(
          "prepare_async_wait",
          [](AllgatherU32& self, int64_t stream) -> int64_t {
            return self.prepare_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def(
          "finish_async_wait",
          [](AllgatherU32& self, int64_t stream) -> double {
            return self.finish_async_wait(reinterpret_cast<hipStream_t>(stream));
          },
          py::arg("stream"))
      .def("is_async_in_progress", &AllgatherU32::is_async_in_progress)
      .def("cancel_async", &AllgatherU32::cancel_async)
      .def("reset_flags", &AllgatherU32::resetFlags)
      .def(
          "get_output_transit_buffer",
          [](AllgatherU32& self) -> py::tuple {
            void* ptr = self.getOutputTransitBuffer();
            size_t size = self.getOutputTransitBufferSize();
            if (ptr == nullptr) throw std::runtime_error("Output transit buffer is null");
            return py::make_tuple(reinterpret_cast<uintptr_t>(ptr), size);
          },
          "Return (ptr, size_bytes) of the output transit buffer")
      .def(
          "register_output_buffer",
          [](AllgatherU32& self, uintptr_t ptr, size_t size) {
            self.register_output_buffer(reinterpret_cast<void*>(ptr), size);
          },
          py::arg("ptr"), py::arg("size"),
          "Register a GPU buffer as direct SDMA output target (collective)")
      .def(
          "deregister_output_buffer",
          [](AllgatherU32& self, uintptr_t ptr) {
            self.deregister_output_buffer(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Deregister a previously registered output buffer (collective)")
      .def(
          "is_output_registered",
          [](AllgatherU32& self, uintptr_t ptr) -> bool {
            return self.is_output_registered(reinterpret_cast<void*>(ptr));
          },
          py::arg("ptr"), "Check whether an output buffer is registered for direct SDMA writes");

  // =========================================================================
  // DataType enum and size_of
  // =========================================================================
  py::enum_<mori::collective::DataType>(m, "DataType")
      .value("Int8", mori::collective::DataType::kInt8)
      .value("Uint8", mori::collective::DataType::kUint8)
      .value("Int16", mori::collective::DataType::kInt16)
      .value("Uint16", mori::collective::DataType::kUint16)
      .value("Int32", mori::collective::DataType::kInt32)
      .value("Uint32", mori::collective::DataType::kUint32)
      .value("Int64", mori::collective::DataType::kInt64)
      .value("Uint64", mori::collective::DataType::kUint64)
      .value("Float16", mori::collective::DataType::kFloat16)
      .value("BFloat16", mori::collective::DataType::kBFloat16)
      .value("Float32", mori::collective::DataType::kFloat32)
      .value("Float64", mori::collective::DataType::kFloat64);
  m.def("size_of", &mori::collective::SizeOf, py::arg("dtype"),
        "Return element size in bytes for a mori_cpp.DataType value");

  // =========================================================================
  // AllGatherIntoTensor — REMOVED
  // The underlying AllgatherSdma<uint32_t>::operator() now throws; this
  // wrapper needs the Python JIT launch path to be ported before it can be
  // re-enabled.
  // =========================================================================

  // =========================================================================
  // AllreduceSdma — JIT launch path (typed instantiations)
  // =========================================================================
  BindAllreduceHandle<uint32_t>(m, "AllreduceSdmaHandle");
  BindAllreduceHandle<int32_t>(m, "AllreduceSdmaHandleInt32");
  BindAllreduceHandle<float>(m, "AllreduceSdmaHandleFp32");
  BindAllreduceHandle<half>(m, "AllreduceSdmaHandleFp16");
  BindAllreduceHandle<hip_bfloat16>(m, "AllreduceSdmaHandleBf16");
}
}  // namespace mori

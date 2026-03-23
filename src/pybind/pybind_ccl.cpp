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

#include <ATen/hip/HIPContext.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/python.h>

#include "mori/application/application.hpp"
#include "mori/collective/collective.hpp"
#include "mori/io/io.hpp"
#include "mori/ops/ops.hpp"
#include "mori/shmem/shmem.hpp"
#include "src/pybind/torch_utils.hpp"

namespace py = pybind11;

namespace {hipStream_t convert_torch_stream_to_hip(py::object stream_obj, int device_index = -1) {
    if (stream_obj.is_none()) {
        // Get current HIP stream for the device
        // This allows using torch.cuda.stream() context manager
        if (device_index < 0) {
            device_index = at::cuda::current_device();
        }
        auto current_stream = at::cuda::getCurrentHIPStream(device_index);
        return current_stream.stream();
    }
    
    // Get the cuda_stream attribute from torch.cuda.Stream object
    // This attribute contains the integer pointer value of the stream
    try {
        uintptr_t stream_ptr = stream_obj.attr("cuda_stream").cast<uintptr_t>();
        return reinterpret_cast<hipStream_t>(stream_ptr);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("Failed to convert torch.cuda.Stream to hipStream_t: ") + e.what());
    }
}

}  // namespace

namespace mori {
void RegisterMoriCcl(pybind11::module_& m) {
    // Bind All2allSdma class (uint32_t version)
    py::class_<mori::collective::All2allSdma<uint32_t>>(m, "All2allSdmaHandle")
        .def(py::init<int, int, size_t, size_t, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("input_buffer_size"),
             py::arg("output_buffer_size"),
             py::arg("copy_output_to_user") = true,
             "Initialize All2allSdma with PE ID, number of PEs, buffer sizes, and copy mode")
        .def(py::init<int, int, size_t, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("transit_buffer_size") = 512 * 1024 * 1024,
             py::arg("copy_output_to_user") = true,
             "Initialize All2allSdma with PE ID, number of PEs, transit buffer size (default 512MB), and copy mode")
        .def("__call__",
            [](mori::collective::All2allSdma<uint32_t>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> double {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }
                // C++ class uses uint32_t template
                // Accept uint32 or int32 (same memory layout)
                uint32_t* input_ptr = nullptr;
                uint32_t* output_ptr = nullptr;
                
                if (input_tensor.scalar_type() == torch::kUInt32) {
                    input_ptr = input_tensor.data_ptr<uint32_t>();
                } else if (input_tensor.scalar_type() == torch::kInt32) {
                    input_ptr = reinterpret_cast<uint32_t*>(input_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Input tensor must be uint32 or int32");
                }
                
                if (output_tensor.scalar_type() == torch::kUInt32) {
                    output_ptr = output_tensor.data_ptr<uint32_t>();
                } else if (output_tensor.scalar_type() == torch::kInt32) {
                    output_ptr = reinterpret_cast<uint32_t*>(output_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Output tensor must be uint32 or int32");
                }

                // Get device index from input tensor and convert stream
                // If stream_obj is None, this will use the current CUDA stream (supports torch.cuda.stream() context)
                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self(input_ptr, output_ptr, count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute All2all SDMA operation with PyTorch CUDA tensors")
        .def("start_async",
            [](mori::collective::All2allSdma<uint32_t>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }
                // C++ class uses uint32_t template
                // Accept uint32 or int32 (same memory layout)
                uint32_t* input_ptr = nullptr;
                uint32_t* output_ptr = nullptr;
                
                if (input_tensor.scalar_type() == torch::kUInt32) {
                    input_ptr = input_tensor.data_ptr<uint32_t>();
                } else if (input_tensor.scalar_type() == torch::kInt32) {
                    input_ptr = reinterpret_cast<uint32_t*>(input_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Input tensor must be uint32 or int32");
                }
                
                if (output_tensor.scalar_type() == torch::kUInt32) {
                    output_ptr = output_tensor.data_ptr<uint32_t>();
                } else if (output_tensor.scalar_type() == torch::kInt32) {
                    output_ptr = reinterpret_cast<uint32_t*>(output_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Output tensor must be uint32 or int32");
                }

                // Get device index from input tensor and convert stream
                // If stream_obj is None, this will use the current CUDA stream (supports torch.cuda.stream() context)
                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self.start_async(input_ptr, output_ptr, count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Start asynchronous All2all SDMA operation (PUT phase)")
        .def("wait_async",
            [](mori::collective::All2allSdma<uint32_t>& self,
               py::object stream_obj) -> double {

                // Convert stream, using current CUDA stream if None
                // This supports torch.cuda.stream() context manager
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj);

                return self.wait_async(stream);
            },
            py::arg("stream") = py::none(),
            "Wait for asynchronous All2all SDMA operation to complete (WAIT phase)")
        .def("is_async_in_progress",
            &mori::collective::All2allSdma<uint32_t>::is_async_in_progress,
            "Check if async operation is in progress")
        .def("cancel_async",
            &mori::collective::All2allSdma<uint32_t>::cancel_async,
            "Cancel ongoing async operation")
        .def("reset_flags",
            &mori::collective::All2allSdma<uint32_t>::resetFlags,
            "Reset synchronization flags")
        .def("get_output_transit_buffer",
            [](mori::collective::All2allSdma<uint32_t>& self, py::object device_obj) -> torch::Tensor {
                void* buffer_ptr = self.getOutputTransitBuffer();
                size_t buffer_size = self.getOutputTransitBufferSize();
                
                if (buffer_ptr == nullptr) {
                    throw std::runtime_error("Output transit buffer is null");
                }
                
                // Convert buffer size from bytes to number of uint32_t elements
                size_t num_elements = buffer_size / sizeof(uint32_t);
                
                // Determine device index
                int device_index = 0;
                if (!device_obj.is_none()) {
                    // Check if it's a PyTorch tensor using Python isinstance
                    py::object torch_module = py::module_::import("torch");
                    py::object tensor_class = torch_module.attr("Tensor");
                    bool is_tensor = py::isinstance(device_obj, tensor_class);
                    
                    if (is_tensor) {
                        // It's a tensor, cast and get device
                        torch::Tensor tensor = device_obj.cast<torch::Tensor>();
                        if (tensor.is_cuda()) {
                            device_index = tensor.device().index();
                        } else {
                            throw std::runtime_error("device tensor must be a CUDA tensor");
                        }
                    } else {
                        // Try to cast as int
                        try {
                            device_index = device_obj.cast<int>();
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("device must be an int, a CUDA tensor, or None");
                        }
                    }
                } else {
                    // Default to current device
                    device_index = at::cuda::current_device();
                }
                
                // Create a tensor from the buffer
                // Note: The buffer is on GPU (CUDA), so we use torch::kCUDA device
                torch::Tensor tensor = torch::from_blob(
                    buffer_ptr,
                    {static_cast<int64_t>(num_elements)},
                    torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA, device_index)
                );
                
                return tensor;
            },
            py::arg("device") = py::none(),
            "Get output transit buffer as a PyTorch tensor. device can be an int (device index) or a CUDA tensor (to use its device), or None (to use current device)");

    // Keep old function-based interface for backward compatibility (optional)
    m.def("all2all_sdma",
        [](int my_pe, int npes,
           py::array_t<uint32_t> input_array,
           py::array_t<uint32_t> output_array,
           size_t count,
           py::object stream_obj) -> double {

            // Validate arrays
            if (input_array.ndim() != 1) {
                throw std::runtime_error("Input array must be 1-dimensional");
            }
            if (output_array.ndim() != 1) {
                throw std::runtime_error("Output array must be 1-dimensional");
            }

            // Get buffer info
            py::buffer_info input_info = input_array.request();
            py::buffer_info output_info = output_array.request();

            // Get data pointers
            uint32_t* input_ptr = static_cast<uint32_t*>(input_info.ptr);
            uint32_t* output_ptr = static_cast<uint32_t*>(output_info.ptr);

            // Handle HIP stream parameter
            hipStream_t stream = nullptr;
            if (!stream_obj.is_none()) {
                // TODO: Convert Python stream object to hipStream_t if needed
            }

            // Call C++ function
            return mori::collective::All2all_sdma<uint32_t>(
                input_ptr, output_ptr, count, stream);
        },
        // Parameter documentation
        py::arg("my_pe"),
        py::arg("npes"),
        py::arg("input"),
        py::arg("output"),
        py::arg("count"),
        py::arg("stream") = py::none(),
        // å‡½æ•°æ–‡æ¡£
        "Execute All2All SDMA operation"
  );

    // Bind AllgatherSdma class (uint32_t version)
    py::class_<mori::collective::AllgatherSdma<uint32_t>>(m, "AllgatherSdmaHandle")
        .def(py::init<int, int, size_t, size_t, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("input_buffer_size"),
             py::arg("output_buffer_size"),
             py::arg("copy_output_to_user") = true,
             "Initialize AllgatherSdma with PE ID, number of PEs, and buffer sizes")
        .def(py::init<int, int, size_t, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("transit_buffer_size") = 512 * 1024 * 1024,
             py::arg("copy_output_to_user") = true,
             "Initialize AllgatherSdma with PE ID, number of PEs, and transit buffer size (default 512MB)")
        .def("__call__",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }
                // AllGather is pure byte-copy (SDMA), no arithmetic on data.
                // Accept any dtype â€” reinterpret as uint32_t* for the C++ API.
                // count is in *elements* of the original dtype; convert to
                // uint32-equivalent count so the kernel copies the right bytes.
                size_t byte_count = count * input_tensor.element_size();
                size_t u32_count = (byte_count + sizeof(uint32_t) - 1) / sizeof(uint32_t);

                uint32_t* input_ptr = reinterpret_cast<uint32_t*>(input_tensor.data_ptr());
                uint32_t* output_ptr = reinterpret_cast<uint32_t*>(output_tensor.data_ptr());

                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self(input_ptr, output_ptr, u32_count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute Allgather SDMA operation (any dtype), synchronization must be done by caller")
        .def("start_async",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }

                size_t byte_count = count * input_tensor.element_size();
                size_t u32_count = (byte_count + sizeof(uint32_t) - 1) / sizeof(uint32_t);

                uint32_t* input_ptr = reinterpret_cast<uint32_t*>(input_tensor.data_ptr());
                uint32_t* output_ptr = reinterpret_cast<uint32_t*>(output_tensor.data_ptr());

                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self.start_async(input_ptr, output_ptr, u32_count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Start asynchronous Allgather SDMA operation (PUT phase)")
        .def("wait_async",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               py::object stream_obj) -> double {

                // Convert stream, using current CUDA stream if None
                // This supports torch.cuda.stream() context manager
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj);

                return self.wait_async(stream);
            },
            py::arg("stream") = py::none(),
            "Wait for asynchronous Allgather SDMA operation to complete (WAIT phase)")
        .def("is_async_in_progress",
            &mori::collective::AllgatherSdma<uint32_t>::is_async_in_progress,
            "Check if async operation is in progress")
        .def("cancel_async",
            &mori::collective::AllgatherSdma<uint32_t>::cancel_async,
            "Cancel ongoing async operation")
        .def("reset_flags",
            &mori::collective::AllgatherSdma<uint32_t>::resetFlags,
            "Reset synchronization flags")
        .def("get_output_transit_buffer",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               py::object device_obj,
               py::object dtype_obj) -> torch::Tensor {
                void* buffer_ptr = self.getOutputTransitBuffer();
                size_t buffer_size = self.getOutputTransitBufferSize();

                if (buffer_ptr == nullptr) {
                    throw std::runtime_error("Output transit buffer is null");
                }

                // Determine torch dtype (default uint32)
                torch::Dtype torch_dtype = torch::kUInt32;
                if (!dtype_obj.is_none()) {
                    torch_dtype = py::cast<torch::Dtype>(dtype_obj);
                }
                size_t elem_size = torch::elementSize(torch_dtype);
                size_t num_elements = buffer_size / elem_size;

                int device_index = 0;
                if (!device_obj.is_none()) {
                    py::object torch_module = py::module_::import("torch");
                    py::object tensor_class = torch_module.attr("Tensor");
                    if (py::isinstance(device_obj, tensor_class)) {
                        torch::Tensor t = device_obj.cast<torch::Tensor>();
                        if (t.is_cuda()) {
                            device_index = t.device().index();
                        } else {
                            throw std::runtime_error("device tensor must be a CUDA tensor");
                        }
                    } else {
                        try {
                            device_index = device_obj.cast<int>();
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("device must be an int, a CUDA tensor, or None");
                        }
                    }
                } else {
                    device_index = at::cuda::current_device();
                }

                return torch::from_blob(
                    buffer_ptr,
                    {static_cast<int64_t>(num_elements)},
                    torch::TensorOptions().dtype(torch_dtype).device(torch::kCUDA, device_index)
                );
            },
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none(),
            "Get output transit buffer as a PyTorch tensor (specify dtype for non-uint32 types)")
        .def("register_output_buffer",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               const torch::Tensor& tensor) {
                if (!tensor.is_cuda()) {
                    throw std::runtime_error("Tensor must be a CUDA tensor");
                }
                self.register_output_buffer(tensor.data_ptr(), tensor.nbytes());
            },
            py::arg("tensor"),
            "Register a CUDA tensor as direct SDMA output target (collective)")
        .def("deregister_output_buffer",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               const torch::Tensor& tensor) {
                self.deregister_output_buffer(tensor.data_ptr());
            },
            py::arg("tensor"),
            "Deregister a previously registered output buffer (collective)")
        .def("is_output_registered",
            [](mori::collective::AllgatherSdma<uint32_t>& self,
               const torch::Tensor& tensor) -> bool {
                return self.is_output_registered(tensor.data_ptr());
            },
            py::arg("tensor"),
            "Check whether an output tensor is registered for direct SDMA writes");

    // Keep old function-based interface for backward compatibility (optional)
    m.def("allgather_sdma",
        [](py::array_t<uint32_t> input_array,
           py::array_t<uint32_t> output_array,
           size_t count,
           py::object stream_obj) -> double {

            // Validate arrays
            if (input_array.ndim() != 1) {
                throw std::runtime_error("Input array must be 1-dimensional");
            }
            if (output_array.ndim() != 1) {
                throw std::runtime_error("Output array must be 1-dimensional");
            }

            // Get buffer info
            py::buffer_info input_info = input_array.request();
            py::buffer_info output_info = output_array.request();

            // Get data pointers
            uint32_t* input_ptr = static_cast<uint32_t*>(input_info.ptr);
            uint32_t* output_ptr = static_cast<uint32_t*>(output_info.ptr);

            // Handle HIP stream parameter
            hipStream_t stream = nullptr;
            if (!stream_obj.is_none()) {
                // TODO: Convert Python stream object to hipStream_t if needed
            }

            // Call C++ function
            return mori::collective::Allgather_sdma<uint32_t>(
                input_ptr, output_ptr, count, stream);
        },
        // Parameter documentation
        py::arg("input"),
        py::arg("output"),
        py::arg("count"),
        py::arg("stream") = py::none(),
        // å‡½æ•°æ–‡æ¡£
        "Execute Allgather SDMA operation"
  );

    // =========================================================================
    // Bind AllreduceSdma class (uint32_t version)
    // =========================================================================
    py::class_<mori::collective::AllreduceSdma<uint32_t>>(m, "AllreduceSdmaHandle")
        .def(py::init<int, int, size_t, size_t, bool, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("input_buffer_size"),
             py::arg("output_buffer_size"),
             py::arg("copy_output_to_user") = true,
             py::arg("use_graph_mode") = false,
             "Initialize AllreduceSdma with PE ID, number of PEs, and buffer sizes")
        .def(py::init<int, int, size_t, bool, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("transit_buffer_size") = 512 * 1024 * 1024,
             py::arg("copy_output_to_user") = true,
             py::arg("use_graph_mode") = false,
             "Initialize AllreduceSdma with PE ID, number of PEs, and transit buffer size (default 512MB)")
        .def("__call__",
            [](mori::collective::AllreduceSdma<uint32_t>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }
                // C++ class uses uint32_t template
                // Accept uint32 or int32 (same memory layout)
                uint32_t* input_ptr = nullptr;
                uint32_t* output_ptr = nullptr;

                if (input_tensor.scalar_type() == torch::kUInt32) {
                    input_ptr = input_tensor.data_ptr<uint32_t>();
                } else if (input_tensor.scalar_type() == torch::kInt32) {
                    input_ptr = reinterpret_cast<uint32_t*>(input_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Input tensor must be uint32 or int32");
                }

                if (output_tensor.scalar_type() == torch::kUInt32) {
                    output_ptr = output_tensor.data_ptr<uint32_t>();
                } else if (output_tensor.scalar_type() == torch::kInt32) {
                    output_ptr = reinterpret_cast<uint32_t*>(output_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Output tensor must be uint32 or int32");
                }

                // Get device index from input tensor and convert stream
                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self(input_ptr, output_ptr, count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute AllReduce SDMA operation (returns bool), synchronization must be done by caller")
        .def("allreduce_inplace",
            [](mori::collective::AllreduceSdma<uint32_t>& self,
               const torch::Tensor& tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (tensor.dim() != 1) {
                    throw std::runtime_error("Tensor must be 1-dimensional");
                }
                if (!tensor.is_cuda()) {
                    throw std::runtime_error("Tensor must be CUDA tensor");
                }

                uint32_t* ptr = nullptr;
                if (tensor.scalar_type() == torch::kUInt32) {
                    ptr = tensor.data_ptr<uint32_t>();
                } else if (tensor.scalar_type() == torch::kInt32) {
                    ptr = reinterpret_cast<uint32_t*>(tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Tensor must be uint32 or int32");
                }

                int device_index = tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self.allreduce_inplace(ptr, count, stream);
            },
            py::arg("data"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute in-place AllReduce SDMA operation (result overwrites input)")
        .def("start_async",
            [](mori::collective::AllreduceSdma<uint32_t>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1 || output_tensor.dim() != 1) {
                    throw std::runtime_error("Tensors must be 1-dimensional");
                }
                if (!input_tensor.is_cuda() || !output_tensor.is_cuda()) {
                    throw std::runtime_error("Tensors must be CUDA tensors");
                }

                uint32_t* input_ptr = nullptr;
                uint32_t* output_ptr = nullptr;

                if (input_tensor.scalar_type() == torch::kUInt32) {
                    input_ptr = input_tensor.data_ptr<uint32_t>();
                } else if (input_tensor.scalar_type() == torch::kInt32) {
                    input_ptr = reinterpret_cast<uint32_t*>(input_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Input tensor must be uint32 or int32");
                }

                if (output_tensor.scalar_type() == torch::kUInt32) {
                    output_ptr = output_tensor.data_ptr<uint32_t>();
                } else if (output_tensor.scalar_type() == torch::kInt32) {
                    output_ptr = reinterpret_cast<uint32_t*>(output_tensor.data_ptr<int32_t>());
                } else {
                    throw std::runtime_error("Output tensor must be uint32 or int32");
                }

                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self.start_async(input_ptr, output_ptr, count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Start asynchronous AllReduce SDMA operation (ReduceScatter + AllGather PUT phase)")
        .def("wait_async",
            [](mori::collective::AllreduceSdma<uint32_t>& self,
               py::object stream_obj) -> double {

                hipStream_t stream = convert_torch_stream_to_hip(stream_obj);

                return self.wait_async(stream);
            },
            py::arg("stream") = py::none(),
            "Wait for asynchronous AllReduce SDMA operation to complete")
        .def("is_async_in_progress",
            &mori::collective::AllreduceSdma<uint32_t>::is_async_in_progress,
            "Check if async operation is in progress")
        .def("cancel_async",
            &mori::collective::AllreduceSdma<uint32_t>::cancel_async,
            "Cancel ongoing async operation")
        .def("reset_flags",
            &mori::collective::AllreduceSdma<uint32_t>::resetFlags,
            "Reset synchronization flags")
        .def("get_output_transit_buffer",
            [](mori::collective::AllreduceSdma<uint32_t>& self, py::object device_obj) -> torch::Tensor {
                void* buffer_ptr = self.getOutputTransitBuffer();
                size_t buffer_size = self.getOutputTransitBufferSize();

                if (buffer_ptr == nullptr) {
                    throw std::runtime_error("Output transit buffer is null");
                }

                // Convert buffer size from bytes to number of uint32_t elements
                size_t num_elements = buffer_size / sizeof(uint32_t);

                // Determine device index
                int device_index = 0;
                if (!device_obj.is_none()) {
                    py::object torch_module = py::module_::import("torch");
                    py::object tensor_class = torch_module.attr("Tensor");
                    bool is_tensor = py::isinstance(device_obj, tensor_class);

                    if (is_tensor) {
                        torch::Tensor tensor = device_obj.cast<torch::Tensor>();
                        if (tensor.is_cuda()) {
                            device_index = tensor.device().index();
                        } else {
                            throw std::runtime_error("device tensor must be a CUDA tensor");
                        }
                    } else {
                        try {
                            device_index = device_obj.cast<int>();
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("device must be an int, a CUDA tensor, or None");
                        }
                    }
                } else {
                    device_index = at::cuda::current_device();
                }

                torch::Tensor tensor = torch::from_blob(
                    buffer_ptr,
                    {static_cast<int64_t>(num_elements)},
                    torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA, device_index)
                );

                return tensor;
            },
            py::arg("device") = py::none(),
            "Get output transit buffer as a PyTorch tensor");

    // =========================================================================
    // Bind AllreduceSdma class (half / fp16 version)
    // =========================================================================
    py::class_<mori::collective::AllreduceSdma<half>>(m, "AllreduceSdmaHandleFp16")
        .def(py::init<int, int, size_t, size_t, bool, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("input_buffer_size"),
             py::arg("output_buffer_size"),
             py::arg("copy_output_to_user") = true,
             py::arg("use_graph_mode") = false,
             "Initialize AllreduceSdma (fp16) with PE ID, number of PEs, and buffer sizes")
        .def(py::init<int, int, size_t, bool, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("transit_buffer_size") = 512 * 1024 * 1024,
             py::arg("copy_output_to_user") = true,
             py::arg("use_graph_mode") = false,
             "Initialize AllreduceSdma (fp16) with PE ID, number of PEs, and transit buffer size")
        .def("__call__",
            [](mori::collective::AllreduceSdma<half>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }
                if (input_tensor.scalar_type() != torch::kFloat16) {
                    throw std::runtime_error("Input tensor must be float16");
                }
                if (output_tensor.scalar_type() != torch::kFloat16) {
                    throw std::runtime_error("Output tensor must be float16");
                }

                half* input_ptr = reinterpret_cast<half*>(input_tensor.data_ptr<at::Half>());
                half* output_ptr = reinterpret_cast<half*>(output_tensor.data_ptr<at::Half>());

                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self(input_ptr, output_ptr, count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute AllReduce SDMA operation (fp16)")
        .def("allreduce_inplace",
            [](mori::collective::AllreduceSdma<half>& self,
               const torch::Tensor& tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (tensor.dim() != 1) {
                    throw std::runtime_error("Tensor must be 1-dimensional");
                }
                if (!tensor.is_cuda()) {
                    throw std::runtime_error("Tensor must be CUDA tensor");
                }
                if (tensor.scalar_type() != torch::kFloat16) {
                    throw std::runtime_error("Tensor must be float16");
                }

                half* ptr = reinterpret_cast<half*>(tensor.data_ptr<at::Half>());

                int device_index = tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self.allreduce_inplace(ptr, count, stream);
            },
            py::arg("data"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute in-place AllReduce SDMA operation (fp16)")
        .def("reset_flags",
            &mori::collective::AllreduceSdma<half>::resetFlags,
            "Reset synchronization flags")
        .def("get_output_transit_buffer",
            [](mori::collective::AllreduceSdma<half>& self, py::object device_obj) -> torch::Tensor {
                void* buffer_ptr = self.getOutputTransitBuffer();
                size_t buffer_size = self.getOutputTransitBufferSize();

                if (buffer_ptr == nullptr) {
                    throw std::runtime_error("Output transit buffer is null");
                }

                size_t num_elements = buffer_size / sizeof(half);

                int device_index = 0;
                if (!device_obj.is_none()) {
                    py::object torch_module = py::module_::import("torch");
                    py::object tensor_class = torch_module.attr("Tensor");
                    bool is_tensor = py::isinstance(device_obj, tensor_class);

                    if (is_tensor) {
                        torch::Tensor tensor = device_obj.cast<torch::Tensor>();
                        if (tensor.is_cuda()) {
                            device_index = tensor.device().index();
                        } else {
                            throw std::runtime_error("device tensor must be a CUDA tensor");
                        }
                    } else {
                        try {
                            device_index = device_obj.cast<int>();
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("device must be an int, a CUDA tensor, or None");
                        }
                    }
                } else {
                    device_index = at::cuda::current_device();
                }

                torch::Tensor tensor = torch::from_blob(
                    buffer_ptr,
                    {static_cast<int64_t>(num_elements)},
                    torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA, device_index)
                );

                return tensor;
            },
            py::arg("device") = py::none(),
            "Get output transit buffer as a PyTorch tensor (fp16)");

    // =========================================================================
    // Bind AllreduceSdma class (__hip_bfloat16 / bf16 version)
    // =========================================================================
    py::class_<mori::collective::AllreduceSdma<__hip_bfloat16>>(m, "AllreduceSdmaHandleBf16")
        .def(py::init<int, int, size_t, size_t, bool, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("input_buffer_size"),
             py::arg("output_buffer_size"),
             py::arg("copy_output_to_user") = true,
             py::arg("use_graph_mode") = false,
             "Initialize AllreduceSdma (bf16) with PE ID, number of PEs, and buffer sizes")
        .def(py::init<int, int, size_t, bool, bool>(),
             py::arg("my_pe"),
             py::arg("npes"),
             py::arg("transit_buffer_size") = 512 * 1024 * 1024,
             py::arg("copy_output_to_user") = true,
             py::arg("use_graph_mode") = false,
             "Initialize AllreduceSdma (bf16) with PE ID, number of PEs, and transit buffer size")
        .def("__call__",
            [](mori::collective::AllreduceSdma<__hip_bfloat16>& self,
               const torch::Tensor& input_tensor,
               const torch::Tensor& output_tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (input_tensor.dim() != 1) {
                    throw std::runtime_error("Input tensor must be 1-dimensional");
                }
                if (output_tensor.dim() != 1) {
                    throw std::runtime_error("Output tensor must be 1-dimensional");
                }
                if (!input_tensor.is_cuda()) {
                    throw std::runtime_error("Input tensor must be CUDA tensor");
                }
                if (!output_tensor.is_cuda()) {
                    throw std::runtime_error("Output tensor must be CUDA tensor");
                }
                if (input_tensor.scalar_type() != torch::kBFloat16) {
                    throw std::runtime_error("Input tensor must be bfloat16");
                }
                if (output_tensor.scalar_type() != torch::kBFloat16) {
                    throw std::runtime_error("Output tensor must be bfloat16");
                }

                __hip_bfloat16* input_ptr = reinterpret_cast<__hip_bfloat16*>(input_tensor.data_ptr<at::BFloat16>());
                __hip_bfloat16* output_ptr = reinterpret_cast<__hip_bfloat16*>(output_tensor.data_ptr<at::BFloat16>());

                int device_index = input_tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self(input_ptr, output_ptr, count, stream);
            },
            py::arg("input"),
            py::arg("output"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute AllReduce SDMA operation (bf16)")
        .def("allreduce_inplace",
            [](mori::collective::AllreduceSdma<__hip_bfloat16>& self,
               const torch::Tensor& tensor,
               size_t count,
               py::object stream_obj) -> bool {

                if (tensor.dim() != 1) {
                    throw std::runtime_error("Tensor must be 1-dimensional");
                }
                if (!tensor.is_cuda()) {
                    throw std::runtime_error("Tensor must be CUDA tensor");
                }
                if (tensor.scalar_type() != torch::kBFloat16) {
                    throw std::runtime_error("Tensor must be bfloat16");
                }

                __hip_bfloat16* ptr = reinterpret_cast<__hip_bfloat16*>(tensor.data_ptr<at::BFloat16>());

                int device_index = tensor.device().index();
                hipStream_t stream = convert_torch_stream_to_hip(stream_obj, device_index);

                return self.allreduce_inplace(ptr, count, stream);
            },
            py::arg("data"),
            py::arg("count"),
            py::arg("stream") = py::none(),
            "Execute in-place AllReduce SDMA operation (bf16)")
        .def("reset_flags",
            &mori::collective::AllreduceSdma<__hip_bfloat16>::resetFlags,
            "Reset synchronization flags")
        .def("get_output_transit_buffer",
            [](mori::collective::AllreduceSdma<__hip_bfloat16>& self, py::object device_obj) -> torch::Tensor {
                void* buffer_ptr = self.getOutputTransitBuffer();
                size_t buffer_size = self.getOutputTransitBufferSize();

                if (buffer_ptr == nullptr) {
                    throw std::runtime_error("Output transit buffer is null");
                }

                size_t num_elements = buffer_size / sizeof(__hip_bfloat16);

                int device_index = 0;
                if (!device_obj.is_none()) {
                    py::object torch_module = py::module_::import("torch");
                    py::object tensor_class = torch_module.attr("Tensor");
                    bool is_tensor = py::isinstance(device_obj, tensor_class);

                    if (is_tensor) {
                        torch::Tensor tensor = device_obj.cast<torch::Tensor>();
                        if (tensor.is_cuda()) {
                            device_index = tensor.device().index();
                        } else {
                            throw std::runtime_error("device tensor must be a CUDA tensor");
                        }
                    } else {
                        try {
                            device_index = device_obj.cast<int>();
                        } catch (const py::cast_error&) {
                            throw std::runtime_error("device must be an int, a CUDA tensor, or None");
                        }
                    }
                } else {
                    device_index = at::cuda::current_device();
                }

                torch::Tensor tensor = torch::from_blob(
                    buffer_ptr,
                    {static_cast<int64_t>(num_elements)},
                    torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, device_index)
                );

                return tensor;
            },
            py::arg("device") = py::none(),
            "Get output transit buffer as a PyTorch tensor (bf16)");

    // Keep old function-based interface for backward compatibility (optional)
    m.def("allreduce_sdma",
        [](py::array_t<uint32_t> input_array,
           py::array_t<uint32_t> output_array,
           size_t count,
           py::object stream_obj) -> double {

            // Validate arrays
            if (input_array.ndim() != 1) {
                throw std::runtime_error("Input array must be 1-dimensional");
            }
            if (output_array.ndim() != 1) {
                throw std::runtime_error("Output array must be 1-dimensional");
            }

            // Get buffer info
            py::buffer_info input_info = input_array.request();
            py::buffer_info output_info = output_array.request();

            // Get data pointers
            uint32_t* input_ptr = static_cast<uint32_t*>(input_info.ptr);
            uint32_t* output_ptr = static_cast<uint32_t*>(output_info.ptr);

            // Handle HIP stream parameter
            hipStream_t stream = nullptr;
            if (!stream_obj.is_none()) {
                // TODO: Convert Python stream object to hipStream_t if needed
            }

            // Call C++ function
            return mori::collective::Allreduce_sdma<uint32_t>(
                input_ptr, output_ptr, count, stream);
        },
        py::arg("input"),
        py::arg("output"),
        py::arg("count"),
        py::arg("stream") = py::none(),
        "Execute AllReduce SDMA operation"
    );

}
}  // namespace mori


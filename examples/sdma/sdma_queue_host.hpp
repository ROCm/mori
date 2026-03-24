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
/**
 * @acknowledgements:
 * - Original implementation by: Sidler, David
 * - Source: https://github.com/AARInternal/shader_sdma
 *
 * @note: This code is adapted/modified from the implementation by Sidler, David
 */

#pragma once

#include <hip/hip_runtime_api.h>

#include <iostream>

#include "hsa/hsa_ext_amd.h"

inline void checkError(hipError_t err, const char* msg, const char* file, int line) {
  if (err != hipSuccess) {
    std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
              << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(cmd) checkError((cmd), #cmd, __FILE__, __LINE__)
#endif

auto checkHsaError = [](hsa_status_t s, const char* msg, const char* file, int line) {
  if (s != HSA_STATUS_SUCCESS) {
    const char* hsa_err_msg;
    hsa_status_string(s, &hsa_err_msg);
    throw(std::runtime_error{std::string("HSA error at ") + file + std::string(":") +
                             std::to_string(line) + std::string(" - ") + hsa_err_msg});
  }
};

#define CHECK_HSA_ERROR(cmd) checkHsaError((cmd), #cmd, __FILE__, __LINE__)

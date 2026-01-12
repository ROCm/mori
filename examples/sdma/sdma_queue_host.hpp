#pragma once

#include "hsa/hsa_ext_amd.h"
#include <hip/hip_runtime_api.h>

#include <iostream>

inline void checkError(hipError_t err, const char* msg, const char* file, int line)
{
   if (err != hipSuccess)
   {
      std::cerr << "HIP error at " << file << ":" << line << " — " << msg << "\n"
                << "  Code: " << err << " (" << hipGetErrorString(err) << ")" << std::endl;
      std::exit(EXIT_FAILURE);
   }
}

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(cmd) checkError((cmd), #cmd, __FILE__, __LINE__)
#endif

auto checkHsaError = [](hsa_status_t s, const char* msg, const char* file, int line) {
   if (s != HSA_STATUS_SUCCESS)
   {
      const char* hsa_err_msg;
      hsa_status_string(s, &hsa_err_msg);
      throw(std::runtime_error{std::string("HSA error at ") + file + std::string(":") + std::to_string(line) +
                               std::string(" - ") + hsa_err_msg});
   }
};

#define CHECK_HSA_ERROR(cmd) checkHsaError((cmd), #cmd, __FILE__, __LINE__)

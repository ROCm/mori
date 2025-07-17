#pragma once

namespace mori {
namespace io {

enum class BackendType : uint32_t {
  Unknown = 0,
  XGMI = 1,
  RDMA = 2,
  TCP = 3,
};

using BackendTypeVec = std::vector<BackendType>;

enum class MemoryLocationType : uint32_t {
  Unknown = 0,
  CPU = 1,
  GPU = 2,
};

enum class StatusCode : uint32_t {
  SUCCESS = 0,
  ERROR = 1,
};

}  // namespace io
}  // namespace mori
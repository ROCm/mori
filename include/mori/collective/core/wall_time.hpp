#pragma once

#include <chrono>

#ifdef MORI_WITH_MPI
#include <mpi.h>
#endif

namespace mori {
namespace collective {

inline double CollectiveWallTime() {
#ifdef MORI_WITH_MPI
  return MPI_Wtime();
#else
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
#endif
}

}  // namespace collective
}  // namespace mori

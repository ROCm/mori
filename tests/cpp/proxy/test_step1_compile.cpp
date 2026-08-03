// Step 1 compile test: verify GpuStates has useProxy + proxyRing fields
#include "mori/shmem/internal.hpp"
#include "mori/core/transport/rdma/proxy/proxy_types.hpp"

#include <cassert>
#include <cstdio>

int main() {
  mori::shmem::GpuStates gs{};
  assert(gs.useProxy == false);
  assert(gs.proxyRing == nullptr);

  mori::core::ProxyRing ring{};
  gs.useProxy = true;
  gs.proxyRing = &ring;
  assert(gs.useProxy == true);
  assert(gs.proxyRing == &ring);

  printf("Step 1 compile test: PASS\n");
  return 0;
}

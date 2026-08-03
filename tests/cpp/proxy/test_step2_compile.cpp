// Step 2 compile test: verify ShmemPutMemNbi proxy path compiles
// This only checks compilation — runtime test comes later
#include "mori/shmem/shmem_ibgda_kernels.hpp"
#include <cstdio>

int main() {
  printf("Step 2 compile test: PASS (shmem_ibgda_kernels.hpp compiles with proxy path)\n");
  return 0;
}

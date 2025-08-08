#include <dlfcn.h>

#include <cassert>

#include "mori/application/topology/topology.hpp"

int TestTopoNodeGpu() {
  mori::io::TopoSystem sys{};
  sys.Load();
  return 0;
}

int main() { return TestTopoNodeGpu(); }
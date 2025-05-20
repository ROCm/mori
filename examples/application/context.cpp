#include "mori/application/application.hpp"

using namespace mori;
using namespace mori::application;

int main() {
  MpiBootstrapNetwork bootNet(MPI_COMM_WORLD);
  bootNet.Initialize();

  Context context(bootNet);
  std::cout << "Local rank: " << context.LocalRank() << std::endl;
  std::cout << "World size: " << context.WorldSize() << std::endl;
  std::cout << "Host Name: " << context.HostName() << std::endl;

  bootNet.Finalize();
}
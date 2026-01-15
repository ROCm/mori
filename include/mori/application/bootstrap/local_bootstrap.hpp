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
#pragma once

#include <vector>
#include <string>

#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace mori {
namespace application {

/**
 * @brief Bootstrap network for local process communication using Unix domain sockets
 * 
 * This implementation uses Unix domain sockets with SCM_RIGHTS for proper
 * file descriptor passing between processes on the same host.
 * 
 * IMPORTANT: This only works for processes on the same host. For cross-host
 * communication, use MpiBootstrapNetwork or TorchBootstrapNetwork.
 * 
 * Use cases:
 * - VMM shareable handle (file descriptor) exchange
 * - Any scenario requiring proper FD passing (cannot use MPI/network for FDs)
 */
class LocalBootstrapNetwork : public BootstrapNetwork {
 public:
  /**
   * @brief Construct a local bootstrap network
   * @param rank Process rank (0 to worldSize-1)
   * @param worldSize Total number of processes
   * @param socketBasePath Base path for Unix domain sockets (default: /tmp/mori_local_bootstrap_)
   */
  LocalBootstrapNetwork(int rank, int worldSize, 
                       const std::string& socketBasePath = "/tmp/mori_local_bootstrap_");
  ~LocalBootstrapNetwork();

  void Initialize() override;
  void Finalize() override;

  /**
   * @brief Allgather for regular data (not file descriptors)
   * This uses the underlying MPI/network bootstrap if available.
   * For FD exchange, use ExchangeFileDescriptors() instead.
   */
  void Allgather(void* sendbuf, void* recvbuf, size_t sendcount) override;
  
  /**
   * @brief AllToAll for regular data (not file descriptors)
   */
  void AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) override;
  
  void Barrier() override;

  /**
   * @brief Exchange file descriptors between all processes
   * 
   * This is the core functionality of LocalBootstrapNetwork.
   * Uses Unix domain socket + SCM_RIGHTS for proper FD passing.
   * 
   * @param localFds Vector of local file descriptors to send
   * @param allFds Output 2D vector to store all FDs [pe][fd_index]
   * @return true on success, false on failure
   * 
   * Example:
   *   std::vector<int> myFds = {fd1, fd2, fd3};
   *   std::vector<std::vector<int>> allFds;
   *   bootstrap.ExchangeFileDescriptors(myFds, allFds);
   *   // Now allFds[peer_rank][i] contains peer's FD
   */
  bool ExchangeFileDescriptors(const std::vector<int>& localFds,
                               std::vector<std::vector<int>>& allFds);

  /**
   * @brief Send a file descriptor to a specific peer
   * @param peer Target peer rank
   * @param fd File descriptor to send
   * @return true on success, false on failure
   */
  bool SendFileDescriptorToPeer(int peer, int fd);

  /**
   * @brief Receive a file descriptor from a specific peer
   * @param peer Source peer rank
   * @return Received file descriptor on success, -1 on failure
   */
  int ReceiveFileDescriptorFromPeer(int peer);

 private:
  std::string socketBasePath_;
  bool initialized_;

  /**
   * @brief Send file descriptor through Unix domain socket using SCM_RIGHTS
   * @param socket_fd Socket file descriptor
   * @param fd File descriptor to send
   * @return 0 on success, -1 on failure
   */
  int SendFD(int socket_fd, int fd);

  /**
   * @brief Receive file descriptor through Unix domain socket using SCM_RIGHTS
   * @param socket_fd Socket file descriptor
   * @return Received file descriptor on success, -1 on failure
   */
  int ReceiveFD(int socket_fd);

  /**
   * @brief Get socket path for communication between two ranks
   * @param rank1 First rank
   * @param rank2 Second rank
   * @return Socket path string
   */
  std::string GetSocketPath(int rank1, int rank2) const;
};

}  // namespace application
}  // namespace mori

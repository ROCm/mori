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

#include "mori/application/bootstrap/local_bootstrap.hpp"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <cstring>
#include <thread>
#include <chrono>
#include <stdexcept>

#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

LocalBootstrapNetwork::LocalBootstrapNetwork(int rank, int worldSize,
                                           const std::string& socketBasePath)
    : socketBasePath_(socketBasePath), initialized_(false) {
  localRank = rank;
  this->worldSize = worldSize;
}

LocalBootstrapNetwork::~LocalBootstrapNetwork() {
  if (initialized_) {
    Finalize();
  }
}

void LocalBootstrapNetwork::Initialize() {
  if (initialized_) {
    return;
  }
  
  MORI_APP_INFO("LocalBootstrapNetwork initializing: rank={}, worldSize={}", 
                localRank, worldSize);
  initialized_ = true;
}

void LocalBootstrapNetwork::Finalize() {
  if (!initialized_) {
    return;
  }
  
  MORI_APP_INFO("LocalBootstrapNetwork finalizing: rank={}", localRank);
  
  // Clean up any remaining socket files created by this rank
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer != localRank) {
      std::string socketPath = GetSocketPath(localRank, peer);
      unlink(socketPath.c_str());
    }
  }
  
  // Clean up barrier files
  std::string arriveFile = socketBasePath_ + "barrier_arrive_" + std::to_string(localRank);
  unlink(arriveFile.c_str());
  
  std::string departFile = socketBasePath_ + "barrier_depart_" + std::to_string(localRank);
  unlink(departFile.c_str());
  
  initialized_ = false;
}

void LocalBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  MORI_APP_ERROR("LocalBootstrapNetwork::Allgather not supported for regular data. "
                "Use ExchangeFileDescriptors() for FD exchange, or use MpiBootstrapNetwork "
                "for regular data.");
  throw std::runtime_error("LocalBootstrapNetwork::Allgather not implemented for regular data");
}

void LocalBootstrapNetwork::AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) {
  MORI_APP_ERROR("LocalBootstrapNetwork::AllToAll not supported. "
                "Use MpiBootstrapNetwork for regular data exchange.");
  throw std::runtime_error("LocalBootstrapNetwork::AllToAll not implemented");
}

void LocalBootstrapNetwork::Barrier() {
  // Two-phase barrier to avoid race conditions
  
  // Phase 1: All processes arrive
  std::string arriveFile = socketBasePath_ + "barrier_arrive_" + std::to_string(localRank);
  int fd = open(arriveFile.c_str(), O_CREAT | O_WRONLY, 0666);
  if (fd >= 0) {
    close(fd);
  }
  
  // Wait for all peers to arrive
  int maxRetries = 1000;  // 10 seconds timeout
  int retries = 0;
  bool allArrived = false;
  
  while (!allArrived && retries < maxRetries) {
    allArrived = true;
    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == localRank) continue;  // Skip self
      
      std::string peerArriveFile = socketBasePath_ + "barrier_arrive_" + std::to_string(peer);
      if (access(peerArriveFile.c_str(), F_OK) != 0) {
        allArrived = false;
        break;
      }
    }
    
    if (!allArrived) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      retries++;
    }
  }
  
  if (!allArrived) {
    MORI_APP_ERROR("LocalBootstrapNetwork::Barrier phase 1 timeout for rank {}", localRank);
    return;
  }
  
  // Phase 2: Signal departure (safe to proceed)
  std::string departFile = socketBasePath_ + "barrier_depart_" + std::to_string(localRank);
  fd = open(departFile.c_str(), O_CREAT | O_WRONLY, 0666);
  if (fd >= 0) {
    close(fd);
  }
  
  // Wait for all peers to depart
  retries = 0;
  bool allDeparted = false;
  
  while (!allDeparted && retries < maxRetries) {
    allDeparted = true;
    for (int peer = 0; peer < worldSize; ++peer) {
      if (peer == localRank) continue;  // Skip self
      
      std::string peerDepartFile = socketBasePath_ + "barrier_depart_" + std::to_string(peer);
      if (access(peerDepartFile.c_str(), F_OK) != 0) {
        allDeparted = false;
        break;
      }
    }
    
    if (!allDeparted) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      retries++;
    }
  }
  
  if (!allDeparted) {
    MORI_APP_ERROR("LocalBootstrapNetwork::Barrier phase 2 timeout for rank {}", localRank);
    return;
  }
  
  // Now safe to cleanup
  unlink(arriveFile.c_str());
  unlink(departFile.c_str());
  
  MORI_APP_TRACE("Rank {} completed barrier", localRank);
}

int LocalBootstrapNetwork::SendFD(int socket_fd, int fd) {
  struct msghdr msg = {0};
  struct cmsghdr *cmsg;
  char buf[CMSG_SPACE(sizeof(int))];
  char data[1] = {'X'};
  struct iovec iov = {
      .iov_base = data,
      .iov_len = sizeof(data)
  };

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));

  memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
  msg.msg_controllen = cmsg->cmsg_len;

  if (sendmsg(socket_fd, &msg, 0) < 0) {
    return -1;
  }
  return 0;
}

int LocalBootstrapNetwork::ReceiveFD(int socket_fd) {
  struct msghdr msg = {0};
  struct cmsghdr *cmsg;
  char buf[CMSG_SPACE(sizeof(int))];
  char data[1];
  struct iovec iov = {
      .iov_base = data,
      .iov_len = sizeof(data)
  };

  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  if (recvmsg(socket_fd, &msg, 0) < 0) {
    return -1;
  }

  cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg == NULL || cmsg->cmsg_type != SCM_RIGHTS) {
    return -1;
  }

  int fd;
  memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
  return fd;
}

std::string LocalBootstrapNetwork::GetSocketPath(int rank1, int rank2) const {
  std::string path = socketBasePath_ + std::to_string(rank1) + "_" + std::to_string(rank2);
  
  // Unix domain socket path limit is 108 bytes (sizeof(sockaddr_un.sun_path))
  if (path.length() >= 108) {
    MORI_APP_ERROR("Socket path too long ({} bytes): {}", path.length(), path);
    throw std::runtime_error("Socket path exceeds maximum length of 108 bytes");
  }
  
  return path;
}

bool LocalBootstrapNetwork::ExchangeFileDescriptors(
    const std::vector<int>& localFds,
    std::vector<std::vector<int>>& allFds) {
  
  if (!initialized_) {
    MORI_APP_ERROR("LocalBootstrapNetwork not initialized");
    return false;
  }
  
  size_t numFds = localFds.size();
  allFds.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    allFds[i].resize(numFds, -1);
  }
  
  // Store our own FDs
  allFds[localRank] = localFds;
  
  // Exchange FDs with each peer
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == localRank) continue;
    
    std::string socketPath = GetSocketPath(localRank, peer);
    
    if (localRank < peer) {
      // Act as server
      
      unlink(socketPath.c_str());
      
      int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
      if (server_fd < 0) {
        MORI_APP_ERROR("Rank {} failed to create socket for peer {}: {}", 
                      localRank, peer, strerror(errno));
        return false;
      }
      
      struct sockaddr_un addr;
      memset(&addr, 0, sizeof(addr));
      addr.sun_family = AF_UNIX;
      strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);
      
      if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        MORI_APP_ERROR("Rank {} failed to bind socket for peer {}: {}", 
                      localRank, peer, strerror(errno));
        close(server_fd);
        return false;
      }
      
      if (listen(server_fd, 1) < 0) {
        MORI_APP_ERROR("Rank {} failed to listen on socket for peer {}: {}", 
                      localRank, peer, strerror(errno));
        close(server_fd);
        return false;
      }
      
      int client_fd = accept(server_fd, NULL, NULL);
      if (client_fd < 0) {
        MORI_APP_ERROR("Rank {} failed to accept connection from peer {}: {}", 
                      localRank, peer, strerror(errno));
        close(server_fd);
        return false;
      }
      
      // Send our FDs
      for (size_t i = 0; i < numFds; ++i) {
        if (SendFD(client_fd, localFds[i]) < 0) {
          MORI_APP_ERROR("Rank {} failed to send FD {} to peer {}: {}", 
                        localRank, i, peer, strerror(errno));
          close(client_fd);
          close(server_fd);
          unlink(socketPath.c_str());
          return false;
        }
      }
      
      // Receive peer's FDs
      for (size_t i = 0; i < numFds; ++i) {
        int receivedFd = ReceiveFD(client_fd);
        if (receivedFd < 0) {
          MORI_APP_ERROR("Rank {} failed to receive FD {} from peer {}: {}", 
                        localRank, i, peer, strerror(errno));
          
          // Close any FDs already received from this peer
          for (size_t j = 0; j < i; ++j) {
            if (allFds[peer][j] >= 0) {
              close(allFds[peer][j]);
              allFds[peer][j] = -1;
            }
          }
          
          close(client_fd);
          close(server_fd);
          unlink(socketPath.c_str());
          return false;
        }
        allFds[peer][i] = receivedFd;
      }
      
      close(client_fd);
      close(server_fd);
      unlink(socketPath.c_str());
      
    } else {
      // Act as client      
      int client_fd = socket(AF_UNIX, SOCK_STREAM, 0);
      if (client_fd < 0) {
        MORI_APP_ERROR("Rank {} failed to create client socket for peer {}: {}", 
                      localRank, peer, strerror(errno));
        return false;
      }
      
      struct sockaddr_un addr;
      memset(&addr, 0, sizeof(addr));
      addr.sun_family = AF_UNIX;
      
      // Client connects to peer's server socket
      std::string peerSocketPath = GetSocketPath(peer, localRank);
      strncpy(addr.sun_path, peerSocketPath.c_str(), sizeof(addr.sun_path) - 1);
      
      // Retry connection (up to 5 seconds)
      const int MAX_CONNECT_RETRIES = 500;
      const int RETRY_INTERVAL_MS = 10;
      bool connected = false;
      
      for (int retry = 0; retry < MAX_CONNECT_RETRIES && !connected; ++retry) {
        if (connect(client_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
          connected = true;
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(RETRY_INTERVAL_MS));
      }
      
      if (!connected) {
        MORI_APP_ERROR("Rank {} failed to connect to peer {}: {}", 
                      localRank, peer, strerror(errno));
        close(client_fd);
        return false;
      }
      
      // Receive peer's FDs first (since peer sends first)
      for (size_t i = 0; i < numFds; ++i) {
        int receivedFd = ReceiveFD(client_fd);
        if (receivedFd < 0) {
          MORI_APP_ERROR("Rank {} failed to receive FD {} from peer {}: {}", 
                        localRank, i, peer, strerror(errno));
          
          // Close any FDs already received from this peer
          for (size_t j = 0; j < i; ++j) {
            if (allFds[peer][j] >= 0) {
              close(allFds[peer][j]);
              allFds[peer][j] = -1;
            }
          }
          
          close(client_fd);
          return false;
        }
        allFds[peer][i] = receivedFd;
      }
      
      // Send our FDs
      for (size_t i = 0; i < numFds; ++i) {
        if (SendFD(client_fd, localFds[i]) < 0) {
          MORI_APP_ERROR("Rank {} failed to send FD {} to peer {}: {}", 
                        localRank, i, peer, strerror(errno));
          
          // Close all FDs received from this peer (send failed after receive)
          for (size_t j = 0; j < numFds; ++j) {
            if (allFds[peer][j] >= 0) {
              close(allFds[peer][j]);
              allFds[peer][j] = -1;
            }
          }
          
          close(client_fd);
          return false;
        }
      }
      
      close(client_fd);
    }
  }
  
  MORI_APP_TRACE("Rank {} completed FD exchange with all peers", localRank);
  return true;
}

bool LocalBootstrapNetwork::SendFileDescriptorToPeer(int peer, int fd) {
  if (!initialized_) {
    MORI_APP_ERROR("LocalBootstrapNetwork not initialized");
    return false;
  }
  
  if (peer == localRank) {
    MORI_APP_ERROR("Cannot send FD to self");
    return false;
  }
  
  std::string socketPath = GetSocketPath(localRank, peer);
  
  MORI_APP_WARN("SendFileDescriptorToPeer not fully implemented. Use ExchangeFileDescriptors instead.");
  return false;
}

int LocalBootstrapNetwork::ReceiveFileDescriptorFromPeer(int peer) {
  if (!initialized_) {
    MORI_APP_ERROR("LocalBootstrapNetwork not initialized");
    return -1;
  }
  
  if (peer == localRank) {
    MORI_APP_ERROR("Cannot receive FD from self");
    return -1;
  }
  
  MORI_APP_WARN("ReceiveFileDescriptorFromPeer not fully implemented. Use ExchangeFileDescriptors instead.");
  return -1;
}

}  // namespace application
}  // namespace mori

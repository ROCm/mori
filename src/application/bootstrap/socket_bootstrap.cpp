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
#include "mori/application/bootstrap/socket_bootstrap.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <thread>

#include "mori/utils/mori_log.hpp"

namespace mori {
namespace application {

SocketBootstrapNetwork::SocketBootstrapNetwork(const UniqueId& unique_id, int rank, int world_size)
    : unique_id_(unique_id), initialized_(false), unexpected_connections_(nullptr) {
  localRank = rank;
  worldSize = world_size;
  peer_addresses_.resize(world_size);
}

SocketBootstrapNetwork::~SocketBootstrapNetwork() { Finalize(); }

void SocketBootstrapNetwork::Initialize() {
  if (initialized_) return;

  // Find local network interface
  if (!FindNetworkInterface(local_interface_)) {
    throw std::runtime_error("Failed to find suitable network interface");
  }

  // Setup communication infrastructure
  if (!SetupCommunicationRing()) {
    throw std::runtime_error("Failed to setup communication ring");
  }

  initialized_ = true;
}

void SocketBootstrapNetwork::Finalize() {
  if (!initialized_) return;

  CloseSocket(listen_socket_);
  CloseSocket(ring_send_socket_);
  CloseSocket(ring_recv_socket_);

  CleanupUnexpectedConnections();

  initialized_ = false;
}

UniqueId SocketBootstrapNetwork::GenerateUniqueId() {
  UniqueId uid;
  uid.version = 1;

  // Generate random magic number
  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<uint64_t> dis;

  UniqueIdData* data = reinterpret_cast<UniqueIdData*>(uid.internal);
  data->magic = dis(gen);
  data->port = 0;  // Will be set later
  data->addr_family = AF_INET;

  return uid;
}

UniqueId SocketBootstrapNetwork::GenerateUniqueId(const std::string& host, int port) {
  UniqueId uid = GenerateUniqueId();
  UniqueIdData* data = reinterpret_cast<UniqueIdData*>(uid.internal);

  data->port = static_cast<uint16_t>(port);

  // Handle IPv6 addresses in brackets
  if (host.front() == '[' && host.back() == ']') {
    std::string hostname = host.substr(1, host.length() - 2);
    data->addr_family = AF_INET6;

    if (inet_pton(AF_INET6, hostname.c_str(), data->addr.ipv6_addr) == 1) {
      return uid;
    }
  } else {
    // Try IPv4 first
    data->addr_family = AF_INET;

    if (inet_pton(AF_INET, host.c_str(), &data->addr.ipv4_addr) == 1) {
      return uid;
    }

    // Try hostname resolution
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result) == 0) {
      if (result->ai_family == AF_INET) {
        data->addr_family = AF_INET;
        struct sockaddr_in* addr4 = reinterpret_cast<struct sockaddr_in*>(result->ai_addr);
        data->addr.ipv4_addr = addr4->sin_addr.s_addr;
      } else if (result->ai_family == AF_INET6) {
        data->addr_family = AF_INET6;
        struct sockaddr_in6* addr6 = reinterpret_cast<struct sockaddr_in6*>(result->ai_addr);
        memcpy(data->addr.ipv6_addr, addr6->sin6_addr.s6_addr, 16);
      }
      freeaddrinfo(result);
      return uid;
    }
  }

  throw std::runtime_error("Failed to parse address: " + host + ":" + std::to_string(port));
}

UniqueId SocketBootstrapNetwork::GenerateUniqueIdWithLocalAddr(int port) {
  std::string local_addr = GetLocalNonLoopbackAddress();
  if (local_addr.empty()) {
    throw std::runtime_error("Failed to find non-loopback network interface");
  }

  return GenerateUniqueId(local_addr, port);
}

UniqueId SocketBootstrapNetwork::GenerateUniqueIdWithInterface(const std::string& interface_name,
                                                               int port) {
  struct ifaddrs *ifaddr, *ifa;
  bool found = false;
  std::string addr_str;

  if (getifaddrs(&ifaddr) == -1) {
    throw std::runtime_error("Failed to get network interfaces");
  }

  // Look for the specified interface
  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (std::string(ifa->ifa_name) != interface_name) continue;

    if (ifa->ifa_addr->sa_family == AF_INET) {
      struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
      char ip_str[INET_ADDRSTRLEN];

      if (inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, INET_ADDRSTRLEN)) {
        addr_str = std::string(ip_str);
        found = true;
        break;
      }
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {
      struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)ifa->ifa_addr;
      char ip_str[INET6_ADDRSTRLEN];

      if (inet_ntop(AF_INET6, &addr_in6->sin6_addr, ip_str, INET6_ADDRSTRLEN)) {
        addr_str = "[" + std::string(ip_str) + "]";
        found = true;
        break;
      }
    }
  }

  freeifaddrs(ifaddr);

  if (!found) {
    throw std::runtime_error("Network interface not found: " + interface_name);
  }

  return GenerateUniqueId(addr_str, port);
}

std::vector<std::string> SocketBootstrapNetwork::GetAvailableNetworkInterfaces() {
  std::vector<std::string> interfaces;
  struct ifaddrs *ifaddr, *ifa;

  if (getifaddrs(&ifaddr) == -1) {
    return interfaces;
  }

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;

    std::string name = ifa->ifa_name;
    std::string addr_info;

    if (ifa->ifa_addr->sa_family == AF_INET) {
      struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;
      char ip_str[INET_ADDRSTRLEN];

      if (inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, INET_ADDRSTRLEN)) {
        addr_info = std::string(ip_str) + " (IPv4)";

        // Check if this interface is already in our list
        bool exists = false;
        for (const auto& iface : interfaces) {
          if (iface.find(name + ":") == 0) {
            exists = true;
            break;
          }
        }

        if (!exists) {
          interfaces.push_back(name + ": " + addr_info);
        }
      }
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {
      struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)ifa->ifa_addr;
      char ip_str[INET6_ADDRSTRLEN];

      if (inet_ntop(AF_INET6, &addr_in6->sin6_addr, ip_str, INET6_ADDRSTRLEN)) {
        addr_info = std::string(ip_str) + " (IPv6)";
        interfaces.push_back(name + ": " + addr_info);
      }
    }
  }

  freeifaddrs(ifaddr);
  return interfaces;
}

std::string SocketBootstrapNetwork::GetLocalNonLoopbackAddress() {
  struct ifaddrs *ifaddr, *ifa;
  std::string result;

  if (getifaddrs(&ifaddr) == -1) {
    return result;
  }

  // Priority order: prefer non-loopback IPv4 addresses first
  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;

    if (ifa->ifa_addr->sa_family == AF_INET) {
      struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;

      // Skip loopback interface
      if (ntohl(addr_in->sin_addr.s_addr) == INADDR_LOOPBACK) continue;

      char ip_str[INET_ADDRSTRLEN];
      if (inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, INET_ADDRSTRLEN)) {
        result = std::string(ip_str);
        break;  // Prefer IPv4, so break immediately
      }
    }
  }

  // If no IPv4 found, try IPv6 (excluding loopback)
  if (result.empty()) {
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;
      if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;

      if (ifa->ifa_addr->sa_family == AF_INET6) {
        struct sockaddr_in6* addr_in6 = (struct sockaddr_in6*)ifa->ifa_addr;

        // Skip loopback (::1) and link-local addresses
        if (IN6_IS_ADDR_LOOPBACK(&addr_in6->sin6_addr) ||
            IN6_IS_ADDR_LINKLOCAL(&addr_in6->sin6_addr))
          continue;

        char ip_str[INET6_ADDRSTRLEN];
        if (inet_ntop(AF_INET6, &addr_in6->sin6_addr, ip_str, INET6_ADDRSTRLEN)) {
          result = std::string(ip_str);
          break;
        }
      }
    }
  }

  freeifaddrs(ifaddr);
  return result;
}

bool SocketBootstrapNetwork::FindNetworkInterface(SocketAddress& interface_addr) {
  struct ifaddrs *ifaddr, *ifa;
  bool found = false;

  if (getifaddrs(&ifaddr) == -1) {
    return false;
  }

  // Check if network interface is specified via environment variable
  const char* ifname = std::getenv("MORI_SOCKET_IFNAME");

  if (ifname) {
    // First try to find the specified interface
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;
      if (std::string(ifa->ifa_name) != ifname) continue;

      if (ifa->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;

        // Skip loopback interface
        if (ntohl(addr_in->sin_addr.s_addr) == INADDR_LOOPBACK) continue;

        // Skip interfaces that are down
        if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;

        memcpy(&interface_addr.sin, addr_in, sizeof(struct sockaddr_in));
        interface_addr.sin.sin_port = 0;  // Let system choose port
        found = true;
        break;
      }
    }
  }

  // If specified interface not found or not specified, fall back to auto-detection
  if (!found) {
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;

      if (ifa->ifa_addr->sa_family == AF_INET) {
        struct sockaddr_in* addr_in = (struct sockaddr_in*)ifa->ifa_addr;

        // Skip loopback interface
        if (ntohl(addr_in->sin_addr.s_addr) == INADDR_LOOPBACK) continue;

        // Skip interfaces that are down
        if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) continue;

        memcpy(&interface_addr.sin, addr_in, sizeof(struct sockaddr_in));
        interface_addr.sin.sin_port = 0;  // Let system choose port
        found = true;
        break;
      }
    }
  }

  freeifaddrs(ifaddr);
  return found;
}

bool SocketBootstrapNetwork::ParseAddress(const std::string& addr_str, SocketAddress& addr) {
  memset(&addr, 0, sizeof(addr));

  size_t colon_pos = addr_str.find_last_of(':');
  if (colon_pos == std::string::npos) return false;

  std::string host = addr_str.substr(0, colon_pos);
  int port = std::stoi(addr_str.substr(colon_pos + 1));

  // Handle IPv6 addresses in brackets
  if (host.front() == '[' && host.back() == ']') {
    host = host.substr(1, host.length() - 2);

    struct sockaddr_in6* addr6 = &addr.sin6;
    addr6->sin6_family = AF_INET6;
    addr6->sin6_port = htons(port);

    if (inet_pton(AF_INET6, host.c_str(), &addr6->sin6_addr) == 1) {
      return true;
    }
  } else {
    // Try IPv4 first
    struct sockaddr_in* addr4 = &addr.sin;
    addr4->sin_family = AF_INET;
    addr4->sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &addr4->sin_addr) == 1) {
      return true;
    }

    // Try hostname resolution
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &result) == 0) {
      if (result->ai_family == AF_INET) {
        memcpy(&addr.sin, result->ai_addr, sizeof(struct sockaddr_in));
      } else if (result->ai_family == AF_INET6) {
        memcpy(&addr.sin6, result->ai_addr, sizeof(struct sockaddr_in6));
      }
      freeaddrinfo(result);
      return true;
    }
  }

  return false;
}

std::string SocketBootstrapNetwork::AddressToString(const SocketAddress& addr) {
  char buf[INET6_ADDRSTRLEN];
  int port;

  if (addr.sa.sa_family == AF_INET) {
    inet_ntop(AF_INET, &addr.sin.sin_addr, buf, INET_ADDRSTRLEN);
    port = ntohs(addr.sin.sin_port);
    return std::string(buf) + ":" + std::to_string(port);
  } else if (addr.sa.sa_family == AF_INET6) {
    inet_ntop(AF_INET6, &addr.sin6.sin6_addr, buf, INET6_ADDRSTRLEN);
    port = ntohs(addr.sin6.sin6_port);
    return "[" + std::string(buf) + "]:" + std::to_string(port);
  }

  return "unknown";
}

void SocketBootstrapNetwork::ExtractAddressFromUniqueId(const UniqueId& uid, SocketAddress& addr) {
  memset(&addr, 0, sizeof(addr));

  const UniqueIdData* data = reinterpret_cast<const UniqueIdData*>(uid.internal);

  if (data->addr_family == AF_INET) {
    struct sockaddr_in* addr4 = &addr.sin;
    addr4->sin_family = AF_INET;
    addr4->sin_port = htons(data->port);            // data->port is in host byte order
    addr4->sin_addr.s_addr = data->addr.ipv4_addr;  // Already in network byte order
  } else if (data->addr_family == AF_INET6) {
    struct sockaddr_in6* addr6 = &addr.sin6;
    addr6->sin6_family = AF_INET6;
    addr6->sin6_port = htons(data->port);  // data->port is in host byte order
    memcpy(&addr6->sin6_addr.s6_addr, data->addr.ipv6_addr, 16);
  }
}

bool SocketBootstrapNetwork::InitializeSocket(Socket& sock, const SocketAddress* addr) {
  CloseSocket(sock);

  int family = AF_INET;
  if (addr && addr->sa.sa_family == AF_INET6) {
    family = AF_INET6;
  }

  sock.fd = socket(family, SOCK_STREAM, 0);
  if (sock.fd == -1) {
    return false;
  }

  // Set socket options
  int opt = 1;
  setsockopt(sock.fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  if (addr) {
    memcpy(&sock.addr, addr, sizeof(SocketAddress));
  } else {
    memcpy(&sock.addr, &local_interface_, sizeof(SocketAddress));
  }

  sock.state = SocketStateInitialized;

  // Set magic from unique_id_
  const UniqueIdData* data = reinterpret_cast<const UniqueIdData*>(unique_id_.internal);
  sock.magic = data->magic;

  return true;
}

bool SocketBootstrapNetwork::ListenSocket(Socket& sock) {
  if (sock.state != SocketStateInitialized) return false;

  int addr_len = (sock.addr.sa.sa_family == AF_INET) ? sizeof(struct sockaddr_in)
                                                     : sizeof(struct sockaddr_in6);

  if (bind(sock.fd, &sock.addr.sa, addr_len) == -1) {
    return false;
  }

  if (listen(sock.fd, 16) == -1) {
    return false;
  }

  // Get the actual bound address (in case port was 0)
  socklen_t len = sizeof(sock.addr);
  if (getsockname(sock.fd, &sock.addr.sa, &len) == -1) {
    return false;
  }

  sock.addr_len = len;
  sock.state = SocketStateListening;

  return true;
}

bool SocketBootstrapNetwork::ConnectSocket(Socket& sock, const SocketAddress& addr) {
  if (sock.state != SocketStateInitialized) return false;

  memcpy(&sock.addr, &addr, sizeof(SocketAddress));

  int addr_len =
      (addr.sa.sa_family == AF_INET) ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);

  // Set socket to non-blocking mode for timeout handling
  int flags = fcntl(sock.fd, F_GETFL, 0);
  fcntl(sock.fd, F_SETFL, flags | O_NONBLOCK);

  int result = connect(sock.fd, &addr.sa, addr_len);

  if (result == -1 && errno != EINPROGRESS) {
    return false;
  }

  if (result == 0) {
    // Connected immediately
    sock.state = SocketStateConnected;
  } else {
    // Wait for connection with timeout
    struct pollfd pfd;
    pfd.fd = sock.fd;
    pfd.events = POLLOUT;

    if (poll(&pfd, 1, 5000) <= 0) {  // 5 second timeout
      return false;
    }

    int error;
    socklen_t len = sizeof(error);
    if (getsockopt(sock.fd, SOL_SOCKET, SO_ERROR, &error, &len) == -1 || error != 0) {
      return false;
    }

    sock.state = SocketStateConnected;
  }

  // Restore blocking mode
  fcntl(sock.fd, F_SETFL, flags);

  return true;
}

bool SocketBootstrapNetwork::AcceptSocket(Socket& listen_sock, Socket& client_sock) {
  if (listen_sock.state != SocketStateListening) return false;

  socklen_t addr_len = sizeof(client_sock.addr);
  client_sock.fd = accept(listen_sock.fd, &client_sock.addr.sa, &addr_len);

  if (client_sock.fd == -1) {
    return false;
  }

  client_sock.addr_len = addr_len;
  client_sock.state = SocketStateConnected;
  client_sock.magic = listen_sock.magic;

  return true;
}

void SocketBootstrapNetwork::CloseSocket(Socket& sock) {
  if (sock.fd != -1) {
    close(sock.fd);
    sock.fd = -1;
  }
  sock.state = SocketStateClosed;
}

bool SocketBootstrapNetwork::SendData(Socket& sock, const void* data, size_t size) {
  if (sock.state != SocketStateConnected) return false;

  const char* buffer = static_cast<const char*>(data);
  size_t bytes_sent = 0;

  while (bytes_sent < size) {
    ssize_t result = send(sock.fd, buffer + bytes_sent, size - bytes_sent, 0);
    if (result <= 0) {
      return false;
    }
    bytes_sent += result;
  }

  return true;
}

bool SocketBootstrapNetwork::ReceiveData(Socket& sock, void* data, size_t size) {
  if (sock.state != SocketStateConnected) return false;

  char* buffer = static_cast<char*>(data);
  size_t bytes_received = 0;

  while (bytes_received < size) {
    ssize_t result = recv(sock.fd, buffer + bytes_received, size - bytes_received, 0);
    if (result <= 0) {
      return false;
    }
    bytes_received += result;
  }

  return true;
}

bool SocketBootstrapNetwork::SetupCommunicationRing() {
  // Initialize listening socket
  if (!InitializeSocket(listen_socket_)) {
    return false;
  }

  if (!ListenSocket(listen_socket_)) {
    return false;
  }

  // Exchange peer addresses using "phone home" protocol
  if (!PhoneHomeProtocol()) {
    return false;
  }

  // Setup ring connections
  int next_rank = (localRank + 1) % worldSize;
  int prev_rank = (localRank - 1 + worldSize) % worldSize;

  // Connect to next rank in ring
  if (!InitializeSocket(ring_send_socket_)) {
    return false;
  }

  if (!ConnectSocket(ring_send_socket_, peer_addresses_[next_rank])) {
    return false;
  }

  // Accept connection from previous rank
  if (!AcceptSocket(listen_socket_, ring_recv_socket_)) {
    return false;
  }

  return true;
}

bool SocketBootstrapNetwork::PhoneHomeProtocol() {
  if (worldSize == 1) {
    // Single process - no communication needed
    memcpy(&peer_addresses_[0], &listen_socket_.addr, sizeof(SocketAddress));
    return true;
  }

  if (localRank == 0) {
    // Root rank: setup root listening socket from UniqueId
    Socket root_listen_socket;
    SocketAddress root_addr;
    ExtractAddressFromUniqueId(unique_id_, root_addr);

    if (!InitializeSocket(root_listen_socket, &root_addr)) {
      return false;
    }

    if (!ListenSocket(root_listen_socket)) {
      return false;
    }

    // Collect addresses from all ranks (including self)
    std::vector<PeerInfo> peer_infos(worldSize);
    peer_infos[0].rank = 0;
    peer_infos[0].world_size = worldSize;
    memcpy(&peer_infos[0].listen_addr, &listen_socket_.addr, sizeof(SocketAddress));

    // Keep client sockets open to avoid race condition
    std::vector<Socket> client_sockets(worldSize);

    // Collect from other ranks - keep sockets open
    for (int i = 1; i < worldSize; i++) {
      Socket client_sock;
      if (!AcceptSocket(root_listen_socket, client_sock)) {
        CloseSocket(root_listen_socket);
        // Close any previously accepted sockets
        for (int j = 1; j < i; j++) {
          CloseSocket(client_sockets[j]);
        }
        return false;
      }

      PeerInfo info;
      if (!ReceiveData(client_sock, &info, sizeof(info))) {
        CloseSocket(client_sock);
        CloseSocket(root_listen_socket);
        // Close any previously accepted sockets
        for (int j = 1; j < i; j++) {
          CloseSocket(client_sockets[j]);
        }
        return false;
      }

      peer_infos[info.rank] = info;
      // Keep socket open and store it by rank
      client_sockets[info.rank] = client_sock;
    }

    // Now send peer addresses to each rank using the same sockets
    for (int i = 1; i < worldSize; i++) {
      int next_rank = (i + 1) % worldSize;

      // First send the next peer address (for ring connection)
      if (!SendData(client_sockets[i], &peer_infos[next_rank].listen_addr, sizeof(SocketAddress))) {
        CloseSocket(root_listen_socket);
        // Close all client sockets
        for (int j = 1; j < worldSize; j++) {
          CloseSocket(client_sockets[j]);
        }
        return false;
      }

      // Then send all peer addresses (for AllToAll operations)
      for (int j = 0; j < worldSize; j++) {
        if (!SendData(client_sockets[i], &peer_infos[j].listen_addr, sizeof(SocketAddress))) {
          CloseSocket(root_listen_socket);
          // Close all client sockets
          for (int k = 1; k < worldSize; k++) {
            CloseSocket(client_sockets[k]);
          }
          return false;
        }
      }

      // Now close the socket after sending all data
      CloseSocket(client_sockets[i]);
    }

    // Store all peer addresses
    for (int i = 0; i < worldSize; i++) {
      memcpy(&peer_addresses_[i], &peer_infos[i].listen_addr, sizeof(SocketAddress));
    }

    CloseSocket(root_listen_socket);

  } else {
    // Non-root rank: connect once, send info, then receive addresses
    SocketAddress root_addr;
    ExtractAddressFromUniqueId(unique_id_, root_addr);

    // Add small delay to ensure root is listening
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    Socket sock;
    if (!InitializeSocket(sock)) {
      return false;
    }

    if (!ConnectSocket(sock, root_addr)) {
      return false;
    }

    // Send my info to root
    PeerInfo my_info;
    my_info.rank = localRank;
    my_info.world_size = worldSize;
    memcpy(&my_info.listen_addr, &listen_socket_.addr, sizeof(SocketAddress));

    if (!SendData(sock, &my_info, sizeof(my_info))) {
      CloseSocket(sock);
      return false;
    }

    // Keep socket open and receive peer addresses on the same connection
    SocketAddress next_addr;
    if (!ReceiveData(sock, &next_addr, sizeof(next_addr))) {
      CloseSocket(sock);
      return false;
    }

    int next_rank = (localRank + 1) % worldSize;
    memcpy(&peer_addresses_[next_rank], &next_addr, sizeof(SocketAddress));

    // Receive all peer addresses from root (for AllToAll operations)
    for (int i = 0; i < worldSize; i++) {
      SocketAddress addr;
      if (!ReceiveData(sock, &addr, sizeof(addr))) {
        CloseSocket(sock);
        return false;
      }
      memcpy(&peer_addresses_[i], &addr, sizeof(SocketAddress));
    }

    CloseSocket(sock);
  }

  return true;
}

bool SocketBootstrapNetwork::SendMessage(int peer, int tag, const void* data, size_t size) {
  Socket sock;
  if (!InitializeSocket(sock)) {
    return false;
  }

  if (!ConnectSocket(sock, peer_addresses_[peer])) {
    return false;
  }

  // Send header: rank, tag, size
  int header[3] = {localRank, tag, static_cast<int>(size)};
  if (!SendData(sock, header, sizeof(header))) {
    CloseSocket(sock);
    return false;
  }

  // Send payload
  if (!SendData(sock, data, size)) {
    CloseSocket(sock);
    return false;
  }

  CloseSocket(sock);
  return true;
}

bool SocketBootstrapNetwork::ReceiveMessage(int peer, int tag, void* data, size_t size) {
  // Check unexpected connections first
  if (DequeueUnexpectedMessage(peer, tag, data, size)) {
    return true;
  }

  // Wait for new connections
  while (true) {
    Socket client_sock;
    if (!AcceptSocket(listen_socket_, client_sock)) {
      return false;
    }

    // Receive header
    int header[3];
    if (!ReceiveData(client_sock, header, sizeof(header))) {
      CloseSocket(client_sock);
      return false;
    }

    int recv_peer = header[0];
    int recv_tag = header[1];
    int recv_size = header[2];

    if (recv_peer == peer && recv_tag == tag && recv_size == static_cast<int>(size)) {
      // Expected message
      bool result = ReceiveData(client_sock, data, size);
      CloseSocket(client_sock);
      return result;
    } else {
      // Unexpected message - receive and store the data
      std::vector<char> buffer(recv_size);
      if (ReceiveData(client_sock, buffer.data(), recv_size)) {
        EnqueueUnexpectedMessage(recv_peer, recv_tag, buffer);
      }
      CloseSocket(client_sock);
    }
  }

  return false;
}

void SocketBootstrapNetwork::EnqueueUnexpectedMessage(int peer, int tag,
                                                      const std::vector<char>& data) {
  UnexpectedConnection* conn = new UnexpectedConnection();
  conn->peer = peer;
  conn->tag = tag;
  conn->data = data;
  conn->next = unexpected_connections_;
  unexpected_connections_ = conn;
}

bool SocketBootstrapNetwork::DequeueUnexpectedMessage(int peer, int tag, void* buffer,
                                                      size_t size) {
  UnexpectedConnection** current = &unexpected_connections_;

  while (*current) {
    if ((*current)->peer == peer && (*current)->tag == tag) {
      UnexpectedConnection* found = *current;
      if (found->data.size() == size) {
        memcpy(buffer, found->data.data(), size);
        *current = found->next;
        delete found;
        return true;
      } else {
        // Size mismatch - this should not happen
        MORI_APP_ERROR("Size mismatch in DequeueUnexpectedMessage: expected={}, got={}", size,
                       found->data.size());
        return false;
      }
    }
    current = &(*current)->next;
  }

  return false;
}

void SocketBootstrapNetwork::CleanupUnexpectedConnections() {
  while (unexpected_connections_) {
    UnexpectedConnection* next = unexpected_connections_->next;
    delete unexpected_connections_;
    unexpected_connections_ = next;
  }
}

void SocketBootstrapNetwork::Allgather(void* sendbuf, void* recvbuf, size_t sendcount) {
  if (!AllgatherImpl(sendbuf, recvbuf, sendcount)) {
    throw std::runtime_error("Allgather operation failed");
  }
}

bool SocketBootstrapNetwork::AllgatherImpl(const void* sendbuf, void* recvbuf, size_t sendcount) {
  if (worldSize == 1) {
    memcpy(recvbuf, sendbuf, sendcount);
    return true;
  }

  // Ring-based allgather algorithm
  char* recv_buffer = static_cast<char*>(recvbuf);

  // Copy own data to receive buffer
  memcpy(recv_buffer + localRank * sendcount, sendbuf, sendcount);

  // Ring algorithm: send data in ring pattern
  for (int step = 0; step < worldSize - 1; step++) {
    int send_rank = (localRank - step + worldSize) % worldSize;
    int recv_rank = (localRank - step - 1 + worldSize) % worldSize;

    // Send data to next rank in ring
    if (!SendData(ring_send_socket_, recv_buffer + send_rank * sendcount, sendcount)) {
      return false;
    }

    // Receive data from previous rank in ring
    if (!ReceiveData(ring_recv_socket_, recv_buffer + recv_rank * sendcount, sendcount)) {
      return false;
    }
  }
  MORI_APP_TRACE("SocketBootstrapNetwork Allgather completed");
  return true;
}

void SocketBootstrapNetwork::AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) {
  if (!AllToAllImpl(sendbuf, recvbuf, sendcount)) {
    throw std::runtime_error("AllToAll operation failed");
  }
}

bool SocketBootstrapNetwork::AllToAllImpl(const void* sendbuf, void* recvbuf, size_t sendcount) {
  if (worldSize == 1) {
    memcpy(recvbuf, sendbuf, sendcount);
    return true;
  }

  const char* send_buffer = static_cast<const char*>(sendbuf);
  char* recv_buffer = static_cast<char*>(recvbuf);

  // Copy own data first
  memcpy(recv_buffer + localRank * sendcount, send_buffer + localRank * sendcount, sendcount);

  // Use a direct pairwise exchange - each process exchanges with every other process
  for (int target = 0; target < worldSize; target++) {
    if (target == localRank) continue;  // Skip self

    // Create a consistent tag for the pair (sender, receiver)
    // Use a deterministic formula: tag = sender_rank * worldSize + receiver_rank + 1
    int send_tag = localRank * worldSize + target + 1;
    int recv_tag = target * worldSize + localRank + 1;

    // Determine order to avoid deadlock - lower rank acts first
    if (localRank < target) {
      // Send data designated for target from sendbuf[target]
      if (!SendMessage(target, send_tag, send_buffer + target * sendcount, sendcount)) {
        MORI_APP_ERROR("Rank {} failed to send to rank {}", localRank, target);
        return false;
      }
      // Receive data from target into recvbuf[target]
      if (!ReceiveMessage(target, recv_tag, recv_buffer + target * sendcount, sendcount)) {
        MORI_APP_ERROR("Rank {} failed to receive from rank {}", localRank, target);
        return false;
      }
    } else {
      // Receive data from target into recvbuf[target]
      if (!ReceiveMessage(target, recv_tag, recv_buffer + target * sendcount, sendcount)) {
        MORI_APP_ERROR("Rank {} failed to receive from rank {}", localRank, target);
        return false;
      }
      // Send data designated for target from sendbuf[target]
      if (!SendMessage(target, send_tag, send_buffer + target * sendcount, sendcount)) {
        MORI_APP_ERROR("Rank {} failed to send to rank {}", localRank, target);
        return false;
      }
    }
  }
  MORI_APP_TRACE("SocketBootstrapNetwork All2All completed");
  return true;
}

void SocketBootstrapNetwork::Barrier() {
  if (!BarrierImpl()) {
    throw std::runtime_error("Barrier operation failed");
  }
}

bool SocketBootstrapNetwork::BarrierImpl() {
  if (worldSize == 1) {
    return true;
  }

  int dummy = 0;

  for (int mask = 1; mask < worldSize; mask <<= 1) {
    int src = (localRank - mask + worldSize) % worldSize;
    int dst = (localRank + mask) % worldSize;

    // Send to dst and receive from src
    if (!SendMessage(dst, mask, &dummy, sizeof(dummy))) {
      return false;
    }

    if (!ReceiveMessage(src, mask, &dummy, sizeof(dummy))) {
      return false;
    }
  }

  return true;
}

}  // namespace application
}  // namespace mori
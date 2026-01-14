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

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "mori/application/bootstrap/base_bootstrap.hpp"

namespace mori {
namespace application {

#define UNIQUEID_PADDING (128 - sizeof(int))
struct UniqueId {
  int version;
  char internal[UNIQUEID_PADDING];  // 124 bytes for internal data

  UniqueId() : version(1) { memset(internal, 0, UNIQUEID_PADDING); }
};
static_assert(sizeof(UniqueId) == 128, "UniqueId must be 128 bytes");

struct UniqueIdData {
  uint64_t magic;
  uint16_t port;
  uint16_t addr_family;
  union {
    uint32_t ipv4_addr;
    uint8_t ipv6_addr[16];
  } addr;
  char padding[96];

  UniqueIdData() : magic(0), port(0), addr_family(AF_INET) {
    memset(&addr, 0, sizeof(addr));
    memset(padding, 0, sizeof(padding));
  }
};

// Socket address union for IPv4/IPv6 support
union SocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
  struct sockaddr_storage storage;
};

// Socket state enumeration
enum SocketState {
  SocketStateNone = 0,
  SocketStateInitialized,
  SocketStateListening,
  SocketStateConnecting,
  SocketStateConnected,
  SocketStateReady,
  SocketStateClosed,
  SocketStateError
};

// Socket structure
struct Socket {
  int fd;
  SocketState state;
  SocketAddress addr;
  int addr_len;
  uint64_t magic;

  Socket() : fd(-1), state(SocketStateNone), addr_len(0), magic(0) {
    memset(&addr, 0, sizeof(addr));
  }
};

// Peer information for communication ring
struct PeerInfo {
  int rank;
  int world_size;
  SocketAddress listen_addr;
};

// Unexpected connection for tag matching
struct UnexpectedConnection {
  int peer;
  int tag;
  std::vector<char> data;  // Store the actual data
  UnexpectedConnection* next;

  UnexpectedConnection() : peer(-1), tag(-1), next(nullptr) {}
};

class SocketBootstrapNetwork : public BootstrapNetwork {
 public:
  // Constructor with UniqueId for initialization
  SocketBootstrapNetwork(const UniqueId& unique_id, int rank, int world_size);
  ~SocketBootstrapNetwork();

  void Initialize() override;
  void Finalize() override;

  void Allgather(void* sendbuf, void* recvbuf, size_t sendcount) override;
  void AllToAll(void* sendbuf, void* recvbuf, size_t sendcount) override;
  void Barrier() override;

  // Static helper functions for UniqueId generation
  static UniqueId GenerateUniqueId();
  static UniqueId GenerateUniqueId(const std::string& host, int port);

  // Advanced UniqueId generation with network interface support
  static UniqueId GenerateUniqueIdWithLocalAddr(int port);
  static UniqueId GenerateUniqueIdWithInterface(const std::string& interface_name, int port);

  // Network interface discovery helpers
  static std::vector<std::string> GetAvailableNetworkInterfaces();
  static std::string GetLocalNonLoopbackAddress();

 private:
  // Core socket operations
  bool InitializeSocket(Socket& sock, const SocketAddress* addr = nullptr);
  bool ListenSocket(Socket& sock);
  bool ConnectSocket(Socket& sock, const SocketAddress& addr);
  bool AcceptSocket(Socket& listen_sock, Socket& client_sock);
  void CloseSocket(Socket& sock);

  // Network discovery
  bool FindNetworkInterface(SocketAddress& interface_addr);
  bool ParseAddress(const std::string& addr_str, SocketAddress& addr);
  std::string AddressToString(const SocketAddress& addr);

  // UniqueId helpers
  void ExtractAddressFromUniqueId(const UniqueId& uid, SocketAddress& addr);

  // Communication primitives
  bool SendData(Socket& sock, const void* data, size_t size);
  bool ReceiveData(Socket& sock, void* data, size_t size);

  // P2P communication with tag matching
  bool SendMessage(int peer, int tag, const void* data, size_t size);
  bool ReceiveMessage(int peer, int tag, void* data, size_t size);

  // Ring communication for collective operations
  bool SetupCommunicationRing();
  bool PhoneHomeProtocol();
  bool ExchangePeerAddresses();

  // Unexpected message management
  void EnqueueUnexpectedMessage(int peer, int tag, const std::vector<char>& data);
  bool DequeueUnexpectedMessage(int peer, int tag, void* buffer, size_t size);
  void CleanupUnexpectedConnections();

  // Collective operation implementations
  bool AllgatherImpl(const void* sendbuf, void* recvbuf, size_t sendcount);
  bool AllToAllImpl(const void* sendbuf, void* recvbuf, size_t sendcount);
  bool BarrierImpl();

 private:
  UniqueId unique_id_;
  bool initialized_;

  // Socket infrastructure
  Socket listen_socket_;
  Socket ring_send_socket_;
  Socket ring_recv_socket_;

  // Peer communication addresses
  std::vector<SocketAddress> peer_addresses_;

  // Unexpected connection queue for tag matching
  UnexpectedConnection* unexpected_connections_;

  // Network interface info
  SocketAddress local_interface_;

  // Constants
  static const uint64_t kDefaultMagic = 0x4D4F524953484D45ULL;  // "MORISHME"
  static const int kMaxRetries = 10;
  static const int kRetryDelay = 100;  // milliseconds
};

}  // namespace application
}  // namespace mori
#include <cassert>
#include <vector>

#include "mori/application/transport/tcp/tcp.hpp"
#include "mori/io/protocol.hpp"

using namespace mori::application;
using namespace mori::io;

using TCPInfo = std::pair<TCPContext*, TCPEndpointHandle>;
using TCPInfoPair = std::pair<TCPInfo, TCPInfo>;

TCPInfoPair PrepareTCPEndpoints() {
  std::string host = "127.0.0.1";

  TCPContext* context1 = new TCPContext(host, 0);
  TCPContext* context2 = new TCPContext(host, 0);

  context1->Listen();
  context2->Listen();
  assert((context1->GetPort() > 0) && (context2->GetPort() > 0));
  assert((context1->GetListenFd() >= 0) && (context2->GetListenFd() >= 0));

  TCPEndpointHandle eph1 = context1->Connect(host, context2->GetPort());
  TCPEndpointHandle eph2 = context2->Accept()[0];

  return {{context1, eph1}, {context2, eph2}};
}

void TestProtocol() {
  auto tcpInfoPair = PrepareTCPEndpoints();
  Protocol initiator(tcpInfoPair.first.second);
  Protocol target(tcpInfoPair.second.second);

  MessageRegEngine msg;
  msg.engineDesc.key = "initiator";
  msg.engineDesc.gpuId = 1;
  msg.engineDesc.hostname = "test";
  msg.engineDesc.backends = BackendBitmap({BackendType::RDMA, BackendType::TCP});
  msg.engineDesc.tcpHandle = tcpInfoPair.first.first->handle;
  msg.rdmaEph.psn = 22;
  msg.rdmaEph.qpn = 35;
  msg.rdmaEph.portId = 9999;
  msg.rdmaEph.ib.lid = 678;
  for (int i = 0; i < sizeof(msg.rdmaEph.eth.gid); i++) msg.rdmaEph.eth.gid[i] = i;
  for (int i = 0; i < sizeof(msg.rdmaEph.eth.mac); i++) msg.rdmaEph.eth.mac[i] = i;

  initiator.WriteMessageRegEngine(msg);

  MessageHeader hdr = target.ReadMessageHeader();
  assert(hdr.type == MessageType::RegEngine);
  MessageRegEngine recv = target.ReadMessageRegEngine(hdr.len);

  assert(recv.engineDesc.key == msg.engineDesc.key);
  assert(recv.engineDesc.gpuId == msg.engineDesc.gpuId);
  assert(recv.engineDesc.hostname == msg.engineDesc.hostname);
  assert(recv.engineDesc.backends == msg.engineDesc.backends);
  assert(recv.engineDesc.tcpHandle == msg.engineDesc.tcpHandle);
  assert(recv.rdmaEph == msg.rdmaEph);

  for (int i = 0; i < sizeof(msg.rdmaEph.eth.gid); i++) assert(recv.rdmaEph.eth.gid[i] == i);
  for (int i = 0; i < sizeof(msg.rdmaEph.eth.mac); i++) assert(recv.rdmaEph.eth.mac[i] == i);
}

int main() { TestProtocol(); }
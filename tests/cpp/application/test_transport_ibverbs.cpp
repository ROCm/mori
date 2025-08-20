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
#include <infiniband/verbs.h>

#include <cassert>
#include <vector>

#include "mori/application/transport/rdma/rdma.hpp"
#include "mori/application/utils/check.hpp"

using namespace mori::application;

void TestCompChannel() {
  RdmaContext context(RdmaBackendType::IBVerbs);
  RdmaDeviceList devices = context.GetRdmaDeviceList();
  ActiveDevicePortList activeDevicePortList = GetActiveDevicePortList(devices);
  assert(activeDevicePortList.size() > 0);

  ActiveDevicePort devicePort1 = activeDevicePortList[0];
  RdmaDevice* device1 = devicePort1.first;
  ActiveDevicePort devicePort2 = activeDevicePortList[1];
  RdmaDevice* device2 = devicePort2.first;

  RdmaDeviceContext* rdmaDeviceContext1 = device1->CreateRdmaDeviceContext();
  RdmaDeviceContext* rdmaDeviceContext2 = device2->CreateRdmaDeviceContext();

  RdmaEndpointConfig config;
  config.portId = devicePort1.second;
  config.withCompChannel = true;
  RdmaEndpoint ep1 = rdmaDeviceContext1->CreateRdmaEndpoint(config);

  config.portId = devicePort2.second;
  RdmaEndpoint ep2 = rdmaDeviceContext2->CreateRdmaEndpoint(config);

  rdmaDeviceContext1->ConnectEndpoint(ep1, ep2);
  rdmaDeviceContext2->ConnectEndpoint(ep2, ep1);

  void *buf1, *buf2;
  size_t bufSize = 1024;
  HIP_RUNTIME_CHECK(hipMalloc(&buf1, bufSize));
  HIP_RUNTIME_CHECK(hipMalloc(&buf2, bufSize));

  RdmaMemoryRegion mr1 = rdmaDeviceContext1->RegisterRdmaMemoryRegion(buf1, bufSize);
  RdmaMemoryRegion mr2 = rdmaDeviceContext2->RegisterRdmaMemoryRegion(buf2, bufSize);

  ibv_sge sge{};
  sge.addr = reinterpret_cast<uint64_t>(buf1);
  sge.length = bufSize;
  sge.lkey = mr1.lkey;

  ibv_send_wr wr{};
  ibv_send_wr* bad_wr = nullptr;
  wr.wr_id = 0;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr.rdma.remote_addr = reinterpret_cast<uint64_t>(buf2);
  wr.wr.rdma.rkey = mr2.rkey;
  ibv_req_notify_cq(ep1.ibvHandle.cq, 0);

  assert(!ibv_post_send(ep1.ibvHandle.qp, &wr, &bad_wr) && "ibv_post_send RDMA READ");
  ibv_cq* ev_cq;
  void* ev_ctx;
  printf("get cq event %d\n", ibv_get_cq_event(ep1.ibvHandle.cq->channel, &ev_cq, &ev_ctx));
}

int main() { TestCompChannel(); }

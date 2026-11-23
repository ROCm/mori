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
// CPU regression checks for injection ordering, resource retirement and staging.
// Link the injector directly with the verbs stand-ins below, never a NIC provider.
#include <cerrno>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>

#include "mori/io/common.hpp"
#include "mori/io/logging.hpp"
#include "src/io/rdma/fault_injector.hpp"

using namespace mori::io;
using namespace std::chrono_literals;

#undef ibv_query_port
extern "C" {
int ibv_query_qp(ibv_qp*, ibv_qp_attr* attr, int, ibv_qp_init_attr*) {
  attr->port_num = 1;
  return 0;
}
int ibv_query_device(ibv_context*, ibv_device_attr* attr) {
  attr->phys_port_cnt = 1;
  return 0;
}
int ibv_modify_qp(ibv_qp*, ibv_qp_attr*, int) { return 0; }
int ibv_query_port(ibv_context*, uint8_t, _compat_ibv_port_attr* attr) {
  reinterpret_cast<ibv_port_attr*>(attr)->state = IBV_PORT_ACTIVE;
  return 0;
}
int ibv_get_async_event(ibv_context*, ibv_async_event*) {
  errno = EAGAIN;
  return -1;
}
void ibv_ack_async_event(ibv_async_event*) {}
int ibv_get_cq_event(ibv_comp_channel*, ibv_cq**, void**) {
  errno = EAGAIN;
  return -1;
}
}

namespace {
void Require(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}

struct Endpoint {
  ibv_context context{};
  ibv_cq cq{};
  ibv_qp qp{};
  ibv_send_wr wr{};
  Endpoint() {
    context.ops.post_send = [](ibv_qp*, ibv_send_wr*, ibv_send_wr**) { return 0; };
    context.ops.poll_cq = [](ibv_cq*, int, ibv_wc*) { return 1; };
    cq.context = &context;
    qp.context = &context;
    qp.send_cq = qp.recv_cq = &cq;
    qp.qp_num = 17;
    wr.opcode = IBV_WR_RDMA_WRITE;
  }
  int Post() {
    ibv_send_wr* bad = nullptr;
    return FaultInjector::Instance().PostSend(&qp, &wr, &bad);
  }
  int Poll() {
    ibv_wc wc{};
    return FaultInjector::Instance().PollCq(&cq, 1, &wc);
  }
  ~Endpoint() { FaultInjector::Instance().ReleaseAsyncContext(&context); }
};

ibv_event_type Take(ibv_context* context) {
  ibv_async_event event{};
  auto& injector = FaultInjector::Instance();
  Require(injector.GetAsyncEvent(context, &event) == 0, "missing synthetic event");
  injector.AckAsyncEvent(&event);
  return event.event_type;
}

void PortRecoveryOrder() {
  Endpoint ep;
  auto& injector = FaultInjector::Instance();
  injector.WatchAsyncContext(&ep.context);
  std::promise<void> entered, release;
  auto enteredFuture = entered.get_future();
  auto released = release.get_future().share();
  struct PausedPost {
    std::promise<void>* entered;
    std::shared_future<void> released;
  } pause{&entered, released};
  ep.qp.qp_context = &pause;
  ep.context.ops.post_send = [](ibv_qp* qp, ibv_send_wr*, ibv_send_wr**) {
    auto* pause = static_cast<PausedPost*>(qp->qp_context);
    pause->entered->set_value();
    pause->released.wait();
    return 0;
  };
  injector.Arm(ParseFaultRule("port_down:op=write"));
  auto post = std::async(std::launch::async, [&] { return ep.Post(); });
  if (enteredFuture.wait_for(1s) != std::future_status::ready) {
    release.set_value();
    throw std::runtime_error("post did not reach the provider");
  }
  injector.Disarm();  // exactly between preparation and the provider's return
  release.set_value();
  Require(post.get() == 0, "port rule changed provider return");
  Require(Take(&ep.context) == IBV_EVENT_PORT_ERR, "ACTIVE overtook PORT_ERR");
  Require(Take(&ep.context) == IBV_EVENT_PORT_ACTIVE, "missing ordered recovery");
  ibv_port_attr attr{};
  Require(injector.QueryPort(&ep.context, 1, &attr) == 0 && attr.state == IBV_PORT_ACTIVE,
          "port override survived Disarm");
}

void ResourceRetirement() {
  Endpoint ep, other;
  auto& injector = FaultInjector::Instance();
  injector.WatchAsyncContext(&ep.context);
  injector.Arm(ParseFaultRule("async_event:value=cq_err:op=write"));
  Require(ep.Post() == 0, "CQ fault changed initial post");
  injector.Disarm();
  Require(ep.Poll() == 0, "Disarm healed a dead CQ");
  injector.ForgetAsyncEndpoint(&ep.context, ep.qp.qp_num, &ep.cq);
  Require(ep.Poll() == 1, "reused CQ address inherited its old fault");

  injector.Arm(ParseFaultRule("async_event:value=cq_err:op=write"));
  ep.Post();
  // Simulate the same CQ memory allocated on a different context. Cleanup must
  // use the recorded owner, without dereferencing that recycled object.
  ep.cq.context = &other.context;
  injector.ReleaseAsyncContext(&ep.context);
  Require(ep.Poll() == 1, "context retirement retained an old CQ marker");
  ep.cq.context = &ep.context;

  injector.Arm(ParseFaultRule("async_event:value=device_fatal:op=write"));
  Require(ep.Post() == 0, "device fault changed triggering post");
  injector.Disarm();
  injector.UnwatchAsyncContext(&ep.context);
  Require(ep.Post() == EIO, "watch removal healed a live failed device");
  injector.ReleaseAsyncContext(&ep.context);
  Require(ep.Post() == 0, "reused context address inherited its old fault");
  Require(!injector.Failed(), "resource retirement reported an injection error");
}

void NotificationStaging() {
  Endpoint ep;
  auto& injector = FaultInjector::Instance();
  injector.WatchAsyncContext(&ep.context);
  injector.Arm(ParseFaultRule("async_event:value=qp_fatal:op=send"));
  TransferStatus data;
  data.SetCode(StatusCode::IN_PROGRESS);
  std::promise<void> waiting;
  auto waitingFuture = waiting.get_future();
  std::atomic<bool> entered{false};
  data.SetProgressCallback([&] {
    if (!entered.exchange(true)) waiting.set_value();
  });
  auto stage = std::async(std::launch::async, [&] { injector.BeforeNotify(&ep.qp, &data); });
  Require(waitingFuture.wait_for(1s) == std::future_status::ready &&
              stage.wait_for(0ms) == std::future_status::timeout,
          "notification fault did not wait for the data");
  data.Update(StatusCode::SUCCESS, "data complete");
  Require(stage.wait_for(1s) == std::future_status::ready, "notification staging did not finish");
  stage.get();
  ep.wr.opcode = IBV_WR_SEND;
  Require(ep.Post() == 0 && injector.Fired() == 1 && data.Succeeded() && !injector.Failed(),
          "notification fault was not staged after successful data");
  injector.Disarm();

  injector.Arm(ParseFaultRule("async_event:value=qp_fatal:op=send"));
  TransferStatus failedData;
  failedData.Update(StatusCode::ERR_RDMA_OP, "data failed");
  injector.BeforeNotify(&ep.qp, &failedData);
  Require(injector.Failed() && injector.Fired() == 0,
          "failed data was treated as a notification-reporting reproduction");
}
}  // namespace

int main() {
  SetLogLevel("error");
  try {
    PortRecoveryOrder();
    ResourceRetirement();
    NotificationStaging();
    std::cout << "PASS: port recovery order, resource retirement, notification staging\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}

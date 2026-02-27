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

#include <poll.h>
#include <sys/eventfd.h>

#include <thread>

#include "src/io/tcp/tcp_types.hpp"

namespace mori {
namespace io {

class DataConnectionWorker {
 public:
  DataConnectionWorker(int fd, EngineKey peer, PinnedStagingPool* staging)
      : fd_(fd), peerKey_(std::move(peer)), staging_(staging) {
    notifyFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    wakeFd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  }

  ~DataConnectionWorker() {
    Stop();
    if (notifyFd_ >= 0) {
      close(notifyFd_);
      notifyFd_ = -1;
    }
    if (wakeFd_ >= 0) {
      close(wakeFd_);
      wakeFd_ = -1;
    }
  }

  DataConnectionWorker(const DataConnectionWorker&) = delete;
  DataConnectionWorker& operator=(const DataConnectionWorker&) = delete;

  void Start() {
    if (running_.load()) return;
    running_.store(true);
    thread_ = std::thread(&DataConnectionWorker::Run, this);
  }

  void Stop() {
    bool was = running_.exchange(false);
    if (!was) return;
    WakeWorker();
    if (thread_.joinable()) thread_.join();
  }

  int NotifyFd() const { return notifyFd_; }
  int Fd() const { return fd_; }

  void SubmitSend(SendItem item) {
    {
      std::lock_guard<std::mutex> lk(sendMu_);
      sendQ_.push_back(std::move(item));
    }
    WakeWorker();
  }

  void RegisterRecvTarget(TransferUniqueId opId, const WorkerRecvTarget& target) {
    std::lock_guard<std::mutex> lk(targetMu_);
    recvTargets_[opId] = target;
  }

  void RemoveRecvTarget(TransferUniqueId opId) {
    std::lock_guard<std::mutex> lk(targetMu_);
    recvTargets_.erase(opId);
  }

  void DrainEvents(std::deque<WorkerEvent>& out) {
    uint64_t v;
    while (::read(notifyFd_, &v, sizeof(v)) > 0) {
    }
    std::lock_guard<std::mutex> lk(eventMu_);
    while (!eventQ_.empty()) {
      out.push_back(std::move(eventQ_.front()));
      eventQ_.pop_front();
    }
  }

 private:
  void WakeWorker() {
    uint64_t one = 1;
    ::write(wakeFd_, &one, sizeof(one));
  }

  void NotifyMain() {
    uint64_t one = 1;
    ::write(notifyFd_, &one, sizeof(one));
  }

  void PostEvent(WorkerEvent ev) {
    {
      std::lock_guard<std::mutex> lk(eventMu_);
      eventQ_.push_back(std::move(ev));
    }
    NotifyMain();
  }

  void PostRecvDone(TransferUniqueId opId, uint8_t lane, uint64_t laneLen, bool discarded = false) {
    WorkerEvent ev;
    ev.type = WorkerEventType::RECV_DONE;
    ev.peerKey = peerKey_;
    ev.opId = opId;
    ev.lane = lane;
    ev.laneLen = laneLen;
    ev.discarded = discarded;
    PostEvent(std::move(ev));
  }

  void PostEarlyData(TransferUniqueId opId, uint8_t lane, uint64_t laneLen,
                     std::shared_ptr<PinnedBuf> buf) {
    WorkerEvent ev;
    ev.type = WorkerEventType::EARLY_DATA;
    ev.peerKey = peerKey_;
    ev.opId = opId;
    ev.lane = lane;
    ev.laneLen = laneLen;
    ev.earlyBuf = std::move(buf);
    PostEvent(std::move(ev));
  }

  void PostCallback(std::function<void()> cb) {
    WorkerEvent ev;
    ev.type = WorkerEventType::SEND_CALLBACK;
    ev.callback = std::move(cb);
    PostEvent(std::move(ev));
  }

  void PostError(const std::string& msg) {
    WorkerEvent ev;
    ev.type = WorkerEventType::CONN_ERROR;
    ev.peerKey = peerKey_;
    ev.errorMsg = msg;
    PostEvent(std::move(ev));
  }

  void Run() {
    MORI_IO_TRACE("TCP: DataWorker fd={} peer={} started", fd_, peerKey_);
    struct pollfd pfds[2];
    pfds[0].fd = fd_;
    pfds[1].fd = wakeFd_;
    pfds[1].events = POLLIN;

    while (running_.load()) {
      bool hasSend;
      {
        std::lock_guard<std::mutex> lk(sendMu_);
        hasSend = !sendQ_.empty();
      }

      pfds[0].events = POLLIN | (hasSend ? POLLOUT : 0);
      pfds[0].revents = 0;
      pfds[1].revents = 0;

      int n = ::poll(pfds, 2, hasSend ? 0 : 1);
      if (n < 0) {
        if (errno == EINTR) continue;
        PostError(std::string("poll failed: ") + strerror(errno));
        break;
      }

      if (pfds[1].revents & POLLIN) {
        uint64_t v;
        while (::read(wakeFd_, &v, sizeof(v)) > 0) {
        }
      }

      if (pfds[0].revents & (POLLERR | POLLHUP | POLLNVAL)) {
        PostError("data connection error/hangup");
        break;
      }

      if (pfds[0].revents & POLLOUT) {
        if (!ProcessSend()) break;
      }

      if (pfds[0].revents & POLLIN) {
        if (!ProcessRecv()) break;
      }
    }
    MORI_IO_TRACE("TCP: DataWorker fd={} peer={} exiting", fd_, peerKey_);
  }

  bool ProcessSend() {
    std::deque<SendItem> batch;
    {
      std::lock_guard<std::mutex> lk(sendMu_);
      batch.swap(sendQ_);
    }

    for (auto& item : batch) {
      while (!item.Done()) {
        constexpr size_t kMaxIov = 64;
        iovec iov[kMaxIov];
        size_t cnt = 0;
        for (size_t i = item.idx; i < item.iov.size() && cnt < kMaxIov; ++i) {
          iov[cnt] = item.iov[i];
          if (i == item.idx && item.off > 0) {
            iov[cnt].iov_base = static_cast<uint8_t*>(iov[cnt].iov_base) + item.off;
            iov[cnt].iov_len -= item.off;
          }
          cnt++;
        }
        msghdr msg{};
        msg.msg_iov = iov;
        msg.msg_iovlen = cnt;
        ssize_t n = ::sendmsg(fd_, &msg, MSG_NOSIGNAL | item.flags);
        if (n < 0) {
          if (IsWouldBlock(errno)) {
            goto requeue_batch;
          }
          PostError(std::string("sendmsg failed: ") + strerror(errno));
          return false;
        }
        if (n == 0) {
          goto requeue_batch;
        }
        item.Advance(static_cast<size_t>(n));
      }
      if (item.onDone) {
        PostCallback(std::move(item.onDone));
      }
    }
    return true;

  requeue_batch: {
    std::lock_guard<std::mutex> lk(sendMu_);
    for (auto rit = batch.rbegin(); rit != batch.rend(); ++rit) {
      if (!rit->Done()) {
        sendQ_.push_front(std::move(*rit));
      }
    }
  }
    return true;
  }

  bool ProcessRecv() {
    while (true) {
      while (hdrGot_ < tcp::kDataHeaderSize) {
        ssize_t n = ::recv(fd_, hdrBuf_ + hdrGot_, tcp::kDataHeaderSize - hdrGot_, 0);
        if (n < 0) {
          if (IsWouldBlock(errno)) return true;
          PostError(std::string("recv header failed: ") + strerror(errno));
          return false;
        }
        if (n == 0) {
          PostError("data connection closed by peer");
          return false;
        }
        hdrGot_ += static_cast<size_t>(n);
      }
      hdrGot_ = 0;

      tcp::DataHeaderView hv;
      if (!tcp::TryParseDataHeader(hdrBuf_, tcp::kDataHeaderSize, &hv)) {
        PostError("bad data header");
        return false;
      }

      const uint8_t lane = static_cast<uint8_t>(hv.opId & kLaneMask);
      const TransferUniqueId userOpId = static_cast<TransferUniqueId>(ToUserOpId(hv.opId));
      const uint64_t payloadLen = hv.payloadLen;

      WorkerRecvTarget target;
      bool hasTarget = false;
      {
        std::lock_guard<std::mutex> lk(targetMu_);
        auto it = recvTargets_.find(userOpId);
        if (it != recvTargets_.end()) {
          target = it->second;
          hasTarget = true;
        }
      }

      if (hasTarget && !target.discard) {
        const LaneSpan span = ComputeLaneSpan(target.totalLen, target.lanesTotal, lane);
        if (span.len != payloadLen) {
          MORI_IO_WARN("TCP: worker recv op {} lane {} len mismatch expected={} got={}", userOpId,
                       (uint32_t)lane, span.len, payloadLen);
          if (!DiscardPayload(payloadLen)) return false;
          PostRecvDone(userOpId, lane, payloadLen, true);
        } else if (target.toGpu) {
          uint8_t* dst = reinterpret_cast<uint8_t*>(target.pinned->ptr) + span.off;
          if (!RecvExact(dst, payloadLen)) return false;
          PostRecvDone(userOpId, lane, payloadLen);
        } else {
          auto segs = SliceSegments(target.segs, span.off, span.len);
          if (!RecvIntoSegments(reinterpret_cast<uint8_t*>(target.cpuBase), segs, payloadLen))
            return false;
          PostRecvDone(userOpId, lane, payloadLen);
        }
      } else if (hasTarget && target.discard) {
        if (!DiscardPayload(payloadLen)) return false;
        PostRecvDone(userOpId, lane, payloadLen, true);
      } else {
        if (payloadLen == 0) {
          PostEarlyData(userOpId, lane, 0, nullptr);
        } else {
          auto buf = staging_->Acquire(static_cast<size_t>(payloadLen));
          if (!buf) {
            if (!DiscardPayload(payloadLen)) return false;
            PostRecvDone(userOpId, lane, payloadLen, true);
          } else {
            if (!RecvExact(reinterpret_cast<uint8_t*>(buf->ptr), payloadLen)) return false;
            PostEarlyData(userOpId, lane, payloadLen, std::move(buf));
          }
        }
      }
    }
    return true;
  }

  bool RecvExact(uint8_t* dst, uint64_t len) {
    uint64_t got = 0;
    while (got < len) {
      const size_t want = static_cast<size_t>(std::min<uint64_t>(len - got, 16ULL * 1024 * 1024));
      ssize_t n = ::recv(fd_, dst + got, want, 0);
      if (n < 0) {
        if (IsWouldBlock(errno)) continue;
        PostError(std::string("recv payload failed: ") + strerror(errno));
        return false;
      }
      if (n == 0) {
        PostError("data connection closed during recv");
        return false;
      }
      got += static_cast<uint64_t>(n);
    }
    return true;
  }

  bool RecvIntoSegments(uint8_t* base, const std::vector<Segment>& segs, uint64_t totalLen) {
    uint64_t remaining = totalLen;
    size_t segIdx = 0;
    uint64_t segOff = 0;
    while (remaining > 0 && segIdx < segs.size()) {
      const Segment& seg = segs[segIdx];
      const uint64_t segRemain = seg.len - segOff;
      const size_t want = static_cast<size_t>(
          std::min<uint64_t>(remaining, std::min<uint64_t>(segRemain, 16ULL * 1024 * 1024)));
      uint8_t* dst = base + seg.off + segOff;
      ssize_t n = ::recv(fd_, dst, want, 0);
      if (n < 0) {
        if (IsWouldBlock(errno)) continue;
        PostError(std::string("recv seg failed: ") + strerror(errno));
        return false;
      }
      if (n == 0) {
        PostError("data connection closed during seg recv");
        return false;
      }
      remaining -= static_cast<uint64_t>(n);
      segOff += static_cast<uint64_t>(n);
      if (segOff >= seg.len) {
        segIdx++;
        segOff = 0;
      }
    }
    return (remaining == 0);
  }

  bool DiscardPayload(uint64_t len) {
    uint8_t tmp[65536];
    uint64_t remaining = len;
    while (remaining > 0) {
      const size_t want = static_cast<size_t>(std::min<uint64_t>(remaining, sizeof(tmp)));
      ssize_t n = ::recv(fd_, tmp, want, 0);
      if (n < 0) {
        if (IsWouldBlock(errno)) continue;
        PostError(std::string("recv discard failed: ") + strerror(errno));
        return false;
      }
      if (n == 0) {
        PostError("data connection closed during discard");
        return false;
      }
      remaining -= static_cast<uint64_t>(n);
    }
    return true;
  }

  int fd_;
  EngineKey peerKey_;
  PinnedStagingPool* staging_;
  std::atomic<bool> running_{false};
  std::thread thread_;
  int notifyFd_{-1};
  int wakeFd_{-1};

  std::mutex sendMu_;
  std::deque<SendItem> sendQ_;

  std::mutex targetMu_;
  std::unordered_map<TransferUniqueId, WorkerRecvTarget> recvTargets_;

  std::mutex eventMu_;
  std::deque<WorkerEvent> eventQ_;

  uint8_t hdrBuf_[tcp::kDataHeaderSize]{};
  size_t hdrGot_{0};
};

}  // namespace io
}  // namespace mori

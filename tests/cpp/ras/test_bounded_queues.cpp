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
#include <gtest/gtest.h>
#include <poll.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "src/ras/event_queue.hpp"

using namespace mori::ras;

namespace {

constexpr size_t kEngineRegistered = 0;
constexpr size_t kEngineUnregistered = 1;

// An event that names its producer and sequence number, with a key longer than the small-string
// buffer, so the queue really copies heap data.
EngineRegistered MakeEvent(int producer, int seq) {
  return EngineRegistered{std::to_string(producer) + "/" + std::to_string(seq) +
                          "/engine-with-a-name-longer-than-sso"};
}

int ProducerOf(const RasEvent& ev) { return std::stoi(std::get<EngineRegistered>(ev).engineKey); }

int SeqOf(const RasEvent& ev) {
  const std::string& key = std::get<EngineRegistered>(ev).engineKey;
  return std::stoi(key.substr(key.find('/') + 1));
}

bool Readable(int fd) {
  pollfd p{fd, POLLIN, 0};
  return poll(&p, 1, 0) == 1 && (p.revents & POLLIN);
}

}  // namespace

TEST(RasEventQueue, CapacityRoundsUpToAPowerOfTwo) {
  EXPECT_EQ(EventQueue(5).Stats().capacity, 8u);
  EXPECT_EQ(EventQueue(8).Stats().capacity, 8u);
  EXPECT_EQ(EventQueue(0).Stats().capacity, 2u);
  EXPECT_EQ(EventQueue(1000).Stats().capacity, 1024u);
}

TEST(RasEventQueue, AFullQueueDropsAndCountsByKind) {
  EventQueue q(8);
  for (int i = 0; i < 8; ++i) EXPECT_TRUE(q.Enqueue(MakeEvent(0, i))) << i;
  EXPECT_FALSE(q.Enqueue(EngineRegistered{"e1"}));
  EXPECT_FALSE(q.Enqueue(EngineUnregistered{"e1"}));
  EXPECT_FALSE(q.Enqueue(EngineUnregistered{"e2"}));

  EventQueueStats s = q.Stats();
  EXPECT_EQ(s.enqueued, 8u);
  EXPECT_EQ(s.dropped, 3u);
  EXPECT_EQ(s.droppedByKind[kEngineRegistered], 1u);
  EXPECT_EQ(s.droppedByKind[kEngineUnregistered], 2u);

  std::vector<int> seen;
  q.ConsumeWakeup();
  EXPECT_EQ(q.Drain([&](RasEvent& ev) { seen.push_back(SeqOf(ev)); }), 8u);
  EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
  EXPECT_EQ(q.Stats().depthHighWater, 8u);

  // Draining frees the slots.
  EXPECT_TRUE(q.Enqueue(EngineUnregistered{"e1"}));
  size_t n = q.Drain([&](RasEvent& ev) {
    EXPECT_EQ(ev.index(), kEngineUnregistered);
    EXPECT_EQ(std::get<EngineUnregistered>(ev).engineKey, "e1");
  });
  EXPECT_EQ(n, 1u);
  EXPECT_EQ(q.Stats().enqueued, 9u);
  EXPECT_EQ(q.Stats().dropped, 3u);
}

TEST(RasEventQueue, DrainHonorsItsLimit) {
  EventQueue q(16);
  for (int i = 0; i < 10; ++i) ASSERT_TRUE(q.Enqueue(MakeEvent(0, i)));
  std::vector<int> seen;
  auto collect = [&](RasEvent& ev) { seen.push_back(SeqOf(ev)); };
  EXPECT_EQ(q.Drain(collect, 4), 4u);
  EXPECT_EQ(q.Drain(collect, 100), 6u);
  EXPECT_EQ(q.Drain(collect), 0u);
  EXPECT_EQ(seen, (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
}

TEST(RasEventQueue, EventsOwnCopiesOfTheirData) {
  EventQueue q(4);
  {
    std::string key(100, 'k');
    ASSERT_TRUE(q.Enqueue(EngineRegistered{key}));
    key.assign(100, 'x');
  }
  q.Drain([&](RasEvent& ev) {
    EXPECT_EQ(std::get<EngineRegistered>(ev).engineKey, std::string(100, 'k'));
  });
}

TEST(RasEventQueue, QueuedEventsAreDestroyedWithTheQueue) {
  // Run under AddressSanitizer this also proves nothing leaks.
  auto q = std::make_unique<EventQueue>(64);
  for (int i = 0; i < 40; ++i) ASSERT_TRUE(q->Enqueue(MakeEvent(1, i)));
  q.reset();
}

TEST(RasEventQueue, TheEventFdWakesOnceAndCoalescesABurst) {
  EventQueue q(64);
  ASSERT_TRUE(q.HasWakeFd());
  EXPECT_FALSE(Readable(q.WakeFd()));
  ASSERT_TRUE(q.Enqueue(EngineRegistered{"a"}));
  EXPECT_TRUE(Readable(q.WakeFd()));
  for (int i = 0; i < 20; ++i) ASSERT_TRUE(q.Enqueue(EngineRegistered{"b"}));
  uint64_t count = 0;
  ASSERT_EQ(read(q.WakeFd(), &count, sizeof(count)), static_cast<ssize_t>(sizeof(count)));
  EXPECT_EQ(count, 1u);  // 21 events, one write

  q.ConsumeWakeup();
  EXPECT_FALSE(Readable(q.WakeFd()));
  EXPECT_EQ(q.Drain([](RasEvent&) {}), 21u);
  ASSERT_TRUE(q.Enqueue(EngineRegistered{"c"}));
  EXPECT_TRUE(Readable(q.WakeFd()));
  // A drop doesn't wake the consumer.
  q.ConsumeWakeup();
  for (int i = 0; i < 63; ++i) ASSERT_TRUE(q.Enqueue(EngineRegistered{"d"}));
  q.ConsumeWakeup();
  EXPECT_FALSE(q.Enqueue(EngineRegistered{"e"}));
  EXPECT_FALSE(Readable(q.WakeFd()));
}

TEST(RasEventQueue, AFullQueueNeverBlocksItsProducers) {
  EventQueue q(16);
  for (int i = 0; i < 16; ++i) ASSERT_TRUE(q.Enqueue(MakeEvent(0, i)));
  auto start = std::chrono::steady_clock::now();
  constexpr int kAttempts = 200000;
  for (int i = 0; i < kAttempts; ++i) ASSERT_FALSE(q.Enqueue(EngineUnregistered{"x"}));
  auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::seconds(2));
  EXPECT_EQ(q.Stats().dropped, static_cast<uint64_t>(kAttempts));
  EXPECT_EQ(q.Stats().droppedByKind[kEngineUnregistered], static_cast<uint64_t>(kAttempts));
}

TEST(RasEventQueue, ConcurrentProducersAccountForEveryEvent) {
  constexpr int kProducers = 8;
  constexpr int kPerProducer = 20000;
  EventQueue q(256);
  std::atomic<bool> producersDone{false};
  std::atomic<int> started{0};
  std::vector<uint64_t> acceptedBy(kProducers, 0);
  std::vector<uint64_t> droppedBy(kProducers, 0);

  std::vector<std::vector<int>> received(kProducers);
  std::thread consumer([&] {
    auto take = [&](RasEvent& ev) { received[ProducerOf(ev)].push_back(SeqOf(ev)); };
    while (!producersDone.load(std::memory_order_acquire)) {
      pollfd pfd{q.WakeFd(), POLLIN, 1};
      poll(&pfd, 1, 1);
      q.ConsumeWakeup();
      q.Drain(take, 64);  // slower than the producers, so the queue fills and drops
      std::this_thread::yield();
    }
    q.ConsumeWakeup();
    q.Drain(take);
  });

  std::vector<std::thread> producers;
  for (int t = 0; t < kProducers; ++t) {
    producers.emplace_back([&, t] {
      started.fetch_add(1);
      while (started.load() < kProducers) {
      }
      for (int i = 0; i < kPerProducer; ++i) {
        if (q.Enqueue(MakeEvent(t, i))) {
          ++acceptedBy[t];
        } else {
          ++droppedBy[t];
        }
      }
    });
  }
  for (auto& th : producers) th.join();
  producersDone.store(true, std::memory_order_release);
  consumer.join();

  uint64_t accepted = 0, dropped = 0;
  for (int t = 0; t < kProducers; ++t) {
    EXPECT_EQ(acceptedBy[t] + droppedBy[t], static_cast<uint64_t>(kPerProducer));
    EXPECT_EQ(received[t].size(), acceptedBy[t]) << "producer " << t;
    // Each producer's accepted events arrive in the order it sent them.
    for (size_t i = 1; i < received[t].size(); ++i) {
      ASSERT_LT(received[t][i - 1], received[t][i]) << "producer " << t;
    }
    accepted += acceptedBy[t];
    dropped += droppedBy[t];
  }
  EventQueueStats s = q.Stats();
  EXPECT_EQ(s.enqueued, accepted);
  EXPECT_EQ(s.dropped, dropped);
  EXPECT_EQ(s.droppedByKind[kEngineRegistered], dropped);
  EXPECT_EQ(accepted + dropped, static_cast<uint64_t>(kProducers) * kPerProducer);
  EXPECT_LE(s.depthHighWater, s.capacity);
  std::printf("producers: %llu accepted, %llu dropped\n", static_cast<unsigned long long>(accepted),
              static_cast<unsigned long long>(dropped));
}

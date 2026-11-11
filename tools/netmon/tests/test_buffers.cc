#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <thread>

#include "batch_queue.h"
#include "guarded_buffer.h"
#include "harness.h"

using namespace netmon;

TEST(GuardedBufferStopsKernelWrites) {
  GuardedBuffer g;
  CHECK(g.Allocate(10000).ok());
  CHECK(g.capacity() >= 10000);
  int p[2];
  CHECK(pipe(p) == 0);
  std::vector<uint8_t> src(8192, 0xab);
  CHECK(write(p[1], src.data(), src.size()) == static_cast<ssize_t>(src.size()));
  // Ask the kernel to copy 8 KiB starting 100 bytes before the guard page. copy_to_user may
  // copy a prefix and return a short count, or fail with EFAULT; it must never go past.
  uint8_t* near_end = g.data() + g.capacity() - 100;
  ssize_t n = read(p[0], near_end, 8192);
  CHECK((n >= 0 && n <= 100) || (n < 0 && errno == EFAULT));
  close(p[0]);
  close(p[1]);
}

TEST(QueueQuotaAndClose) {
  BatchQueue q(2, 2);
  auto mk = [](int s) { return std::make_unique<Batch>(s, 5, 4); };
  std::unique_ptr<Batch> b;
  for (int i = 0; i < 2; i++) {
    b = mk(0);
    CHECK(q.Push(b) == BatchQueue::PushResult::kOk);
    CHECK(!b);
  }
  b = mk(0);
  CHECK(q.Push(b) == BatchQueue::PushResult::kFull);
  CHECK(b);  // caller keeps it on kFull
  auto other = mk(1);
  CHECK(q.Push(other) == BatchQueue::PushResult::kOk);  // quota is per stream
  bool drained;
  auto got = q.Pop(std::chrono::milliseconds(1), &drained);
  CHECK(got && got->stream == 0);
  CHECK(q.Push(b) == BatchQueue::PushResult::kOk);  // room again
  q.Close();
  auto late = mk(1);
  CHECK(q.Push(late) == BatchQueue::PushResult::kClosed);
  int n = 0;
  while (q.Pop(std::chrono::milliseconds(1), &drained)) n++;
  CHECK_EQ(n, 3);
  CHECK(drained);
}

TEST(QueueConcurrentProducers) {
  constexpr int kStreams = 8, kPer = 2000;
  BatchQueue q(kStreams, 64);
  std::atomic<long> pushed{0}, full{0};
  std::vector<std::thread> ts;
  for (int s = 0; s < kStreams; s++) {
    ts.emplace_back([&, s] {
      for (int i = 0; i < kPer; i++) {
        auto b = std::make_unique<Batch>(s, 4, 1);
        b->count = 1;
        b->words[0] = static_cast<uint64_t>(i);
        if (q.Push(b) == BatchQueue::PushResult::kOk)
          pushed++;
        else
          full++;
      }
    });
  }
  long popped = 0;
  std::vector<long> last(kStreams, -1);
  bool ordered = true;
  std::thread consumer([&] {
    bool drained = false;
    while (!drained) {
      auto b = q.Pop(std::chrono::milliseconds(10), &drained);
      if (!b) continue;
      popped++;
      long v = static_cast<long>(b->words[0]);
      if (v <= last[static_cast<size_t>(b->stream)]) ordered = false;
      last[static_cast<size_t>(b->stream)] = v;
    }
  });
  for (auto& t : ts) t.join();
  q.Close();
  consumer.join();
  CHECK_EQ(pushed.load() + full.load(), static_cast<long>(kStreams * kPer));
  CHECK_EQ(popped, pushed.load());
  CHECK(ordered);  // per-stream FIFO
}

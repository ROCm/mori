// The writer thread owns ALL file I/O of a recording: output directory and lock, stream files,
// initial and final meta.json, record writes, truncation on failure. The main thread never
// touches the filesystem after startup, so an NFS hang can only stall this thread and the
// main thread's deadlines keep working.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "batch_queue.h"
#include "util.h"

namespace netmon {

struct WriterStream {
  std::string file;
  std::string label;  // "benic1p1.rdma"
  size_t width = 0;   // u64 words per record
  int key_tx = -1;    // value index (after the header) for the console summary, or -1
  int key_rx = -1;
};

class Writer {
 public:
  // Test hook: replaces ::write for record and meta writes.
  using WriteFn = ssize_t (*)(int, const void*, size_t);
  static WriteFn write_fn;

  Writer(BatchQueue* queue, std::string dir, bool quiet);
  ~Writer();

  // Each Request* posts a command; Wait() blocks until it completes or the deadline passes
  // (returns false on timeout; the command may still be running).
  void RequestPrepare();  // create/verify DIR, exclusive .lock
  void RequestCreate(std::vector<WriterStream> streams, std::string initial_meta);
  // Final status is "failed" if a write error happened, else `status`.
  using MetaBuilder =
      std::function<std::string(const std::string& status, const std::string& write_error)>;
  void RequestFinish(bool drain, std::string status, MetaBuilder build_meta);
  void RequestAbort();  // remove only what this run created, then exit
  bool Wait(uint64_t deadline_mono_ns, Status* result);

  // Join the thread after Finish/Abort completed (Wait returned true); safe on _exit paths.
  void Join() {
    if (thread_.joinable()) thread_.join();
  }

  bool failed() const { return failed_.load(); }
  std::string error();

 private:
  enum class Cmd { kNone, kPrepare, kCreate, kFinish, kAbort };
  void Main();
  Status DoPrepare();
  Status DoCreate();
  Status DoFinish();
  void DoAbort();
  void WriteBatch(const Batch& b);
  Status WriteAll(int fd, const void* p, size_t n);
  Status WriteMetaAtomic(const std::string& text);
  void Summary(const Batch& b);
  void SetFailed(const std::string& msg);

  BatchQueue* queue_;
  std::string dir_;
  bool quiet_;

  std::mutex mu_;
  std::condition_variable cv_;
  Cmd cmd_ = Cmd::kNone;  // pending command
  bool done_ = false;     // result of the last command is available
  Status result_;
  bool running_ = false;  // between Create and Finish: consume the queue

  // Command arguments (written by main before posting, read by the writer).
  std::vector<WriterStream> streams_;
  std::string meta_text_;
  bool drain_ = true;
  std::string status_;
  MetaBuilder build_meta_;

  // Writer-private state.
  bool created_dir_ = false;
  bool created_lock_ = false;
  bool created_meta_ = false;
  std::vector<int> fds_;
  std::vector<bool> created_files_;
  std::vector<uint64_t> bytes_ok_;
  struct Rate {
    uint64_t t0 = 0, tx0 = 0, rx0 = 0, t = 0, tx = 0, rx = 0;
    bool have = false;
  };
  std::vector<Rate> rates_;
  uint64_t last_print_ = 0;

  std::atomic<bool> failed_{false};
  std::mutex err_mu_;
  std::string error_;
  std::thread thread_;
};

}  // namespace netmon

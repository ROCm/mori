// End-to-end recorder tests with fake sources. Each case runs RunRecording in a forked child
// (it installs process-wide signal state and may _exit), and the parent checks the exit code
// and the files left behind.
#include <dirent.h>
#include <ftw.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <sstream>

#include "harness.h"
#include "privilege.h"
#include "recorder.h"
#include "writer.h"

using namespace netmon;

namespace {

[[noreturn]] void BlockForever() {
  // A condition-variable wait is restarted after signals, so SIGUSR1 cannot end it: this
  // stands in for an uninterruptible driver call.
  static std::mutex m;
  static std::condition_variable cv;
  std::unique_lock<std::mutex> lk(m);
  for (;;) cv.wait(lk);
}

enum class Mode { kOk, kStuckInQuery, kStuckInProbe, kProbeFails };

class FakeSource : public Source {
 public:
  FakeSource(const char* kind, Mode mode, int stuck_after = 0)
      : kind_(kind), mode_(mode), stuck_after_(stuck_after) {
    for (int i = 0; i < 3; i++) names_.push_back("c" + std::to_string(i));
  }
  const char* kind() const override { return kind_; }
  const std::vector<std::string>& names() const override { return names_; }
  double snapshot_fraction() const override { return 0.5; }
  Status Probe(std::string* detail) override {
    if (mode_ == Mode::kStuckInProbe) BlockForever();
    if (mode_ == Mode::kProbeFails) return Status::Error("fake probe failure");
    *detail = "ok";
    return Status::Ok();
  }
  Status Query(uint64_t* out) override {
    if (mode_ == Mode::kStuckInQuery && calls_ >= static_cast<uint64_t>(stuck_after_))
      BlockForever();
    calls_++;
    for (size_t i = 0; i < names_.size(); i++) out[i] = calls_ * (i + 1);
    return Status::Ok();
  }

 private:
  const char* kind_;
  Mode mode_;
  int stuck_after_;
  uint64_t calls_ = 0;
  std::vector<std::string> names_;
};

struct Spec {
  const char* kind;
  Mode mode;
  int stuck_after = 0;
  bool required = false;
};

int RemoveEntry(const char* p, const struct stat*, int, struct FTW*) { return remove(p); }

// Temporary directory for one test; removed when the test passed (kept for debugging if not).
class TempDir {
 public:
  TempDir() : failures_(nt::g_failures) {
    char tmpl[] = "/tmp/netmon-test-XXXXXX";
    char* d = mkdtemp(tmpl);
    path_ = d ? d : "";
  }
  ~TempDir() {
    if (!path_.empty() && nt::g_failures == failures_) {
      nftw(path_.c_str(), RemoveEntry, 16, FTW_DEPTH | FTW_PHYS);
    }
  }
  const std::string& path() const { return path_; }

 private:
  int failures_;
  std::string path_;
};

std::string ReadAll(const std::string& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool Exists(const std::string& p) {
  struct stat sb;
  return stat(p.c_str(), &sb) == 0;
}

off_t FileSize(const std::string& p) {
  struct stat sb;
  return stat(p.c_str(), &sb) == 0 ? sb.st_size : -1;
}

// Write hooks (installed in the child only).
std::atomic<long> g_budget{0};
ssize_t FailAfterBudget(int fd, const void* p, size_t n) {
  long left = g_budget.fetch_sub(static_cast<long>(n));
  if (left <= 0) {
    errno = ENOSPC;
    return -1;
  }
  if (static_cast<long>(n) > left) n = static_cast<size_t>(left);  // short write, then ENOSPC
  return ::write(fd, p, n);
}
std::atomic<int> g_calls_before_block{0};
ssize_t BlockAfterCalls(int fd, const void* p, size_t n) {
  if (g_calls_before_block.fetch_sub(1) <= 0) BlockForever();
  return ::write(fd, p, n);
}

void CheckRecordFile(const std::string& path, size_t width, bool expect_records) {
  off_t sz = FileSize(path);
  CHECK(sz >= 0);
  CHECK(sz % static_cast<off_t>(width * 8) == 0);
  if (expect_records) CHECK(sz > 0);
  std::ifstream f(path, std::ios::binary);
  std::vector<uint64_t> w(static_cast<size_t>(sz) / 8);
  f.read(reinterpret_cast<char*>(w.data()), sz);
  uint64_t prev = 0;
  for (size_t r = 0; r + width <= w.size(); r += width) {
    CHECK(w[r + 1] >= w[r]);          // t1 >= t0
    CHECK(w[r] >= prev);              // monotonic sample order
    CHECK(w[r + 3] < 1000000000ull);  // anchor read span is small
    prev = w[r];
  }
}

struct ChildCase {
  std::vector<Spec> specs;
  uint64_t duration_ns;
  void (*before)();
};

void InstallEnospc() {
  g_budget = 12000;  // the lock and the initial meta fit, then ENOSPC partway through records
  Writer::write_fn = FailAfterBudget;
}
void InstallBlockingWrites() {
  g_calls_before_block = 6;  // lock, meta and a few batches, then every write blocks
  Writer::write_fn = BlockAfterCalls;
}

const std::map<std::string, ChildCase>& Cases() {
  static const std::map<std::string, ChildCase> cases = {
      {"normal", {{{"rdma", Mode::kOk}, {"ipc", Mode::kOk}}, 300000000ull, nullptr}},
      {"until_signal", {{{"rdma", Mode::kOk}}, 0, nullptr}},
      {"stuck_sampler",
       {{{"rdma", Mode::kOk}, {"ipc", Mode::kStuckInQuery, 20}}, 300000000ull, nullptr}},
      {"stuck_probe", {{{"rdma", Mode::kOk}, {"ipc", Mode::kStuckInProbe}}, 300000000ull, nullptr}},
      {"optional_probe_fails",
       {{{"rdma", Mode::kOk}, {"ipc", Mode::kProbeFails}}, 200000000ull, nullptr}},
      {"required_probe_fails",
       {{{"rdma", Mode::kOk}, {"ipc", Mode::kProbeFails, 0, true}}, 200000000ull, nullptr}},
      {"short", {{{"rdma", Mode::kOk}}, 100000000ull, nullptr}},
      {"enospc", {{{"rdma", Mode::kOk}}, 2000000000ull, InstallEnospc}},
      {"enospc_and_stuck",
       {{{"rdma", Mode::kOk}, {"ipc", Mode::kStuckInQuery, 5}}, 2000000000ull, InstallEnospc}},
      {"writer_blocks", {{{"rdma", Mode::kOk}}, 300000000ull, InstallBlockingWrites}},
  };
  return cases;
}

// Runs a recording case in a fresh process (fork + exec of this test binary), so the child
// starts from main() with a clean runtime (sanitizers do not support running threads in a
// child forked from a multi-threaded parent). Returns the exit code, or 128 + signal.
int RunCase(const std::string& name, const std::string& out, int send_sigint_after_ms = -1) {
  std::string self = SelfExePath();
  pid_t pid = fork();
  if (pid == 0) {
    execl(self.c_str(), self.c_str(), "--recorder-case", name.c_str(), out.c_str(),
          static_cast<char*>(nullptr));
    _exit(127);
  }
  if (send_sigint_after_ms >= 0) {
    usleep(static_cast<useconds_t>(send_sigint_after_ms) * 1000);
    kill(pid, SIGINT);
  }
  int status = 0;
  for (int waited = 0; waited < 600; waited++) {  // 60 s cap (sanitizer builds are slow)
    if (waitpid(pid, &status, WNOHANG) == pid) {
      if (WIFEXITED(status)) return WEXITSTATUS(status);
      return 128 + WTERMSIG(status);
    }
    usleep(100000);
  }
  kill(pid, SIGKILL);
  waitpid(pid, &status, 0);
  return -1;
}

}  // namespace

// Child side, called from main() for "--recorder-case NAME OUT".
int RunRecorderCase(const char* name, const char* out) {
  auto it = Cases().find(name);
  if (it == Cases().end()) return 126;
  const ChildCase& c = it->second;
  if (c.before) c.before();
  std::vector<std::unique_ptr<Stream>> streams;
  for (size_t i = 0; i < c.specs.size(); i++) {
    auto s = std::make_unique<Stream>();
    s->netdev = "fake" + std::to_string(i);
    s->src = std::make_unique<FakeSource>(c.specs[i].kind, c.specs[i].mode, c.specs[i].stuck_after);
    s->interval_ns = 2000000;
    s->required = c.specs[i].required;
    streams.push_back(std::move(s));
  }
  RecordOptions o;
  o.out_dir = out;
  o.duration_ns = c.duration_ns;
  o.quiet = true;
  o.probe_timeout_ns = 1000000000ull;
  o.stop_timeout_ns = 500000000ull;
  o.io_timeout_ns = 1500000000ull;
  o.host_id = "test";
  return RunRecording(&streams, &o);
}

TEST(RecorderNormalRun) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";  // does not exist yet: created with parents
  int code = RunCase("normal", dir);
  CHECK_EQ(code, 0);
  std::string meta = ReadAll(dir + "/meta.json");
  CHECK(meta.find("\"status\": \"complete\"") != std::string::npos);
  CHECK(meta.find("\"format_version\": 2") != std::string::npos);
  CHECK(Exists(dir + "/.lock"));
  CheckRecordFile(dir + "/fake0.rdma.bin", 7, true);
  CheckRecordFile(dir + "/fake1.ipc.bin", 7, true);
}

TEST(RecorderSigintIsInterrupted) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("until_signal", dir, 400);
  CHECK_EQ(code, 0);
  CHECK(ReadAll(dir + "/meta.json").find("\"status\": \"interrupted\"") != std::string::npos);
  CheckRecordFile(dir + "/fake0.rdma.bin", 7, true);
}

TEST(RecorderStuckSamplerIsAbandoned) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("stuck_sampler", dir);
  CHECK_EQ(code, 4);
  std::string meta = ReadAll(dir + "/meta.json");
  CHECK(meta.find("\"abandoned\": [\"fake1.ipc.bin\"]") != std::string::npos);
  CHECK(meta.find("\"status\": \"complete\"") != std::string::npos);
  CheckRecordFile(dir + "/fake0.rdma.bin", 7, true);
  CheckRecordFile(dir + "/fake1.ipc.bin", 7, false);
}

TEST(RecorderStuckProbeLeavesNothing) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("stuck_probe", dir);
  CHECK_EQ(code, 1);
  CHECK(!Exists(dir));  // the directory this run created is removed again
}

TEST(RecorderOptionalProbeFailureDegrades) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("optional_probe_fails", dir);
  CHECK_EQ(code, 0);
  std::string meta = ReadAll(dir + "/meta.json");
  CHECK(meta.find("fake probe failure") != std::string::npos);  // listed under degraded
  CHECK(!Exists(dir + "/fake1.ipc.bin"));
}

TEST(RecorderRequiredProbeFailureAborts) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("required_probe_fails", dir);
  CHECK_EQ(code, 1);
  CHECK(!Exists(dir));
}

TEST(RecorderRefusesNonEmptyDirectory) {
  TempDir tmp;
  auto dir = tmp.path();
  {
    std::ofstream(dir + "/precious.txt") << "keep";
  }
  int code = RunCase("short", dir);
  CHECK_EQ(code, 1);
  CHECK(ReadAll(dir + "/precious.txt") == "keep");
  CHECK(!Exists(dir + "/.lock"));
  CHECK(!Exists(dir + "/meta.json"));
}

TEST(RecorderWriteFailure) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("enospc", dir);
  CHECK_EQ(code, 3);
  CheckRecordFile(dir + "/fake0.rdma.bin", 7, false);  // truncated to whole records
  // The final meta cannot be written either (no space): it stays "running", which the
  // report treats as an unfinished recording.
  CHECK(ReadAll(dir + "/meta.json").find("\"status\": \"running\"") != std::string::npos);
}

TEST(RecorderWriteFailureWithAbandonedThread) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("enospc_and_stuck", dir);
  CHECK_EQ(code, 3);  // write failure outranks abandoned
}

TEST(RecorderWriterBlocks) {
  TempDir tmp;
  auto dir = tmp.path() + "/run";
  int code = RunCase("writer_blocks", dir);
  CHECK_EQ(code, 3);
}

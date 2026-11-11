#include "writer.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace netmon {

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
              "record files are little-endian; batches are written as-is");

Writer::WriteFn Writer::write_fn = ::write;

Writer::Writer(BatchQueue* queue, std::string dir, bool quiet)
    : queue_(queue), dir_(std::move(dir)), quiet_(quiet) {
  thread_ = std::thread([this] { Main(); });
}

Writer::~Writer() {
  // Only reached on the normal path, where Main() has returned after Finish/Abort.
  if (thread_.joinable()) thread_.join();
}

void Writer::RequestPrepare() {
  std::lock_guard<std::mutex> lk(mu_);
  cmd_ = Cmd::kPrepare;
  done_ = false;
  cv_.notify_all();
}

void Writer::RequestCreate(std::vector<WriterStream> streams, std::string initial_meta) {
  std::lock_guard<std::mutex> lk(mu_);
  streams_ = std::move(streams);
  meta_text_ = std::move(initial_meta);
  cmd_ = Cmd::kCreate;
  done_ = false;
  cv_.notify_all();
}

void Writer::RequestFinish(bool drain, std::string status, MetaBuilder build_meta) {
  std::lock_guard<std::mutex> lk(mu_);
  drain_ = drain;
  status_ = std::move(status);
  build_meta_ = std::move(build_meta);
  cmd_ = Cmd::kFinish;
  done_ = false;
  cv_.notify_all();
}

void Writer::RequestAbort() {
  std::lock_guard<std::mutex> lk(mu_);
  cmd_ = Cmd::kAbort;
  done_ = false;
  cv_.notify_all();
}

bool Writer::Wait(uint64_t deadline_mono_ns, Status* result) {
  std::unique_lock<std::mutex> lk(mu_);
  while (!done_) {
    uint64_t now = MonoNs();
    if (now >= deadline_mono_ns) return false;
    cv_.wait_for(
        lk, std::chrono::nanoseconds(std::min<uint64_t>(deadline_mono_ns - now, 100000000ull)));
  }
  *result = result_;
  return true;
}

std::string Writer::error() {
  std::lock_guard<std::mutex> lk(err_mu_);
  return error_;
}

void Writer::SetFailed(const std::string& msg) {
  {
    std::lock_guard<std::mutex> lk(err_mu_);
    if (error_.empty()) error_ = msg;
  }
  failed_.store(true);
}

void Writer::Main() {
  for (;;) {
    Cmd cmd;
    {
      std::unique_lock<std::mutex> lk(mu_);
      if (running_ && cmd_ == Cmd::kNone) {
        lk.unlock();
        bool drained;
        auto b = queue_->Pop(std::chrono::milliseconds(50), &drained);
        if (b && !failed_.load()) {
          WriteBatch(*b);
          Summary(*b);
        }
        continue;  // b is freed here, outside any lock
      }
      cv_.wait(lk, [&] { return cmd_ != Cmd::kNone; });
      cmd = cmd_;
      cmd_ = Cmd::kNone;
    }
    Status st;
    bool exit_after = false;
    switch (cmd) {
      case Cmd::kPrepare:
        st = DoPrepare();
        break;
      case Cmd::kCreate:
        st = DoCreate();
        break;
      case Cmd::kFinish:
        st = DoFinish();
        exit_after = true;
        break;
      case Cmd::kAbort:
        DoAbort();
        exit_after = true;
        break;
      case Cmd::kNone:
        break;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      result_ = st;
      done_ = true;
      if (cmd == Cmd::kCreate && st.ok()) running_ = true;
      if (exit_after) running_ = false;
      cv_.notify_all();
    }
    if (exit_after) return;
  }
}

Status Writer::DoPrepare() {
  struct stat sb;
  if (stat(dir_.c_str(), &sb) != 0) {
    if (errno != ENOENT) return Status::Errno("stat " + dir_, errno);
    // Create missing parents too (e.g. /shared/run1/{host}).
    std::string cur;
    for (const auto& part : Split(dir_, '/')) {
      if (part.empty()) {
        if (cur.empty()) cur = "/";
        continue;
      }
      cur += (cur.empty() || cur.back() == '/') ? part : "/" + part;
      if (mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) {
        return Status::Errno("mkdir " + cur, errno);
      }
    }
    created_dir_ = true;
  } else {
    if (!S_ISDIR(sb.st_mode)) return Status::Error(dir_ + " exists and is not a directory");
    DIR* d = opendir(dir_.c_str());
    if (!d) return Status::Errno("opendir " + dir_, errno);
    bool empty = true;
    while (dirent* e = readdir(d)) {
      if (strcmp(e->d_name, ".") != 0 && strcmp(e->d_name, "..") != 0) {
        empty = false;
        break;
      }
    }
    closedir(d);
    if (!empty) {
      return Status::Error(dir_ + " is not empty; refusing to overwrite an existing recording");
    }
  }
  const std::string lock = dir_ + "/.lock";
  int fd = open(lock.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
  if (fd < 0) {
    // Another recorder got here first: do not touch anything it owns.
    return Status::Errno("create " + lock + " (another recording in this directory?)", errno);
  }
  created_lock_ = true;
  std::string content = "pid " + std::to_string(getpid()) + "\nhost " + Hostname() +
                        "\nstart_realtime_ns " + std::to_string(RealNs()) + "\n";
  Status st = WriteAll(fd, content.data(), content.size());
  close(fd);
  return st;
}

Status Writer::DoCreate() {
  fds_.assign(streams_.size(), -1);
  created_files_.assign(streams_.size(), false);
  bytes_ok_.assign(streams_.size(), 0);
  rates_.assign(streams_.size(), Rate{});
  for (size_t i = 0; i < streams_.size(); i++) {
    const std::string path = dir_ + "/" + streams_[i].file;
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) return Status::Errno("create " + path, errno);
    fds_[i] = fd;
    created_files_[i] = true;
  }
  Status st = WriteMetaAtomic(meta_text_);
  if (st.ok()) created_meta_ = true;
  last_print_ = MonoNs();
  return st;
}

Status Writer::WriteAll(int fd, const void* p, size_t n) {
  const uint8_t* c = static_cast<const uint8_t*>(p);
  size_t done = 0;
  while (done < n) {
    ssize_t w = write_fn(fd, c + done, n - done);
    if (w < 0) {
      if (errno == EINTR) continue;
      return Status::Errno("write", errno);
    }
    if (w == 0) return Status::Error("write returned 0");
    done += static_cast<size_t>(w);
  }
  return Status::Ok();
}

void Writer::WriteBatch(const Batch& b) {
  const size_t i = static_cast<size_t>(b.stream);
  const size_t rec_bytes = b.width * 8;
  const size_t n = b.count * rec_bytes;
  if (n == 0 || fds_[i] < 0) return;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(b.words.data());
  size_t done = 0;
  while (done < n) {
    ssize_t w = write_fn(fds_[i], p + done, n - done);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) {
      const int err = w < 0 ? errno : EIO;
      // Publish the failure first: the truncate below may itself block (NFS), and the main
      // thread must still see the failure and start its supervised shutdown.
      SetFailed("write " + streams_[i].file + ": " + std::strerror(err));
      // Keep only whole records so the file stays a valid prefix.
      const uint64_t keep = (bytes_ok_[i] + done) / rec_bytes * rec_bytes;
      if (ftruncate(fds_[i], static_cast<off_t>(keep)) == 0) bytes_ok_[i] = keep;
      return;
    }
    done += static_cast<size_t>(w);
  }
  bytes_ok_[i] += n;
}

void Writer::Summary(const Batch& b) {
  if (quiet_) return;
  const size_t i = static_cast<size_t>(b.stream);
  const WriterStream& ws = streams_[i];
  if (ws.key_tx >= 0 && ws.key_rx >= 0) {
    Rate& r = rates_[i];
    for (size_t k = 0; k < b.count; k++) {
      const uint64_t* rec = b.Record(k);
      // Request end time; over a ~1 s window its error (< one request) is negligible.
      // Per-sample peaks are left to the offline report, which applies the trusted-sample
      // rule (design §7.4); a console peak from raw neighbours would show false spikes.
      const uint64_t t = rec[1];
      const uint64_t tx = rec[4 + ws.key_tx], rx = rec[4 + ws.key_rx];
      if (!r.have || tx < r.tx || rx < r.rx) {  // first sample or counter reset
        r.t0 = t;
        r.tx0 = tx;
        r.rx0 = rx;
        r.have = true;
      }
      r.t = t;
      r.tx = tx;
      r.rx = rx;
    }
  }
  const uint64_t now = MonoNs();
  if (now - last_print_ < 1000000000ull) return;
  last_print_ = now;
  std::string line;
  for (size_t j = 0; j < streams_.size(); j++) {
    Rate& r = rates_[j];
    if (!r.have || r.t <= r.t0) continue;
    const double dt = static_cast<double>(r.t - r.t0);
    char buf[160];
    snprintf(buf, sizeof(buf), "%s tx %.1f rx %.1f", streams_[j].label.c_str(),
             static_cast<double>(r.tx - r.tx0) * 8 / dt,
             static_cast<double>(r.rx - r.rx0) * 8 / dt);
    if (!line.empty()) line += " | ";
    line += buf;
    r.t0 = r.t;
    r.tx0 = r.tx;
    r.rx0 = r.rx;
  }
  if (!line.empty()) fprintf(stderr, "[Gb/s, ~1 s average] %s\n", line.c_str());
}

Status Writer::WriteMetaAtomic(const std::string& text) {
  const std::string tmp = dir_ + "/meta.json.tmp";
  int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  if (fd < 0) return Status::Errno("create " + tmp, errno);
  Status st = WriteAll(fd, text.data(), text.size());
  if (close(fd) != 0 && st.ok()) st = Status::Errno("close " + tmp, errno);
  if (!st.ok()) {
    unlink(tmp.c_str());
    return st;
  }
  if (rename(tmp.c_str(), (dir_ + "/meta.json").c_str()) != 0) {
    st = Status::Errno("rename meta.json", errno);
    unlink(tmp.c_str());
  }
  return st;
}

Status Writer::DoFinish() {
  if (drain_ && !failed_.load()) {
    for (;;) {
      bool drained;
      auto b = queue_->Pop(std::chrono::milliseconds(50), &drained);
      if (b) {
        if (!failed_.load()) WriteBatch(*b);
        continue;
      }
      if (drained) break;
    }
  } else {
    queue_->DiscardAll();
  }
  for (size_t i = 0; i < fds_.size(); i++) {
    if (fds_[i] >= 0 && close(fds_[i]) != 0) SetFailed("close " + streams_[i].file);
    fds_[i] = -1;
  }
  const std::string status = failed_.load() ? "failed" : status_;
  Status st = WriteMetaAtomic(build_meta_(status, error()));
  if (!st.ok()) SetFailed(st.message());
  return failed_.load() ? Status::Error(error()) : Status::Ok();
}

void Writer::DoAbort() {
  for (size_t i = 0; i < fds_.size(); i++) {
    if (fds_[i] >= 0) close(fds_[i]);
    if (created_files_[i]) unlink((dir_ + "/" + streams_[i].file).c_str());
  }
  if (created_meta_) unlink((dir_ + "/meta.json").c_str());
  if (created_lock_) unlink((dir_ + "/.lock").c_str());
  if (created_dir_) rmdir(dir_.c_str());  // only removes it if it is still empty
}

}  // namespace netmon

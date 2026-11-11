#include "guarded_buffer.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>

namespace netmon {

GuardedBuffer::~GuardedBuffer() {
  if (base_) munmap(base_, mapped_);
}

Status GuardedBuffer::Allocate(size_t bytes) {
  if (base_) return Status::Error("GuardedBuffer already allocated");
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  if (bytes == 0 || bytes > (size_t{1} << 30)) return Status::Error("GuardedBuffer: bad size");
  const size_t usable = (bytes + page - 1) / page * page;
  const size_t total = usable + page;
  void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return Status::Errno("mmap", errno);
  uint8_t* base = static_cast<uint8_t*>(p);
  if (mprotect(base + usable, page, PROT_NONE) != 0) {
    int err = errno;
    munmap(p, total);
    return Status::Errno("mprotect", err);
  }
  base_ = base;
  data_ = base;
  capacity_ = usable;
  mapped_ = total;
  return Status::Ok();
}

}  // namespace netmon

#include "rdma_netlink.h"

#include <linux/netlink.h>
#include <rdma/rdma_netlink.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace netmon {
namespace nl {

namespace {

constexpr uint16_t kTypeMask = 0x3fff;  // NLA_TYPE_MASK without the flag bits

uint16_t NlType(uint16_t op) { return static_cast<uint16_t>(RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, op)); }

// Bounds-checked netlink attribute walker over [p, p+len).
class AttrIter {
 public:
  AttrIter(const uint8_t* p, size_t len) : p_(p), left_(len) {}
  // Returns false at the end or on a malformed attribute (*bad = true).
  bool Next(uint16_t* type, const uint8_t** payload, size_t* plen, bool* bad) {
    if (left_ == 0) return false;
    if (left_ < NLA_HDRLEN) {
      *bad = true;
      return false;
    }
    nlattr a;
    std::memcpy(&a, p_, sizeof(a));
    if (a.nla_len < NLA_HDRLEN || a.nla_len > left_) {
      *bad = true;
      return false;
    }
    *type = a.nla_type & kTypeMask;
    *payload = p_ + NLA_HDRLEN;
    *plen = a.nla_len - NLA_HDRLEN;
    size_t adv = NLA_ALIGN(a.nla_len);
    if (adv > left_) adv = left_;
    p_ += adv;
    left_ -= adv;
    return true;
  }

 private:
  const uint8_t* p_;
  size_t left_;
};

std::string CStr(const uint8_t* p, size_t n) {
  size_t e = 0;
  while (e < n && p[e] != 0) e++;
  return std::string(reinterpret_cast<const char*>(p), e);
}

void PutU32Attr(std::vector<uint8_t>* m, uint16_t type, uint32_t v) {
  nlattr a{};
  a.nla_len = NLA_HDRLEN + 4;
  a.nla_type = type;
  const size_t at = m->size();
  m->resize(at + NLA_HDRLEN + 4);
  std::memcpy(m->data() + at, &a, sizeof(a));
  std::memcpy(m->data() + at + NLA_HDRLEN, &v, 4);
}

std::vector<uint8_t> BuildMsg(uint16_t type, uint16_t flags, uint32_t seq,
                              const std::vector<uint8_t>& body) {
  nlmsghdr h{};
  h.nlmsg_len = static_cast<uint32_t>(NLMSG_HDRLEN + body.size());
  h.nlmsg_type = type;
  h.nlmsg_flags = flags;
  h.nlmsg_seq = seq;
  std::vector<uint8_t> m(NLMSG_HDRLEN + body.size(), 0);
  std::memcpy(m.data(), &h, sizeof(h));
  if (!body.empty()) std::memcpy(m.data() + NLMSG_HDRLEN, body.data(), body.size());
  return m;
}

}  // namespace

std::vector<uint8_t> BuildStatGetRequest(uint32_t dev_index, uint32_t port, uint32_t seq) {
  std::vector<uint8_t> body;
  PutU32Attr(&body, RDMA_NLDEV_ATTR_DEV_INDEX, dev_index);
  PutU32Attr(&body, RDMA_NLDEV_ATTR_PORT_INDEX, port);
  return BuildMsg(NlType(RDMA_NLDEV_CMD_STAT_GET), NLM_F_REQUEST, seq, body);
}

Status ParseStatGetReply(const uint8_t* buf, size_t len, uint32_t expect_seq, bool* found,
                         bool* stale, std::vector<std::string>* names, uint64_t* vals,
                         size_t max_vals, size_t* count) {
  *found = false;
  *stale = false;
  *count = 0;
  size_t off = 0;
  while (off + NLMSG_HDRLEN <= len) {
    nlmsghdr h;
    std::memcpy(&h, buf + off, sizeof(h));
    if (h.nlmsg_len < NLMSG_HDRLEN || h.nlmsg_len > len - off) {
      return Status::Error("netlink: malformed message length");
    }
    const uint8_t* payload = buf + off + NLMSG_HDRLEN;
    const size_t plen = h.nlmsg_len - NLMSG_HDRLEN;
    size_t adv = NLMSG_ALIGN(h.nlmsg_len);
    off += adv > len - off ? len - off : adv;

    if (h.nlmsg_seq != expect_seq) {
      *stale = true;
      continue;
    }
    if (h.nlmsg_type == NLMSG_ERROR) {
      nlmsgerr e;
      if (plen < sizeof(int)) return Status::Error("netlink: short NLMSG_ERROR");
      std::memcpy(&e.error, payload, sizeof(int));
      if (e.error == 0) continue;  // ACK
      return Status::Errno("netlink STAT_GET", -e.error);
    }
    if (h.nlmsg_type != NlType(RDMA_NLDEV_CMD_STAT_GET)) continue;

    *found = true;
    AttrIter top(payload, plen);
    uint16_t t;
    const uint8_t* p;
    size_t n;
    bool bad = false;
    while (top.Next(&t, &p, &n, &bad)) {
      if (t != RDMA_NLDEV_ATTR_STAT_HWCOUNTERS) continue;
      AttrIter entries(p, n);
      uint16_t et;
      const uint8_t* ep;
      size_t en;
      while (entries.Next(&et, &ep, &en, &bad)) {
        if (et != RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY) continue;
        AttrIter fields(ep, en);
        uint16_t ft;
        const uint8_t* fp;
        size_t fn;
        bool have_val = false, have_name = false;
        uint64_t v = 0;
        std::string name;
        while (fields.Next(&ft, &fp, &fn, &bad)) {
          if (ft == RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_VALUE) {
            if (fn != 8) return Status::Error("netlink: counter value is not u64");
            std::memcpy(&v, fp, 8);
            have_val = true;
          } else if (ft == RDMA_NLDEV_ATTR_STAT_HWCOUNTER_ENTRY_NAME) {
            name = CStr(fp, fn);
            have_name = true;
          }
        }
        if (bad) return Status::Error("netlink: malformed counter entry");
        if (!have_val || !have_name) return Status::Error("netlink: counter entry incomplete");
        if (names) names->push_back(name);
        if (*count < max_vals) vals[*count] = v;
        (*count)++;
      }
      if (bad) return Status::Error("netlink: malformed counter table");
    }
    if (bad) return Status::Error("netlink: malformed attributes");
    return Status::Ok();
  }
  return Status::Ok();
}

}  // namespace nl

RdmaNetlinkSource::~RdmaNetlinkSource() {
  if (fd_ >= 0) close(fd_);
}

Status RdmaNetlinkSource::Open(const std::string& ibdev, uint32_t port, uint64_t query_timeout_ns) {
  port_ = port;
  timeout_ns_ = query_timeout_ns;
  rx_.resize(64 * 1024);
  fd_ = socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_RDMA);
  if (fd_ < 0) return Status::Errno("socket(NETLINK_RDMA)", errno);
  sockaddr_nl sa{};
  sa.nl_family = AF_NETLINK;
  if (bind(fd_, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
    return Status::Errno("bind(NETLINK_RDMA)", errno);
  }
  ibdev_ = ibdev;
  return Status::Ok();  // no query here: the first one runs in Probe(), under supervision
}

Status RdmaNetlinkSource::Probe(std::string* detail) {
  Status st = ResolveDevIndex(ibdev_, &dev_index_);
  if (!st.ok()) return st;
  size_t count = 0;
  std::vector<uint64_t> tmp(1024);
  std::vector<std::string> names;
  st = Exchange(&names, tmp.data(), tmp.size(), &count);
  if (!st.ok()) return st;
  if (names.empty()) return Status::Error(ibdev_ + ": STAT_GET returned no hw counters");
  if (count > tmp.size()) return Status::Error(ibdev_ + ": too many hw counters");
  names_ = std::move(names);
  *detail = std::to_string(names_.size()) + " counters";
  return Status::Ok();
}

Status RdmaNetlinkSource::ResolveDevIndex(const std::string& ibdev, uint32_t* index) {
  nlmsghdr h{};
  h.nlmsg_len = NLMSG_HDRLEN;
  h.nlmsg_type = static_cast<uint16_t>(RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_GET));
  h.nlmsg_flags = NLM_F_REQUEST | NLM_F_DUMP;
  h.nlmsg_seq = ++seq_;
  if (send(fd_, &h, sizeof(h), 0) < 0) return Status::Errno("netlink send", errno);
  timeval tv{static_cast<time_t>(timeout_ns_ / 1000000000ull),
             static_cast<suseconds_t>(timeout_ns_ % 1000000000ull / 1000)};
  setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  bool found = false;
  for (;;) {
    ssize_t n = recv(fd_, rx_.data(), rx_.size(), 0);
    if (n < 0) return Status::Errno("netlink recv (device dump)", errno);
    size_t off = 0, len = static_cast<size_t>(n);
    while (off + NLMSG_HDRLEN <= len) {
      nlmsghdr m;
      std::memcpy(&m, rx_.data() + off, sizeof(m));
      if (m.nlmsg_len < NLMSG_HDRLEN || m.nlmsg_len > len - off) {
        return Status::Error("netlink: malformed device dump");
      }
      if (m.nlmsg_seq == seq_) {
        if (m.nlmsg_type == NLMSG_DONE) {
          if (!found) return Status::Error("RDMA device " + ibdev + " not found via netlink");
          return Status::Ok();
        }
        if (m.nlmsg_type == NLMSG_ERROR) return Status::Error("netlink: device dump failed");
        uint32_t idx = 0;
        std::string name;
        bool have_idx = false, bad = false;
        uint16_t t;
        const uint8_t* p;
        size_t pn;
        size_t aoff = NLMSG_HDRLEN;
        while (aoff + NLA_HDRLEN <= m.nlmsg_len) {
          nlattr a;
          std::memcpy(&a, rx_.data() + off + aoff, sizeof(a));
          if (a.nla_len < NLA_HDRLEN || a.nla_len > m.nlmsg_len - aoff) {
            bad = true;
            break;
          }
          t = a.nla_type & NLA_TYPE_MASK;
          p = rx_.data() + off + aoff + NLA_HDRLEN;
          pn = a.nla_len - NLA_HDRLEN;
          if (t == RDMA_NLDEV_ATTR_DEV_INDEX && pn >= 4) {
            std::memcpy(&idx, p, 4);
            have_idx = true;
          } else if (t == RDMA_NLDEV_ATTR_DEV_NAME) {
            size_t e = 0;
            while (e < pn && p[e] != 0) e++;
            name.assign(reinterpret_cast<const char*>(p), e);
          }
          aoff += NLA_ALIGN(a.nla_len);
        }
        if (bad) return Status::Error("netlink: malformed device attributes");
        if (have_idx && name == ibdev) {
          *index = idx;
          found = true;
        }
      }
      size_t adv = NLMSG_ALIGN(m.nlmsg_len);
      off += adv > len - off ? len - off : adv;
    }
  }
}

Status RdmaNetlinkSource::Exchange(std::vector<std::string>* names, uint64_t* vals, size_t max_vals,
                                   size_t* count) {
  const uint64_t deadline = MonoNs() + timeout_ns_;
  const uint32_t seq = ++seq_;
  std::vector<uint8_t> req = nl::BuildStatGetRequest(dev_index_, port_, seq);
  if (send(fd_, req.data(), req.size(), 0) < 0) return Status::Errno("netlink send", errno);
  for (;;) {
    uint64_t now = MonoNs();
    if (now >= deadline) return Status::Error("netlink STAT_GET timed out");
    uint64_t left = deadline - now;
    timeval tv{static_cast<time_t>(left / 1000000000ull),
               static_cast<suseconds_t>(left % 1000000000ull / 1000)};
    if (tv.tv_sec == 0 && tv.tv_usec == 0) tv.tv_usec = 1;
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    ssize_t n = recv(fd_, rx_.data(), rx_.size(), MSG_TRUNC);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        return Status::Error("netlink STAT_GET timed out");
      }
      return Status::Errno("netlink recv", errno);
    }
    if (static_cast<size_t>(n) > rx_.size()) return Status::Error("netlink reply truncated");
    bool found = false, stale = false;
    if (names) names->clear();
    Status st = nl::ParseStatGetReply(rx_.data(), static_cast<size_t>(n), seq, &found, &stale,
                                      names, vals, max_vals, count);
    if (!st.ok()) return st;
    if (found) return Status::Ok();
    // Only stale replies (an earlier timed-out request answered late): keep receiving.
  }
}

Status RdmaNetlinkSource::Query(uint64_t* out) {
  if (names_.empty()) return Status::Error("rdma source queried before Probe()");
  size_t count = 0;
  Status st = Exchange(nullptr, out, names_.size(), &count);
  if (!st.ok()) return st;
  if (count != names_.size()) {
    return Status::Error("hw counter count changed: " + std::to_string(names_.size()) + " -> " +
                         std::to_string(count));
  }
  return Status::Ok();
}

}  // namespace netmon

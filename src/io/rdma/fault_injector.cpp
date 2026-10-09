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
#include "src/io/rdma/fault_injector.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mori/io/logging.hpp"

namespace mori {
namespace io {

namespace {

struct NamedValue {
  const char* name;
  int value;
};

constexpr NamedValue kKinds[] = {
    {"post_send_fail", static_cast<int>(FaultKind::PostSendFail)},
    {"post_recv_fail", static_cast<int>(FaultKind::PostRecvFail)},
    {"qp_error", static_cast<int>(FaultKind::QpError)},
    {"cqe_error", static_cast<int>(FaultKind::CqeError)},
    {"cqe_drop", static_cast<int>(FaultKind::CqeDrop)},
};

constexpr NamedValue kOpcodes[] = {
    {"write", IBV_WC_RDMA_WRITE},
    {"read", IBV_WC_RDMA_READ},
    {"send", IBV_WC_SEND},
    {"recv", IBV_WC_RECV},
};

constexpr NamedValue kWcStatuses[] = {
    {"retry_exc", IBV_WC_RETRY_EXC_ERR},   {"rnr_retry_exc", IBV_WC_RNR_RETRY_EXC_ERR},
    {"rem_access", IBV_WC_REM_ACCESS_ERR}, {"rem_op", IBV_WC_REM_OP_ERR},
    {"loc_prot", IBV_WC_LOC_PROT_ERR},     {"flush", IBV_WC_WR_FLUSH_ERR},
    {"fatal", IBV_WC_FATAL_ERR},
};

template <size_t N>
int Lookup(const NamedValue (&table)[N], const std::string& token, bool allowNumber) {
  for (const NamedValue& nv : table) {
    if (token == nv.name) return nv.value;
  }
  if (allowNumber) {
    char* end = nullptr;
    long v = std::strtol(token.c_str(), &end, 0);
    if (end != token.c_str() && *end == '\0') return static_cast<int>(v);
  }
  throw std::invalid_argument("MORI_IO_FAULT: unknown token '" + token + "'");
}

uint64_t ParseU64(const std::string& key, const std::string& token) {
  char* end = nullptr;
  unsigned long long v = std::strtoull(token.c_str(), &end, 0);
  if (end == token.c_str() || *end != '\0') {
    throw std::invalid_argument("MORI_IO_FAULT: bad number for " + key + ": '" + token + "'");
  }
  return v;
}

std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> out;
  size_t start = 0;
  while (true) {
    size_t pos = s.find(sep, start);
    out.push_back(s.substr(start, pos - start));
    if (pos == std::string::npos) return out;
    start = pos + 1;
  }
}

}  // namespace

const char* FaultKindName(FaultKind kind) {
  for (const NamedValue& nv : kKinds) {
    if (nv.value == static_cast<int>(kind)) return nv.name;
  }
  return "unknown";
}

FaultRule ParseFaultRule(const std::string& spec) {
  std::vector<std::string> parts = Split(spec, ':');
  FaultRule rule;
  rule.kind = static_cast<FaultKind>(Lookup(kKinds, parts[0], false));
  if (rule.kind == FaultKind::PostSendFail || rule.kind == FaultKind::PostRecvFail) {
    rule.value = EIO;
  } else if (rule.kind == FaultKind::CqeError) {
    rule.value = IBV_WC_RETRY_EXC_ERR;
  }

  for (size_t i = 1; i < parts.size(); ++i) {
    size_t eq = parts[i].find('=');
    if (eq == std::string::npos) {
      throw std::invalid_argument("MORI_IO_FAULT: expected key=value, got '" + parts[i] + "'");
    }
    std::string key = parts[i].substr(0, eq);
    std::string val = parts[i].substr(eq + 1);
    if (key == "skip") {
      rule.skip = ParseU64(key, val);
    } else if (key == "count") {
      rule.count = ParseU64(key, val);
    } else if (key == "qpn") {
      rule.qpn = static_cast<uint32_t>(ParseU64(key, val));
    } else if (key == "op") {
      rule.wcOpcode = Lookup(kOpcodes, val, true);
    } else if (key == "value") {
      rule.value = rule.kind == FaultKind::CqeError ? Lookup(kWcStatuses, val, true)
                                                    : static_cast<int>(ParseU64(key, val));
    } else {
      throw std::invalid_argument("MORI_IO_FAULT: unknown key '" + key + "'");
    }
  }
  return rule;
}

FaultInjector& FaultInjector::Instance() {
  static FaultInjector injector;
  return injector;
}

FaultInjector::FaultInjector() {
  const char* spec = std::getenv("MORI_IO_FAULT");
  if (spec != nullptr && spec[0] != '\0') Arm(ParseFaultRule(spec));
}

void FaultInjector::Arm(const FaultRule& rule) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    rule_ = rule;
    matched_ = 0;
  }
  fired_.store(0, std::memory_order_relaxed);
  armed_.store(true, std::memory_order_release);
  MORI_IO_WARN("Fault injection armed: kind={} skip={} count={} value={} op={} qpn={}",
               FaultKindName(rule.kind), rule.skip, rule.count, rule.value, rule.wcOpcode,
               rule.qpn);
}

void FaultInjector::Disarm() { armed_.store(false, std::memory_order_release); }

bool FaultInjector::ShouldFire(FaultKind kind, uint32_t qpn, int wcOpcode, int* value) {
  std::lock_guard<std::mutex> lock(mu_);
  if (!armed_.load(std::memory_order_acquire) || rule_.kind != kind) return false;
  if (rule_.qpn != 0 && rule_.qpn != qpn) return false;
  if (rule_.wcOpcode >= 0 && rule_.wcOpcode != wcOpcode) return false;
  if (matched_++ < rule_.skip) return false;
  uint64_t fired = fired_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (rule_.count != 0 && fired >= rule_.count) armed_.store(false, std::memory_order_release);
  *value = rule_.value;
  MORI_IO_WARN("Fault injected: kind={} qpn={} opcode={} value={} (#{})", FaultKindName(kind), qpn,
               wcOpcode, rule_.value, fired);
  return true;
}

int FaultInjector::PostSend(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad) {
  int value = 0;
  if (armed_.load(std::memory_order_relaxed)) {
    if (ShouldFire(FaultKind::PostSendFail, qp->qp_num, -1, &value)) {
      *bad = wr;
      return value;
    }
    if (ShouldFire(FaultKind::QpError, qp->qp_num, -1, &value)) {
      ibv_qp_attr attr{};
      attr.qp_state = IBV_QPS_ERR;
      int ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
      if (ret != 0) MORI_IO_ERROR("Fault injection: ibv_modify_qp(ERR) failed: {}", ret);
    }
  }
  return direct_.PostSend(qp, wr, bad);
}

int FaultInjector::PostRecv(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad) {
  int value = 0;
  if (armed_.load(std::memory_order_relaxed) &&
      ShouldFire(FaultKind::PostRecvFail, qp->qp_num, -1, &value)) {
    *bad = wr;
    return value;
  }
  return direct_.PostRecv(qp, wr, bad);
}

int FaultInjector::PollCq(ibv_cq* cq, int numEntries, ibv_wc* wc) {
  int n = direct_.PollCq(cq, numEntries, wc);
  if (n <= 0 || !armed_.load(std::memory_order_relaxed)) return n;

  // Only successful CQEs carry a valid opcode, and only they can be "lost" or
  // turned into errors; real error CQEs pass through untouched.
  int kept = 0;
  for (int i = 0; i < n; ++i) {
    int value = 0;
    if (wc[i].status == IBV_WC_SUCCESS) {
      if (ShouldFire(FaultKind::CqeDrop, wc[i].qp_num, wc[i].opcode, &value)) continue;
      if (ShouldFire(FaultKind::CqeError, wc[i].qp_num, wc[i].opcode, &value)) {
        wc[i].status = static_cast<ibv_wc_status>(value);
        wc[i].vendor_err = 0;
      }
    }
    wc[kept++] = wc[i];
  }
  return kept;
}

VerbsOps& Verbs() { return FaultInjector::Instance(); }

}  // namespace io
}  // namespace mori

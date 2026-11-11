#include "ionic_ipc.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

namespace netmon {
namespace ipc {

const std::vector<std::string>& FieldNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> v = {"frames_rx_ok",
                                  "frames_rx_all",
                                  "frames_rx_bad_fcs",
                                  "frames_rx_bad_all",
                                  "octets_rx_ok",
                                  "octets_rx_all",
                                  "frames_rx_unicast",
                                  "frames_rx_multicast",
                                  "frames_rx_broadcast",
                                  "frames_rx_pause",
                                  "frames_rx_bad_length",
                                  "frames_rx_undersized",
                                  "frames_rx_oversized",
                                  "frames_rx_fragments",
                                  "frames_rx_jabber",
                                  "frames_rx_pripause",
                                  "frames_rx_stomped_crc",
                                  "frames_rx_too_long",
                                  "frames_rx_vlan_good",
                                  "frames_rx_dropped",
                                  "frames_rx_less_than_64b",
                                  "frames_rx_64b",
                                  "frames_rx_65b_127b",
                                  "frames_rx_128b_255b",
                                  "frames_rx_256b_511b",
                                  "frames_rx_512b_1023b",
                                  "frames_rx_1024b_1518b",
                                  "frames_rx_1519b_2047b",
                                  "frames_rx_2048b_4095b",
                                  "frames_rx_4096b_8191b",
                                  "frames_rx_8192b_9215b",
                                  "frames_rx_other",
                                  "frames_tx_ok",
                                  "frames_tx_all",
                                  "frames_tx_bad",
                                  "octets_tx_ok",
                                  "octets_tx_total",
                                  "frames_tx_unicast",
                                  "frames_tx_multicast",
                                  "frames_tx_broadcast",
                                  "frames_tx_pause",
                                  "frames_tx_pripause",
                                  "frames_tx_vlan",
                                  "frames_tx_less_than_64b",
                                  "frames_tx_64b",
                                  "frames_tx_65b_127b",
                                  "frames_tx_128b_255b",
                                  "frames_tx_256b_511b",
                                  "frames_tx_512b_1023b",
                                  "frames_tx_1024b_1518b",
                                  "frames_tx_1519b_2047b",
                                  "frames_tx_2048b_4095b",
                                  "frames_tx_4096b_8191b",
                                  "frames_tx_8192b_9215b",
                                  "frames_tx_other",
                                  "frames_tx_pri_0",
                                  "frames_tx_pri_1",
                                  "frames_tx_pri_2",
                                  "frames_tx_pri_3",
                                  "frames_tx_pri_4",
                                  "frames_tx_pri_5",
                                  "frames_tx_pri_6",
                                  "frames_tx_pri_7",
                                  "frames_rx_pri_0",
                                  "frames_rx_pri_1",
                                  "frames_rx_pri_2",
                                  "frames_rx_pri_3",
                                  "frames_rx_pri_4",
                                  "frames_rx_pri_5",
                                  "frames_rx_pri_6",
                                  "frames_rx_pri_7",
                                  "tx_pripause_0_1us_count",
                                  "tx_pripause_1_1us_count",
                                  "tx_pripause_2_1us_count",
                                  "tx_pripause_3_1us_count",
                                  "tx_pripause_4_1us_count",
                                  "tx_pripause_5_1us_count",
                                  "tx_pripause_6_1us_count",
                                  "tx_pripause_7_1us_count",
                                  "rx_pripause_0_1us_count",
                                  "rx_pripause_1_1us_count",
                                  "rx_pripause_2_1us_count",
                                  "rx_pripause_3_1us_count",
                                  "rx_pripause_4_1us_count",
                                  "rx_pripause_5_1us_count",
                                  "rx_pripause_6_1us_count",
                                  "rx_pripause_7_1us_count",
                                  "rx_pause_1us_count",
                                  "frames_tx_truncated"};
    for (size_t i = v.size(); i < kWords; i++) v.push_back("unknown_" + std::to_string(i));
    return v;
  }();
  return names;
}

const std::vector<VerifiedFirmware>& VerifiedFirmwares() {
  // n06-21, 2026-11-11: verify PASS on 7 cards (89/89 equal idle) and rates match perftest
  // (evidence 01/02).
  static const std::vector<VerifiedFirmware> v = {{"1.117.5-a-77", {0x01, 0x11}}};
  return v;
}

const VerifiedFirmware* FindVerified(const std::string& fw_version) {
  for (const auto& f : VerifiedFirmwares()) {
    if (fw_version == f.fw_version) return &f;
  }
  return nullptr;
}

std::array<uint8_t, kRequestLen> BuildRequest(uint8_t seq, std::array<uint8_t, 2> port) {
  return {0x02, seq, 0x0c, 0x00, 0x11, 0x00, 0x00, 0x00, 'I',     'P',
          'C',  'M', 0x01, 0x01, 0x02, 0x00, 0x00, 0x00, port[0], port[1]};
}

Status ParseReply(const uint8_t* b, size_t n, uint8_t seq, uint64_t* out) {
  if (n != kReplyLen) return Status::Error("ipc reply length " + std::to_string(n));
  const size_t len_field = static_cast<size_t>(b[2]) | (static_cast<size_t>(b[3]) << 8);
  if (b[0] != 0x03) return Status::Error("ipc reply: bad type byte");
  if (b[1] != seq) return Status::Error("ipc reply: sequence mismatch");
  if (len_field + 8 != n) return Status::Error("ipc reply: bad length field");
  if (b[4] != 0 || b[5] != 0 || b[6] != 0 || b[7] != 0) {
    return Status::Error("ipc reply: unexpected header bytes 4..7");
  }
  if (std::memcmp(b + 8, "IPCM", 4) != 0) return Status::Error("ipc reply: bad magic");
  if (b[12] != 0x01 || b[13] != 0x01 || b[14] != 0x02 || b[15] != 0x00) {
    return Status::Error("ipc reply: unexpected opcode");
  }
  uint32_t status;
  std::memcpy(&status, b + 16, 4);
  if (status != 0) return Status::Error("ipc reply status " + std::to_string(status));
  for (size_t i = 0; i < kWords; i++) {
    uint64_t v = 0;
    for (int k = 7; k >= 0; k--) v = (v << 8) | b[kHeaderLen + i * 8 + k];
    out[i] = v;
  }
  return Status::Ok();
}

std::string ConsistencyResult::Summary() const {
  const char* v = verdict == Consistency::kPass   ? "pass"
                  : verdict == Consistency::kFail ? "FAIL"
                                                  : "insufficient evidence";
  std::string s = std::string(v) + " (" + std::to_string(compared) +
                  " compared: " + std::to_string(equal) + " equal, " + std::to_string(ahead) +
                  " ahead, " + std::to_string(behind) + " behind)";
  if (looks_like_reset) s += "; counters look reset (ipc much smaller than ethtool)";
  for (size_t i = 0; i < behind_fields.size() && i < 5; i++) s += "; " + behind_fields[i];
  return s;
}

ConsistencyResult CheckConsistency(const uint64_t* ipc_vals,
                                   const std::vector<std::string>& eth_names,
                                   const uint64_t* eth_vals) {
  ConsistencyResult r;
  const auto& names = FieldNames();
  size_t reset_like = 0;
  for (size_t i = 0; i < kKnownFields; i++) {
    size_t j = 0;
    while (j < eth_names.size() && eth_names[j] != names[i]) j++;
    if (j == eth_names.size()) continue;  // older ethtool string table: not comparable
    const uint64_t e = eth_vals[j], v = ipc_vals[i];
    if (e == 0) continue;
    r.compared++;
    if (v == e) {
      r.equal++;
    } else if (v > e) {
      r.ahead++;
    } else {
      r.behind++;
      if (v < e / 2) reset_like++;
      r.behind_fields.push_back(names[i] + ": ipc " + std::to_string(v) + " < ethtool " +
                                std::to_string(e));
    }
  }
  if (r.compared < 4) {
    r.verdict = Consistency::kInsufficient;
  } else if (r.behind == 0) {
    r.verdict = Consistency::kPass;
  } else if (reset_like == r.behind) {
    // Everything that went backwards collapsed: counters were cleared, not misaligned.
    r.looks_like_reset = true;
    r.verdict = Consistency::kInsufficient;
  } else {
    r.verdict = Consistency::kFail;
  }
  return r;
}

}  // namespace ipc

IonicIpcSource::~IonicIpcSource() {
  if (fd_ >= 0) close(fd_);
}

Status IonicIpcSource::Open(const std::string& dev, std::array<uint8_t, 2> port,
                            const std::string& netdev) {
  dev_ = dev;
  port_ = port;
  fd_ = open(dev.c_str(), O_RDWR | O_CLOEXEC);
  if (fd_ < 0) return Status::Errno("open " + dev, errno);
  eth_ = std::make_unique<EthtoolSource>();
  return eth_->Open(netdev);
}

Status IonicIpcSource::Probe(std::string* detail) {
  std::vector<uint64_t> ev(eth_->names().size()), iv(ipc::kWords);
  Status st = eth_->Query(ev.data());  // ethtool first: its copy can only be older
  if (!st.ok()) return st;
  st = Query(iv.data());
  if (!st.ok()) return st;
  ipc::ConsistencyResult r = ipc::CheckConsistency(iv.data(), eth_->names(), ev.data());
  *detail = "consistency " + r.Summary();
  if (r.verdict == ipc::Consistency::kFail) {
    return Status::Error("ipc layout check failed: " + r.Summary());
  }
  return Status::Ok();
}

Status IonicIpcSource::Query(uint64_t* out) {
  const uint8_t seq = ++seq_;
  auto req = ipc::BuildRequest(seq, port_);
  // EINTR is reported rather than retried, so a stop request can proceed.
  ssize_t w = write(fd_, req.data(), req.size());
  if (w < 0) return Status::Errno("ipc write " + dev_, errno);
  if (static_cast<size_t>(w) != req.size()) return Status::Error("ipc short write");
  uint8_t buf[1040];
  ssize_t r = read(fd_, buf, sizeof(buf));
  if (r < 0) return Status::Errno("ipc read " + dev_, errno);
  return ipc::ParseReply(buf, static_cast<size_t>(r), seq, out);
}

}  // namespace netmon

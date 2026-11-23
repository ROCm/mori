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
// Firmware IPC port statistics (the channel `nicctl show port statistics` uses).
//
// Undocumented protocol, reverse-engineered from nicctl on firmware 1.117.5-a-77
// (work-docs projects/ainic-monitor evidence 01, raw/nicctl-port-stats-ipc.strace.txt).
// Every reply is a live MAC counter snapshot (~1.6 ms). Needs root to open /dev/tawkipcdevN.
//
//   request (20 B): 02 <seq> 0c 00 11 00 00 00 'IPCM' 01 01 02 00 00 00 <port: 2 B>
//   reply  (800 B): 03 <seq> <len-8: u16 le> 00 00 00 00 'IPCM' 01 01 02 00 <status u32>
//                   <4 B> then struct ionic_port_stats as little-endian u64s (97 words).
// Anything that does not match is rejected; the layout is never guessed.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ethtool_stats.h"
#include "source.h"

namespace netmon {

namespace ipc {

constexpr size_t kRequestLen = 20;
constexpr size_t kReplyLen = 800;
constexpr size_t kHeaderLen = 24;
constexpr size_t kWords = (kReplyLen - kHeaderLen) / 8;  // 97
constexpr size_t kKnownFields = 89;                      // struct ionic_port_stats

// Field names of struct ionic_port_stats (ionic driver ionic_if.h), then unknown_89..96.
const std::vector<std::string>& FieldNames();

struct VerifiedFirmware {
  const char* fw_version;
  std::array<uint8_t, 2> port;
};
// Firmware versions on which the layout was verified on hardware. Add an entry only after
// `ainic-netmon verify` passes and the counters were cross-checked against traffic.
const std::vector<VerifiedFirmware>& VerifiedFirmwares();
const VerifiedFirmware* FindVerified(const std::string& fw_version);

std::array<uint8_t, kRequestLen> BuildRequest(uint8_t seq, std::array<uint8_t, 2> port);
// Validate a reply and decode its 97 words into out. Pure function.
Status ParseReply(const uint8_t* buf, size_t len, uint8_t seq, uint64_t* out);

enum class Consistency { kPass, kFail, kInsufficient };
struct ConsistencyResult {
  Consistency verdict = Consistency::kInsufficient;
  size_t compared = 0;  // fields where ethtool is non-zero
  size_t equal = 0;
  size_t ahead = 0;   // ipc > ethtool (ethtool's copy is stale)
  size_t behind = 0;  // ipc < ethtool
  bool looks_like_reset = false;
  std::vector<std::string> behind_fields;
  std::string Summary() const;
};
// Detects gross misalignment only; it is not a layout proof (that is what the firmware
// whitelist is for). ethtool must be read before IPC. Pure function.
ConsistencyResult CheckConsistency(const uint64_t* ipc_vals,
                                   const std::vector<std::string>& eth_names,
                                   const uint64_t* eth_vals);

}  // namespace ipc

class IonicIpcSource : public Source {
 public:
  ~IonicIpcSource() override;
  // Opens the device (needs root). `netdev` is used for the startup consistency check.
  Status Open(const std::string& dev, std::array<uint8_t, 2> port, const std::string& netdev);

  const char* kind() const override { return "ipc"; }
  const std::vector<std::string>& names() const override { return ipc::FieldNames(); }
  double snapshot_fraction() const override { return 0.25; }
  Status Probe(std::string* detail) override;
  Status Query(uint64_t* out) override;

 private:
  int fd_ = -1;
  std::string dev_;
  std::array<uint8_t, 2> port_{};
  uint8_t seq_ = 0;
  std::unique_ptr<EthtoolSource> eth_;
};

}  // namespace netmon

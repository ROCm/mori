// Parsers against real captures (n06-21, FW 1.117.5-a-77) and corrupted variants.
#include <linux/netlink.h>
#include <rdma/rdma_netlink.h>

#include <cstring>

#include "harness.h"
#include "ionic_ipc.h"
#include "rdma_netlink.h"

using namespace netmon;

namespace {

constexpr uint32_t kFixtureSeq = 7;  // sequence number of the captured STAT_GET

Status ParseNl(const std::vector<uint8_t>& b, uint32_t seq, bool* found, bool* stale,
               std::vector<std::string>* names, std::vector<uint64_t>* vals, size_t* count) {
  vals->assign(256, 0);
  return nl::ParseStatGetReply(b.data(), b.size(), seq, found, stale, names, vals->data(),
                               vals->size(), count);
}

}  // namespace

TEST(NetlinkRealReply) {
  auto b = nt::ReadFixture("netlink_stat_get_ionic0_reply.bin");
  CHECK(b.size() == 2304);
  bool found, stale;
  std::vector<std::string> names;
  std::vector<uint64_t> vals;
  size_t count;
  Status st = ParseNl(b, kFixtureSeq, &found, &stale, &names, &vals, &count);
  CHECK(st.ok());
  CHECK(found);
  CHECK_EQ(count, size_t{54});
  CHECK_EQ(names.size(), size_t{54});
  bool has_tx = false;
  for (auto& n : names) has_tx |= n == "tx_rdma_ucast_bytes";
  CHECK(has_tx);
}

TEST(NetlinkStaleSequence) {
  auto b = nt::ReadFixture("netlink_stat_get_ionic0_reply.bin");
  bool found, stale;
  std::vector<std::string> names;
  std::vector<uint64_t> vals;
  size_t count;
  CHECK(ParseNl(b, kFixtureSeq + 1, &found, &stale, &names, &vals, &count).ok());
  CHECK(!found);
  CHECK(stale);
}

TEST(NetlinkTruncationNeverCrashes) {
  auto b = nt::ReadFixture("netlink_stat_get_ionic0_reply.bin");
  for (size_t n = 0; n < b.size(); n++) {
    std::vector<uint8_t> t(b.begin(), b.begin() + static_cast<long>(n));
    bool found, stale;
    std::vector<std::string> names;
    std::vector<uint64_t> vals;
    size_t count;
    Status st = ParseNl(t, kFixtureSeq, &found, &stale, &names, &vals, &count);
    // A cut message is either rejected or not recognised; never a full 54-counter success.
    CHECK(!(st.ok() && found && count == 54));
  }
}

TEST(NetlinkCorruptAttributeLengths) {
  auto b = nt::ReadFixture("netlink_stat_get_ionic0_reply.bin");
  int rejected = 0;
  // Flip every nla_len-sized u16 position in turn to a huge value; must never read out of
  // bounds (run under ASan/valgrind for a stronger signal) and must mostly be rejected.
  for (size_t off = NLMSG_HDRLEN; off + 2 <= b.size(); off += 4) {
    auto c = b;
    c[off] = 0xff;
    c[off + 1] = 0xff;
    bool found, stale;
    std::vector<std::string> names;
    std::vector<uint64_t> vals;
    size_t count;
    if (!ParseNl(c, kFixtureSeq, &found, &stale, &names, &vals, &count).ok()) rejected++;
  }
  CHECK(rejected > 0);
}

TEST(NetlinkErrorMessage) {
  std::vector<uint8_t> m(NLMSG_HDRLEN + sizeof(nlmsgerr), 0);
  nlmsghdr h{};
  h.nlmsg_len = static_cast<uint32_t>(m.size());
  h.nlmsg_type = NLMSG_ERROR;
  h.nlmsg_seq = 5;
  std::memcpy(m.data(), &h, sizeof(h));
  int err = -95;  // EOPNOTSUPP
  std::memcpy(m.data() + NLMSG_HDRLEN, &err, sizeof(err));
  bool found, stale;
  std::vector<std::string> names;
  std::vector<uint64_t> vals;
  size_t count;
  CHECK(!ParseNl(m, 5, &found, &stale, &names, &vals, &count).ok());
}

TEST(NetlinkRequestShape) {
  auto r = nl::BuildStatGetRequest(3, 1, 42);
  nlmsghdr h;
  std::memcpy(&h, r.data(), sizeof(h));
  CHECK_EQ(h.nlmsg_len, static_cast<uint32_t>(r.size()));
  CHECK_EQ(h.nlmsg_seq, 42u);
  CHECK_EQ(h.nlmsg_type,
           static_cast<uint16_t>(RDMA_NL_GET_TYPE(RDMA_NL_NLDEV, RDMA_NLDEV_CMD_STAT_GET)));
}

TEST(IpcRealReply) {
  auto b = nt::ReadFixture("ipc_port_stats_reply.bin");
  CHECK_EQ(b.size(), ipc::kReplyLen);
  std::vector<uint64_t> v(ipc::kWords);
  CHECK(ipc::ParseReply(b.data(), b.size(), 0x02, v.data()).ok());
  CHECK_EQ(v[0], uint64_t{80729736039});  // frames_rx_ok in the capture
  CHECK(v[4] > v[0]);                     // octets_rx_ok > frames_rx_ok
  CHECK_EQ(ipc::FieldNames().size(), ipc::kWords);
  CHECK(ipc::FieldNames()[35] == "octets_tx_ok");
  CHECK(ipc::FieldNames()[89] == "unknown_89");
}

TEST(IpcRequestMatchesCapture) {
  auto cap = nt::ReadFixture("ipc_port_stats_request.bin");
  auto r = ipc::BuildRequest(0x02, {0x01, 0x11});
  CHECK(cap.size() == r.size() && std::memcmp(cap.data(), r.data(), r.size()) == 0);
}

TEST(IpcRejectsEveryHeaderMutation) {
  auto b = nt::ReadFixture("ipc_port_stats_reply.bin");
  std::vector<uint64_t> v(ipc::kWords);
  // Bytes 20..23 are not validated (they carry a serial-number fragment, "BBFP").
  for (size_t i = 0; i < 20; i++) {
    auto c = b;
    c[i] ^= 0x5a;
    CHECK(!ipc::ParseReply(c.data(), c.size(), 0x02, v.data()).ok());
  }
  CHECK(!ipc::ParseReply(b.data(), b.size() - 8, 0x02, v.data()).ok());
  CHECK(!ipc::ParseReply(b.data(), b.size(), 0x03, v.data()).ok());  // wrong sequence
}

TEST(IpcConsistencyVerdicts) {
  std::vector<std::string> eth_names = ipc::FieldNames();
  eth_names.resize(ipc::kKnownFields);
  std::vector<uint64_t> eth(ipc::kKnownFields), live(ipc::kWords, 0);
  for (size_t i = 0; i < ipc::kKnownFields; i++) eth[i] = live[i] = 1000 + i;

  auto r = ipc::CheckConsistency(live.data(), eth_names, eth.data());
  CHECK(r.verdict == ipc::Consistency::kPass);
  CHECK_EQ(r.equal, ipc::kKnownFields);

  for (size_t i = 0; i < 10; i++) live[i] += 50;  // live copy ahead of the stale ethtool copy
  r = ipc::CheckConsistency(live.data(), eth_names, eth.data());
  CHECK(r.verdict == ipc::Consistency::kPass);
  CHECK_EQ(r.ahead, size_t{10});

  // Shifted layout: neighbouring fields swapped, one goes backwards by a little.
  std::swap(live[20], live[21]);
  r = ipc::CheckConsistency(live.data(), eth_names, eth.data());
  CHECK(r.verdict == ipc::Consistency::kFail);

  // Everything collapsed: counters were cleared, not misaligned.
  std::vector<uint64_t> cleared(ipc::kWords, 1);
  r = ipc::CheckConsistency(cleared.data(), eth_names, eth.data());
  CHECK(r.verdict == ipc::Consistency::kInsufficient);
  CHECK(r.looks_like_reset);

  // Few comparable fields, but one goes backwards: a contradiction outranks too little
  // evidence.
  std::vector<uint64_t> few_eth(ipc::kKnownFields, 0), few_live(ipc::kWords, 0);
  few_eth[0] = 1000;
  few_eth[4] = 5000;
  few_live[0] = 1000;
  few_live[4] = 4000;  // octets_rx_ok behind, but not collapsed
  r = ipc::CheckConsistency(few_live.data(), eth_names, few_eth.data());
  CHECK(r.verdict == ipc::Consistency::kFail);

  // New card, all-zero ethtool copy: nothing to compare.
  std::vector<uint64_t> zeros(ipc::kKnownFields, 0);
  r = ipc::CheckConsistency(live.data(), eth_names, zeros.data());
  CHECK(r.verdict == ipc::Consistency::kInsufficient);
}

TEST(IpcFirmwareWhitelist) {
  CHECK(ipc::FindVerified("1.117.5-a-77") != nullptr);
  CHECK(ipc::FindVerified("1.117.5-a-78") == nullptr);
  CHECK(ipc::FindVerified("") == nullptr);
}

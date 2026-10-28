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
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "src/ras/identity.hpp"
#include "src/ras/protocol.hpp"

using namespace mori::ras;

namespace {

// A scratch directory tree, removed when the fixture goes away.
class FixtureTree {
 public:
  FixtureTree() {
    char tmpl[] = "/tmp/mori_ras_identity_XXXXXX";
    root_ = mkdtemp(tmpl);
  }
  ~FixtureTree() { std::system(("rm -rf " + root_).c_str()); }
  const std::string& Root() const { return root_; }

  void Write(const std::string& rel, const std::string& content) {
    std::string path = root_ + "/" + rel;
    std::system(("mkdir -p $(dirname " + path + ")").c_str());
    std::ofstream(path) << content;
  }
  void Symlink(const std::string& rel, const std::string& target) {
    std::string path = root_ + "/" + rel;
    std::system(("mkdir -p $(dirname " + path + ")").c_str());
    ASSERT_EQ(symlink(target.c_str(), path.c_str()), 0);
  }
  // One RDMA port with GIDs given as {index, gid, type, ndev}.
  struct Gid {
    int index;
    std::string gid;
    std::string type;
    std::string ndev;
  };
  void Port(const std::string& dev, int port, const std::string& linkLayer,
            const std::vector<Gid>& gids) {
    std::string base = "sys/class/infiniband/" + dev + "/ports/" + std::to_string(port) + "/";
    Write(base + "link_layer", linkLayer + "\n");
    Write(base + "state", "4: ACTIVE\n");
    for (const Gid& g : gids) {
      Write(base + "gids/" + std::to_string(g.index), g.gid + "\n");
      Write(base + "gid_attrs/types/" + std::to_string(g.index), g.type + "\n");
      Write(base + "gid_attrs/ndevs/" + std::to_string(g.index), g.ndev + "\n");
    }
  }

 private:
  std::string root_;
};

IdentitySources FixtureSources(const FixtureTree& t) {
  IdentitySources s;
  s.procRoot = t.Root() + "/proc";
  s.sysRoot = t.Root() + "/sys";
  s.prefixLength = [](const std::string& netdev, const std::string& address) {
    if (netdev == "enp9s0" && address == "10.3.0.7") return 24;
    if (netdev == "enp10s0" && address == "fd00:4::7") return 64;
    return -1;
  };
  return s;
}

void BuildNics(FixtureTree& t) {
  const std::string zero = "0000:0000:0000:0000:0000:0000:0000:0000";
  t.Port("ionic_0", 1, "Ethernet",
         {{0, "fe80:0000:0000:0000:0262:0bff:fe12:3456", "IB/RoCE v1", "enp9s0"},
          {1, "fe80:0000:0000:0000:0262:0bff:fe12:3456", "RoCE v2", "enp9s0"},
          {2, "0000:0000:0000:0000:0000:ffff:0a03:0007", "IB/RoCE v1", "enp9s0"},
          {3, "0000:0000:0000:0000:0000:ffff:0a03:0007", "RoCE v2", "enp9s0"},
          {4, zero, "RoCE v2", "enp9s0"}});
  t.Port("ionic_1", 1, "Ethernet",
         {{0, "fe80:0000:0000:0000:0262:0bff:fe12:3457", "RoCE v2", "enp10s0"},
          {1, "fd00:0004:0000:0000:0000:0000:0000:0007", "RoCE v2", "enp10s0"}});
  t.Port("mlx5_0", 1, "InfiniBand", {{0, "fe80:0000:0000:0000:0002:c903:00ab:cdef", "", ""}});
}

const NicAttachment* FindNic(const std::vector<NicAttachment>& nics, const std::string& dev) {
  for (const auto& n : nics) {
    if (n.device == dev) return &n;
  }
  return nullptr;
}

}  // namespace

TEST(RasIdentity, ReadsProcessFactsFromAFixture) {
  FixtureTree t;
  // The comm field holds spaces and parentheses; parsing must start after the last ')'.
  std::string stat = "4242 (my (odd) proc) S 1 4242 4242 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 3 0 ";
  stat += "987654321 1000000 100 18446744073709551615 0 0 0 0 0 0 0 0 0 17 5 0 0 0 0 0\n";
  t.Write("proc/4242/stat", stat);
  t.Symlink("proc/4242/ns/pid", "pid:[4026532111]");
  t.Write("proc/sys/kernel/random/boot_id", "1b6c9a52-3f0e-4c8e-9a31-5d2c7e8f9a01\n");
  ProcessFacts f = ReadProcessFacts(t.Root() + "/proc", "4242");
  EXPECT_EQ(f.startTimeTicks, 987654321u);
  EXPECT_EQ(f.pidNamespace, 4026532111u);
  EXPECT_EQ(f.bootId, "1b6c9a52-3f0e-4c8e-9a31-5d2c7e8f9a01");

  ProcessFacts missing = ReadProcessFacts(t.Root() + "/proc", "999");
  EXPECT_EQ(missing.startTimeTicks, 0u);
  EXPECT_EQ(missing.pidNamespace, 0u);
}

TEST(RasIdentity, ReadsTheLiveProcess) {
  ProcessFacts self = ReadProcessFacts("/proc", "self");
  EXPECT_NE(self.pidNamespace, 0u);
  EXPECT_NE(self.startTimeTicks, 0u);
  EXPECT_EQ(self.bootId.size(), 36u);
  ProcessFacts byPid = ReadProcessFacts("/proc", std::to_string(getpid()));
  EXPECT_EQ(byPid.startTimeTicks, self.startTimeTicks);
  EXPECT_EQ(byPid.pidNamespace, self.pidNamespace);
}

TEST(RasIdentity, ReadsRoceAndInfinibandPorts) {
  FixtureTree t;
  BuildNics(t);
  std::vector<NicAttachment> nics = ReadNicInventory(FixtureSources(t));
  ASSERT_EQ(nics.size(), 3u);

  const NicAttachment* v4 = FindNic(nics, "ionic_0");
  ASSERT_NE(v4, nullptr);
  EXPECT_EQ(v4->port, 1u);
  EXPECT_EQ(v4->address, "10.3.0.7");  // the RoCEv2 IPv4-mapped GID, not the link-local one
  EXPECT_EQ(v4->netdev, "enp9s0");
  EXPECT_EQ(v4->rail, "10.3.0.0/24");

  const NicAttachment* v6 = FindNic(nics, "ionic_1");
  ASSERT_NE(v6, nullptr);
  EXPECT_EQ(v6->address, "fd00:4::7");  // the global RoCEv2 GID, skipping fe80::
  EXPECT_EQ(v6->netdev, "enp10s0");
  EXPECT_EQ(v6->rail, "fd00:4::/64");

  const NicAttachment* ib = FindNic(nics, "mlx5_0");
  ASSERT_NE(ib, nullptr);
  EXPECT_EQ(ib->address, "fe80::2:c903:ab:cdef");
  EXPECT_EQ(ib->rail, "");  // an InfiniBand subnet prefix says nothing about rails
}

TEST(RasIdentity, FiltersNicsLikeMoriRdmaDevices) {
  FixtureTree t;
  BuildNics(t);
  IdentitySources s = FixtureSources(t);
  s.rdmaDevices = "ionic_1";
  std::vector<NicAttachment> only = ReadNicInventory(s);
  ASSERT_EQ(only.size(), 1u);
  EXPECT_EQ(only[0].device, "ionic_1");

  s.rdmaDevices = "^ionic_1, mlx5_0";
  std::vector<NicAttachment> rest = ReadNicInventory(s);
  ASSERT_EQ(rest.size(), 1u);
  EXPECT_EQ(rest[0].device, "ionic_0");

  s.rdmaDevices = "no_such_device";
  EXPECT_TRUE(ReadNicInventory(s).empty());
  s.sysRoot = t.Root() + "/missing";
  s.rdmaDevices.clear();
  EXPECT_TRUE(ReadNicInventory(s).empty());
}

TEST(RasIdentity, AppliesTheTopologyFile) {
  Topology topo;
  topo.nics = {{"ionic_0", 1, "enp9s0", "10.3.0.7", "10.3.0.0/24", "", ""},
               {"ionic_1", 1, "enp10s0", "fd00:4::7", "fd00:4::/64", "", ""},
               {"mlx5_0", 1, "", "fe80::1", "", "", ""}};
  const std::string file =
      "# written by the job prolog\n"
      "rack = r12\n"
      "leaf=leaf-r12-a\n"
      "rail.enp9s0=rail-3\n"
      "rail.mlx5_0=ib-plane-0\n"
      "this line is not key=value? it is, but the key is unknown\n"
      "garbage without an equals sign\n"
      "\n"
      "lldp.enp9s0.via=LLDP\n"
      "lldp.enp9s0.chassis.name=sw-r12-a\n"
      "lldp.enp9s0.port.descr=Ethernet7 to host\n"
      "lldp.enp9s0.port.ifname=Ethernet7\n"
      "lldp.enp10s0.chassis.name=sw-r12-b\n"
      "lldp.enp10s0.port.mac=00:11:22:33:44:55\n"
      "lldp.enp11s0.chassis.name=not-a-rdma-port\n";
  ApplyTopologyFile(file, &topo);
  EXPECT_EQ(topo.rack, "r12");
  EXPECT_EQ(topo.leaf, "leaf-r12-a");
  EXPECT_EQ(topo.nics[0].rail, "rail-3");
  EXPECT_EQ(topo.nics[0].switchName, "sw-r12-a");
  EXPECT_EQ(topo.nics[0].switchPort, "Ethernet7");  // port.ifname beats port.descr
  EXPECT_EQ(topo.nics[1].rail, "fd00:4::/64");
  EXPECT_EQ(topo.nics[1].switchName, "sw-r12-b");
  EXPECT_EQ(topo.nics[1].switchPort, "00:11:22:33:44:55");
  EXPECT_EQ(topo.nics[2].rail, "ib-plane-0");
  EXPECT_EQ(topo.source, "file,lldp");
}

TEST(RasIdentity, TakesTheLeafFromSlurmUnlessTheFileNamesOne) {
  Topology topo;
  ApplySlurmTopology("spine1.leaf7.node12", "switch.switch.node", &topo);
  EXPECT_EQ(topo.leaf, "leaf7");
  EXPECT_EQ(topo.source, "slurm");

  Topology fromFile;
  ApplyTopologyFile("leaf=leaf-from-file\n", &fromFile);
  ApplySlurmTopology("spine1.leaf7.node12", "switch.switch.node", &fromFile);
  EXPECT_EQ(fromFile.leaf, "leaf-from-file");
  EXPECT_EQ(fromFile.source, "file");

  Topology mismatched;
  ApplySlurmTopology("leaf7.node12", "switch.switch.node", &mismatched);
  EXPECT_EQ(mismatched.leaf, "");
  ApplySlurmTopology("", "", &mismatched);
  EXPECT_EQ(mismatched.source, "");
}

TEST(RasIdentity, CollectsEverySourceIntoOneIdentity) {
  FixtureTree t;
  BuildNics(t);
  t.Write("proc/self/stat", "77 (x) S 1 1 1 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 555 0 0\n");
  t.Symlink("proc/self/ns/pid", "pid:[4026531836]");
  t.Write("proc/sys/kernel/random/boot_id", "b00710d0-0000-4000-8000-000000000001\n");
  t.Write("topology", "rack=r1\nlldp.enp9s0.chassis.name=sw1\nlldp.enp9s0.port.local=7\n");
  IdentitySources s = FixtureSources(t);
  s.topologyFile = t.Root() + "/topology";
  s.slurmTopologyAddr = "s0.l3.n1";
  s.slurmTopologyPattern = "switch.switch.node";

  RuntimeUuid uuid{0x1234, 0x8000000000005678ull};
  RasProcessId id = CollectIdentity("node-1", "host-1", "10.0.0.1", 77, uuid, s);
  EXPECT_EQ(id.uuid, uuid);
  EXPECT_EQ(id.nodeId, "node-1");
  EXPECT_EQ(id.hostname, "host-1");
  EXPECT_EQ(id.host, "10.0.0.1");
  EXPECT_EQ(id.pid, 77);
  EXPECT_EQ(id.startTimeTicks, 555u);
  EXPECT_EQ(id.pidNamespace, 4026531836u);
  EXPECT_EQ(id.bootId, "b00710d0-0000-4000-8000-000000000001");
  EXPECT_EQ(id.topology.rack, "r1");
  EXPECT_EQ(id.topology.leaf, "l3");
  EXPECT_EQ(id.topology.nics.size(), 3u);
  EXPECT_EQ(FindNic(id.topology.nics, "ionic_0")->switchName, "sw1");
  EXPECT_EQ(FindNic(id.topology.nics, "ionic_0")->switchPort, "7");
  EXPECT_EQ(id.topology.source, "sysfs,file,lldp,slurm");
  EXPECT_TRUE(id.labels.empty());
  EXPECT_TRUE(id.attachments.empty());  // they arrive later, from the engines

  // A missing topology file is skipped, not an error.
  s.topologyFile = t.Root() + "/does-not-exist";
  RasProcessId noFile = CollectIdentity("node-1", "host-1", "10.0.0.1", 77, uuid, s);
  EXPECT_EQ(noFile.topology.rack, "");
  EXPECT_EQ(noFile.topology.source, "sysfs,slurm");
}

TEST(RasIdentity, RuntimeUuidsAreRandomVersion4) {
  std::set<std::pair<uint64_t, uint64_t>> seen;
  for (int i = 0; i < 1000; ++i) {
    RuntimeUuid u = NewRuntimeUuid();
    EXPECT_EQ((u.hi >> 12) & 0xf, 4u);  // version 4
    EXPECT_EQ(u.lo >> 62, 2u);          // RFC 4122 variant
    EXPECT_NE(u.lo, 0u);
    EXPECT_TRUE(seen.insert({u.hi, u.lo}).second);
  }
  RuntimeUuid u{0x0123456789abcdefull, 0x8edcba9876543210ull};
  EXPECT_EQ(u.ToString(), "01234567-89ab-cdef-8edc-ba9876543210");
}

TEST(RasIdentity, TokensAre128RandomBits) {
  std::set<std::string> seen;
  for (int i = 0; i < 1000; ++i) {
    std::string tok = NewToken();
    ASSERT_EQ(tok.size(), 32u);
    EXPECT_EQ(tok.find_first_not_of("0123456789abcdef"), std::string::npos);
    EXPECT_TRUE(seen.insert(tok).second);
  }
}

TEST(RasIdentity, LabelLimitsMatchWhatPeersAccept) {
  EXPECT_TRUE(ValidLabels({}));
  EXPECT_TRUE(ValidLabels({{"role", "prefill"}}));
  Labels many;
  for (size_t i = 0; i < proto::kMaxLabels; ++i) many["k" + std::to_string(i)] = "v";
  EXPECT_TRUE(ValidLabels(many));
  many["one-more"] = "v";
  EXPECT_FALSE(ValidLabels(many));
  EXPECT_FALSE(ValidLabels({{std::string(proto::kMaxLabelKey + 1, 'k'), "v"}}));
  EXPECT_FALSE(ValidLabels({{"k", std::string(proto::kMaxLabelValue + 1, 'v')}}));
  EXPECT_FALSE(ValidLabels({{"", "v"}}));
}

namespace {

NicAttachment Picked(const std::string& device, uint32_t port) {
  NicAttachment n;
  n.device = device;
  n.port = port;
  return n;
}

}  // namespace

TEST(RasIdentity, KeepsOneAttachmentPerEngineAndGpu) {
  RasProcessId id;
  AttachGpu("engine-b", 1, "0000:15:00.0", {}, &id);
  AttachGpu("engine-a", 1, "0000:15:00.0", {}, &id);
  AttachGpu("engine-a", 0, "0000:05:00.0", {}, &id);
  AttachGpu("engine-a", 0, "0000:05:00.0", {}, &id);  // another registration on GPU 0
  ASSERT_EQ(id.attachments.size(), 3u);
  // Ordered by engine, then GPU.
  EXPECT_EQ(id.attachments[0].owner, "engine-a");
  EXPECT_EQ(id.attachments[0].gpu, 0);
  EXPECT_EQ(id.attachments[0].pciBusId, "0000:05:00.0");
  EXPECT_EQ(id.attachments[1].owner, "engine-a");
  EXPECT_EQ(id.attachments[1].gpu, 1);
  EXPECT_EQ(id.attachments[2].owner, "engine-b");
  EXPECT_EQ(id.attachments[2].gpu, 1);
  for (const GpuAttachment& a : id.attachments) EXPECT_TRUE(a.nics.empty());
}

TEST(RasIdentity, AttachedNicsComeFromTheHostInventory) {
  RasProcessId id;
  const NicAttachment inventory{"ionic_0", 1, "enp9s0", "10.3.0.7", "10.3.0.0/24", "sw1", "7"};
  id.topology.nics.push_back(inventory);
  id.topology.nics.push_back({"ionic_0", 2, "enp9s1", "10.4.0.7", "10.4.0.0/24", "sw2", "9"});

  // A session's NICs can arrive before the registration, if RAS dropped that event.
  AttachGpu("engine-a", 0, "0000:05:00.0", {Picked("ionic_0", 1)}, &id);
  AttachGpu("engine-a", 0, "", {}, &id);
  AttachGpu("engine-a", 0, "0000:05:00.0", {Picked("ionic_0", 1), Picked("ionic_9", 1)}, &id);
  ASSERT_EQ(id.attachments.size(), 1u);
  const GpuAttachment& a = id.attachments[0];
  EXPECT_EQ(a.pciBusId, "0000:05:00.0");
  ASSERT_EQ(a.nics.size(), 2u);
  EXPECT_EQ(a.nics[0], inventory);  // matched by device and port, so not ionic_0 port 2
  // A NIC missing from the inventory keeps its device and port only.
  EXPECT_EQ(a.nics[1], Picked("ionic_9", 1));
}

TEST(RasIdentity, DetachingAnEngineDropsOnlyItsAttachments) {
  RasProcessId id;
  AttachGpu("engine-a", 0, "0000:05:00.0", {}, &id);
  AttachGpu("engine-a", 1, "0000:15:00.0", {}, &id);
  AttachGpu("engine-b", 0, "0000:05:00.0", {}, &id);
  DetachOwner("engine-a", &id);
  ASSERT_EQ(id.attachments.size(), 1u);
  EXPECT_EQ(id.attachments[0].owner, "engine-b");
  DetachOwner("engine-unknown", &id);
  EXPECT_EQ(id.attachments.size(), 1u);
}

TEST(RasIdentity, AttachmentsAreBounded) {
  RasProcessId id;
  for (size_t i = 0; i < kMaxAttachments + 3; ++i) {
    AttachGpu("engine-" + std::to_string(1000 + i), 0, "", {}, &id);
  }
  EXPECT_EQ(id.attachments.size(), kMaxAttachments);
  std::vector<NicAttachment> nics;
  for (size_t i = 0; i < kMaxAttachmentNics + 3; ++i)
    nics.push_back(Picked("nic" + std::to_string(i), 1));
  AttachGpu("engine-1000", 0, "", nics, &id);
  EXPECT_EQ(id.attachments[0].nics.size(), kMaxAttachmentNics);
}

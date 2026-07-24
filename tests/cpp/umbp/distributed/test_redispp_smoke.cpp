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

// redis-plus-plus build/link smoke test.
//
// This does NOT test the UMBP store — it only proves that the redis++ submodule
// compiles with this toolchain and links (static redis++ + system hiredis)
// before any store code depends on it. It is the de-risk gate for adopting
// redis++ as the Redis-backend client library (cluster / sentinel / pooling):
// if this builds and links, the later seam migration rests on a proven base.
//
// Runtime behaviour: connects to UMBP_REDIS_URI (default tcp://127.0.0.1:6379)
// and, if a cluster seed is given via UMBP_REDIS_CLUSTER_SEEDS, to that cluster;
// both skip cleanly when nothing is reachable, so BUILD_TESTS on a host without
// a store never fails.

#include <gtest/gtest.h>
#include <sw/redis++/redis++.h>

#include <cstdlib>
#include <string>

namespace {

std::string EnvOr(const char* key, const std::string& def) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string(v) : def;
}

// Compile check: constructing ConnectionOptions exercises the redis++ headers.
// Actual linkage of the static archive is proven by the ping tests below, which
// reference symbols defined in redis++'s .cpp files (Redis::ping /
// RedisCluster) regardless of whether they run or skip at runtime.
TEST(RedisPlusPlusSmoke, HeadersCompile) {
  sw::redis::ConnectionOptions opts;
  opts.host = "127.0.0.1";
  opts.port = 6379;
  EXPECT_EQ(opts.port, 6379);
}

TEST(RedisPlusPlusSmoke, PingSingleIfReachable) {
  const std::string uri = EnvOr("UMBP_REDIS_URI", "tcp://127.0.0.1:6379");
  try {
    sw::redis::Redis redis(uri);
    const std::string pong = redis.ping();
    EXPECT_EQ(pong, "PONG");
  } catch (const sw::redis::Error& e) {
    GTEST_SKIP() << "no RESP store reachable at " << uri << " (" << e.what() << "); skipping";
  }
}

TEST(RedisPlusPlusSmoke, PingClusterIfSeedGiven) {
  const char* seeds = std::getenv("UMBP_REDIS_CLUSTER_SEEDS");
  if (seeds == nullptr || *seeds == '\0') {
    GTEST_SKIP() << "UMBP_REDIS_CLUSTER_SEEDS not set; skipping cluster smoke";
  }
  try {
    // RedisCluster fetches CLUSTER SLOTS on construction, proving cluster-path
    // symbols link and a real cluster is reachable.
    sw::redis::RedisCluster cluster{std::string(seeds)};
    const std::string pong = cluster.redis("smoke", false).ping();
    EXPECT_EQ(pong, "PONG");
  } catch (const sw::redis::Error& e) {
    GTEST_SKIP() << "no Redis Cluster reachable at " << seeds << " (" << e.what() << "); skipping";
  }
}

}  // namespace

// Plain C++ test for SpdkSsdTier.
// Requires SPDK available at runtime (malloc bdev by default).
// Usage: ./test_spdk_ssd_tier
//        or: UMBP_SSD_BACKEND=spdk UMBP_SPDK_BDEV=NVMe0n1 ./test_spdk_ssd_tier

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "umbp/common/config.h"
#include "umbp/storage/spdk_ssd_tier.h"

static UMBPConfig MakeTestConfig() {
    UMBPConfig cfg;
    cfg.ssd_backend = "spdk";
    cfg.ssd_capacity_bytes = 512ULL * 1024 * 1024;  // 512 MB

    const char* bdev = std::getenv("UMBP_SPDK_BDEV");
    if (bdev) {
        cfg.spdk_bdev_name = bdev;
    } else {
        cfg.spdk_bdev_name = "Malloc0";
    }
    const char* mask = std::getenv("UMBP_SPDK_REACTOR_MASK");
    if (mask) cfg.spdk_reactor_mask = mask;
    const char* mem = std::getenv("UMBP_SPDK_MEM_MB");
    if (mem) cfg.spdk_mem_size_mb = std::atoi(mem);
    else cfg.spdk_mem_size_mb = 256;
    const char* pci = std::getenv("UMBP_SPDK_NVME_PCI");
    if (pci) cfg.spdk_nvme_pci_addr = pci;

    return cfg;
}

static void test_basic_write_read() {
    printf("  test_basic_write_read...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid() && "SpdkSsdTier should be initialized");

    const std::string key = "test_key_001";
    const size_t data_size = 8192;
    std::vector<char> write_data(data_size);
    for (size_t i = 0; i < data_size; ++i)
        write_data[i] = static_cast<char>(i & 0xFF);

    assert(!tier.Exists(key));
    bool ok = tier.Write(key, write_data.data(), data_size);
    assert(ok && "Write should succeed");
    assert(tier.Exists(key));

    std::vector<char> read_data(data_size, 0);
    ok = tier.ReadIntoPtr(key, reinterpret_cast<uintptr_t>(read_data.data()), data_size);
    assert(ok && "Read should succeed");
    assert(std::memcmp(write_data.data(), read_data.data(), data_size) == 0 &&
           "Data mismatch");

    printf("  PASS\n");
}

static void test_evict_and_capacity() {
    printf("  test_evict_and_capacity...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid());

    const std::string key = "evict_test_key";
    std::vector<char> data(4096, 0x55);

    tier.Write(key, data.data(), data.size());
    assert(tier.Exists(key));

    auto [used1, total] = tier.Capacity();
    assert(total > 0);
    assert(used1 > 0);

    bool evicted = tier.Evict(key);
    assert(evicted);
    assert(!tier.Exists(key));

    auto [used2, _] = tier.Capacity();
    assert(used2 < used1);

    printf("  PASS\n");
}

static void test_batch_write_read() {
    printf("  test_batch_write_read...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid());

    constexpr int N = 32;
    constexpr size_t ITEM_SIZE = 16384;  // 16KB per item

    std::vector<std::string> keys(N);
    std::vector<std::vector<char>> write_bufs(N);
    std::vector<const void*> write_ptrs(N);
    std::vector<size_t> sizes(N, ITEM_SIZE);

    for (int i = 0; i < N; ++i) {
        keys[i] = "batch_key_" + std::to_string(i);
        write_bufs[i].resize(ITEM_SIZE);
        for (size_t j = 0; j < ITEM_SIZE; ++j)
            write_bufs[i][j] = static_cast<char>((i + j) & 0xFF);
        write_ptrs[i] = write_bufs[i].data();
    }

    auto write_results = tier.BatchWrite(keys, write_ptrs, sizes);
    for (int i = 0; i < N; ++i) {
        assert(write_results[i] && "Batch write item should succeed");
        assert(tier.Exists(keys[i]));
    }

    // Batch read
    std::vector<std::vector<char>> read_bufs(N);
    std::vector<uintptr_t> read_ptrs(N);
    for (int i = 0; i < N; ++i) {
        read_bufs[i].resize(ITEM_SIZE, 0);
        read_ptrs[i] = reinterpret_cast<uintptr_t>(read_bufs[i].data());
    }

    auto read_results = tier.BatchReadIntoPtr(keys, read_ptrs, sizes);
    for (int i = 0; i < N; ++i) {
        assert(read_results[i] && "Batch read item should succeed");
        assert(std::memcmp(write_bufs[i].data(), read_bufs[i].data(), ITEM_SIZE) == 0 &&
               "Batch data mismatch");
    }

    printf("  PASS\n");
}

static void test_dedup() {
    printf("  test_dedup...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid());

    const std::string key = "dedup_key";
    std::vector<char> data(4096, 0xAA);

    bool ok1 = tier.Write(key, data.data(), data.size());
    assert(ok1);

    // Write again — should return true (key already exists)
    bool ok2 = tier.Write(key, data.data(), data.size());
    assert(ok2);

    printf("  PASS\n");
}

static void test_lru() {
    printf("  test_lru...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid());

    std::vector<char> data(4096, 0);

    tier.Write("lru_a", data.data(), data.size());
    tier.Write("lru_b", data.data(), data.size());
    tier.Write("lru_c", data.data(), data.size());

    // LRU tail should be "lru_a" (oldest)
    assert(tier.GetLRUKey() == "lru_a");

    auto candidates = tier.GetLRUCandidates(2);
    assert(candidates.size() == 2);
    assert(candidates[0] == "lru_a");
    assert(candidates[1] == "lru_b");

    // Access "lru_a" → moves to front
    std::vector<char> buf(4096);
    tier.ReadIntoPtr("lru_a", reinterpret_cast<uintptr_t>(buf.data()), 4096);

    assert(tier.GetLRUKey() == "lru_b");

    printf("  PASS\n");
}

static void test_large_batch() {
    printf("  test_large_batch (deep queue exercise)...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid());

    constexpr int N = 256;
    constexpr size_t ITEM_SIZE = 65536;  // 64KB

    std::vector<std::string> keys(N);
    std::vector<std::vector<char>> bufs(N);
    std::vector<const void*> ptrs(N);
    std::vector<size_t> sizes(N, ITEM_SIZE);

    for (int i = 0; i < N; ++i) {
        keys[i] = "lg_" + std::to_string(i);
        bufs[i].resize(ITEM_SIZE);
        std::memset(bufs[i].data(), static_cast<unsigned char>(i), ITEM_SIZE);
        ptrs[i] = bufs[i].data();
    }

    auto wr = tier.BatchWrite(keys, ptrs, sizes);
    int write_ok = 0;
    for (auto b : wr) if (b) ++write_ok;
    printf("    wrote %d/%d items\n", write_ok, N);
    assert(write_ok > 0);

    // Read back the successful ones
    std::vector<std::string> rkeys;
    std::vector<uintptr_t> rptrs;
    std::vector<size_t> rsizes;
    std::vector<std::vector<char>> rbufs;

    for (int i = 0; i < N; ++i) {
        if (!wr[i]) continue;
        rkeys.push_back(keys[i]);
        rbufs.emplace_back(ITEM_SIZE, 0);
        rptrs.push_back(reinterpret_cast<uintptr_t>(rbufs.back().data()));
        rsizes.push_back(ITEM_SIZE);
    }

    auto rd = tier.BatchReadIntoPtr(rkeys, rptrs, rsizes);
    int read_ok = 0;
    for (size_t j = 0; j < rd.size(); ++j) {
        if (rd[j]) ++read_ok;
    }
    printf("    read back %d/%zu items\n", read_ok, rd.size());
    assert(read_ok == write_ok);

    printf("  PASS\n");
}

static void test_clear() {
    printf("  test_clear...\n");
    auto cfg = MakeTestConfig();
    SpdkSsdTier tier(cfg);
    assert(tier.IsValid());

    std::vector<char> data(4096, 0x77);
    tier.Write("clear_key", data.data(), data.size());
    assert(tier.Exists("clear_key"));

    tier.Clear();
    assert(!tier.Exists("clear_key"));
    printf("  PASS\n");
}

int main() {
    printf("=== test_spdk_ssd_tier ===\n");
    test_basic_write_read();
    test_evict_and_capacity();
    test_batch_write_read();
    test_dedup();
    test_lru();
    test_large_batch();
    test_clear();
    printf("=== ALL PASSED ===\n");
    return 0;
}

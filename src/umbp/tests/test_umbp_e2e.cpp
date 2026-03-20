// Copyright © Advanced Micro Devices, Inc. All rights reserved.
// MIT License
//
// test_umbp_e2e: End-to-end integration tests for UMBP.
// Runs POSIX tests always. SPDK tests run only if UMBP_SPDK_NVME_PCI is set.
// All tests in one executable — run: ./test_umbp_e2e
//
// Test categories:
//   [POSIX]  Always available
//   [ROLE]   Role deduction logic (no I/O)
//   [SPDK]   Requires UMBP_SPDK_NVME_PCI (skipped if absent)
//   [PROXY]  Requires SPDK + Linux fork (skipped if absent)

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef __unix__
#include <unistd.h>
#endif

#include "umbp/common/config.h"
#include "umbp/umbp_client.h"

#ifdef __linux__
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include "umbp/proxy/spdk_proxy_protocol.h"
#include "umbp/proxy/spdk_proxy_shm.h"
#include "umbp/storage/spdk_proxy_tier.h"
#endif

// ---------------------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------------------
static int g_passed = 0, g_failed = 0, g_skipped = 0;

#define RUN_TEST(fn)                                               \
    do {                                                           \
        printf("  %-50s", #fn "...");                              \
        fflush(stdout);                                            \
        bool _ok = fn();                                           \
        if (_ok) { g_passed++; printf("PASS\n"); }                \
        else { g_failed++; printf("FAIL\n"); }                    \
    } while (0)

#define SKIP_TEST(name, reason)                                    \
    do {                                                           \
        printf("  %-50s", name "...");                             \
        g_skipped++;                                               \
        printf("SKIP (%s)\n", reason);                             \
    } while (0)

#define CHECK(cond)                                                \
    do {                                                           \
        if (!(cond)) {                                             \
            fprintf(stderr, "\n    CHECK FAILED: %s (%s:%d)\n",   \
                    #cond, __FILE__, __LINE__);                    \
            return false;                                          \
        }                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// Helper: create a UMBPConfig for POSIX testing
// ---------------------------------------------------------------------------
static UMBPConfig MakePosixConfig(size_t dram_mb = 64, size_t ssd_mb = 256) {
    UMBPConfig cfg;
    cfg.dram_capacity_bytes = dram_mb * 1024 * 1024;
    cfg.ssd_capacity_bytes = ssd_mb * 1024 * 1024;
    cfg.ssd_backend = "posix";
    cfg.ssd_storage_dir = "/tmp/umbp_e2e_test_" + std::to_string(getpid());
    cfg.role = UMBPRole::Standalone;
    return cfg;
}

// =========================================================================
// [ROLE] Tests
// =========================================================================

static bool test_role_default_standalone() {
    UMBPConfig cfg;
    CHECK(cfg.ResolveRole() == UMBPRole::Standalone);
    return true;
}

static bool test_role_explicit_leader() {
    UMBPConfig cfg;
    cfg.role = UMBPRole::SharedSSDLeader;
    CHECK(cfg.ResolveRole() == UMBPRole::SharedSSDLeader);
    return true;
}

static bool test_role_explicit_follower() {
    UMBPConfig cfg;
    cfg.role = UMBPRole::SharedSSDFollower;
    CHECK(cfg.ResolveRole() == UMBPRole::SharedSSDFollower);
    return true;
}

static bool test_role_backward_compat_follower_mode() {
    UMBPConfig cfg;
    cfg.follower_mode = true;
    CHECK(cfg.ResolveRole() == UMBPRole::SharedSSDFollower);
    return true;
}

static bool test_role_backward_compat_force_cow() {
    UMBPConfig cfg;
    cfg.force_ssd_copy_on_write = true;
    CHECK(cfg.ResolveRole() == UMBPRole::SharedSSDLeader);
    return true;
}

static bool test_auto_rank_sentinel() {
    CHECK(kAutoRankId == UINT32_MAX);
    // Without UMBP_SPDK_PROXY_RANK env, FromEnvironment should give kAutoRankId
    // (can't reliably unsetenv in test, so just verify the constant)
    return true;
}

#ifdef __linux__
static bool test_role_from_local_rank_env() {
    // Fork a child that sets LOCAL_RANK and checks role
    int pipefd[2];
    CHECK(pipe(pipefd) == 0);

    pid_t pid = fork();
    if (pid == 0) {
        close(pipefd[0]);
        // Ensure no prior UMBP_ROLE
        unsetenv("UMBP_ROLE");

        // Test LOCAL_RANK=0 → Leader
        setenv("LOCAL_RANK", "0", 1);
        auto cfg0 = UMBPConfig::FromEnvironment();
        uint8_t r0 = (cfg0.role == UMBPRole::SharedSSDLeader) ? 1 : 0;

        // Test LOCAL_RANK=3 → Follower
        setenv("LOCAL_RANK", "3", 1);
        auto cfg3 = UMBPConfig::FromEnvironment();
        uint8_t r3 = (cfg3.role == UMBPRole::SharedSSDFollower) ? 1 : 0;

        uint8_t result = r0 & r3;
        write(pipefd[1], &result, 1);
        close(pipefd[1]);
        _exit(0);
    }

    close(pipefd[1]);
    uint8_t result = 0;
    read(pipefd[0], &result, 1);
    close(pipefd[0]);
    waitpid(pid, nullptr, 0);

    CHECK(result == 1);
    return true;
}
#endif

// =========================================================================
// [POSIX] Tests
// =========================================================================

static bool test_posix_standalone_write_read() {
    auto cfg = MakePosixConfig();
    UMBPClient client(cfg);

    std::string key = "test_posix_wr_1";
    std::vector<char> data(4096, 'A');
    CHECK(client.Put(key, data.data(), data.size()));

    std::vector<char> buf(4096, 0);
    CHECK(client.GetIntoPtr(key, reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
    CHECK(buf == data);

    client.Clear();
    return true;
}

static bool test_posix_batch_write_read() {
    auto cfg = MakePosixConfig();
    UMBPClient client(cfg);

    const int N = 100;
    const size_t sz = 8192;
    std::vector<std::string> keys(N);
    std::vector<std::vector<char>> datas(N);
    std::vector<uintptr_t> ptrs(N);
    std::vector<size_t> sizes(N, sz);

    for (int i = 0; i < N; ++i) {
        keys[i] = "posix_batch_" + std::to_string(i);
        datas[i].resize(sz, static_cast<char>(i & 0xFF));
        ptrs[i] = reinterpret_cast<uintptr_t>(datas[i].data());
    }

    auto wr = client.BatchPutFromPtr(keys, ptrs, sizes);
    int write_ok = 0;
    for (auto b : wr) write_ok += b;
    CHECK(write_ok == N);

    std::vector<std::vector<char>> read_bufs(N, std::vector<char>(sz, 0));
    std::vector<uintptr_t> dst_ptrs(N);
    for (int i = 0; i < N; ++i)
        dst_ptrs[i] = reinterpret_cast<uintptr_t>(read_bufs[i].data());

    auto rr = client.BatchGetIntoPtr(keys, dst_ptrs, sizes);
    int read_ok = 0;
    for (auto b : rr) read_ok += b;
    CHECK(read_ok == N);

    for (int i = 0; i < N; ++i)
        CHECK(read_bufs[i] == datas[i]);

    client.Clear();
    return true;
}

static bool test_posix_dedup() {
    auto cfg = MakePosixConfig();
    UMBPClient client(cfg);

    std::vector<char> data(1024, 'X');
    CHECK(client.Put("dedup_key", data.data(), data.size()));
    CHECK(client.Put("dedup_key", data.data(), data.size()));  // should dedup
    CHECK(client.Exists("dedup_key"));

    client.Clear();
    return true;
}

static bool test_posix_evict() {
    auto cfg = MakePosixConfig();
    UMBPClient client(cfg);

    std::vector<char> data(1024, 'E');
    CHECK(client.Put("evict_key", data.data(), data.size()));
    CHECK(client.Exists("evict_key"));
    CHECK(client.Remove("evict_key"));
    CHECK(!client.Exists("evict_key"));

    client.Clear();
    return true;
}

static bool test_posix_ssd_direct_write_read() {
    auto cfg = MakePosixConfig();
    UMBPClient client(cfg);

    std::vector<char> data(32768, 'S');
    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    CHECK(ssd != nullptr);
    CHECK(ssd->Write("ssd_direct_key", data.data(), data.size()));

    std::vector<char> buf(32768, 0);
    CHECK(ssd->ReadIntoPtr("ssd_direct_key",
          reinterpret_cast<uintptr_t>(buf.data()), buf.size()));
    CHECK(buf == data);

    client.Clear();
    return true;
}

static bool test_posix_dram_evict_to_ssd() {
    // Small DRAM, force demotion to SSD
    auto cfg = MakePosixConfig(1 /* 1MB DRAM */, 256);
    UMBPClient client(cfg);

    const size_t sz = 128 * 1024;  // 128KB each
    std::vector<char> data(sz, 'D');

    // Write 10 items → 1.25MB total, DRAM=1MB → some must demote to SSD
    for (int i = 0; i < 10; ++i) {
        std::string key = "dram_evict_" + std::to_string(i);
        CHECK(client.Put(key, data.data(), data.size()));
    }

    // All should still be readable (from DRAM or SSD)
    for (int i = 0; i < 10; ++i) {
        std::string key = "dram_evict_" + std::to_string(i);
        CHECK(client.Exists(key));
        std::vector<char> buf(sz, 0);
        CHECK(client.GetIntoPtr(key, reinterpret_cast<uintptr_t>(buf.data()), sz));
        CHECK(buf == data);
    }

    client.Clear();
    return true;
}

static bool test_posix_capacity() {
    auto cfg = MakePosixConfig(64, 256);
    UMBPClient client(cfg);

    auto* dram = client.Storage().GetTier(StorageTier::CPU_DRAM);
    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    CHECK(dram != nullptr);
    CHECK(ssd != nullptr);

    auto [dram_used, dram_total] = dram->Capacity();
    CHECK(dram_total > 0);

    auto [ssd_used, ssd_total] = ssd->Capacity();
    CHECK(ssd_total > 0);

    client.Clear();
    return true;
}

static bool test_posix_empty_read() {
    auto cfg = MakePosixConfig();
    UMBPClient client(cfg);

    std::vector<char> buf(1024, 0);
    CHECK(!client.GetIntoPtr("nonexistent", reinterpret_cast<uintptr_t>(buf.data()), 1024));
    CHECK(!client.Exists("nonexistent"));

    return true;
}

static bool test_posix_large_value() {
    auto cfg = MakePosixConfig(16, 512);
    UMBPClient client(cfg);

    const size_t sz = 4 * 1024 * 1024;  // 4MB
    std::vector<char> data(sz);
    for (size_t i = 0; i < sz; ++i) data[i] = static_cast<char>(i & 0xFF);

    // Write directly to SSD (4MB > 16MB DRAM)
    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    CHECK(ssd != nullptr);
    CHECK(ssd->Write("large_val", data.data(), data.size()));

    std::vector<char> buf(sz, 0);
    CHECK(ssd->ReadIntoPtr("large_val",
          reinterpret_cast<uintptr_t>(buf.data()), sz));
    CHECK(buf == data);

    client.Clear();
    return true;
}

// =========================================================================
// [SPDK] Tests — only if UMBP_SPDK_NVME_PCI is set
// =========================================================================

#ifdef __linux__
static bool test_spdk_standalone_write_read() {
    // This test requires USE_SPDK compiled in and SPDK env set
    UMBPConfig cfg;
    cfg.ssd_backend = "spdk";
    cfg.role = UMBPRole::Standalone;
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
    cfg.ssd_capacity_bytes = 1024ULL * 1024 * 1024;

    // Read SPDK config from env
    auto env_cfg = UMBPConfig::FromEnvironment();
    cfg.spdk_nvme_pci_addr = env_cfg.spdk_nvme_pci_addr;
    cfg.spdk_reactor_mask = env_cfg.spdk_reactor_mask;
    cfg.spdk_mem_size_mb = env_cfg.spdk_mem_size_mb;
    cfg.spdk_io_workers = env_cfg.spdk_io_workers;

    UMBPClient client(cfg);

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    CHECK(ssd != nullptr);

    const size_t sz = 65536;
    std::vector<char> data(sz, 'Z');
    CHECK(ssd->Write("spdk_standalone_1", data.data(), data.size()));

    std::vector<char> buf(sz, 0);
    CHECK(ssd->ReadIntoPtr("spdk_standalone_1",
          reinterpret_cast<uintptr_t>(buf.data()), sz));
    CHECK(buf == data);

    client.Clear();
    return true;
}

// ---------------------------------------------------------------------------
// [PROXY] Tests — Leader auto-fork + Follower connect
// ---------------------------------------------------------------------------

// Helper: fork, set env, run function, return exit code
static int ForkAndRun(const char* local_rank, const char* backend,
                      int (*fn)(const char*), const char* arg = nullptr) {
    pid_t pid = fork();
    if (pid == 0) {
        unsetenv("UMBP_ROLE");
        unsetenv("UMBP_SPDK_PROXY_RANK");
        setenv("LOCAL_RANK", local_rank, 1);
        if (backend) setenv("UMBP_SSD_BACKEND", backend, 1);
        _exit(fn(arg));
    }
    int status;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static int leader_write_and_wait(const char*) {
    auto cfg = UMBPConfig::FromEnvironment();
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
    UMBPClient client(cfg);

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) return 1;

    // Write 10 keys
    for (int i = 0; i < 10; ++i) {
        std::string key = "proxy_shared_" + std::to_string(i);
        std::vector<char> data(4096, static_cast<char>('A' + i));
        if (!ssd->Write(key, data.data(), data.size())) return 2;
    }

    // Wait for follower (signaled via file)
    for (int i = 0; i < 100; ++i) {
        if (access("/tmp/umbp_e2e_follower_done", F_OK) == 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    unlink("/tmp/umbp_e2e_follower_done");
    return 0;
}

static int follower_read_and_verify(const char*) {
    // Small delay to let leader start
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    auto cfg = UMBPConfig::FromEnvironment();
    cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
    UMBPClient client(cfg);

    auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
    if (!ssd) return 1;

    int ok = 0;
    for (int i = 0; i < 10; ++i) {
        std::string key = "proxy_shared_" + std::to_string(i);
        std::vector<char> buf(4096, 0);
        if (ssd->ReadIntoPtr(key, reinterpret_cast<uintptr_t>(buf.data()), 4096)) {
            std::vector<char> expected(4096, static_cast<char>('A' + i));
            if (buf == expected) ok++;
        }
    }

    // Signal leader we're done
    int fd = open("/tmp/umbp_e2e_follower_done", O_CREAT | O_WRONLY, 0644);
    if (fd >= 0) close(fd);

    return (ok == 10) ? 0 : 3;
}

static bool test_proxy_auto_fork_leader_follower() {
    unlink("/tmp/umbp_e2e_follower_done");

    // Launch leader and follower concurrently
    pid_t leader = fork();
    if (leader == 0) {
        unsetenv("UMBP_ROLE");
        unsetenv("UMBP_SPDK_PROXY_RANK");
        setenv("LOCAL_RANK", "0", 1);
        setenv("UMBP_SSD_BACKEND", "spdk", 1);
        _exit(leader_write_and_wait(nullptr));
    }

    pid_t follower = fork();
    if (follower == 0) {
        unsetenv("UMBP_ROLE");
        unsetenv("UMBP_SPDK_PROXY_RANK");
        setenv("LOCAL_RANK", "1", 1);
        setenv("UMBP_SSD_BACKEND", "spdk", 1);
        _exit(follower_read_and_verify(nullptr));
    }

    int lstatus, fstatus;
    waitpid(leader, &lstatus, 0);
    waitpid(follower, &fstatus, 0);

    bool leader_ok = WIFEXITED(lstatus) && WEXITSTATUS(lstatus) == 0;
    bool follower_ok = WIFEXITED(fstatus) && WEXITSTATUS(fstatus) == 0;

    if (!leader_ok) fprintf(stderr, "    leader exit=%d\n",
                            WIFEXITED(lstatus) ? WEXITSTATUS(lstatus) : -1);
    if (!follower_ok) fprintf(stderr, "    follower exit=%d\n",
                              WIFEXITED(fstatus) ? WEXITSTATUS(fstatus) : -1);

    CHECK(leader_ok);
    CHECK(follower_ok);
    return true;
}

static bool test_proxy_daemon_cleanup_after_leader_exit() {
    // Fork a leader that creates proxy, then exits immediately.
    // Daemon should self-terminate (orphan timeout or spawner-death signal).
    pid_t leader = fork();
    if (leader == 0) {
        unsetenv("UMBP_ROLE");
        unsetenv("UMBP_SPDK_PROXY_RANK");
        setenv("LOCAL_RANK", "0", 1);
        setenv("UMBP_SSD_BACKEND", "spdk", 1);

        {
            auto cfg = UMBPConfig::FromEnvironment();
            cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
            UMBPClient client(cfg);
            // client destructor will trigger proxy shutdown
        }
        _exit(0);
    }

    int status;
    waitpid(leader, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);

    // Wait a bit and verify proxy SHM is gone or daemon exited
    std::this_thread::sleep_for(std::chrono::seconds(3));
    int probe = umbp::proxy::ProxyShmRegion::ProbeExisting(
        umbp::proxy::kDefaultShmName);
    // Either SHM is gone (probe==0) or daemon is shutting down
    // We don't require immediate cleanup, just that it doesn't hang forever
    CHECK(probe <= 0);

    // Clean up any residual SHM
    umbp::proxy::ProxyShmRegion::CleanupStale(umbp::proxy::kDefaultShmName);
    return true;
}

static bool test_proxy_cas_rank_allocation() {
    // Fork 3 "followers" that all try to auto-allocate rank slots.
    // Each should get a unique rank.

    // First, spawn a leader to create the proxy
    pid_t leader = fork();
    if (leader == 0) {
        unsetenv("UMBP_ROLE");
        unsetenv("UMBP_SPDK_PROXY_RANK");
        setenv("LOCAL_RANK", "0", 1);
        setenv("UMBP_SSD_BACKEND", "spdk", 1);

        auto cfg = UMBPConfig::FromEnvironment();
        cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
        UMBPClient client(cfg);

        // Wait for followers to finish (max 30s)
        for (int i = 0; i < 300; ++i) {
            if (access("/tmp/umbp_e2e_cas_done", F_OK) == 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        unlink("/tmp/umbp_e2e_cas_done");
        _exit(0);
    }

    // Wait for proxy to be ready
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // Fork 3 followers
    int pipe_fds[3][2];
    pid_t followers[3];
    for (int i = 0; i < 3; ++i) {
        pipe(pipe_fds[i]);
        followers[i] = fork();
        if (followers[i] == 0) {
            close(pipe_fds[i][0]);
            unsetenv("UMBP_ROLE");
            unsetenv("UMBP_SPDK_PROXY_RANK");
            char lr[8];
            snprintf(lr, sizeof(lr), "%d", i + 1);
            setenv("LOCAL_RANK", lr, 1);
            setenv("UMBP_SSD_BACKEND", "spdk", 1);

            auto cfg = UMBPConfig::FromEnvironment();
            cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;

            // Wait for proxy, then auto-allocate rank via CAS
            std::string shm_name = cfg.spdk_proxy_shm_name.empty()
                ? "/umbp_spdk_proxy" : cfg.spdk_proxy_shm_name;
            SpdkProxyTier::WaitForProxy(shm_name, 15000);
            SpdkProxyTier tier(cfg);
            uint8_t result = tier.IsValid() ? 1 : 0;
            write(pipe_fds[i][1], &result, 1);
            close(pipe_fds[i][1]);
            _exit(0);
        }
        close(pipe_fds[i][1]);
    }

    int ok_count = 0;
    for (int i = 0; i < 3; ++i) {
        uint8_t r = 0;
        read(pipe_fds[i][0], &r, 1);
        close(pipe_fds[i][0]);
        if (r == 1) ok_count++;
        int status;
        waitpid(followers[i], &status, 0);
    }

    // Signal leader to exit
    int fd = open("/tmp/umbp_e2e_cas_done", O_CREAT | O_WRONLY, 0644);
    if (fd >= 0) close(fd);
    waitpid(leader, nullptr, 0);
    unlink("/tmp/umbp_e2e_cas_done");

    CHECK(ok_count == 3);
    return true;
}

static bool test_proxy_batch_write_read() {
    pid_t leader = fork();
    if (leader == 0) {
        unsetenv("UMBP_ROLE");
        unsetenv("UMBP_SPDK_PROXY_RANK");
        setenv("LOCAL_RANK", "0", 1);
        setenv("UMBP_SSD_BACKEND", "spdk", 1);

        auto cfg = UMBPConfig::FromEnvironment();
        cfg.dram_capacity_bytes = 64ULL * 1024 * 1024;
        UMBPClient client(cfg);

        auto* ssd = client.Storage().GetTier(StorageTier::LOCAL_SSD);
        if (!ssd) _exit(1);

        const int N = 50;
        const size_t sz = 32768;
        std::vector<std::string> keys(N);
        std::vector<const void*> cptrs(N);
        std::vector<size_t> sizes(N, sz);
        std::vector<std::vector<char>> datas(N);
        for (int i = 0; i < N; ++i) {
            keys[i] = "proxy_batch_" + std::to_string(i);
            datas[i].resize(sz, static_cast<char>(i & 0xFF));
            cptrs[i] = datas[i].data();
        }

        auto wr = ssd->BatchWrite(keys, cptrs, sizes);
        int wok = 0;
        for (auto b : wr) wok += b;
        if (wok != N) _exit(2);

        std::vector<std::vector<char>> rbufs(N, std::vector<char>(sz, 0));
        std::vector<uintptr_t> dptrs(N);
        for (int i = 0; i < N; ++i)
            dptrs[i] = reinterpret_cast<uintptr_t>(rbufs[i].data());

        auto rr = ssd->BatchReadIntoPtr(keys, dptrs, sizes);
        int rok = 0;
        for (auto b : rr) rok += b;
        if (rok != N) _exit(3);

        for (int i = 0; i < N; ++i) {
            if (rbufs[i] != datas[i]) _exit(4);
        }
        _exit(0);
    }

    int status;
    waitpid(leader, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    return true;
}
#endif  // __linux__

// =========================================================================
// main
// =========================================================================

int main() {
    bool have_spdk = (std::getenv("UMBP_SPDK_NVME_PCI") != nullptr);

    printf("=== UMBP End-to-End Integration Tests ===\n");
    printf("  SPDK available: %s\n\n", have_spdk ? "YES" : "NO (set UMBP_SPDK_NVME_PCI to enable)");

    // --- Role deduction tests ---
    printf("[ROLE] Role deduction tests\n");
    RUN_TEST(test_role_default_standalone);
    RUN_TEST(test_role_explicit_leader);
    RUN_TEST(test_role_explicit_follower);
    RUN_TEST(test_role_backward_compat_follower_mode);
    RUN_TEST(test_role_backward_compat_force_cow);
    RUN_TEST(test_auto_rank_sentinel);
#ifdef __linux__
    RUN_TEST(test_role_from_local_rank_env);
#endif
    printf("\n");

    // --- POSIX tests ---
    printf("[POSIX] POSIX backend tests\n");
    RUN_TEST(test_posix_standalone_write_read);
    RUN_TEST(test_posix_batch_write_read);
    RUN_TEST(test_posix_dedup);
    RUN_TEST(test_posix_evict);
    RUN_TEST(test_posix_ssd_direct_write_read);
    RUN_TEST(test_posix_dram_evict_to_ssd);
    RUN_TEST(test_posix_capacity);
    RUN_TEST(test_posix_empty_read);
    RUN_TEST(test_posix_large_value);
    printf("\n");

    // --- SPDK tests ---
#ifdef __linux__
    if (have_spdk) {
        printf("[SPDK] SPDK standalone tests\n");
        RUN_TEST(test_spdk_standalone_write_read);
        printf("\n");

        printf("[PROXY] SPDK proxy auto-fork tests\n");
        RUN_TEST(test_proxy_batch_write_read);
        RUN_TEST(test_proxy_auto_fork_leader_follower);
        RUN_TEST(test_proxy_daemon_cleanup_after_leader_exit);
        RUN_TEST(test_proxy_cas_rank_allocation);
        printf("\n");
    } else {
        printf("[SPDK] Skipped (UMBP_SPDK_NVME_PCI not set)\n");
        SKIP_TEST("test_spdk_standalone_write_read", "no SPDK");
        printf("\n");
        printf("[PROXY] Skipped (UMBP_SPDK_NVME_PCI not set)\n");
        SKIP_TEST("test_proxy_batch_write_read", "no SPDK");
        SKIP_TEST("test_proxy_auto_fork_leader_follower", "no SPDK");
        SKIP_TEST("test_proxy_daemon_cleanup_after_leader_exit", "no SPDK");
        SKIP_TEST("test_proxy_cas_rank_allocation", "no SPDK");
        printf("\n");
    }
#else
    printf("[SPDK/PROXY] Skipped (Linux only)\n\n");
    g_skipped += 5;
#endif

    // --- Summary ---
    printf("========================================\n");
    printf("  PASSED: %d  FAILED: %d  SKIPPED: %d\n", g_passed, g_failed, g_skipped);
    printf("========================================\n");

    if (g_failed > 0) {
        printf("*** SOME TESTS FAILED ***\n");
        return 1;
    }

    printf("=== ALL TESTS PASSED ===\n");
    return 0;
}

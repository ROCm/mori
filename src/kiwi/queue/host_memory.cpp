#include <mori/kiwi/queue/detail/host_memory.hpp>
#include <mori/kiwi/queue/error.hpp>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <string>

namespace mori::kiwi::queue::detail {
namespace {

void check_hsa(hsa_status_t status, const char* call) {
    if (status == HSA_STATUS_SUCCESS) return;
    const char* text = nullptr;
    hsa_status_string(status, &text);
    throw queue_error(std::string(call) + " failed: " +
                      (text ? text : "unknown HSA error"));
}

struct HsaRuntime {
    HsaRuntime() { check_hsa(hsa_init(), "hsa_init"); }
    ~HsaRuntime() { (void)hsa_shut_down(); }
};

void ensure_hsa_runtime() {
    static HsaRuntime runtime;
    (void)runtime;
}

struct GpuLookup {
    int domain;
    int bus;
    int device;
    hsa_agent_t result{};
};

hsa_status_t find_gpu(hsa_agent_t agent, void* data) {
    hsa_device_type_t type{};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) !=
            HSA_STATUS_SUCCESS ||
        type != HSA_DEVICE_TYPE_GPU) {
        return HSA_STATUS_SUCCESS;
    }

    uint32_t domain = 0;
    uint32_t bdf = 0;
    if (hsa_agent_get_info(
            agent,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
            &domain) != HSA_STATUS_SUCCESS ||
        hsa_agent_get_info(
            agent,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
            &bdf) != HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }

    auto& lookup = *static_cast<GpuLookup*>(data);
    if (domain == static_cast<uint32_t>(lookup.domain) &&
        ((bdf >> 8) & 0xffu) == static_cast<uint32_t>(lookup.bus) &&
        ((bdf >> 3) & 0x1fu) == static_cast<uint32_t>(lookup.device)) {
        lookup.result = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

hsa_agent_t current_gpu() {
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));

    GpuLookup lookup{
        properties.pciDomainID,
        properties.pciBusID,
        properties.pciDeviceID,
    };
    hsa_status_t status = hsa_iterate_agents(find_gpu, &lookup);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        check_hsa(status, "hsa_iterate_agents");
    }
    if (!lookup.result.handle) {
        throw queue_error(
            "Unable to match the current HIP device to an HSA GPU agent");
    }
    return lookup.result;
}

hsa_status_t find_pool(hsa_amd_memory_pool_t pool, void* data) {
    hsa_amd_segment_t segment{};
    uint32_t flags = 0;
    bool alloc_allowed = false;
    if (hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) !=
            HSA_STATUS_SUCCESS ||
        segment != HSA_AMD_SEGMENT_GLOBAL) {
        return HSA_STATUS_SUCCESS;
    }
    if (hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) !=
            HSA_STATUS_SUCCESS ||
        !(flags &
          HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED)) {
        return HSA_STATUS_SUCCESS;
    }
    if (hsa_amd_memory_pool_get_info(
            pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
            &alloc_allowed) != HSA_STATUS_SUCCESS ||
        !alloc_allowed) {
        return HSA_STATUS_SUCCESS;
    }

    *static_cast<hsa_amd_memory_pool_t*>(data) = pool;
    return HSA_STATUS_INFO_BREAK;
}

hsa_amd_memory_pool_t extended_scope_pool(hsa_agent_t gpu) {
    hsa_agent_t cpu{};
    check_hsa(
        hsa_agent_get_info(
            gpu,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NEAREST_CPU),
            &cpu),
        "hsa_agent_get_info(HSA_AMD_AGENT_INFO_NEAREST_CPU)");

    hsa_amd_memory_pool_t pool{};
    hsa_status_t status =
        hsa_amd_agent_iterate_memory_pools(cpu, find_pool, &pool);
    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        check_hsa(status, "hsa_amd_agent_iterate_memory_pools");
    }
    if (!pool.handle) {
        throw queue_error(
            "No allocatable extended-scope fine-grained host memory pool");
    }
    return pool;
}

} // namespace

void* host_malloc_uc(std::size_t size) {
    ensure_hsa_runtime();
    hsa_agent_t gpu = current_gpu();
    void* ptr = nullptr;
    check_hsa(hsa_amd_memory_pool_allocate(
                  extended_scope_pool(gpu), size, 0, &ptr),
              "hsa_amd_memory_pool_allocate");

    hsa_status_t status =
        hsa_amd_agents_allow_access(1, &gpu, nullptr, ptr);
    if (status != HSA_STATUS_SUCCESS) {
        hsa_amd_memory_pool_free(ptr);
        check_hsa(status, "hsa_amd_agents_allow_access");
    }
    return ptr;
}

void host_free_uc(void* ptr) noexcept {
    if (ptr) (void)hsa_amd_memory_pool_free(ptr);
}

} // namespace kiwi::queue::detail

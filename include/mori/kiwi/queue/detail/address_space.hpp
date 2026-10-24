/**
 * @file address_space.hpp
 * @brief AMDGPU global address-space pointer alias for device handles
 *
 * Queue memory (host-mapped slots, progress words, device counters) is always
 * reachable through the global address space. A generic pointer that the
 * compiler cannot trace to a kernel argument (for example, a DeviceHandle
 * loaded from a device-resident context) lowers to flat_* instructions. A flat
 * access also counts against lgkmcnt, so a later LDS wait blocks on the
 * flat store's PCIe acknowledgement. Declaring the handle pointers in the
 * global address space makes every access a global_* instruction no matter
 * how the handle reaches the kernel.
 */

#pragma once

namespace mori::kiwi::queue::detail {

/// Pointer to T in the AMDGPU global address space (address space 1).
template<typename T>
using global_ptr = T __attribute__((address_space(1)))*;

/// Convert a generic pointer to global memory into a global_ptr.
template<typename T>
inline global_ptr<T> to_global(T* p) {
    return (global_ptr<T>)p;
}

} // namespace kiwi::queue::detail

#pragma once

#include <cstddef>

namespace mori::kiwi::queue::detail {

void* host_malloc_uc(std::size_t size);
void host_free_uc(void* ptr) noexcept;

template<typename T>
T* host_malloc_uc(std::size_t count) {
    return static_cast<T*>(host_malloc_uc(count * sizeof(T)));
}

} // namespace kiwi::queue::detail

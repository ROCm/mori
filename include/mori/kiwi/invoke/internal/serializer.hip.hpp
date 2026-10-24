#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <cstring>

namespace mori::kiwi::invoke {
namespace internal {

// Helper to serialize a single value into byte buffer (device-compatible)
template<typename T>
__host__ __device__ inline void pack_one(uint8_t*& dst, const T& arg) {
    __builtin_memcpy(dst, &arg, sizeof(T));
    dst += sizeof(T);
}

// Helper to deserialize a single value from byte buffer (host-side)
template<typename T>
__host__ inline void unpack_one(const uint8_t*& src, T& arg) {
    std::memcpy(&arg, src, sizeof(T));
    src += sizeof(T);
}

// Helper to calculate total size (handles empty pack)
template<typename... Args>
struct TotalSize {
    static constexpr size_t value = (sizeof(Args) + ...);
};

template<>
struct TotalSize<> {
    static constexpr size_t value = 0;
};

// Argument serializer for variadic template arguments
template<typename... Args>
class ArgumentSerializer {
public:
    // Compile-time calculations
    static constexpr size_t total_bytes = TotalSize<Args...>::value;

    // Serialize arguments into byte buffer
    __host__ __device__ static void serialize(uint8_t* buffer, const Args&... args) {
        uint8_t* ptr = buffer;
        // Fold expression: pack each argument sequentially
        (pack_one(ptr, args), ...);
    }

    // Deserialize byte buffer into tuple of arguments (host-side)
    __host__ static std::tuple<Args...> deserialize(const uint8_t* buffer) {
        std::tuple<Args...> result;
        const uint8_t* ptr = buffer;
        deserialize_impl(result, ptr, std::index_sequence_for<Args...>{});
        return result;
    }

private:
    // Helper to deserialize into tuple using index sequence
    template<size_t... Is>
    __host__ static void deserialize_impl(
        std::tuple<Args...>& result,
        const uint8_t*& ptr,
        std::index_sequence<Is...>
    ) {
        // Fold expression: unpack each tuple element sequentially
        (unpack_one(ptr, std::get<Is>(result)), ...);
    }
};

// Specialization for no arguments
template<>
class ArgumentSerializer<> {
public:
    static constexpr size_t total_bytes = 0;

    __host__ __device__ static void serialize(uint8_t* buffer) {
        // No-op: no arguments to serialize
        (void)buffer;
    }

    __host__ static std::tuple<> deserialize(const uint8_t* buffer) {
        (void)buffer;
        return std::tuple<>();
    }
};

}  // namespace internal
}  // namespace kiwi::invoke

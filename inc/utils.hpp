#pragma once

#include <concepts>
#include <cstdint>
#include <cstdlib>

namespace utils
{
struct AlignedDeleter
{
    void operator()(void* ptr) const noexcept
    {
        std::free(ptr);
    }
};

template <std::integral T>
[[nodiscard]] constexpr T align(T value, size_t alignment) noexcept
{
    const T mask = static_cast<T>(alignment-1);
    return (value+mask)&~mask;
}
        
template <std::integral T>
[[nodiscard]] constexpr T align_64(T value) noexcept
{
    return align(value, 64);
}
}

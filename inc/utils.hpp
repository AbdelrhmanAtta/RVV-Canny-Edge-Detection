/**
 * @file    utils.hpp
 * @brief   Utility functions for memory management and alignment.
 */

#pragma once

#include <concepts>
#include <cstdlib>

namespace utils::memory
{
struct deleter
{
    void operator()(void* ptr) const noexcept
    {
        std::free(ptr);
    }
};

/**
 * @brief   Allocates memory with a specified alignment.
 * @param   alignment       The alignment boundary (must be a power of 2).
 * @param   size            The number of bytes to allocate.
 * @return                  Pointer to the allocated memory, or nullptr on failure.
 */
inline void* aligned_alloc(size_t alignment, size_t size) noexcept
{
    size_t remainder = size % alignment;
    if (remainder != 0)
        size += (alignment - remainder);
    return std::aligned_alloc(alignment, size);
}

template <std::integral T>
[[nodiscard]] constexpr T align(T value, size_t alignment) noexcept
{
    const T mask = static_cast<T>(alignment - 1);
    return (value + mask) & ~mask;
}
   
template <std::integral T>
[[nodiscard]] constexpr T align_64(T value) noexcept
{
    return align(value, 64);
}

} // namespace utils::memory

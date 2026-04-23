/**
 * @file    utils.hpp
 * @brief   Utility functions for memory management and alignment.
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <cstdlib>

namespace utils::memory
{
/**
 * @brief   Struct-like function object for aligned memory deallocation.
 */
struct deleter
{
    void operator()(void* ptr) const noexcept
    {
        std::free(ptr);
    }
};

/**
 * @brief   Aligns a given value up to the nearest multiple of the specified alignment.
 * @tparam  T               An integral type to be aligned.
 * @param   value           The value to align.
 * @param   alignment       The alignment boundary.
 * @return                  Aligned value, rounded up to the nearest multiple of alignment.
 */
template <std::integral T>
[[nodiscard]] constexpr T align(T value, size_t alignment) noexcept
{
    const T mask = static_cast<T>(alignment-1);
    return (value+mask)&~mask;
}
   
/**
 * @brief   Aligns a given value up to the nearest multiple of 64.
 * @tparam  T               An integral type to be aligned.
 * @param   value           The value to align.
 * @return                  Aligned value, rounded up to 64.
 */
template <std::integral T>
[[nodiscard]] constexpr T align_64(T value) noexcept
{
    return align(value, 64);
}

} // namespace utils::memory

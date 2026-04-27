#pragma once

#include "utils.hpp"
#include <cstddef>
#include <memory>
#include <cstdint>
#include <expected>

/**
 * @file std_types.hpp
 * @brief Standard type definitions and common status codes.
 *
 * This header provides basic platform-independent types and return
 * status values used across the project.
 */

/**
 * @enum Status
 * @brief Represents the execution result of a function.
 *
 * Used to indicate whether an operation succeeded or failed.
 */
enum class Status : uint8_t
{
    E_OK  = 0,              /**< Operation completed successfully */
    E_NOK = 1,              /**< Operation failed */
    E_INVAL_PTR,            /**< Invalid Pointer */
    E_ALLOC_FAIL,           /**< Memory Allocation Failure */
    E_INVAL_DIR,            /**< Invalid Direction */
};

namespace image::io
{
/**
 * @brief   Metadata and buffer management for an image.
 * @tparam  PixelT The underlying data type of each pixel (default is uint8_t).
 */
template <typename PixelT = uint8_t>
struct metadata_t
{
    /** @brief RAII pointer to 64-byte aligned pixel data. */
    std::unique_ptr<PixelT[], utils::memory::deleter> image_buffer;
    uint32_t width;
    uint32_t height;
    size_t pixel_count;
    size_t aligned_buffer_size;
};

} // namespace image::io

namespace processing::convolution
{
template <typename KernelT = uint8_t, typename AccumulatorT = int32_t>
struct kernel_t
{
    const KernelT* data;
    uint8_t width;
    uint8_t height;
    AccumulatorT sum;
};

template <typename PixelT = uint8_t>
struct clamp_t
{
    PixelT clamp_min;
    PixelT clamp_max;
};

} // namespace processing::convolution

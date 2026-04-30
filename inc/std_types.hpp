/**
 * @file std_types.hpp
 * @brief Standard type definitions and common status codes.
 */

#pragma once

#include "utils.hpp"
#include <cstddef>
#include <memory>
#include <cstdint>
#include <expected>


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
    std::unique_ptr<PixelT[], utils::memory::deleter> buffer;
    uint32_t width;
    uint32_t height;
    size_t pixel_count;
    size_t aligned_buffer_size;
};

} // namespace image::io

namespace processing
{
/**
 * @brief   Structure representing a convolution kernel, including its data, dimensions, and normalization sum.
 * @tparam  KernelT         The data type of the kernel elements (e.g., uint8_t, uint16_t).
 * @tparam  AccumulatorT    The data type used for accumulating convolution sums to prevent overflow (e.g., int32_t, uint32_t).
 */
template <typename KernelT = uint8_t, typename AccumulatorT = int32_t>
struct kernel_t
{
    const KernelT* data;
    uint8_t width;
    uint8_t height;
    AccumulatorT sum;
};

/**
 * @brief   Structure for defining clamping parameters to constrain output pixel values within a specified range.
 * @tparam  PixelT The data type of the pixel values (e.g., uint8_t, uint16_t).
 */
template <typename PixelT = uint8_t>
struct clamp_t
{
    PixelT clamp_min;
    PixelT clamp_max;
};

} // namespace processing::convolution

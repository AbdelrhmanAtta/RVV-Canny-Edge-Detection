/**
 * @file std_types.hpp
 * @brief Standard type definitions and common status codes.
 */

#pragma once

#include "utils.hpp"

#include <cstddef>
#include <memory>
#include <cstdint>


/**
 * @enum Status
 * @brief Represents the execution result of a function.
 * E_OK: Operation completed successfully.
 * E_NOK: Operation failed due to a non-specific error.
 * E_INVAL_PTR: Operation failed due to an invalid pointer argument.
 * E_ALLOC_FAIL: Operation failed due to memory allocation failure.
 * E_INVAL_DIR: Operation failed due to an invalid direction argument.
 */
enum class Status : uint8_t
{
    E_OK  = 0,
    E_NOK = 1,
    E_INVAL_PTR,
    E_ALLOC_FAIL,
    E_INVAL_DIR,
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
    std::unique_ptr<PixelT[], utils::memory::deleter> buffer;
    uint32_t width;
    uint32_t height;
    size_t pixel_count;
    size_t aligned_buffer_size;
};

} // namespace image::io

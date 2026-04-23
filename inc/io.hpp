/**
 * @file    io.hpp
 * @brief   Utilities for loading and saving raw grayscale image data for RISC-V.
 *
 * This module provides type-safe, aligned memory allocation and IO operations
 * designed for high-performance image processing.
 */

#pragma once

#include "std_types.hpp"
#include "utils.hpp"
#include <string_view>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <fstream>
#include <expected>

namespace image::io
{
/**
 * @brief   Metadata and buffer management for an image.
 * @tparam  PixelT The underlying data type of each pixel (default is uint8_t).
 */
template <typename PixelT = uint8_t>
struct image_metadata
{
    /** @brief RAII pointer to 64-byte aligned pixel data. */
    std::unique_ptr<PixelT[], utils::AlignedDeleter> image_buffer;
    uint32_t image_width;
    uint32_t image_height;
    size_t pixel_count;
    size_t aligned_buffer_size;
};

/** @brief Result type alias for image loading operations. */
template <typename PixelT = uint8_t>
using load_result_t = std::expected<image_metadata<PixelT>, Status>;

/**
 * @brief   Loads a raw binary file from disk into an aligned RAII buffer.
 * Allocates 64-byte aligned memory to satisfy RISC-V vector and cache line requirements.
 * Files are expected to be located in the `./assets/` directory.
 * @tparam  PixelT              The pixel component type (e.g., uint8_t, uint16_t, float).
 * @param   file_name           Name of the raw file to load.
 * @param   image_width         Expected width of the image.
 * @param   image_height        Expected height of the image.
 * @return  load_result_t       A struct containing the managed buffer and metadata, 
 * or a Status error code on failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] load_result_t<PixelT> load_raw(const std::string_view file_name, 
                                            uint32_t image_width, 
                                            uint32_t image_height)
{
    const size_t pixel_count = static_cast<size_t>(image_width) * image_height;
    const size_t total_bytes = pixel_count * sizeof(PixelT);
    const size_t aligned_buffer_size = utils::memory::align_64(total_bytes);

    // Allocate aligned memory for RISC-V Vector/Cache efficiency
    void* raw_ptr = std::aligned_alloc(64, aligned_buffer_size);
    if (!raw_ptr) 
    {
        return std::unexpected(Status::E_ALLOC_FAIL);
    }

    // Initialize Metadata and transfer ownership to unique_ptr
    image_metadata<PixelT> meta
    {
        .image_buffer = std::unique_ptr<PixelT[], utils::memory::deleter>(static_cast<PixelT*>(raw_ptr)),
        .image_width = image_width,
        .image_height = image_height,
        .pixel_count = pixel_count,
        .aligned_buffer_size = aligned_buffer_size
    };

    std::filesystem::path file_path = std::filesystem::path("./assets/") / file_name;
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file) 
    {
        return std::unexpected(Status::E_INVAL_DIR);
    }

    // Read the actual image data into the aligned buffer
    if (!file.read(reinterpret_cast<char*>(meta.image_buffer.get()), total_bytes))
    {
        return std::unexpected(Status::E_NOK);
    }

    return meta;
}

/**
 * @brief   Saves pixel data from a buffer to a raw binary file.
 * Writes the specified raw pixel data to the `./assets/` directory.
 * Does not write the alignment padding, only the actual pixel data.
 * @tparam  PixelT          The pixel component type (e.g., uint8_t, uint16_t, float).
 * @param   file_name       Name of the file to create or overwrite.
 * @param   image_buffer    Pointer to the source pixel data.
 * @param   image_width     Width of the image in pixels.
 * @param   image_height    Height of the image in pixels.
 * @return  Status          E_OK on success, or a Status error code on failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status save_raw(const std::string_view file_name,
                            const PixelT* image_buffer,
                            uint32_t image_width,
                            uint32_t image_height)
{
    // Validate input pointer
    if (!image_buffer)
    {
        return Status::E_INVAL_PTR;
    }

    std::filesystem::path file_path = std::filesystem::path("./assets/") / file_name;
    std::ofstream file(file_path, std::ios::binary);

    if (!file)
    {
        return Status::E_INVAL_DIR;
    }

    // Calculate actual data size
    const size_t total_bytes = static_cast<size_t>(image_width) * image_height * sizeof(PixelT);
    
    // Write raw bytes to disk
    if (!file.write(reinterpret_cast<const char*>(image_buffer), total_bytes))
    {
        return Status::E_NOK;
    }
    
    return Status::E_OK;
}

} // namespace image::io
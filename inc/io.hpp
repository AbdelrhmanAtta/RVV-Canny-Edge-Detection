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
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <fstream>

namespace image::io
{
/**
 * @brief   Loads a raw binary file from disk into an aligned RAII buffer.
 * Allocates 64-byte aligned memory to satisfy RISC-V vector and cache line requirements.
 * Files are expected to be located in the `./assets/` directory.
 * @tparam  PixelT              The pixel component type (e.g., uint8_t, uint16_t, float).
 * @param   file_name           Name of the raw file to load.
 * @param   metadata            Reference to the metadata struct to be populated.
 * @return  Status              E_OK on success, or a Status error code on failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status load_raw(const std::string_view file_name, 
                              metadata_t<PixelT>& metadata)
{
    const size_t pixel_count = static_cast<size_t>(metadata.width) * metadata.height;
    const size_t total_bytes = pixel_count * sizeof(PixelT);
    const size_t aligned_buffer_size = utils::memory::align_64(total_bytes);

    if(!pixel_count)
    {
        return Status::E_NOK;
    }

    // Allocate aligned memory for the image buffer
    void* raw_ptr = std::aligned_alloc(64, aligned_buffer_size);
    if (!raw_ptr) 
    {
        return Status::E_ALLOC_FAIL;
    }

    // Update the existing Metadata struct in-place
    metadata.buffer.reset(static_cast<PixelT*>(raw_ptr));
    metadata.pixel_count = pixel_count;
    metadata.aligned_buffer_size = aligned_buffer_size;

#if defined(TESTS)
    std::filesystem::path file_path = std::filesystem::path("./tests_assets/") / file_name;
#else
    std::filesystem::path file_path = std::filesystem::path("./assets/") / file_name;
#endif
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file) 
    {
        return Status::E_INVAL_DIR;
    }

    if (!file.read(reinterpret_cast<char*>(metadata.buffer.get()), total_bytes))
    {
        return Status::E_NOK;
    }

    return Status::E_OK;
}

/**
 * @brief   Saves pixel data from a buffer to a raw binary file.
 * Writes the specified raw pixel data to the `./assets/` directory.
 * Does not write the alignment padding, only the actual pixel data.
 * @tparam  PixelT          The pixel component type (e.g., uint8_t, uint16_t, float).
 * @param   file_name       Name of the file to create or overwrite.
 * @param   metadata        The metadata struct containing the image buffer and dimensions.
 * @return  Status          E_OK on success, or a Status error code on failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status save_raw(const std::string_view file_name,
                              const metadata_t<PixelT>& metadata)
{
    if(!metadata.buffer)
    {
        return Status::E_INVAL_PTR;
    }
    if(!metadata.pixel_count)
    {
        return Status::E_NOK;
    }

#if defined(TESTS)
    std::filesystem::path file_path = std::filesystem::path("./tests_assets/") / file_name;
#else
    std::filesystem::path file_path = std::filesystem::path("./assets/") / file_name;
#endif
    std::ofstream file(file_path, std::ios::binary);
    if (!file)
    {
        return Status::E_INVAL_DIR;
    }

    // Calculate actual data size
    const size_t total_bytes = metadata.pixel_count * sizeof(PixelT);
    
    // Write raw bytes to disk
    if (!file.write(reinterpret_cast<const char*>(metadata.buffer.get()), total_bytes))
    {
        return Status::E_NOK;
    }
    
    return Status::E_OK;
}

} // namespace image::io
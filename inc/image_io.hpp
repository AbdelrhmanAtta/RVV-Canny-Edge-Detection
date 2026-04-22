/**
 * @file image_io.hpp
 * @brief Utilities for loading and saving raw grayscale image data for RISC-V.
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

namespace io
{
/**
 * @brief Metadata and buffer management for an image.
 * @tparam PixelT The underlying data type of each pixel (default is uint8_t).
 */
template <typename PixelT = uint8_t>
struct ImageMetaData
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
using load_result_t = std::expected<ImageMetaData<PixelT>, Status>;

/**
 * @brief   Loads a raw binary file from disk into an aligned RAII buffer.
 *          Allocates 64-byte aligned memory to satisfy RISC-V vector and cache line requirements.
 *          Files are expected to be located in the `./assets/` directory.
 * @tparam  PixelT     The pixel component type (e.g., uint8_t, uint16_t, float).
 * @param   file_name  Name of the raw file to load.
 * @param   image_width Expected width of the image.
 * @param   image_height Expected height of the image.
 * @return  load_result_t A struct containing the managed buffer and metadata, 
 *          or a Status error code on failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] load_result_t<PixelT> image_io_loadRaw(const std::string_view file_name, 
                                                    uint32_t image_width, 
                                                    uint32_t image_height)
{
    const size_t pixel_count = static_cast<size_t>(image_width) * image_height;
    const size_t total_bytes = pixel_count * sizeof(PixelT);
    const size_t aligned_buffer_size = utils::align_64(total_bytes);

    // Allocate aligned memory
    void* raw_ptr = std::aligned_alloc(64, aligned_buffer_size);
    if (!raw_ptr) 
    {
        return std::unexpected(Status::E_ALLOC_FAIL);
    }

    // Initialize Metadata
    ImageMetaData<PixelT> meta
    {
        .image_buffer = std::unique_ptr<PixelT[], utils::AlignedDeleter>(static_cast<PixelT*>(raw_ptr)),
        .image_width = image_width,
        .image_height = image_height,
        .pixel_count = pixel_count,
        .aligned_buffer_size = aligned_buffer_size
    };

    // Open file using filesystem path
    std::filesystem::path file_path = std::filesystem::path("./assets/") / file_name;
    std::ifstream file(file_path, std::ios::binary);
    
    if (!file) 
    {
        return std::unexpected(Status::E_INVAL_DIR);
    }

    // Read the actual data
    if (!file.read(reinterpret_cast<char*>(meta.image_buffer.get()), total_bytes))
    {
        return std::unexpected(Status::E_NOK);
    }

    return meta;
}

template <typename PixelT = uint8_t>
[[nodiscard]] load_result_t<PixelT> image_io_saveRaw(const std::string_view file_name,
                                                    const PixelT* image_buffer
                                                    uint32_t image_width
                                                    uint32_t image_height)
{

    return meta;
}
} // namespace io

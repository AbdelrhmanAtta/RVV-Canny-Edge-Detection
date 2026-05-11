/**
 * @file    double_thresholding.hpp
 * @brief   Double thresholding implementation for categorizing edge pixels.
 */

#pragma once

#include "std_types.hpp"
#include <cstdint>
#include <memory>

namespace processing
{

/** @brief  Categorizes pixels into strong, weak, and suppressed based on two thresholds.
 *
 *  @param  image The input/output image metadata (modified in-place).
 *  @param  low_threshold The lower bound for weak edges.
 *  @param  high_threshold The upper bound for strong edges.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status double_thresholding(image::io::metadata_t<PixelT>& image,
                                         PixelT low_threshold,
                                         PixelT high_threshold)
{
    if(!image.height || !image.width || !image.aligned_buffer_size || !image.buffer)
    {
        return Status::E_INVAL_PTR;
    }

    const uint32_t width = image.width;
    const uint32_t height = image.height;
    const size_t total_pixels = image.pixel_count;
    PixelT* __restrict ptr = image.buffer.get();

    for(size_t i = 0; i < total_pixels; ++i)
    {
        const PixelT val = ptr[i];

        // Strong edges: pixels >= high threshold
        if(val >= high_threshold)
        {
            ptr[i] = static_cast<PixelT>(255);
        }
        // Weak edges: pixels between low and high thresholds
        else if(val >= low_threshold)
        {
            ptr[i] = static_cast<PixelT>(128);
        }
        // Suppressed: pixels below low threshold
        else
        {
            ptr[i] = static_cast<PixelT>(0);
        }
    }

    return Status::E_OK;
}

} // namespace processing

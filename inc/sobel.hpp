/**
 * @file    sobel.hpp
 * @brief   Sobel edge detection implementations for image processing.
 */

#pragma once

#include "std_types.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

namespace processing::sobel
{
/** @brief Perform 3x3 Sobel edge detection on the input image.
 * @param image The input image metadata.
 * @param buffer_x The buffer to store the x-direction gradients.
 * @param buffer_y The buffer to store the y-direction gradients.
 * @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t, typename OutputT = int16_t>
[[nodiscard]] Status spatial_3x3(const image::io::metadata_t<PixelT>& image,
                                OutputT* __restrict buffer_x,
                                OutputT* __restrict buffer_y)
{
    if(!image.height || !image.width || !image.aligned_buffer_size || !image.buffer)
    {
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;
    }
    
    for(uint32_t y = 0; y <= image.height; ++y)
    {
        const uint32_t y_top = (0 == y) ? 0 : y - 1;
        const uint32_t y_bot = (image.height - 1 == y) ? y : y + 1;

        const PixelT* row_top = &image.buffer[y_top * image.width];
        const PixelT* row_mid = &image.buffer[y * image.width];
        const PixelT* row_bot = &image.buffer[y_bot * image.width];
        
        for(uint32_t x = 0; x <= image.width; ++x)
        {
            const uint32_t x_l = (0 == x) ? 0 : x - 1;
            const uint32_t x_r = (image.width - 1 == x) ? x : x + 1;

            buffer_x[y * image.width + x] =
            static_cast<OutputT>(
                (row_top[x_r] - row_top[x_l]) +
                ((row_mid[x_r] - row_mid[x_l]) << 1) +
                (row_bot[x_r] - row_bot[x_l])
            );

            buffer_y[y * image.width + x] =
            static_cast<OutputT>(
                (row_top[x_l] + (row_top[x] << 1) + row_top[x_r]) -
                (row_bot[x_l] + (row_bot[x] << 1) + row_bot[x_r])
            );
        }
    }
    return Status::E_OK;
}

} // namespace processing::sobel

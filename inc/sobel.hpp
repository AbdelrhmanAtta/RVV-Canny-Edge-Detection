/**
 * @file    sobel.hpp
 * @brief   Sobel edge detection implementations for image processing.
 */

#pragma once

#include "std_types.hpp"
#include <algorithm>
#include <vector>
#include <cmath>

namespace processing
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

    const uint32_t width = image.width;
    const uint32_t height = image.height;
    
    const uint32_t pw = width + 2;
    const uint32_t ph = height + 2;

    auto padded_input = std::make_unique<PixelT[]>(pw * ph);
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), 0);

    for (uint32_t y = 0; y < height; ++y) {
        std::copy_n(&image.buffer[y * width], width, &padded_input[(y + 1) * pw + 1]);
    }

    for(uint32_t y = 0; y < height; ++y)
    {
        const PixelT* row_top = &padded_input[y * pw];
        const PixelT* row_mid = &padded_input[(y + 1) * pw];
        const PixelT* row_bot = &padded_input[(y + 2) * pw];
        
        for(uint32_t x = 0; x < width; ++x)
        {

            const uint32_t xl = x;
            const uint32_t xc = x + 1;
            const uint32_t xr = x + 2;

            buffer_x[y * width + x] = static_cast<OutputT>(
                (row_top[xr] - row_top[xl]) +
                ((row_mid[xr] - row_mid[xl]) << 1) +
                (row_bot[xr] - row_bot[xl])
            );

            buffer_y[y * width + x] = static_cast<OutputT>(
                (row_top[xl] + (row_top[xc] << 1) + row_top[xr]) -
                (row_bot[xl] + (row_bot[xc] << 1) + row_bot[xr])
            );
        }
    }
    return Status::E_OK;
}

} // namespace processing

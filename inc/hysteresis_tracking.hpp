/**
 * @file    hysteresis_tracking.hpp
 * @brief   Hysteresis Double Thresholding and Edge Tracking for Canny Edge Detection.
 */

#pragma once

#include "std_types.hpp"
#include <vector>
#include <cstdint>
#include <memory>
#include <cstring>

namespace processing
{

/** @brief  Perform Hysteresis Double Thresholding and Edge Tracking.
 *
 *  @param  image The input image metadata (usually output of Non-Maximum Suppression).
 *  @param  low_threshold The lower bound threshold for weak edges.
 *  @param  high_threshold The upper bound threshold for strong edges.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status hysteresis(const image::io::metadata_t<PixelT>& image, 
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

    // PASS 1: Scalar Double Thresholding
    for (size_t i = 0; i < total_pixels; ++i)
    {
        if (ptr[i] >= high_threshold)
        {
            ptr[i] = 255;
        }
        else if (ptr[i] >= low_threshold)
        {
            ptr[i] = 128;
        }
        else
        {
            ptr[i] = 0;
        }
    }

    // PASS 2: Scalar Edge Tracking (Depth First Search)
    std::vector<uint32_t> stack;
    stack.reserve(total_pixels / 20); 

    for (uint32_t y = 1; y < height - 1; ++y) 
    {
        for (uint32_t x = 1; x < width - 1; ++x) 
        {
            const uint32_t idx = y * width + x;
            if (ptr[idx] == 255) 
            {
                stack.push_back(idx);
            }
        }
    }

    const int dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    while (!stack.empty()) 
    {
        const uint32_t idx = stack.back();
        stack.pop_back();

        const uint32_t x = idx % width;
        const uint32_t y = idx / width;

        for (int i = 0; i < 8; ++i) 
        {
            const uint32_t nx = x + dx[i];
            const uint32_t ny = y + dy[i];

            if (nx < width && ny < height) 
            {
                const uint32_t n_idx = ny * width + nx;
                
                if (ptr[n_idx] == 128) 
                {
                    ptr[n_idx] = 255;
                    stack.push_back(n_idx);
                }
            }
        }
    }

    // PASS 3: Final Suppression
    for (size_t i = 0; i < total_pixels; ++i)
    {
        if (ptr[i] == 128)
        {
            ptr[i] = 0;
        }
    }

    return Status::E_OK;
}

} // namespace processing

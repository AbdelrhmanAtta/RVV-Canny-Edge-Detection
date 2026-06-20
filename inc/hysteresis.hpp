/**
 * @file    hysteresis.hpp
 * @brief   Hysteresis double thresholding and edge tracking for Canny edge detection.
 */

#pragma once

#include "std_types.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace processing::hysteresis
{
/**
 * @brief   Performs hysteresis double thresholding and edge tracking.
 * Classifies pixels as strong, weak, or non-edges based on thresholds,
 * then connects weak edges to strong edges using depth-first search.
 * @tparam  PixelT              The pixel component type (e.g., uint8_t, uint16_t, float).
 * @param   image               The input image metadata (usually output of non-maximum suppression).
 * @param   low_threshold       The lower bound threshold for weak edges.
 * @param   high_threshold      The upper bound threshold for strong edges.
 * @return  Status              E_OK on success, or a Status error code on failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status mag(const image::io::metadata_t<PixelT>& image,
                         PixelT low_threshold,
                         PixelT high_threshold)
{
    if (!image.buffer || !image.height || !image.width || !image.aligned_buffer_size)
        return (!image.buffer) ? Status::E_INVAL_PTR : Status::E_NOK;

    const uint32_t width        = image.width;
    const uint32_t height       = image.height;
    const size_t   total_pixels = image.pixel_count;

    PixelT* __restrict ptr = image.buffer.get();

    // Double Thresholding
    for (size_t i = 0; i < total_pixels; ++i)
    {
        if (ptr[i] >= high_threshold)
            ptr[i] = 255;
        else if (ptr[i] >= low_threshold)
            ptr[i] = 128;
        else
            ptr[i] = 0;
    }

    // Edge Tracking (Depth First Search)
    std::vector<uint32_t> stack;
    stack.reserve(total_pixels / 20);

    for (uint32_t y = 1; y < height - 1; ++y)
    {
        for (uint32_t x = 1; x < width - 1; ++x)
        {
            const uint32_t idx = y * width + x;
            if (ptr[idx] == 255)
                stack.push_back(idx);
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

    // Final Suppression
    for (size_t i = 0; i < total_pixels; ++i)
    {
        if (ptr[i] == 128)
            ptr[i] = 0;
    }

    return Status::E_OK;
}

} // namespace processing::hysteresis

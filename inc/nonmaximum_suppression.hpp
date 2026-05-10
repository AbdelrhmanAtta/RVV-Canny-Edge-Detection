/**
 * @file    nonmaximum_suppression.hpp
 * @brief   Non-maximum suppression implementation for thinning edges.
 */
#pragma once
#include "std_types.hpp"
#include <cstdint>
#include <memory>
#include <cstring>

namespace processing
{
/** @brief  Performs non-maximum suppression to thin edges.
 *
 *  @param  magnitude The input gradient magnitude image.
 *  @param  direction The input gradient direction image (angles 0, 45, 90, 135).
 *  @param  out The output thinned image.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status nms(const image::io::metadata_t<PixelT>& magnitude,
                         const image::io::metadata_t<PixelT>& direction,
                         image::io::metadata_t<PixelT>& out)
{
    if(!magnitude.height || !magnitude.width || !magnitude.aligned_buffer_size || !magnitude.buffer
       || !direction.buffer || !out.buffer)
    {
        return (magnitude.buffer && direction.buffer && out.buffer) ? Status::E_NOK : Status::E_INVAL_PTR;
    }

    if(magnitude.width != direction.width || magnitude.height != direction.height ||
       magnitude.width != out.width || magnitude.height != out.height)
    {
        return Status::E_NOK;
    }

    const uint32_t width = magnitude.width;
    const uint32_t height = magnitude.height;

    const PixelT* __restrict mag_ptr = magnitude.buffer.get();
    const PixelT* __restrict dir_ptr = direction.buffer.get();
    PixelT* __restrict out_ptr = out.buffer.get();

    std::memset(out_ptr, 0, out.aligned_buffer_size * sizeof(PixelT));

    for(uint32_t y = 1; y < height - 1; ++y)
    {
        for(uint32_t x = 1; x < width - 1; ++x)
        {
            const uint32_t idx = y * width + x;
            const PixelT mag = mag_ptr[idx];
            const PixelT angle = dir_ptr[idx];

            PixelT q = 255;
            PixelT r = 255;

            // Angle 0: Horizontal - compare left and right neighbors
            if(angle == 0)
            {
                q = mag_ptr[idx + 1];
                r = mag_ptr[idx - 1];
            }
            // Angle 45: Diagonal / - compare top-right and bottom-left neighbors
            else if(angle == 45)
            {
                q = mag_ptr[(y - 1) * width + (x + 1)];
                r = mag_ptr[(y + 1) * width + (x - 1)];
            }
            // Angle 90: Vertical - compare top and bottom neighbors
            else if(angle == 90)
            {
                q = mag_ptr[(y - 1) * width + x];
                r = mag_ptr[(y + 1) * width + x];
            }
            // Angle 135: Diagonal \ - compare top-left and bottom-right neighbors
            else if(angle == 135)
            {
                q = mag_ptr[(y - 1) * width + (x - 1)];
                r = mag_ptr[(y + 1) * width + (x + 1)];
            }

            out_ptr[idx] = (mag >= q && mag >= r) ? mag : static_cast<PixelT>(0);
        }
    }
    return Status::E_OK;
}
} // namespace processing

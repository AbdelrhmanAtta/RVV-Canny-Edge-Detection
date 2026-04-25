/**
 * @file    math.hpp
 * @brief   Mathematical utilities for image processing operations.
 */

#pragma once

#include "std_types.hpp"
#include <algorithm>
#include <expected>

namespace processing::convolution
{
/** @brief Result type alias for image convolution operations. */
template <typename PixelT = uint8_t>
using convolution_t = std::expected<PixelT*, Status>;

/**
 * @brief   Applies a spatial convolution to an image using the provided kernel.
 * The function supports edge clamping to handle borders and optional output clamping.
 * @tparam  PixelT          The pixel component type (e.g., uint8_t, uint16_t, float).
 * @tparam  KernelT         The kernel component type (e.g., uint8_t, uint16_t).
 * @tparam  AccumulatorT    The type used for accumulating convolution sums to prevent overflow (e.g., int32_t, uint32_t).
 * @tparam  ClampT          Whether to apply clamping to the output values.
 * @param   image           The input image metadata and buffer.
 * @param   image_output    Pointer to the output buffer where the convolved image will be stored.
 * @param   kernel          The convolution kernel structure (contains data, dimensions, and sum).
 * @param   clamp           The clamping parameters (min and max values) to apply if ClampT is true.
 * @return  convolution_t   A pointer to the output image buffer on success, or a Status error code on failure.
 * @note    The output buffer must be pre-allocated and large enough to hold the convolved image data.
 * @note    If clamping is required, ClampT must be set to true.
 */
template <typename PixelT = uint8_t, typename KernelT = uint8_t, typename AccumulatorT = int32_t, bool ClampT = false>
[[nodiscard]] convolution_t<PixelT> spatial(const image::io::metadata_t<PixelT>& image,
                                            PixelT* image_output,
                                            const kernel_t<KernelT, AccumulatorT>& kernel,
                                            const clamp_t<PixelT>& clamp)
{
    if(0==kernel.kernel_height || 0==kernel.kernel_width ||
        !image.image_height || !image.image_width)
    {
        return std::unexpected(Status::E_NOK);
    }
    if (!kernel.kernel_data || !image.image_buffer)
    {
        return std::unexpected(Status::E_INVAL_PTR); 
    }

    uint8_t kernel_horizontal_radius = (kernel.kernel_width-1)/2;
    uint8_t kernel_vertical_radius = (kernel.kernel_height-1)/2;

    for(uint32_t y=0; y<image.image_height; ++y)
    {
        for(uint32_t x=0; x<image.image_width; ++x)
        {
            AccumulatorT convolution_sum=0;

            for(int32_t ky=-kernel_vertical_radius; ky<=kernel_vertical_radius; ++ky)
            {
                /** @brief Edge Clamping */
                int32_t ny = std::max(0, std::min(ky+static_cast<int32_t>(y), static_cast<int32_t>(image.image_height)-1));
                uint32_t row_offset = ny * image.image_width; 
                int32_t k_row_offset = (ky + kernel_vertical_radius) * kernel.kernel_width;
                
                for(int32_t kx=-kernel_horizontal_radius; kx<=kernel_horizontal_radius; ++kx)
                {
                    int32_t nx = std::max(0, std::min(kx+static_cast<int32_t>(x), static_cast<int32_t>(image.image_width)-1));
                    convolution_sum += static_cast<AccumulatorT>(image.image_buffer[row_offset+nx])
                                        *kernel.kernel_data[k_row_offset+(kx+kernel_horizontal_radius)];
                }
            }
            if(kernel.kernel_sum)
            {
                convolution_sum/=kernel.kernel_sum;
            }
    
            if constexpr (ClampT)
            {
                convolution_sum = std::max(static_cast<AccumulatorT>(clamp.clamp_min), 
                                            std::min(convolution_sum, static_cast<AccumulatorT>(clamp.clamp_max)));
            }
    
            image_output[y*image.image_width+x] = static_cast<PixelT>(convolution_sum);
        }
    }
    return image_output;
}

} // namespace processing::convolution

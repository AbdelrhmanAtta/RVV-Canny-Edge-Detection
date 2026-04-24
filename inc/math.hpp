/**
 * @file    math.hpp
 * @brief   Mathematical utilities for image processing operations.
 */

#pragma once

#include "std_types.hpp"
#include <algorithm>
#include <expected>
#include <optional>

namespace processing::convolution
{
/** @brief Result type alias for image processing operations. */
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
 * @param   kernel_data     Pointer to the convolution kernel data (flattened 2D array).
 * @param   kernel_width    Width of the convolution kernel.
 * @param   kernel_height   Height of the convolution kernel.
 * @param   kernel_sum      The sum of the kernel elements for normalization (if zero, no normalization is applied).
 * @param   clamp           Clamping parameters for output values (used if ClampT is true).
 * @return  convolution_t   A pointer to the output image buffer on success, or a Status error code on failure.
 */
template <typename PixelT = uint8_t, typename KernelT = uint8_t, typename AccumulatorT = int32_t, bool ClampT = false>
[[nodiscard]] convolution_t<PixelT> spatial(const metadata_t<PixelT>& image,
                                            PixelT* image_output,
                                            const KernelT* kernel_data,
                                            uint8_t kernel_width,
                                            uint8_t kernel_height,
                                            AccumulatorT kernel_sum,
                                            const clamp_t<PixelT>& clamp)
{
    if(kernel_height<=1 || kernel_width<=1 ||
        !image.image_height || !image.image_width)
    {
        return std::unexpected(Status::E_NOK);
    }
    if (!kernel_data || !image.image_buffer)
    {
        return std::unexpected(Status::E_INVAL_PTR); 
    }

    uint8_t kernel_horizontal_radius = (kernel_width-1)/2;
    uint8_t kernel_vertical_radius = (kernel_height-1)/2;

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
                int32_t k_row_offset = (ky + kernel_vertical_radius) * kernel_width;
                
                for(int32_t kx=-kernel_horizontal_radius; kx<=kernel_horizontal_radius; ++kx)
                {
                    int32_t nx = std::max(0, std::min(kx+static_cast<int32_t>(x), static_cast<int32_t>(image.image_width)-1));
                    convolution_sum += static_cast<AccumulatorT>(image.image_buffer[row_offset+nx])
                                        *kernel_data[k_row_offset+(kx+kernel_horizontal_radius)];
                }
            }
            if(kernel_sum)
            {
                convolution_sum/=kernel_sum;
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

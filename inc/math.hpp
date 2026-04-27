/**
 * @file    math.hpp
 * @brief   Mathematical utilities for image processing operations.
 */

#pragma once

#include "std_types.hpp"
#include <algorithm>
#include <vector>

namespace processing::convolution
{
/**
 * @brief   Applies a spatial convolution to an image using the provided kernel.
 * The function supports edge clamping to handle borders and optional output clamping.
 * @tparam  ClampT          Whether to apply clamping to the output values.
 * @tparam  PixelT          The pixel component type (e.g., uint8_t, uint16_t, float).
 * @tparam  KernelT         The kernel component type (e.g., uint8_t, uint16_t).
 * @tparam  AccumulatorT    The type used for accumulating convolution sums to prevent overflow (e.g., int32_t, uint32_t).
 * @param   image           The input image metadata and buffer.
 * @param   image_output    Pointer to the output buffer where the convolved image will be stored.
 * @param   kernel          The convolution kernel structure (contains data, dimensions, and sum).
 * @param   clamp           The clamping parameters (min and max values) to apply if ClampT is true.
 * @return  Status          Status error code on failure.
 * @note    The output buffer must be pre-allocated and large enough to hold the convolved image data.
 * @note    If clamping is required, ClampT must be set to true.
 */
template <bool ClampT = false, typename PixelT = uint8_t, typename KernelT = uint8_t, typename AccumulatorT = int32_t>
[[nodiscard]] Status spatial(const image::io::metadata_t<PixelT>& image,
                            PixelT* image_output,
                            const kernel_t<KernelT, AccumulatorT>& kernel,
                            const clamp_t<PixelT>& clamp)
{
    if(!kernel.height || !kernel.width || !kernel.data
        || !image.height || !image.width || !image.aligned_buffer_size)
    {
        return Status::E_NOK;
    }
    if (!image.image_buffer)
    {
        return Status::E_INVAL_PTR; 
    }
 
    const uint8_t kernel_horizontal_radius = (kernel.width-1)/2;
    const uint8_t kernel_vertical_radius = (kernel.height-1)/2;
    const bool normalize = kernel.sum != 0;
 
    for(uint32_t y=0; y<image.height; ++y)
    {
        for(uint32_t x=0; x<image.width; ++x)
        {
            AccumulatorT convolution_sum=0;
 
            for(int32_t ky=-kernel_vertical_radius; ky<=kernel_vertical_radius; ++ky)
            {
                const int32_t ny = std::max(0, std::min(ky+static_cast<int32_t>(y), static_cast<int32_t>(image.height)-1));
                const uint32_t row_offset = ny * image.width; 
                const int32_t k_row_offset = (ky + kernel_vertical_radius) * kernel.width;
                
                for(int32_t kx=-kernel_horizontal_radius; kx<=kernel_horizontal_radius; ++kx)
                {
                    const int32_t nx = std::max(0, std::min(kx+static_cast<int32_t>(x), static_cast<int32_t>(image.width)-1));
                    convolution_sum += static_cast<AccumulatorT>(image.image_buffer[row_offset+nx])
                                        *kernel.data[k_row_offset+(kx+kernel_horizontal_radius)];
                }
            }
            if(normalize)
            {
                convolution_sum/=kernel.sum;
            }
    
            if constexpr(ClampT)
            {
                convolution_sum = std::max(static_cast<AccumulatorT>(clamp.clamp_min), 
                                            std::min(convolution_sum, static_cast<AccumulatorT>(clamp.clamp_max)));
            }
    
            image_output[y*image.width+x] = static_cast<PixelT>(convolution_sum);
        }
    }
    return Status::E_OK;
}
 
/**
 * @brief   Applies a separable convolution to an image using the provided kernel.
 * The function supports edge clamping to handle borders and optional output clamping.
 * @tparam  ClampT          Whether to apply clamping to the output values.
 * @tparam  PixelT          The pixel component type (e.g., uint8_t, uint16_t, float).
 * @tparam  KernelT         The kernel component type (e.g., uint8_t, uint16_t).
 * @tparam  AccumulatorT    The type used for accumulating convolution sums to prevent overflow (e.g., int32_t, uint32_t).
 * @param   image           The input image metadata and buffer.
 * @param   image_output    Pointer to the output buffer where the convolved image will be stored.
 * @param   kernel          The convolution kernel structure (contains data, dimensions, and sum).
 * @param   clamp           The clamping parameters (min and max values) to apply if ClampT is true.
 * @return  Status          Status error code on failure.
 * @note    The output buffer must be pre-allocated and large enough to hold the convolved image data.
 * @note    If clamping is required, ClampT must be set to true.
 */
template <bool ClampT = false, typename PixelT = uint8_t, typename KernelT = uint8_t, typename AccumulatorT = int32_t>
[[nodiscard]] Status separable(const image::io::metadata_t<PixelT>& image,
                                PixelT* image_output,
                                const kernel_t<KernelT, AccumulatorT>& kernel,
                                const clamp_t<PixelT>& clamp)
{
    if(!kernel.height || !kernel.width || !kernel.data
        || !image.height || !image.width || !image.aligned_buffer_size)
    {
        return Status::E_NOK;
    }
    if (!image.image_buffer)
    {
        return Status::E_INVAL_PTR; 
    }
 
    std::vector<AccumulatorT> output_buffer(image.width*image.height, 0);
    const uint8_t kernel_radius = std::max((kernel.width-1)/2, (kernel.height-1)/2);
    const bool normalize = kernel.sum != 0;
 
    for(uint32_t y=0; y<image.height; ++y)
    {
        const uint32_t offset = y * image.width;
        for(uint32_t x=0; x<image.width; ++x)
        {
            AccumulatorT convolution_sum=0;
            for(int32_t kx=-kernel_radius; kx<=kernel_radius; ++kx)
            {
                const int32_t nx = std::max(0, std::min(kx+static_cast<int32_t>(x), static_cast<int32_t>(image.width)-1));
                convolution_sum += static_cast<AccumulatorT>(image.image_buffer[offset+nx])
                                                            *kernel.data[kx+kernel_radius];
            }
            output_buffer[y*image.width+x] = convolution_sum;
        }
    }
 
    for(uint32_t y=0; y<image.height; ++y)
    {
        const uint32_t offset = y * image.width;
        for(uint32_t x=0; x<image.width; ++x)
        {
            AccumulatorT convolution_sum=0;
            for(int32_t ky=-kernel_radius; ky<=kernel_radius; ++ky)
            {
                const int32_t ny = std::max(0, std::min(ky+static_cast<int32_t>(y), static_cast<int32_t>(image.height)-1));
                convolution_sum += static_cast<AccumulatorT>(output_buffer[ny*image.width+x])
                                                            *kernel.data[ky+kernel_radius];
            }
            if(normalize)
            {
                convolution_sum/=(kernel.sum*kernel.sum);
            }
            if constexpr(ClampT)
            {
                convolution_sum = std::max(static_cast<AccumulatorT>(clamp.clamp_min), 
                                            std::min(convolution_sum, static_cast<AccumulatorT>(clamp.clamp_max)));
            }
            image_output[offset+x] = static_cast<PixelT>(convolution_sum);
        }
    }
    return Status::E_OK;
}
 
} // namespace processing::convolution

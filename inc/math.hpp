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
                            const clamp_t<PixelT>& clamp={})
{
    if(!kernel.height || !kernel.width || !kernel.data
        || !image.height || !image.width || !image.aligned_buffer_size)
    {
        return Status::E_NOK;
    }
    if (!image.buffer)
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
                    convolution_sum += static_cast<AccumulatorT>(image.buffer[row_offset+nx])
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
                                const clamp_t<PixelT>& clamp={})
{
    if(!kernel.height || !kernel.width || !kernel.data
        || !image.height || !image.width || !image.aligned_buffer_size)
    {
        return Status::E_NOK;
    }
    if (!image.buffer)
    {
        return Status::E_INVAL_PTR; 
    }
 
    std::vector<AccumulatorT> output_buffer(image.width*image.height);
    const uint8_t kernel_radius = std::max((kernel.width-1)/2, (kernel.height-1)/2);
    const bool normalize = kernel.sum != 0;
 
    for(uint32_t y=0; y<image.height; ++y)
    {
        const uint32_t offset = y*image.width;
        for(uint32_t x=0; x<image.width; ++x)
        {
            AccumulatorT convolution_sum=0;
            for(int32_t kx=-kernel_radius; kx<=kernel_radius; ++kx)
            {
                const int32_t nx = std::max(0, std::min(kx+static_cast<int32_t>(x), static_cast<int32_t>(image.width)-1));
                convolution_sum += static_cast<AccumulatorT>(image.buffer[offset+nx])
                                                            *kernel.data[kx+kernel_radius];
            }
            output_buffer[y*image.width+x] = convolution_sum;
        }
    }
 
    for(uint32_t y=0; y<image.height; ++y)
    {
        const uint32_t offset = y*image.width;
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
 
/**
 * @brief   3x3 Gaussian blur kernel and its 1D separable counterpart.
 * The 3x3 kernel is normalized by a factor of 16 and sigma approximately 0.85.
 */
inline constexpr uint8_t GAUSSIAN_3x3_DATA[] =
{
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
};
inline constexpr uint8_t GAUSSIAN_3x3_1D_DATA[3] = {1, 2, 1};
inline constexpr kernel_t<uint8_t, uint16_t> GAUSSIAN_3x3 = 
{
    .data = GAUSSIAN_3x3_DATA,
    .width = 3,
    .height = 3,
    .sum = 16
};
inline constexpr kernel_t<uint8_t, uint16_t> GAUSSIAN_3x3_1D = 
{
    .data = GAUSSIAN_3x3_1D_DATA,
    .width = 3,
    .height = 1,
    .sum = 4
};

/**
 * @brief   5x5 Gaussian blur kernel and its 1D separable counterpart.
 * The 5x5 kernel is normalized by a factor of 273 and sigma approximately 1.
 */
inline constexpr uint8_t GAUSSIAN_5x5_DATA[] = 
{
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};
inline constexpr uint8_t GAUSSIAN_5x5_1D_DATA[] = {1, 4, 7, 4, 1};
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_5x5 = 
{
    .data = GAUSSIAN_5x5_DATA,
    .width = 5,
    .height = 5,
    .sum = 273
};  
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_5x5_1D = 
{
    .data = GAUSSIAN_5x5_1D_DATA,
    .width = 5,
    .height = 1,
    .sum = 17
};

/**
 * @brief   7x7 Gaussian blur kernel and its 1D separable counterpart.
 * The 7x7 kernel is normalized by a factor of 1111 and sigma approximately 1.4.
 */
inline constexpr uint8_t GAUSSIAN_7x7_DATA[] = 
{
    0,  0,  1,   2,  1,  0, 0,
    0,  3, 13,  22, 13,  3, 0,
    1, 13, 59,  97, 59, 13, 1,
    2, 22, 97, 159, 97, 22, 2,
    1, 13, 59,  97, 59, 13, 1,
    0,  3, 13,  22, 13,  3, 0,
    0,  0,  1,   2,  1,  0, 0
};
inline constexpr uint8_t GAUSSIAN_7x7_1D_DATA[] = {1, 3, 7, 11, 7, 3, 1};
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_7x7 = 
{
    .data = GAUSSIAN_7x7_DATA,
    .width = 7,
    .height = 7,
    .sum = 1111
};
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_7x7_1D = 
{
    .data = GAUSSIAN_7x7_1D_DATA,
    .width = 7,
    .height = 1,
    .sum = 33
};

/**
 * @brief   3x3 Sobel Filter kernel and its 1D separable counterpart.
 * The Sobel filter is used for edge detection, with the horizontal kernel detecting vertical edges and the vertical kernel detecting horizontal edges.
 * The kernels are not normalized, and the sum is set to 0 to indicate that no normalization should be applied.
 */
inline constexpr int8_t SOBEL_HORIZONTAL_3x3_DATA[] = 
{
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};
inline constexpr kernel_t<int8_t, uint16_t> SOBEL_HORIZONTAL_3x3 =
{
    .data = SOBEL_HORIZONTAL_3x3_DATA,
    .width = 3,
    .height = 3,
    .sum = 0
};
inline constexpr int8_t SOBEL_VERTICAL_3x3_DATA[] = 
{
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
};
inline constexpr kernel_t<int8_t, uint16_t> SOBEL_VERTICAL_3x3 =
{
    .data = SOBEL_VERTICAL_3x3_DATA,
    .width = 3,
    .height = 3,
    .sum = 0
};

/**
 * @brief   Clamping parameters for 8-bit grayscale images, with a minimum of 0 and a maximum of 255.
 * This can be used as the clamp argument for convolution functions when processing grayscale images.
 */
inline constexpr clamp_t<uint8_t> GREYSCALE_CLAMP = 
{
    .clamp_min = 0,
    .clamp_max = 255
};

} // namespace processing::convolution

/**
 * @file    filters.hpp
 * @brief   Filter implementations for image processing operations.
 */

#pragma once

#include "std_types.hpp"
#include "math.hpp"
#include <algorithm>

namespace processing::filters
{
/**
 * @brief   Applies a Gaussian blur to an image using either a spatial or separable convolution approach based on the kernel dimensions.
 * @tparam  ClampT          Whether to apply clamping to the output values.
 * @tparam  PixelT          The pixel component type (e.g., uint8_t, uint16_t, float).
 * @tparam  KernelT         The kernel component type (e.g., uint8_t, uint16_t).
 * @tparam  AccumulatorT    The type used for accumulating convolution sums to prevent overflow (e.g., int32_t, uint32_t).
 * @param   image           The input image metadata and buffer.
 * @param   image_output    Pointer to the output buffer where the blurred image will be stored.
 * @param   kernel          The Gaussian blur kernel structure (contains data, dimensions, and sum).
 * @param   clamp           The clamping parameters (min and max values) to apply if ClampT is true.
 * @return  Status          Status error code on failure.
 * @note    The output buffer must be pre-allocated and large enough to hold the blurred image data.
 * @note    If clamping is required, ClampT must be set to true
 */
template <bool ClampT = false, typename PixelT = uint8_t, typename KernelT = uint8_t, typename AccumulatorT = int32_t>
[[nodiscard]] Status gaussian_blur(const image::io::metadata_t<PixelT>& image,
                                PixelT* image_output,
                                const processing::convolution::kernel_t<KernelT, AccumulatorT>& kernel,
                                const processing::convolution::clamp_t<PixelT>& clamp={})
{
    if(std::min(kernel.width, kernel.height)==1)
    {
        return separable<ClampT, PixelT, KernelT, AccumulatorT>(image, image_output, kernel, clamp);
    }
    else
    {
        return spatial<ClampT, PixelT, KernelT, AccumulatorT>(image, image_output, kernel, clamp);
    }
    return Status::E_OK;
}

template <bool ClampT = false, typename PixelT = uint8_t, typename KernelT = int8_t, typename AccumulatorT = int16_t>
[[nodiscard]] Status sobel_filter(const image::io::metadata_t<PixelT>& image,
                                AccumulatorT* Gx_output,
                                AccumulatorT* Gy_output,
                                const processing::convolution::kernel_t<KernelT, AccumulatorT>& kernel_x,
                                const processing::convolution::kernel_t<KernelT, AccumulatorT>& kernel_y,
                                const processing::convolution::clamp_t<PixelT>& clamp={})
{
    Status statusX = processing::convolution::spatial<ClampT, AccumulatorT, KernelT, AccumulatorT>(image, Gx_output, processing::convolution::SOBEL_HORIZONTAL_3x3);
    Status statusY = processing::convolution::spatial<ClampT, AccumulatorT, KernelT, AccumulatorT>(image, Gy_output, processing::convolution::SOBEL_VERTICAL_3x3);
    if(Status::E_OK!=statusX || Status::E_OK!=statusY)
    {
        return Status::E_NOK;
    }
    else return Status::E_OK;
}

} // namespace processing::filters

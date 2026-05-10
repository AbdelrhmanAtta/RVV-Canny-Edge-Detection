/**
 * @file    gaussian.hpp
 * @brief   Gaussian blur implementations for image processing.
 */

#pragma once

#include "std_types.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>

namespace processing
{
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

/**
 * @brief   Applies a 5x5 Gaussian blur to the input image using direct convolution.
 * @param   image The input image metadata, which will be modified in-place with the blurred output.
 * @return  Status indicating success or failure of the operation.
 */
template <typename PixelT = uint8_t, typename AccumulatorT = int32_t>
[[nodiscard]] Status spatial_5x5(image::io::metadata_t<PixelT>& image)
{
    if (!image.height || !image.width || !image.buffer)
    {
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;
    }

    const int32_t width = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);
    const int32_t kernel_radius = 2;
    
    const uint32_t pw = width + 2 * kernel_radius;
    const uint32_t ph = height + 2 * kernel_radius;
    auto padded_input = std::make_unique<PixelT[]>(pw * ph); 
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), 0);

    for (int32_t y = 0; y < height; ++y) {
        std::copy_n(&image.buffer[y * width], width, &padded_input[(y + kernel_radius) * pw + kernel_radius]);
    }

    auto raw_out = static_cast<PixelT*>(utils::memory::aligned_alloc(64, image.aligned_buffer_size));
    if (!raw_out)
    {
        return Status::E_ALLOC_FAIL;
    }

    std::unique_ptr<PixelT[], utils::memory::deleter> output_buffer(raw_out);

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            AccumulatorT sum = 0;
            
            for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky) {
                const uint32_t row_offset = (y + kernel_radius + ky) * pw;
                const uint32_t kernel_offset = (ky + kernel_radius) * 5;
                
                for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                    sum += static_cast<AccumulatorT>(padded_input[row_offset + (x + kernel_radius + kx)]) * 
                           static_cast<AccumulatorT>(GAUSSIAN_5x5_DATA[kernel_offset + (kx + kernel_radius)]);
                }
            }

            sum = static_cast<AccumulatorT>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            sum = (sum < 0) ? 0 : (sum > 255) ? 255 : sum;
            
            output_buffer[y * width + x] = static_cast<PixelT>(sum);
        }
    }

    image.buffer = std::move(output_buffer);
    return Status::E_OK;
}

/**
 * @brief   Applies a 5x5 Gaussian blur to the input image using separable convolution.
 * @param   image The input image metadata, which will be modified in-place with the blurred output.
 * @return  Status indicating success or failure of the operation.
 */
template <typename PixelT = uint8_t, typename AccumulatorT = int32_t>
[[nodiscard]] Status separable_5x5(image::io::metadata_t<PixelT>& image)
{
    if (!image.height || !image.width || !image.buffer)
    {
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;
    }

    const int32_t width = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);
    const int32_t kernel_radius = 2;

    const uint32_t pw = width + 2 * kernel_radius;
    const uint32_t ph = height + 2 * kernel_radius;
    
    auto padded_input = std::make_unique<PixelT[]>(pw * ph);
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), 0);

    for (int32_t y = 0; y < height; ++y) {
        std::copy_n(&image.buffer[y * width], width, &padded_input[(y + kernel_radius) * pw + kernel_radius]);
    }

    auto inter_buffer = std::make_unique<AccumulatorT[]>(pw * ph);

    for (uint32_t y = 0; y < ph; ++y) {
        for (uint32_t x = kernel_radius; x < pw - kernel_radius; ++x) {
            AccumulatorT sum = 0;
            const uint32_t row_offset = y * pw;
            for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                sum += static_cast<AccumulatorT>(padded_input[row_offset + (x + kx)]) * 
                       static_cast<AccumulatorT>(GAUSSIAN_5x5_1D_DATA[kx + kernel_radius]);
            }
            inter_buffer[row_offset + x] = sum;
        }
    }

    auto raw_out = static_cast<PixelT*>(utils::memory::aligned_alloc(64, image.aligned_buffer_size));
    if (!raw_out) return Status::E_ALLOC_FAIL;
    std::unique_ptr<PixelT[], utils::memory::deleter> output_buffer(raw_out);

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            AccumulatorT sum = 0;
            const uint32_t col_x = x + kernel_radius;
            for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky) {
                sum += inter_buffer[(y + kernel_radius + ky) * pw + col_x] * 
                       static_cast<AccumulatorT>(GAUSSIAN_5x5_1D_DATA[ky + kernel_radius]);
            }

            sum = static_cast<AccumulatorT>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            sum = (sum < 0) ? 0 : (sum > 255) ? 255 : sum;

            output_buffer[y * width + x] = static_cast<PixelT>(sum);
        }
    }

    image.buffer = std::move(output_buffer);
    return Status::E_OK;
}

} // namespace processing

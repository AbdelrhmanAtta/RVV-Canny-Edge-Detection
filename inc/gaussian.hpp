/**
 * @file    gaussian.hpp
 * @brief   Gaussian blur implementations for image processing.
 */

#pragma once

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 12000
#include <riscv_vector.h>
#endif

#include "std_types.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>
#include <type_traits>


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
template <typename PixelT = uint8_t, typename AccumulatorT = uint32_t>
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

    for (int32_t y = 0; y < height; ++y) 
    {
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

    for (int32_t y = 0; y < height; ++y) 
    {
        for (int32_t x = 0; x < width; ) 
        {
            #if defined(__riscv_v) && __riscv_v >= 1000000
            size_t vl;
            vuint32m4_t sum;
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                vl = __riscv_vsetvl_e8m1(width - x);
                sum = __riscv_vmv_v_x_u32m4(0, vl);
            }
            #else
            AccumulatorT sum = 0;    
            #endif

            for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky) 
            {
                const uint32_t row_offset = (y + kernel_radius + ky) * pw;
                const uint32_t kernel_offset = (ky + kernel_radius) * 5;
                
                for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx) 
                {
                    #if defined(__riscv_v) && __riscv_v >= 1000000
                    if constexpr (std::is_same_v<uint8_t, PixelT>)
                    {
                        vuint8m1_t pixels = __riscv_vle8_v_u8m1(&padded_input[row_offset + (x + kernel_radius + kx)], vl);
                        vuint16m2_t pixels_extended = __riscv_vzext_vf2_u16m2(pixels, vl);
                        sum = __riscv_vwmaccu_vx_u32m4(sum, GAUSSIAN_5x5_DATA[kernel_offset + (kx + kernel_radius)], 
                                                        pixels_extended, vl);
                    }
                    #else
                    sum += static_cast<AccumulatorT>(padded_input[row_offset + (x + kernel_radius + kx)]) * static_cast<AccumulatorT>(GAUSSIAN_5x5_DATA[kernel_offset + (kx + kernel_radius)]);
                    #endif
                }
            }

            #if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                sum = __riscv_vmul_vx_u32m4(sum, 240, vl);
                sum = __riscv_vsrl_vx_u32m4(sum, 16, vl);
                
                vuint16m2_t narrow16 = __riscv_vncvt_x_x_w_u16m2(sum, vl);
                vuint8m1_t final_pixels = __riscv_vncvt_x_x_w_u8m1(narrow16, vl);

                __riscv_vse8_v_u8m1(&output_buffer[y * width + x], final_pixels, vl);
                x += vl;
            }
            #else
            sum = static_cast<AccumulatorT>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            sum = (sum < 0) ? 0 : (sum > 255) ? 255 : sum;
            
            output_buffer[y * width + x] = static_cast<PixelT>(sum);
            x += 1;
            #endif
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
        for (uint32_t x = kernel_radius; x < pw - kernel_radius; ) {
            #if defined(__riscv_v) && __riscv_v >= 1000000
            size_t vl;
            vuint32m4_t sum;
            if constexpr (std::is_same_v<uint8_t, PixelT>) {
                vl = __riscv_vsetvl_e8m1(pw - kernel_radius - x);
                sum = __riscv_vmv_v_x_u32m4(0, vl);
            }
            #else
            AccumulatorT sum = 0;
            #endif

            const uint32_t row_offset = y * pw;
            for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                #if defined(__riscv_v) && __riscv_v >= 1000000
                if constexpr (std::is_same_v<uint8_t, PixelT>) {
                    vuint8m1_t pixels = __riscv_vle8_v_u8m1(&padded_input[row_offset + (x + kx)], vl);
                    vuint16m2_t pixels16 = __riscv_vzext_vf2_u16m2(pixels, vl);
                    sum = __riscv_vwmaccu_vx_u32m4(sum, GAUSSIAN_5x5_1D_DATA[kx + kernel_radius], pixels16, vl);
                }
                #else
                sum += static_cast<AccumulatorT>(padded_input[row_offset + (x + kx)]) * static_cast<AccumulatorT>(GAUSSIAN_5x5_1D_DATA[kx + kernel_radius]);
                #endif
            }

            #if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>) {
                __riscv_vse32_v_u32m4(reinterpret_cast<uint32_t*>(&inter_buffer[row_offset + x]), sum, vl);
                x += vl;
            }
            #else
            inter_buffer[row_offset + x] = sum;
            x += 1;
            #endif
        }
    }

    auto raw_out = static_cast<PixelT*>(utils::memory::aligned_alloc(64, image.aligned_buffer_size));
    if (!raw_out) return Status::E_ALLOC_FAIL;
    std::unique_ptr<PixelT[], utils::memory::deleter> output_buffer(raw_out);

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ) {
            #if defined(__riscv_v) && __riscv_v >= 1000000
            size_t vl;
            vuint32m4_t sum;
            if constexpr (std::is_same_v<uint8_t, PixelT>) {
                vl = __riscv_vsetvl_e32m4(width - x);
                sum = __riscv_vmv_v_x_u32m4(0, vl);
            }
            #else
            AccumulatorT sum = 0;
            #endif

            const uint32_t col_x = x + kernel_radius;
            for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky) {
                #if defined(__riscv_v) && __riscv_v >= 1000000
                if constexpr (std::is_same_v<uint8_t, PixelT>) {
                    vuint32m4_t inter_vals = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t*>(&inter_buffer[(y + kernel_radius + ky) * pw + col_x]), vl);
                    sum = __riscv_vmacc_vx_u32m4(sum, GAUSSIAN_5x5_1D_DATA[ky + kernel_radius], inter_vals, vl);
                }
                #else
                sum += inter_buffer[(y + kernel_radius + ky) * pw + col_x] * static_cast<AccumulatorT>(GAUSSIAN_5x5_1D_DATA[ky + kernel_radius]);
                #endif
            }

            #if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>) {
                sum = __riscv_vmul_vx_u32m4(sum, 240, vl);
                sum = __riscv_vsrl_vx_u32m4(sum, 16, vl);
                vuint16m2_t narrow16 = __riscv_vncvt_x_x_w_u16m2(sum, vl);
                vuint8m1_t final_pixels = __riscv_vncvt_x_x_w_u8m1(narrow16, vl);
                __riscv_vse8_v_u8m1(&output_buffer[y * width + x], final_pixels, vl);
                x += vl;
            }
            #else
            sum = static_cast<AccumulatorT>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            sum = (sum < 0) ? 0 : (sum > 255) ? 255 : sum;
            output_buffer[y * width + x] = static_cast<PixelT>(sum);
            x += 1;
            #endif
        }
    }

    image.buffer = std::move(output_buffer);
    return Status::E_OK;
}

} // namespace processing
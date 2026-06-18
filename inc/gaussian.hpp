/**
 * @file    gaussian.hpp
 * @brief   Gaussian blur implementations for image processing.
 */

#pragma once

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 11000
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


#if defined(__riscv_v) && __riscv_v >= 1000000

template <int LMUL> struct rvv_traits;

template <> struct rvv_traits<1>
{
    using u8  = vuint8m1_t;
    using u16 = vuint16m2_t;
    using u32 = vuint32m4_t;

    static size_t setvl_e8 (size_t n) { return __riscv_vsetvl_e8m1(n);  }
    static size_t setvl_e32(size_t n) { return __riscv_vsetvl_e32m4(n); }

    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m1(p, vl);       }
    static u32 vmv0 (size_t vl)                    { return __riscv_vmv_v_x_u32m4(0, vl);      }
    static u16 vzext(u8 v, size_t vl)              { return __riscv_vzext_vf2_u16m2(v, vl);    }

    static u32 vwmaccu(u32 acc, uint8_t s, u16 v, size_t vl)
    { return __riscv_vwmaccu_vx_u32m4(acc, s, v, vl); }

    static u32 vmacc(u32 acc, uint8_t s, u32 v, size_t vl)
    { return __riscv_vmacc_vx_u32m4(acc, s, v, vl); }

    static u32 vmul(u32 v, uint32_t s, size_t vl) { return __riscv_vmul_vx_u32m4(v, s, vl);  }
    static u32 vsrl(u32 v, uint32_t s, size_t vl) { return __riscv_vsrl_vx_u32m4(v, s, vl);  }

    static u8 narrow_u32_to_u8(u32 v, size_t vl)
    { return __riscv_vncvt_x_x_w_u8m1(__riscv_vncvt_x_x_w_u16m2(v, vl), vl); }

    static void vse8 (uint8_t*  p, u8  v, size_t vl) { __riscv_vse8_v_u8m1(p, v, vl);   }
    static void vse32(uint32_t* p, u32 v, size_t vl) { __riscv_vse32_v_u32m4(p, v, vl); }

    static u32 vle32(const uint32_t* p, size_t vl) { return __riscv_vle32_v_u32m4(p, vl); }
};

template <> struct rvv_traits<2>
{
    using u8  = vuint8m2_t;
    using u16 = vuint16m4_t;
    using u32 = vuint32m8_t;

    static size_t setvl_e8 (size_t n) { return __riscv_vsetvl_e8m2(n);  }
    static size_t setvl_e32(size_t n) { return __riscv_vsetvl_e32m8(n); }

    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m2(p, vl);       }
    static u32 vmv0 (size_t vl)                    { return __riscv_vmv_v_x_u32m8(0, vl);      }
    static u16 vzext(u8 v, size_t vl)              { return __riscv_vzext_vf2_u16m4(v, vl);    }

    static u32 vwmaccu(u32 acc, uint8_t s, u16 v, size_t vl)
    { return __riscv_vwmaccu_vx_u32m8(acc, s, v, vl); }

    static u32 vmacc(u32 acc, uint8_t s, u32 v, size_t vl)
    { return __riscv_vmacc_vx_u32m8(acc, s, v, vl); }

    static u32 vmul(u32 v, uint32_t s, size_t vl) { return __riscv_vmul_vx_u32m8(v, s, vl);  }
    static u32 vsrl(u32 v, uint32_t s, size_t vl) { return __riscv_vsrl_vx_u32m8(v, s, vl);  }

    static u8 narrow_u32_to_u8(u32 v, size_t vl)
    { return __riscv_vncvt_x_x_w_u8m2(__riscv_vncvt_x_x_w_u16m4(v, vl), vl); }

    static void vse8 (uint8_t*  p, u8  v, size_t vl) { __riscv_vse8_v_u8m2(p, v, vl);   }
    static void vse32(uint32_t* p, u32 v, size_t vl) { __riscv_vse32_v_u32m8(p, v, vl); }

    static u32 vle32(const uint32_t* p, size_t vl) { return __riscv_vle32_v_u32m8(p, vl); }
};

template <> struct rvv_traits<4>
{
    // u8m4 -> u16m8 -> u32: widening to m16 is invalid.
    // Accumulator is capped at m8; pixel LMUL is 4, accumulator LMUL is 8.
    using u8  = vuint8m2_t;
    using u16 = vuint16m4_t;
    using u32 = vuint32m8_t;

    static size_t setvl_e8 (size_t n) { return __riscv_vsetvl_e8m2(n);  }
    static size_t setvl_e32(size_t n) { return __riscv_vsetvl_e32m8(n); }

    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m2(p, vl);       }
    static u32 vmv0 (size_t vl)                    { return __riscv_vmv_v_x_u32m8(0, vl);      }
    static u16 vzext(u8 v, size_t vl)              { return __riscv_vzext_vf2_u16m4(v, vl);    }

    static u32 vwmaccu(u32 acc, uint8_t s, u16 v, size_t vl)
    { return __riscv_vwmaccu_vx_u32m8(acc, s, v, vl); }

    static u32 vmacc(u32 acc, uint8_t s, u32 v, size_t vl)
    { return __riscv_vmacc_vx_u32m8(acc, s, v, vl); }

    static u32 vmul(u32 v, uint32_t s, size_t vl) { return __riscv_vmul_vx_u32m8(v, s, vl);  }
    static u32 vsrl(u32 v, uint32_t s, size_t vl) { return __riscv_vsrl_vx_u32m8(v, s, vl);  }

    static u8 narrow_u32_to_u8(u32 v, size_t vl)
    { return __riscv_vncvt_x_x_w_u8m2(__riscv_vncvt_x_x_w_u16m4(v, vl), vl); }

    static void vse8 (uint8_t*  p, u8  v, size_t vl) { __riscv_vse8_v_u8m2(p, v, vl);   }
    static void vse32(uint32_t* p, u32 v, size_t vl) { __riscv_vse32_v_u32m8(p, v, vl); }

    static u32 vle32(const uint32_t* p, size_t vl) { return __riscv_vle32_v_u32m8(p, vl); }
};

#endif // __riscv_v


/**
 * @brief   Applies a 5x5 Gaussian blur to the input image using direct convolution.
 * @param   image The input image metadata, which will be modified in-place with the blurred output.
 * @return  Status indicating success or failure of the operation.
 */
template <typename PixelT = uint8_t, typename AccumulatorT = uint32_t, int LMUL = 1>
[[nodiscard]] Status spatial_5x5(image::io::metadata_t<PixelT>& image)
{
    if (!image.height || !image.width || !image.buffer)
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;

    const int32_t width  = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);
    const int32_t kernel_radius = 2;

    const uint32_t pw = width  + 2 * kernel_radius;
    const uint32_t ph = height + 2 * kernel_radius;

    auto padded_input = std::make_unique<PixelT[]>(pw * ph);
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), PixelT{0});
    for (int32_t y = 0; y < height; ++y)
        std::copy_n(&image.buffer[y * width], width,
                    &padded_input[(y + kernel_radius) * pw + kernel_radius]);

    auto raw_out = static_cast<PixelT*>(
        utils::memory::aligned_alloc(64, image.aligned_buffer_size));
    if (!raw_out) return Status::E_ALLOC_FAIL;
    std::unique_ptr<PixelT[], utils::memory::deleter> output_buffer(raw_out);

    constexpr uint32_t shift      = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; )
        {
#if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                using T = rvv_traits<LMUL>;
                const size_t vl  = T::setvl_e8(width - x);
                auto sum = T::vmv0(vl);

                for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky)
                {
                    const uint32_t row_off    = (y + kernel_radius + ky) * pw;
                    const uint32_t kernel_off = (ky + kernel_radius) * 5;
                    for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx)
                    {
                        auto px  = T::vle8(&padded_input[row_off + (x + kernel_radius + kx)], vl);
                        auto px16 = T::vzext(px, vl);
                        sum = T::vwmaccu(sum,
                                         GAUSSIAN_5x5_DATA[kernel_off + (kx + kernel_radius)],
                                         px16, vl);
                    }
                }

                sum = T::vmul(sum, 240, vl);
                sum = T::vsrl(sum,  16, vl);
                T::vse8(&output_buffer[y * width + x], T::narrow_u32_to_u8(sum, vl), vl);
                x += static_cast<int32_t>(vl);
                continue;
            }
#endif

            AccumulatorT sum = 0;
            for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky)
            {
                const uint32_t row_off    = (y + kernel_radius + ky) * pw;
                const uint32_t kernel_off = (ky + kernel_radius) * 5;
                for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx)
                    sum += static_cast<AccumulatorT>(
                               padded_input[row_off + (x + kernel_radius + kx)]) *
                           static_cast<AccumulatorT>(
                               GAUSSIAN_5x5_DATA[kernel_off + (kx + kernel_radius)]);
            }
            sum = static_cast<AccumulatorT>(
                (static_cast<uint64_t>(sum) * multiplier) >> shift);
            sum = (sum > 255) ? 255 : sum;
            output_buffer[y * width + x] = static_cast<PixelT>(sum);
            x += 1;
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
template <typename PixelT = uint8_t, typename AccumulatorT = uint32_t, int LMUL = 1>
[[nodiscard]] Status separable_5x5(image::io::metadata_t<PixelT>& image)
{
    if (!image.height || !image.width || !image.buffer)
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;

    const int32_t width  = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);
    const int32_t kernel_radius = 2;

    const uint32_t pw = width  + 2 * kernel_radius;
    const uint32_t ph = height + 2 * kernel_radius;

    auto padded_input = std::make_unique<PixelT[]>(pw * ph);
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), PixelT{0});
    for (int32_t y = 0; y < height; ++y)
        std::copy_n(&image.buffer[y * width], width,
                    &padded_input[(y + kernel_radius) * pw + kernel_radius]);

    auto inter_buffer = std::make_unique<AccumulatorT[]>(pw * ph);

    for (uint32_t y = 0; y < ph; ++y)
    {
        for (uint32_t x = kernel_radius; x < pw - kernel_radius; )
        {
            const uint32_t row_off = y * pw;

#if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                using T = rvv_traits<LMUL>;
                const size_t vl  = T::setvl_e8(pw - kernel_radius - x);
                auto sum = T::vmv0(vl);

                for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx)
                {
                    auto px   = T::vle8(&padded_input[row_off + (x + kx)], vl);
                    auto px16 = T::vzext(px, vl);
                    sum = T::vwmaccu(sum, GAUSSIAN_5x5_1D_DATA[kx + kernel_radius], px16, vl);
                }

                T::vse32(reinterpret_cast<uint32_t*>(&inter_buffer[row_off + x]), sum, vl);
                x += static_cast<uint32_t>(vl);
                continue;
            }
#endif

            AccumulatorT sum = 0;
            for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx)
                sum += static_cast<AccumulatorT>(padded_input[row_off + (x + kx)]) *
                       static_cast<AccumulatorT>(GAUSSIAN_5x5_1D_DATA[kx + kernel_radius]);
            inter_buffer[row_off + x] = sum;
            x += 1;
        }
    }

    auto raw_out = static_cast<PixelT*>(
        utils::memory::aligned_alloc(64, image.aligned_buffer_size));
    if (!raw_out) return Status::E_ALLOC_FAIL;
    std::unique_ptr<PixelT[], utils::memory::deleter> output_buffer(raw_out);

    constexpr uint32_t shift      = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; )
        {
            const uint32_t col_x = x + kernel_radius;

#if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                using T = rvv_traits<LMUL>;
                const size_t vl  = T::setvl_e32(width - x);
                auto sum = T::vmv0(vl);

                for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky)
                {
                    auto iv = T::vle32(reinterpret_cast<uint32_t*>(
                        &inter_buffer[(y + kernel_radius + ky) * pw + col_x]), vl);
                    sum = T::vmacc(sum, GAUSSIAN_5x5_1D_DATA[ky + kernel_radius], iv, vl);
                }

                sum = T::vmul(sum, 240, vl);
                sum = T::vsrl(sum,  16, vl);
                T::vse8(&output_buffer[y * width + x], T::narrow_u32_to_u8(sum, vl), vl);
                x += static_cast<int32_t>(vl);
                continue;
            }
#endif

            AccumulatorT sum = 0;
            for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky)
                sum += inter_buffer[(y + kernel_radius + ky) * pw + col_x] *
                       static_cast<AccumulatorT>(GAUSSIAN_5x5_1D_DATA[ky + kernel_radius]);
            sum = static_cast<AccumulatorT>(
                (static_cast<uint64_t>(sum) * multiplier) >> shift);
            sum = (sum > 255) ? 255 : sum;
            output_buffer[y * width + x] = static_cast<PixelT>(sum);
            x += 1;
        }
    }

    image.buffer = std::move(output_buffer);
    return Status::E_OK;
}

} // namespace processing
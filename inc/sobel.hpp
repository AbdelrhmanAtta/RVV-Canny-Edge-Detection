/**
 * @file    sobel.hpp
 * @brief   RVV-optimized Sobel Edge Detection
 */

#pragma once

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 11000
#include <riscv_vector.h>
#endif

#include "std_types.hpp"
#include <cstdint>
#include <algorithm>

namespace processing
{

#if defined(__riscv_v) && __riscv_v >= 1000000

template <int LMUL> struct sobel_rvv_traits;

template <> struct sobel_rvv_traits<1>
{
    using u8  = vuint8m1_t;
    using u16 = vuint16m2_t;
    using i16 = vint16m2_t;

    static size_t setvl(size_t n) { return __riscv_vsetvl_e8m1(n); }
    
    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m1(p, vl); }
    
    // Zero-extend 8-bit unsigned to 16-bit signed
    static i16 vzext(u8 v, size_t vl) { 
        return __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v, vl)); 
    }
    
    static i16 vadd(i16 a, i16 b, size_t vl) { return __riscv_vadd_vv_i16m2(a, b, vl); }
    static i16 vsub(i16 a, i16 b, size_t vl) { return __riscv_vsub_vv_i16m2(a, b, vl); }
    
    // Multiply by 2 is just a logical shift left by 1
    static i16 vsll1(i16 a, size_t vl)       { return __riscv_vsll_vx_i16m2(a, 1, vl); }
    
    static void vse16(int16_t* p, i16 v, size_t vl) { __riscv_vse16_v_i16m2(p, v, vl); }
};

template <> struct sobel_rvv_traits<2>
{
    using u8  = vuint8m2_t;
    using u16 = vuint16m4_t;
    using i16 = vint16m4_t;

    static size_t setvl(size_t n) { return __riscv_vsetvl_e8m2(n); }
    
    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m2(p, vl); }
    
    static i16 vzext(u8 v, size_t vl) { 
        return __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(v, vl)); 
    }
    
    static i16 vadd(i16 a, i16 b, size_t vl) { return __riscv_vadd_vv_i16m4(a, b, vl); }
    static i16 vsub(i16 a, i16 b, size_t vl) { return __riscv_vsub_vv_i16m4(a, b, vl); }
    static i16 vsll1(i16 a, size_t vl)       { return __riscv_vsll_vx_i16m4(a, 1, vl); }
    
    static void vse16(int16_t* p, i16 v, size_t vl) { __riscv_vse16_v_i16m4(p, v, vl); }
};

template <> struct sobel_rvv_traits<4>
{
    // LMUL=4 is perfectly legal here because 8-bit to 16-bit widening only hits m8!
    using u8  = vuint8m4_t;
    using u16 = vuint16m8_t;
    using i16 = vint16m8_t;

    static size_t setvl(size_t n) { return __riscv_vsetvl_e8m4(n); }
    
    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m4(p, vl); }
    
    static i16 vzext(u8 v, size_t vl) { 
        return __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2_u16m8(v, vl)); 
    }
    
    static i16 vadd(i16 a, i16 b, size_t vl) { return __riscv_vadd_vv_i16m8(a, b, vl); }
    static i16 vsub(i16 a, i16 b, size_t vl) { return __riscv_vsub_vv_i16m8(a, b, vl); }
    static i16 vsll1(i16 a, size_t vl)       { return __riscv_vsll_vx_i16m8(a, 1, vl); }
    
    static void vse16(int16_t* p, i16 v, size_t vl) { __riscv_vse16_v_i16m8(p, v, vl); }
};

#endif // __riscv_v

/**
 * @brief   Applies a 3x3 Sobel edge detection filter.
 * @param   image Input image metadata (8-bit grayscale).
 * @param   gx Output buffer for X gradients (16-bit signed).
 * @param   gy Output buffer for Y gradients (16-bit signed).
 */
template <int LMUL = 1>
[[nodiscard]] Status spatial_3x3(const image::io::metadata_t<uint8_t>& image, int16_t* gx, int16_t* gy)
{
    if (!image.height || !image.width || !image.buffer || !gx || !gy)
        return Status::E_INVAL_PTR;

    const int32_t width  = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);

    std::fill_n(gx, width * height, 0);
    std::fill_n(gy, width * height, 0);

    for (int32_t y = 1; y < height - 1; ++y)
    {
        for (int32_t x = 1; x < width - 1; )
        {
#if defined(__riscv_v) && __riscv_v >= 1000000
            using T = sobel_rvv_traits<LMUL>;
            size_t vl = T::setvl(width - 1 - x);

            auto t0_u8 = T::vle8(&image.buffer[(y - 1) * width + (x - 1)], vl);
            auto t1_u8 = T::vle8(&image.buffer[(y - 1) * width + x], vl);
            auto t2_u8 = T::vle8(&image.buffer[(y - 1) * width + (x + 1)], vl);
            auto m0_u8 = T::vle8(&image.buffer[y * width + (x - 1)], vl);
            auto m2_u8 = T::vle8(&image.buffer[y * width + (x + 1)], vl);
            auto b0_u8 = T::vle8(&image.buffer[(y + 1) * width + (x - 1)], vl); 
            auto b1_u8 = T::vle8(&image.buffer[(y + 1) * width + x], vl);
            auto b2_u8 = T::vle8(&image.buffer[(y + 1) * width + (x + 1)], vl);

            auto t0 = T::vzext(t0_u8, vl);
            auto t1 = T::vzext(t1_u8, vl);
            auto t2 = T::vzext(t2_u8, vl);
            auto m0 = T::vzext(m0_u8, vl);
            auto m2 = T::vzext(m2_u8, vl);
            auto b0 = T::vzext(b0_u8, vl);
            auto b1 = T::vzext(b1_u8, vl);
            auto b2 = T::vzext(b2_u8, vl);

            auto gx_vec = T::vadd(T::vadd(T::vsub(t2, t0, vl), T::vsll1(T::vsub(m2, m0, vl), vl), vl), T::vsub(b2, b0, vl), vl);
            auto gy_vec = T::vadd(T::vadd(T::vsub(b0, t0, vl), T::vsll1(T::vsub(b1, t1, vl), vl), vl), T::vsub(b2, t2, vl), vl);

            T::vse16(&gx[y * width + x], gx_vec, vl);
            T::vse16(&gy[y * width + x], gy_vec, vl);
            
            x += static_cast<int32_t>(vl);
#else
            // Everything below is strictly for the scalar fallback path
            int16_t t0 = image.buffer[(y - 1) * width + (x - 1)];
            int16_t t1 = image.buffer[(y - 1) * width + x];
            int16_t t2 = image.buffer[(y - 1) * width + (x + 1)];
            int16_t m0 = image.buffer[y * width + (x - 1)];
            int16_t m2 = image.buffer[y * width + (x + 1)];
            int16_t b0 = image.buffer[(y + 1) * width + (x - 1)];
            int16_t b1 = image.buffer[(y + 1) * width + x];
            int16_t b2 = image.buffer[(y + 1) * width + (x + 1)];

            gx[y * width + x] = (t2 - t0) + ((m2 - m0) << 1) + (b2 - b0);
            gy[y * width + x] = (b0 - t0) + ((b1 - t1) << 1) + (b2 - t2);

            x += 1;
#endif
        }
    }
    return Status::E_OK;
}

} // namespace processing

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

namespace processing::sobel
{

#if defined(__riscv_v) && __riscv_v >= 1000000

/**
 * @brief   Specializations of RISC-V Vector intrinsics for different LMUL settings.
 *          Loads 8-bit pixels, widens to signed 16-bit, and computes Sobel gradient
 *          sums via add/sub/shift.
 *
 *          LMUL choice: each specialization groups LMUL register multiples into one
 *          register group. LMUL=1/2 are conservative (more concurrent register groups
 *          left for the compiler to schedule); LMUL=4 maximizes pixels processed per
 *          instruction since the 8-bit -> 16-bit widening here only reaches m8, which
 *          is still legal (unlike the gaussian 8->16->32 chain, which would need m16
 *          at LMUL=4 and is therefore capped).
 *
 *          VLEN dependency: vsetvl_e8mX returns the actual runtime vl, which scales
 *          linearly with VLEN for a fixed LMUL. A larger VLEN means more pixels are
 *          processed per outer-loop iteration (fewer iterations to cover a row); a
 *          smaller VLEN means more iterations. No source change is needed either way.
 */
template <int LMUL> struct rvv_traits;

template <> struct rvv_traits<1>
{
    using u8  = vuint8m1_t;
    using u16 = vuint16m2_t;
    using i16 = vint16m2_t;

    // Computes vl for 8-bit elements at LMUL=1, the most conservative grouping
    // (one physical register per group). VLEN scaling: vl grows linearly with
    // VLEN at this fixed LMUL, so a wider VLEN covers more pixels per call
    // without any source change.
    static size_t setvl(size_t n) { return __riscv_vsetvl_e8m1(n); }

    // Loads vl 8-bit pixels into an m1 group — one of the eight 3x3-window
    // taps loaded per iteration. LMUL=1 matches the pixel group width used
    // throughout this specialization. VLEN scaling: more pixels per load as
    // VLEN grows.
    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m1(p, vl); }

    // Zero-extends 8-bit pixels (m1) to 16-bit (m2), then reinterprets as
    // signed so gradient subtraction below can go negative. LMUL doubles
    // because each 16-bit element needs twice the storage of its 8-bit
    // source at the same element count. VLEN scaling does not change this
    // doubling relationship, only the absolute lane count per call.
    static i16 vzext(u8 v, size_t vl)
    {
        return __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(v, vl));
    }

    // Signed 16-bit (m2) element-wise add, used to sum the three Sobel taps
    // for gx/gy. LMUL=2 matches the widened pixel group from vzext. VLEN
    // scaling: more gradient lanes summed per instruction as VLEN grows.
    static i16 vadd(i16 a, i16 b, size_t vl) { return __riscv_vadd_vv_i16m2(a, b, vl); }
    // Signed 16-bit (m2) element-wise subtract, used for each directional
    // tap difference (e.g. right column minus left column). Same LMUL/VLEN
    // behavior as vadd above.
    static i16 vsub(i16 a, i16 b, size_t vl) { return __riscv_vsub_vv_i16m2(a, b, vl); }

    // Multiply by 2 via logical shift left by 1 — applies the Sobel kernel's
    // center-row/column weight of 2. Operates on m2 to match the other
    // 16-bit ops. VLEN scaling: more lanes shifted per call as VLEN grows.
    static i16 vsll1(i16 a, size_t vl)       { return __riscv_vsll_vx_i16m2(a, 1, vl); }

    // Stores vl signed 16-bit gradient results (m2) to the gx/gy output
    // buffers. Matches the LMUL used throughout this trait's arithmetic.
    // VLEN scaling: more results committed per store as VLEN grows.
    static void vse16(int16_t* p, i16 v, size_t vl) { __riscv_vse16_v_i16m2(p, v, vl); }
};

template <> struct rvv_traits<2>
{
    using u8  = vuint8m2_t;
    using u16 = vuint16m4_t;
    using i16 = vint16m4_t;

    // Computes vl for 8-bit elements at LMUL=2: double the register-group
    // width of LMUL=1, covering roughly twice as many pixels per call at
    // the cost of more register file pressure. VLEN scaling is linear,
    // same as LMUL=1, with a larger absolute vl per call.
    static size_t setvl(size_t n) { return __riscv_vsetvl_e8m2(n); }

    // Loads vl 8-bit pixels into an m2 group — same role as LMUL=1's load,
    // double the elements per call. VLEN scaling: more pixels per load as
    // VLEN grows, same as any LMUL.
    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m2(p, vl); }

    // Zero-extends 8-bit (m2) to 16-bit (m4), then reinterprets as signed.
    // LMUL doubles for the same reason as in the LMUL=1 trait: doubled
    // element width needs doubled register-group width at constant element
    // count. VLEN scaling is unaffected by this relationship.
    static i16 vzext(u8 v, size_t vl)
    {
        return __riscv_vreinterpret_v_u16m4_i16m4(__riscv_vzext_vf2_u16m4(v, vl));
    }

    // Signed 16-bit (m4) add for summing Sobel taps. LMUL matches the
    // widened pixel group from vzext. VLEN scaling: more lanes per call
    // as VLEN grows.
    static i16 vadd(i16 a, i16 b, size_t vl) { return __riscv_vadd_vv_i16m4(a, b, vl); }
    // Signed 16-bit (m4) subtract for tap differences. Same LMUL/VLEN
    // behavior as vadd above.
    static i16 vsub(i16 a, i16 b, size_t vl) { return __riscv_vsub_vv_i16m4(a, b, vl); }
    // Multiply by 2 via shift left by 1, applying the center weight.
    // Operates on m4 to match the other 16-bit ops in this trait. VLEN
    // scaling: more lanes shifted per call as VLEN grows.
    static i16 vsll1(i16 a, size_t vl)       { return __riscv_vsll_vx_i16m4(a, 1, vl); }

    // Stores vl signed 16-bit results (m4) to gx/gy. Matches this trait's
    // arithmetic LMUL. VLEN scaling: more results per store as VLEN grows.
    static void vse16(int16_t* p, i16 v, size_t vl) { __riscv_vse16_v_i16m4(p, v, vl); }
};

template <> struct rvv_traits<4>
{
    // LMUL=4 is legal here (unlike gaussian's accumulator) because the
    // widening chain is only 8-bit -> 16-bit, reaching m8 — the maximum
    // legal LMUL — rather than needing a third widening step to 32-bit
    // that would exceed it. No cap is required.
    using u8  = vuint8m4_t;
    using u16 = vuint16m8_t;
    using i16 = vint16m8_t;

    // Computes vl for 8-bit elements at LMUL=4, the maximum usable pixel
    // group width for this trait set, covering the most pixels per call
    // at the cost of the most register file pressure. VLEN scaling is
    // linear, same as any LMUL, with the largest absolute vl per call.
    static size_t setvl(size_t n) { return __riscv_vsetvl_e8m4(n); }

    // Loads vl 8-bit pixels into an m4 group — maximum pixel throughput
    // per load in this trait set. VLEN scaling: more pixels per load as
    // VLEN grows.
    static u8  vle8 (const uint8_t* p, size_t vl) { return __riscv_vle8_v_u8m4(p, vl); }

    // Zero-extends 8-bit (m4) to 16-bit (m8), then reinterprets as signed.
    // LMUL doubles to the ISA maximum because each 16-bit element needs
    // twice the storage of its 8-bit source at constant element count.
    // VLEN scaling unaffected by this relationship.
    static i16 vzext(u8 v, size_t vl)
    {
        return __riscv_vreinterpret_v_u16m8_i16m8(__riscv_vzext_vf2_u16m8(v, vl));
    }

    // Signed 16-bit (m8) add for summing Sobel taps, at the maximum legal
    // LMUL. VLEN scaling: more lanes per call as VLEN grows.
    static i16 vadd(i16 a, i16 b, size_t vl) { return __riscv_vadd_vv_i16m8(a, b, vl); }
    // Signed 16-bit (m8) subtract for tap differences. Same LMUL/VLEN
    // behavior as vadd above.
    static i16 vsub(i16 a, i16 b, size_t vl) { return __riscv_vsub_vv_i16m8(a, b, vl); }
    // Multiply by 2 via shift left by 1, applying the center weight, on
    // the m8 group. VLEN scaling: more lanes shifted per call as VLEN grows.
    static i16 vsll1(i16 a, size_t vl)       { return __riscv_vsll_vx_i16m8(a, 1, vl); }

    // Stores vl signed 16-bit results (m8) to gx/gy, the maximum lanes
    // committed per store in this trait set. VLEN scaling: more results
    // per store as VLEN grows.
    static void vse16(int16_t* p, i16 v, size_t vl) { __riscv_vse16_v_i16m8(p, v, vl); }
};

#endif // __riscv_v

/**
 * @brief   Applies a 3x3 Sobel edge detection filter.
 * @tparam  LMUL    RVV register group multiple to use for the vectorized path
 *                   (see rvv_traits for the throughput/register-pressure
 *                   tradeoff at each value).
 * @param   image   Input image metadata (8-bit grayscale).
 * @param   gx      Output buffer for X gradients (16-bit signed).
 * @param   gy      Output buffer for Y gradients (16-bit signed).
 */
template <int LMUL = 2>
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
            // vl pixels processed this iteration; scales with VLEN and the
            // chosen LMUL (see trait comments above for the per-LMUL
            // throughput/register-pressure tradeoff and VLEN scaling rule).
            using T = rvv_traits<LMUL>;
            size_t vl = T::setvl(width - 1 - x);

            // Load the eight 3x3-window neighbor taps (all but the center
            // pixel, which the Sobel kernel weights as zero) as 8-bit (u8).
            // Same LMUL/VLEN behavior as T::vle8 documented in the traits.
            auto t0_u8 = T::vle8(&image.buffer[(y - 1) * width + (x - 1)], vl);
            auto t1_u8 = T::vle8(&image.buffer[(y - 1) * width + x], vl);
            auto t2_u8 = T::vle8(&image.buffer[(y - 1) * width + (x + 1)], vl);
            auto m0_u8 = T::vle8(&image.buffer[y * width + (x - 1)], vl);
            auto m2_u8 = T::vle8(&image.buffer[y * width + (x + 1)], vl);
            auto b0_u8 = T::vle8(&image.buffer[(y + 1) * width + (x - 1)], vl);
            auto b1_u8 = T::vle8(&image.buffer[(y + 1) * width + x], vl);
            auto b2_u8 = T::vle8(&image.buffer[(y + 1) * width + (x + 1)], vl);

            // Widen all eight taps to signed 16-bit so tap differences below
            // can go negative. Same LMUL/VLEN behavior as T::vzext documented
            // in the traits.
            auto t0 = T::vzext(t0_u8, vl);
            auto t1 = T::vzext(t1_u8, vl);
            auto t2 = T::vzext(t2_u8, vl);
            auto m0 = T::vzext(m0_u8, vl);
            auto m2 = T::vzext(m2_u8, vl);
            auto b0 = T::vzext(b0_u8, vl);
            auto b1 = T::vzext(b1_u8, vl);
            auto b2 = T::vzext(b2_u8, vl);

            // Gx = (t2 - t0) + 2*(m2 - m0) + (b2 - b0): right-column minus
            // left-column taps, center row doubled via vsll1. Each vadd/vsub/
            // vsll1 call follows the LMUL/VLEN behavior documented in the
            // traits above.
            auto gx_vec = T::vadd(T::vadd(T::vsub(t2, t0, vl), T::vsll1(T::vsub(m2, m0, vl), vl), vl), T::vsub(b2, b0, vl), vl);
            // Gy = (b0 - t0) + 2*(b1 - t1) + (b2 - t2): bottom-row minus
            // top-row taps, center column doubled via vsll1. Same op/LMUL/
            // VLEN behavior as gx_vec above.
            auto gy_vec = T::vadd(T::vadd(T::vsub(b0, t0, vl), T::vsll1(T::vsub(b1, t1, vl), vl), vl), T::vsub(b2, t2, vl), vl);

            // Store both gradient vectors; see T::vse16 in the traits for
            // LMUL/VLEN behavior.
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

} // namespace processing::sobel

/**
 * @file    gradient.hpp
 * @brief   Gradient computation implementations for image processing.
 */

#pragma once

#include "std_types.hpp"
#include "utils.hpp"
#include <cstdint>
#include <math.h>
#include <memory>

namespace processing
{

#if defined(__riscv_v) && __riscv_v >= 1000000

template <int LMUL> struct gradient_rvv_traits;

// ---- LMUL = 1 ---------------------------------------------------------
template <> struct gradient_rvv_traits<1>
{
    using i16 = vint16m1_t;
    using u16 = vuint16m1_t;   // native-width unsigned 16, same LMUL as i16
    using u32 = vuint32m2_t;   // widened from u16: m1 -> m2
    using u8  = vuint8mf2_t;   // narrowed from u16: m1 / 2 = mf2

    static size_t setvl_e16(size_t n) { return __riscv_vsetvl_e16m1(n); }

    static i16 vle16_i(const int16_t* p, size_t vl)  { return __riscv_vle16_v_i16m1(p, vl); }
    static u16 vle16_u(const uint16_t* p, size_t vl) { return __riscv_vle16_v_u16m1(p, vl); }
    static i16 vneg_i(i16 v, size_t vl)              { return __riscv_vneg_v_i16m1(v, vl); }
    static i16 vmax_i(i16 a, i16 b, size_t vl)       { return __riscv_vmax_vv_i16m1(a, b, vl); }
    static u16 reinterpret(i16 v)                    { return __riscv_vreinterpret_v_i16m1_u16m1(v); }
    static u16 vadd_u(u16 a, u16 b, size_t vl)       { return __riscv_vadd_vv_u16m1(a, b, vl); }
    static void vse16_u(uint16_t* p, u16 v, size_t vl) { __riscv_vse16_v_u16m1(p, v, vl); }

    // Reduction dest is always m1 — ISA rule, not parameterized by LMUL.
    static vuint16m1_t vredmax(u16 v, vuint16m1_t acc, size_t vl)
    {
        return __riscv_vredmax_vs_u16m1_u16m1(v, acc, vl);
    }

    static u32 vwmulu_u(u16 v, uint16_t s, size_t vl) { return __riscv_vwmulu_vx_u32m2(v, s, vl); }
    static u32 vdivu_u(u32 v, uint32_t s, size_t vl)  { return __riscv_vdivu_vx_u32m2(v, s, vl); }
    static u16 vncvt_u32_u16(u32 v, size_t vl)        { return __riscv_vncvt_x_x_w_u16m1(v, vl); }
    static u8  vncvt_u16_u8(u16 v, size_t vl)         { return __riscv_vncvt_x_x_w_u8mf2(v, vl); }
    static void vse8_u(uint8_t* p, u8 v, size_t vl)   { __riscv_vse8_v_u8mf2(p, v, vl); }
};

// ---- LMUL = 2 ---------------------------------------------------------
template <> struct gradient_rvv_traits<2>
{
    using i16 = vint16m2_t;
    using u16 = vuint16m2_t;
    using u32 = vuint32m4_t;   // widened from u16: m2 -> m4
    using u8  = vuint8m1_t;    // narrowed from u16: m2 / 2 = m1

    static size_t setvl_e16(size_t n) { return __riscv_vsetvl_e16m2(n); }

    static i16 vle16_i(const int16_t* p, size_t vl)  { return __riscv_vle16_v_i16m2(p, vl); }
    static u16 vle16_u(const uint16_t* p, size_t vl) { return __riscv_vle16_v_u16m2(p, vl); }
    static i16 vneg_i(i16 v, size_t vl)              { return __riscv_vneg_v_i16m2(v, vl); }
    static i16 vmax_i(i16 a, i16 b, size_t vl)       { return __riscv_vmax_vv_i16m2(a, b, vl); }
    static u16 reinterpret(i16 v)                    { return __riscv_vreinterpret_v_i16m2_u16m2(v); }
    static u16 vadd_u(u16 a, u16 b, size_t vl)       { return __riscv_vadd_vv_u16m2(a, b, vl); }
    static void vse16_u(uint16_t* p, u16 v, size_t vl) { __riscv_vse16_v_u16m2(p, v, vl); }

    static vuint16m1_t vredmax(u16 v, vuint16m1_t acc, size_t vl)
    {
        return __riscv_vredmax_vs_u16m2_u16m1(v, acc, vl);
    }

    static u32 vwmulu_u(u16 v, uint16_t s, size_t vl) { return __riscv_vwmulu_vx_u32m4(v, s, vl); }
    static u32 vdivu_u(u32 v, uint32_t s, size_t vl)  { return __riscv_vdivu_vx_u32m4(v, s, vl); }
    static u16 vncvt_u32_u16(u32 v, size_t vl)        { return __riscv_vncvt_x_x_w_u16m2(v, vl); }
    static u8  vncvt_u16_u8(u16 v, size_t vl)         { return __riscv_vncvt_x_x_w_u8m1(v, vl); }
    static void vse8_u(uint8_t* p, u8 v, size_t vl)   { __riscv_vse8_v_u8m1(p, v, vl); }
};

// ---- LMUL = 4 ---------------------------------------------------------
template <> struct gradient_rvv_traits<4>
{
    using i16 = vint16m4_t;
    using u16 = vuint16m4_t;
    using u32 = vuint32m8_t;   // widened from u16: m4 -> m8
    using u8  = vuint8m2_t;    // narrowed from u16: m4 / 2 = m2

    static size_t setvl_e16(size_t n) { return __riscv_vsetvl_e16m4(n); }

    static i16 vle16_i(const int16_t* p, size_t vl)  { return __riscv_vle16_v_i16m4(p, vl); }
    static u16 vle16_u(const uint16_t* p, size_t vl) { return __riscv_vle16_v_u16m4(p, vl); }
    static i16 vneg_i(i16 v, size_t vl)              { return __riscv_vneg_v_i16m4(v, vl); }
    static i16 vmax_i(i16 a, i16 b, size_t vl)       { return __riscv_vmax_vv_i16m4(a, b, vl); }
    static u16 reinterpret(i16 v)                    { return __riscv_vreinterpret_v_i16m4_u16m4(v); }
    static u16 vadd_u(u16 a, u16 b, size_t vl)       { return __riscv_vadd_vv_u16m4(a, b, vl); }
    static void vse16_u(uint16_t* p, u16 v, size_t vl) { __riscv_vse16_v_u16m4(p, v, vl); }

    static vuint16m1_t vredmax(u16 v, vuint16m1_t acc, size_t vl)
    {
        return __riscv_vredmax_vs_u16m4_u16m1(v, acc, vl);
    }

    static u32 vwmulu_u(u16 v, uint16_t s, size_t vl) { return __riscv_vwmulu_vx_u32m8(v, s, vl); }
    static u32 vdivu_u(u32 v, uint32_t s, size_t vl)  { return __riscv_vdivu_vx_u32m8(v, s, vl); }
    static u16 vncvt_u32_u16(u32 v, size_t vl)        { return __riscv_vncvt_x_x_w_u16m4(v, vl); }
    static u8  vncvt_u16_u8(u16 v, size_t vl)         { return __riscv_vncvt_x_x_w_u8m2(v, vl); }
    static void vse8_u(uint8_t* p, u8 v, size_t vl)   { __riscv_vse8_v_u8m2(p, v, vl); }
};

#endif // __riscv_v


/** @brief  Compute the L1 gradient magnitude of an image.
 *  @param  image The input image metadata.
 *  @param  Gx The x-component of the gradient.
 *  @param  Gy The y-component of the gradient.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t, typename GradientT = int16_t, typename MagT = uint16_t, int LMUL = 4>
[[nodiscard]] Status l1(const image::io::metadata_t<PixelT>& image,
                        const GradientT* __restrict Gx,
                        const GradientT* __restrict Gy)
{
    if (!image.height || !image.width || !image.aligned_buffer_size || !image.buffer || !Gx || !Gy)
        return (image.buffer && Gx && Gy) ? Status::E_NOK : Status::E_INVAL_PTR;

    auto pointer_buffer = std::unique_ptr<MagT[], utils::memory::deleter>(
        static_cast<MagT*>(utils::memory::aligned_alloc(64, image.aligned_buffer_size * sizeof(MagT)))
    );
    if (!pointer_buffer) return Status::E_ALLOC_FAIL;

    const uint32_t count = image.pixel_count;
    MagT max_value = 0;

#if defined(__riscv_v) && __riscv_v >= 1000000
    using T = gradient_rvv_traits<LMUL>;

    // v_max_magnitude: reduction destination accumulator — ALWAYS m1 regardless
    // of LMUL. This is an ISA rule, not a choice, so it lives outside the traits.
    vuint16m1_t v_max_magnitude = __riscv_vmv_v_x_u16m1(0, 1);
    size_t vl;

    // --- Pass 1: |Gx| + |Gy|, track running max via vector reduction ---
    for (uint32_t i = 0; i < count; i += vl)
    {
        // vsetvl: ask the hardware how many i16 elements (at this LMUL) fit in one
        // vector register group for the remaining tail. VLA-safe: works unmodified
        // whether VLEN is 128, 256, or 512 -- the tail (count - i) just shrinks faster
        // on wider hardware.
        vl = T::setvl_e16(count - i);

        // vle16: load Gx/Gy chunks as signed 16-bit. Sobel output on 8-bit pixels
        // fits comfortably in int16_t, so no widening is needed on load.
        typename T::i16 vgx = T::vle16_i(Gx + i, vl);
        typename T::i16 vgy = T::vle16_i(Gy + i, vl);

        // vmax(v, -v): elementwise absolute value. RVV has no dedicated vabs,
        // so abs(x) = max(x, -x) is the standard idiom.
        typename T::i16 vgx_abs = T::vmax_i(vgx, T::vneg_i(vgx, vl), vl);
        typename T::i16 vgy_abs = T::vmax_i(vgy, T::vneg_i(vgy, vl), vl);

        // reinterpret + vadd: L1 magnitude = |Gx| + |Gy|. The sum of two
        // non-negative i16 values is always representable as u16, so we
        // reinterpret the absolute values before adding as unsigned.
        typename T::u16 vmag = T::vadd_u(T::reinterpret(vgx_abs), T::reinterpret(vgy_abs), vl);

        T::vse16_u(pointer_buffer.get() + i, vmag, vl);

        // vredmax: vector reduction. Collapses the whole vl-wide chunk into a
        // single scalar max. Per the RVV spec, reduction destinations are always
        // m1 regardless of the source LMUL -- this is independent of our LMUL choice.
        v_max_magnitude = T::vredmax(vmag, v_max_magnitude, vl);
    }

    // vmv_x_s: extract the reduced scalar out of element 0 of the m1 result
    // register into a plain C++ value so we can compare/accumulate it.
    max_value = static_cast<MagT>(__riscv_vmv_x_s_u16m1_u16(v_max_magnitude));

    if (max_value == 0)
    {
        std::fill(image.buffer.get(), image.buffer.get() + count, PixelT{0});
        return Status::E_OK;
    }

    // --- Pass 2: normalize to [0, 255] — widen u16->u32, scale, divide, narrow x2 ---
    for (uint32_t i = 0; i < count; i += vl)
    {
        // Second pass re-strip-mines at the same LMUL as the first pass for
        // consistency, though it is independent: each pass picks its own vl.
        vl = T::setvl_e16(count - i);

        typename T::u16 vmag = T::vle16_u(pointer_buffer.get() + i, vl);

        // vwmulu (widen-multiply): magnitude * 255, widening u16 -> u32 in one
        // instruction so the multiply can't overflow before the divide.
        typename T::u32 vscaled = T::vwmulu_u(vmag, 255, vl);

        // vdivu: divide by the global max found in pass 1 to land in [0, 255].
        vscaled = T::vdivu_u(vscaled, max_value, vl);

        // vncvt (u32 -> u16 -> u8): two narrowing converts back down to 8-bit.
        // Values are already bounded to [0, 255] by construction, so this is a
        // plain truncating narrow, not a saturating clip.
        typename T::u16 vnarrow1 = T::vncvt_u32_u16(vscaled, vl);
        typename T::u8  vnarrow2 = T::vncvt_u16_u8(vnarrow1, vl);

        T::vse8_u(reinterpret_cast<uint8_t*>(image.buffer.get()) + i, vnarrow2, vl);
    }

#else
    for (uint32_t i = 0; i < count; ++i)
    {
        MagT value = static_cast<MagT>(std::abs(Gx[i]) + std::abs(Gy[i]));
        pointer_buffer[i] = value;
        if (value > max_value) max_value = value;
    }

    float scale = (max_value > 0) ? 255.0f / static_cast<float>(max_value) : 0.0f;

    for (uint32_t i = 0; i < count; ++i)
        image.buffer[i] = static_cast<PixelT>(pointer_buffer[i] * scale);
#endif

    return Status::E_OK;
}

/** @brief  Compute the L2 gradient magnitude of an image.
 *  @param  image The input image metadata.
 *  @param  Gx The x-component of the gradient.
 *  @param  Gy The y-component of the gradient.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t, typename GradientT = int16_t, typename MagT = float>
[[nodiscard]] Status l2(const image::io::metadata_t<PixelT>& image,
                        const GradientT* __restrict Gx,
                        const GradientT* __restrict Gy)
{
    if(!image.height || !image.width || !image.aligned_buffer_size || !image.buffer
        || !Gx || !Gy)
    {
        return (image.buffer && Gx && Gy) ? Status::E_NOK : Status::E_INVAL_PTR;
    }

    auto pointer_buffer = std::unique_ptr<MagT[], utils::memory::deleter>(
        static_cast<MagT*>(utils::memory::aligned_alloc(64, image.aligned_buffer_size * sizeof(MagT)))
    );

    if(!pointer_buffer)
    {
        return Status::E_ALLOC_FAIL;
    }

    MagT max_value = 0.0f;
    for(uint32_t i = 0; i < image.pixel_count; ++i)
    {
        float x = static_cast<float>(Gx[i]);
        float y = static_cast<float>(Gy[i]);
        MagT value = std::sqrt(x * x + y * y);
        
        pointer_buffer[i] = value;
        if(value > max_value)
        {
            max_value = value;
        }
    }

    float scale = (max_value > 0.0f) ? 255.0f / max_value : 0.0f;

    for(uint32_t i = 0; i < image.pixel_count; ++i) 
    {
        image.buffer[i] = static_cast<PixelT>(pointer_buffer[i] * scale);
    }

    return Status::E_OK;
}

/** @brief  Compute the direction of the gradient of an image.
 *  @param  image The input image metadata.
 *  @param  Gx The x-component of the gradient.
 *  @param  Gy The y-component of the gradient.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t, typename GradientT = int32_t>
[[nodiscard]] Status direction(const image::io::metadata_t<PixelT>& image,
                                const GradientT* __restrict Gx,
                                const GradientT* __restrict Gy)
{
    if(!image.height || !image.width || !image.aligned_buffer_size || !image.buffer
    || !Gx || !Gy)
    {
        return (image.buffer && Gx && Gy) ? Status::E_NOK : Status::E_INVAL_PTR;
    }

    uint8_t angle;
    for(uint32_t i = 0; i < image.pixel_count; ++i)
    {
        const GradientT cur_x = Gx[i];
        const GradientT cur_y = Gy[i];
        const uint16_t abs_x = static_cast<uint16_t>(std::abs(cur_x));
        const uint16_t abs_y = static_cast<uint16_t>(std::abs(cur_y));

        if ((abs_y * 5) > (abs_x * 12)) // > 2.4 (Vertical gradient)
        {
            angle = 90;
        }
        else if ((abs_y * 2) > abs_x) // > 0.5 (Diagonal range)
        {
            // In image coordinates, y increases downwards.
            // cur_x > 0, cur_y > 0 is Down-Right (135 deg)
            // cur_x > 0, cur_y < 0 is Up-Right (45 deg)
            if (((cur_x > 0) && (cur_y > 0)) || ((cur_x < 0) && (cur_y < 0)))
            {
                angle = 135;
            }
            else
            {
                angle = 45;
            }
        }
        else // < 0.5 (Horizontal range)
        {
            angle = 0;
        }
        image.buffer[i] = angle;
    }

    return Status::E_OK;
}

} // namespace processing


/**
 * @file    gradient.hpp
 * @brief   Gradient computation implementations for image processing.
 */

#pragma once

#if defined(__riscv_v_intrinsic) && __riscv_v_intrinsic >= 11000
#include <riscv_vector.h>
#endif 

#include "std_types.hpp"
#include "utils.hpp"

#include <cstdint>
#include <math.h>
#include <memory>

namespace processing::gradient
{

#if defined(__riscv_v) && __riscv_v >= 1000000

/**
 * @brief   Specializations of RISC-V Vector intrinsics for different LMUL settings.
 *          Loads signed 16-bit gradient components, computes their absolute value,
 *          sums them (L1 magnitude), reduces to find the maximum, then scales and
 *          narrows the result to 8-bit for output.
 *
 *          LMUL choice: each specialization groups LMUL register multiples into one
 *          register group. LMUL=1 is conservative (more concurrent groups available);
 *          LMUL=2/4 trade register pressure for more elements processed per
 *          instruction. The accumulator (u32) and the reduction result (always m1,
 *          fixed by the RVV reduction instruction semantics) widen alongside LMUL,
 *          same widening-chain reasoning as gaussian_rvv_traits.
 *
 *          VLEN dependency: vsetvl_e16mX returns the actual runtime vl, scaling
 *          linearly with VLEN at fixed LMUL. A larger VLEN processes more pixels
 *          per outer-loop iteration (fewer iterations to cover the image); a smaller
 *          VLEN means more iterations. No source change is required either way.
 */
template <int LMUL> struct rvv_traits;

template <> struct rvv_traits<1>
{
    using i16 = vint16m1_t;
    using u16 = vuint16m1_t;
    using u32 = vuint32m2_t;
    using u8  = vuint8mf2_t;

    // Computes vl for 16-bit elements at LMUL=1, the baseline grouping with
    // the most concurrent register groups available. VLEN scaling: vl grows
    // linearly with VLEN at this fixed LMUL.
    static size_t setvl_e16(size_t n) { return __riscv_vsetvl_e16m1(n); }

    // Loads vl signed 16-bit gradient components (m1) from Gx/Gy. LMUL=1
    // matches this trait's baseline group width. VLEN scaling: more
    // components loaded per call as VLEN grows.
    static i16 vle16_i(const int16_t* p, size_t vl)  { return __riscv_vle16_v_i16m1(p, vl); }
    // Loads vl unsigned 16-bit magnitudes (m1) back from the scratch buffer
    // in the second pass. Same LMUL/VLEN behavior as vle16_i.
    static u16 vle16_u(const uint16_t* p, size_t vl) { return __riscv_vle16_v_u16m1(p, vl); }
    // Negates a signed 16-bit (m1) vector — paired with vmax_i below to
    // compute |v| without a branch. VLEN scaling: more lanes negated per call.
    static i16 vneg_i(i16 v, size_t vl)              { return __riscv_vneg_v_i16m1(v, vl); }
    // Element-wise max of v and -v (m1), i.e. absolute value. VLEN scaling:
    // more lanes compared per call as VLEN grows.
    static i16 vmax_i(i16 a, i16 b, size_t vl)       { return __riscv_vmax_vv_i16m1(a, b, vl); }
    // Reinterprets a signed 16-bit (m1) vector as unsigned with no data
    // movement — |Gx| and |Gy| are non-negative after vmax_i, so this is a
    // free bit-pattern relabel before the unsigned add. LMUL is unchanged
    // since reinterpret never alters register-group width.
    static u16 reinterpret(i16 v)                    { return __riscv_vreinterpret_v_i16m1_u16m1(v); }
    // Unsigned 16-bit (m1) add: |Gx| + |Gy|, the L1 magnitude. VLEN scaling:
    // more lanes summed per call as VLEN grows.
    static u16 vadd_u(u16 a, u16 b, size_t vl)       { return __riscv_vadd_vv_u16m1(a, b, vl); }
    // Stores vl unsigned 16-bit magnitudes (m1) to the scratch buffer for
    // the second (scaling) pass. VLEN scaling: more lanes stored per call.
    static void vse16_u(uint16_t* p, u16 v, size_t vl) { __riscv_vse16_v_u16m1(p, v, vl); }

    // Reduces the magnitude vector (m1) to a running scalar max, accumulated
    // in a fixed m1 destination/source register (the RVV reduction ISA
    // always produces an m1 result regardless of source LMUL). LMUL of the
    // *source* vector is 1 here, matching this trait. VLEN scaling: more
    // lanes folded into the running max per call, but the result width and
    // semantics (single scalar in lane 0) are unaffected by VLEN.
    static vuint16m1_t vredmax(u16 v, vuint16m1_t acc, size_t vl)
    { return __riscv_vredmax_vs_u16m1_u16m1(v, acc, vl);}

    // Widening unsigned multiply: magnitude (m1) * 255 (scalar) -> 32-bit
    // (m2) product, the first step of the 0-255 rescale. LMUL doubles
    // because each 32-bit product needs twice the storage of its 16-bit
    // source at constant element count. VLEN scaling: more products per call.
    static u32 vwmulu_u(u16 v, uint16_t s, size_t vl) { return __riscv_vwmulu_vx_u32m2(v, s, vl); }
    // Unsigned divide of the scaled product (m2) by the runtime max_value
    // scalar, completing the rescale to [0,255]. LMUL unchanged (m2 in,
    // m2 out). VLEN scaling: more divisions per call.
    static u32 vdivu_u(u32 v, uint32_t s, size_t vl)  { return __riscv_vdivu_vx_u32m2(v, s, vl); }
    // Narrows 32-bit (m2) to 16-bit (m1), the first narrowing step back
    // toward 8-bit output. LMUL halves because each output element is half
    // the byte-width of its input at constant element count. VLEN scaling:
    // more lanes narrowed per call.
    static u16 vncvt_u32_u16(u32 v, size_t vl)        { return __riscv_vncvt_x_x_w_u16m1(v, vl); }
    // Narrows 16-bit (m1) to 8-bit (mf2, a fractional LMUL since one more
    // halving below m1 requires a fractional group). LMUL continues to
    // halve following the narrowing chain. VLEN scaling: more lanes
    // narrowed per call.
    static u8  vncvt_u16_u8(u16 v, size_t vl)         { return __riscv_vncvt_x_x_w_u8mf2(v, vl); }
    // Stores vl 8-bit final magnitudes (mf2) into the image's pixel buffer.
    // VLEN scaling: more pixels written per call.
    static void vse8_u(uint8_t* p, u8 v, size_t vl)   { __riscv_vse8_v_u8mf2(p, v, vl); }
};

template <> struct rvv_traits<2>
{
    using i16 = vint16m2_t;
    using u16 = vuint16m2_t;
    using u32 = vuint32m4_t;
    using u8  = vuint8m1_t;

    // Computes vl for 16-bit elements at LMUL=2, double the group width of
    // LMUL=1, covering roughly twice the elements per call at the cost of
    // more register file pressure. VLEN scaling is linear, same as LMUL=1.
    static size_t setvl_e16(size_t n) { return __riscv_vsetvl_e16m2(n); }

    // Loads vl signed 16-bit gradient components (m2). Same role as
    // LMUL=1's load, double the elements per call. VLEN scaling unchanged.
    static i16 vle16_i(const int16_t* p, size_t vl)  { return __riscv_vle16_v_i16m2(p, vl); }
    // Loads vl unsigned 16-bit magnitudes (m2) for the second pass.
    static u16 vle16_u(const uint16_t* p, size_t vl) { return __riscv_vle16_v_u16m2(p, vl); }
    // Negates a signed 16-bit (m2) vector, paired with vmax_i for |v|.
    static i16 vneg_i(i16 v, size_t vl)              { return __riscv_vneg_v_i16m2(v, vl); }
    // Element-wise max (m2), i.e. absolute value.
    static i16 vmax_i(i16 a, i16 b, size_t vl)       { return __riscv_vmax_vv_i16m2(a, b, vl); }
    // Reinterprets signed (m2) as unsigned, free relabel, no LMUL change.
    static u16 reinterpret(i16 v)                    { return __riscv_vreinterpret_v_i16m2_u16m2(v); }
    // Unsigned 16-bit (m2) add: |Gx| + |Gy|.
    static u16 vadd_u(u16 a, u16 b, size_t vl)       { return __riscv_vadd_vv_u16m2(a, b, vl); }
    // Stores vl unsigned 16-bit magnitudes (m2) to the scratch buffer.
    static void vse16_u(uint16_t* p, u16 v, size_t vl) { __riscv_vse16_v_u16m2(p, v, vl); }

    // Reduces the magnitude vector (m2 source) to a running scalar max in a
    // fixed m1 register (reduction result is always m1 regardless of source
    // LMUL, per the RVV reduction ISA). VLEN scaling: more lanes folded per
    // call; result width/semantics unaffected by VLEN.
    static vuint16m1_t vredmax(u16 v, vuint16m1_t acc, size_t vl)
    { return __riscv_vredmax_vs_u16m2_u16m1(v, acc, vl); }

    // Widening multiply: magnitude (m2) * 255 -> 32-bit (m4) product.
    static u32 vwmulu_u(u16 v, uint16_t s, size_t vl) { return __riscv_vwmulu_vx_u32m4(v, s, vl); }
    // Unsigned divide (m4) by max_value, completing the [0,255] rescale.
    static u32 vdivu_u(u32 v, uint32_t s, size_t vl)  { return __riscv_vdivu_vx_u32m4(v, s, vl); }
    // Narrows 32-bit (m4) to 16-bit (m2), first narrowing step.
    static u16 vncvt_u32_u16(u32 v, size_t vl)        { return __riscv_vncvt_x_x_w_u16m2(v, vl); }
    // Narrows 16-bit (m2) to 8-bit (m1), second narrowing step.
    static u8  vncvt_u16_u8(u16 v, size_t vl)         { return __riscv_vncvt_x_x_w_u8m1(v, vl); }
    // Stores vl 8-bit final magnitudes (m1) into the image's pixel buffer.
    static void vse8_u(uint8_t* p, u8 v, size_t vl)   { __riscv_vse8_v_u8m1(p, v, vl); }
};

template <> struct rvv_traits<4>
{
    using i16 = vint16m4_t;
    using u16 = vuint16m4_t;
    using u32 = vuint32m8_t;
    using u8  = vuint8m2_t;

    // Computes vl for 16-bit elements at LMUL=4, the maximum group width
    // used in this trait set, covering the most elements per call at the
    // cost of the most register file pressure. VLEN scaling is linear,
    // same as any LMUL, with the largest absolute vl per call.
    static size_t setvl_e16(size_t n) { return __riscv_vsetvl_e16m4(n); }

    // Loads vl signed 16-bit gradient components (m4), maximum throughput
    // per load in this trait set.
    static i16 vle16_i(const int16_t* p, size_t vl)  { return __riscv_vle16_v_i16m4(p, vl); }
    // Loads vl unsigned 16-bit magnitudes (m4) for the second pass.
    static u16 vle16_u(const uint16_t* p, size_t vl) { return __riscv_vle16_v_u16m4(p, vl); }
    // Negates a signed 16-bit (m4) vector, paired with vmax_i for |v|.
    static i16 vneg_i(i16 v, size_t vl)              { return __riscv_vneg_v_i16m4(v, vl); }
    // Element-wise max (m4), i.e. absolute value.
    static i16 vmax_i(i16 a, i16 b, size_t vl)       { return __riscv_vmax_vv_i16m4(a, b, vl); }
    // Reinterprets signed (m4) as unsigned, free relabel, no LMUL change.
    static u16 reinterpret(i16 v)                    { return __riscv_vreinterpret_v_i16m4_u16m4(v); }
    // Unsigned 16-bit (m4) add: |Gx| + |Gy|.
    static u16 vadd_u(u16 a, u16 b, size_t vl)       { return __riscv_vadd_vv_u16m4(a, b, vl); }
    // Stores vl unsigned 16-bit magnitudes (m4) to the scratch buffer.
    static void vse16_u(uint16_t* p, u16 v, size_t vl) { __riscv_vse16_v_u16m4(p, v, vl); }

    // Reduces the magnitude vector (m4 source) to a running scalar max in a
    // fixed m1 register (reduction result is always m1 regardless of source
    // LMUL). VLEN scaling: more lanes folded per call; result width/
    // semantics unaffected by VLEN.
    static vuint16m1_t vredmax(u16 v, vuint16m1_t acc, size_t vl)
    { return __riscv_vredmax_vs_u16m4_u16m1(v, acc, vl); }

    // Widening multiply: magnitude (m4) * 255 -> 32-bit (m8) product, the
    // maximum legal LMUL for the accumulator in this trait set.
    static u32 vwmulu_u(u16 v, uint16_t s, size_t vl) { return __riscv_vwmulu_vx_u32m8(v, s, vl); }
    // Unsigned divide (m8) by max_value, completing the [0,255] rescale.
    static u32 vdivu_u(u32 v, uint32_t s, size_t vl)  { return __riscv_vdivu_vx_u32m8(v, s, vl); }
    // Narrows 32-bit (m8) to 16-bit (m4), first narrowing step.
    static u16 vncvt_u32_u16(u32 v, size_t vl)        { return __riscv_vncvt_x_x_w_u16m4(v, vl); }
    // Narrows 16-bit (m4) to 8-bit (m2), second narrowing step.
    static u8  vncvt_u16_u8(u16 v, size_t vl)         { return __riscv_vncvt_x_x_w_u8m2(v, vl); }
    // Stores vl 8-bit final magnitudes (m2) into the image's pixel buffer.
    static void vse8_u(uint8_t* p, u8 v, size_t vl)   { __riscv_vse8_v_u8m2(p, v, vl); }
};

#endif // __riscv_v

/** @brief  Compute the L1 gradient magnitude of an image.
 *  @tparam LMUL    RVV register group multiple to use for the vectorized
 *                   path (see rvv_traits for the throughput/
 *                   register-pressure tradeoff at each value).
 *  @param  image   The input image metadata.
 *  @param  Gx      The x-component of the gradient.
 *  @param  Gy      The y-component of the gradient.
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
    // vl elements processed per iteration; scales with VLEN and the chosen
    // LMUL (see rvv_traits comments above for the per-LMUL
    // throughput/register-pressure tradeoff and VLEN scaling rule).
    using T = rvv_traits<LMUL>;

    // Running max accumulator for the reduction below, always an m1 vector
    // per the RVV reduction ISA regardless of the source vector's LMUL.
    vuint16m1_t v_max_magnitude = __riscv_vmv_v_x_u16m1(0, 1);
    size_t vl;

    // Pass 1: compute |Gx| + |Gy| per pixel, store to scratch, and track
    // the running maximum via reduction (needed to rescale into 0-255).
    for (uint32_t i = 0; i < count; i += vl)
    {
        vl = T::setvl_e16(count - i);

        // Load this batch's gradient components; see T::vle16_i for
        // LMUL/VLEN behavior.
        typename T::i16 vgx = T::vle16_i(Gx + i, vl);
        typename T::i16 vgy = T::vle16_i(Gy + i, vl);

        // Absolute value via negate + max; see T::vneg_i/T::vmax_i.
        typename T::i16 vgx_abs = T::vmax_i(vgx, T::vneg_i(vgx, vl), vl);
        typename T::i16 vgy_abs = T::vmax_i(vgy, T::vneg_i(vgy, vl), vl);

        // L1 magnitude = |Gx| + |Gy|; reinterpret to unsigned first since
        // both operands are non-negative. See T::reinterpret/T::vadd_u.
        typename T::u16 vmag = T::vadd_u(T::reinterpret(vgx_abs), T::reinterpret(vgy_abs), vl);

        // Spill the unnormalized magnitude for pass 2 to rescale.
        T::vse16_u(pointer_buffer.get() + i, vmag, vl);

        // Fold this batch's max into the running reduction accumulator.
        v_max_magnitude = T::vredmax(vmag, v_max_magnitude, vl);
    }

    // Extract the scalar max from lane 0 of the reduction result.
    max_value = static_cast<MagT>(__riscv_vmv_x_s_u16m1_u16(v_max_magnitude));

    if (max_value == 0)
    {
        std::fill(image.buffer.get(), image.buffer.get() + count, PixelT{0});
        return Status::E_OK;
    }

    // Pass 2: rescale each magnitude to [0,255] and narrow to 8-bit output.
    for (uint32_t i = 0; i < count; i += vl)
    {
        vl = T::setvl_e16(count - i);

        // Reload the magnitude computed in pass 1.
        typename T::u16 vmag = T::vle16_u(pointer_buffer.get() + i, vl);

        // scaled = vmag * 255 / max_value, computed via widening multiply
        // then divide to avoid overflow; see T::vwmulu_u/T::vdivu_u.
        typename T::u32 vscaled = T::vwmulu_u(vmag, 255, vl);
        vscaled = T::vdivu_u(vscaled, max_value, vl);

        // Narrow 32-bit -> 16-bit -> 8-bit back to pixel width; see
        // T::vncvt_u32_u16/T::vncvt_u16_u8.
        typename T::u16 vnarrow1 = T::vncvt_u32_u16(vscaled, vl);
        typename T::u8  vnarrow2 = T::vncvt_u16_u8(vnarrow1, vl);

        // Write the final 8-bit magnitude directly into the image buffer.
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
    if (!image.height || !image.width || !image.aligned_buffer_size || !image.buffer || !Gx || !Gy)
        return (image.buffer && Gx && Gy) ? Status::E_NOK : Status::E_INVAL_PTR;

    auto pointer_buffer = std::unique_ptr<MagT[], utils::memory::deleter>(
        static_cast<MagT*>(utils::memory::aligned_alloc(64, image.aligned_buffer_size * sizeof(MagT)))
    );
    if (!pointer_buffer) return Status::E_ALLOC_FAIL;

    MagT max_value = 0.0f;
    for (uint32_t i = 0; i < image.pixel_count; ++i)
    {
        float x = static_cast<float>(Gx[i]);
        float y = static_cast<float>(Gy[i]);
        MagT value = std::sqrt(x * x + y * y);

        pointer_buffer[i] = value;
        if (value > max_value) max_value = value;
    }

    float scale = (max_value > 0.0f) ? 255.0f / max_value : 0.0f;

    for (uint32_t i = 0; i < image.pixel_count; ++i)
        image.buffer[i] = static_cast<PixelT>(pointer_buffer[i] * scale);

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
    if (!image.height || !image.width || !image.aligned_buffer_size || !image.buffer || !Gx || !Gy)
        return (image.buffer && Gx && Gy) ? Status::E_NOK : Status::E_INVAL_PTR;

    uint8_t angle;
    for (uint32_t i = 0; i < image.pixel_count; ++i)
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
                angle = 135;
            else
                angle = 45;
        }
        else // < 0.5 (Horizontal range)
        {
            angle = 0;
        }
        image.buffer[i] = angle;
    }

    return Status::E_OK;
}

} // namespace processing::gradient

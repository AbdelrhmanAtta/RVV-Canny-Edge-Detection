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
#include <type_traits>

namespace processing::gaussian
{
/**
 * @brief   5x5 Gaussian blur kernel and its 1D separable counterpart.
 *          The 5x5 kernel is normalized by a factor of 273 and sigma approximately 1.
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

/**
 * @brief   Specializations of RISC-V Vector intrinsics for different LMUL settings.
 *          The implementations include loading, storing, widening, and narrowing operations
 *          tailored for 8-bit pixel data and 32-bit accumulators.
 *
 *          LMUL choice: each specialization groups LMUL register multiples into one
 *          register group, so a larger LMUL processes more elements per instruction
 *          at the cost of using more of the register file (fewer groups available
 *          for the compiler to schedule/overlap). LMUL=1 is the safe default with
 *          the most register groups available; LMUL=2/4 trade register pressure for
 *          higher per-instruction throughput. LMUL=4 is capped to an m8 accumulator
 *          (see specialization below) since u8m4 -> u16m8 -> u32m16 is not a valid
 *          widening chain (max LMUL is 8).
 *
 *          VLEN dependency: none of these intrinsics hardcode an element count.
 *          `vsetvl_e8mX`/`vsetvl_e32mX` return the actual vector length (vl) the
 *          hardware can process this iteration, which scales linearly with VLEN
 *          (e.g. doubling VLEN doubles vl for a fixed LMUL). All loop strides use
 *          this runtime vl, so the same binary runs correctly — just with fewer
 *          loop iterations — on any VLEN-compliant RVV implementation without
 *          recompilation.
 */
template <int LMUL> struct gaussian_rvv_traits;

template <> struct gaussian_rvv_traits<1>
{
    using u8  = vuint8m1_t;
    using u16 = vuint16m2_t;
    using u32 = vuint32m4_t;

    // Computes the vector length (vl) for 8-bit elements at LMUL=1.
    // LMUL=1: one register group = 1 physical vector register, the baseline
    // grouping with the most independent register groups available to the
    // compiler. If VLEN were larger, vl would grow proportionally and fewer
    // outer-loop iterations would be needed to cover the same row; if VLEN
    // were smaller, vl shrinks and more iterations run, but no source change
    // is required either way.
    static size_t setvl_e8 (size_t n) { return __riscv_vsetvl_e8m1(n);  }
    // Computes the vector length (vl) for 32-bit elements at LMUL=4 (the
    // accumulator group for this trait set). Used in the separable vertical
    // pass where the working values are already 32-bit. Same VLEN scaling
    // as above: vl tracks VLEN/32 * LMUL at runtime.
    static size_t setvl_e32(size_t n) { return __riscv_vsetvl_e32m4(n); }

    // Loads vl 8-bit pixels from memory into an m1 register group.
    // LMUL=1 chosen to match the pixel-group width used throughout this
    // specialization. A wider VLEN means each load pulls in more pixels
    // per instruction (vl scales with VLEN), so fewer loads are issued
    // for the same row width.
    static u8  vle8 (const uint8_t* p, size_t vl)  { return __riscv_vle8_v_u8m1(p, vl);     }
    // Initializes a 32-bit accumulator register group to zero at LMUL=4,
    // matching the widened accumulator type after two widening steps from
    // u8m1. LMUL=4 here is forced by the widening chain (m1 -> m2 -> m4),
    // not chosen independently. A different VLEN changes how many lanes
    // this zero-fill covers but not the LMUL relationship.
    static u32 vmv0 (size_t vl)                    { return __riscv_vmv_v_x_u32m4(0, vl);   }
    // Zero-extends 8-bit pixels (m1) to 16-bit (m2) — the first widening
    // step toward the 32-bit accumulator. LMUL doubles here because each
    // 16-bit element needs twice the storage of its 8-bit source while
    // keeping the same element count, so the register group must double
    // in width. VLEN changes do not affect this relationship, only the
    // absolute number of lanes processed per call.
    static u16 vzext(u8 v, size_t vl)              { return __riscv_vzext_vf2_u16m2(v, vl); }

    // Widening multiply-accumulate: multiplies each 16-bit pixel (m2) by an
    // 8-bit scalar kernel coefficient and accumulates into the 32-bit (m4)
    // sum, in one instruction. LMUL=4 on the result is required because the
    // product of a 16-bit value and an 8-bit scalar needs 32-bit headroom.
    // This is the core multiply-accumulate of the convolution; a larger
    // VLEN lets one call cover more output pixels at once (vl scales),
    // reducing the number of times this 5x5 (or 1x5) tap loop executes.
    static u32 vwmaccu(u32 acc, uint8_t s, u16 v, size_t vl)
    { return __riscv_vwmaccu_vx_u32m4(acc, s, v, vl); }

    // Non-widening multiply-accumulate used in the separable vertical pass,
    // where inputs are already 32-bit intermediate sums (m4) and the kernel
    // tap is an 8-bit scalar. LMUL stays at 4 since no widening occurs here.
    // VLEN scaling: same as above, more lanes per call with larger VLEN.
    static u32 vmacc(u32 acc, uint8_t s, u32 v, size_t vl)
    { return __riscv_vmacc_vx_u32m4(acc, s, v, vl); }

    // Multiplies the accumulated sum by a fixed-point reciprocal constant
    // (240), part of the shift-multiply approximation of division by 273.
    // Operates on the m4 accumulator group to match vwmaccu's output LMUL.
    // VLEN only affects how many pixels' sums are scaled per instruction.
    static u32 vmul(u32 v, uint32_t s, size_t vl) { return __riscv_vmul_vx_u32m4(v, s, vl); }
    // Logical right-shift by 16 to complete the fixed-point division
    // approximation (sum * 240 >> 16 ≈ sum / 273). Stays at m4 to match
    // the accumulator. VLEN does not change the math, only throughput.
    static u32 vsrl(u32 v, uint32_t s, size_t vl) { return __riscv_vsrl_vx_u32m4(v, s, vl); }

    // Clamps the 32-bit (m4) result to [0,255] then narrows in two steps:
    // 32-bit -> 16-bit (m4 -> m2) -> 8-bit (m2 -> m1), the inverse of the
    // widening chain used to build the accumulator. LMUL halves at each
    // narrowing step because each output element needs half the storage
    // of its input. VLEN changes the lane count per call but not this
    // halving relationship between stages.
    static u8 narrow_u32_to_u8(u32 v, size_t vl)
    {
        v = __riscv_vminu_vx_u32m4(v, 255, vl);
        return __riscv_vncvt_x_x_w_u8m1(__riscv_vncvt_x_x_w_u16m2(v, vl), vl);
    }

    // Stores vl 8-bit results (m1) to memory — the final write of a blurred
    // output row segment. Matches the pixel LMUL used for loads. Larger
    // VLEN means more output pixels committed per store instruction.
    static void vse8 (uint8_t*  p, u8  v, size_t vl) { __riscv_vse8_v_u8m1(p, v, vl);   }
    // Stores vl 32-bit intermediate sums (m4) to memory — used to spill the
    // horizontal-pass accumulator in the separable path. Matches the
    // accumulator LMUL throughout this trait. VLEN scales lanes per store.
    static void vse32(uint32_t* p, u32 v, size_t vl) { __riscv_vse32_v_u32m4(p, v, vl); }

    // Loads vl 32-bit intermediate values (m4) back from memory for the
    // vertical pass of the separable filter. Matches the accumulator LMUL
    // so the result feeds directly into vmacc above without conversion.
    // VLEN scaling is the same as the other 32-bit ops.
    static u32 vle32(const uint32_t* p, size_t vl) { return __riscv_vle32_v_u32m4(p, vl); }
};

template <> struct gaussian_rvv_traits<2>
{
    using u8  = vuint8m2_t;
    using u16 = vuint16m4_t;
    using u32 = vuint32m8_t;

    // Computes vl for 8-bit elements at LMUL=2: double the register-group
    // width of LMUL=1, so each call covers roughly twice as many pixels
    // per instruction at the cost of using more of the vector register
    // file (fewer concurrent groups available). VLEN scaling is linear,
    // same as the LMUL=1 case, just with a larger absolute vl per call.
    static size_t setvl_e8 (size_t n) { return __riscv_vsetvl_e8m2(n);  }
    // Computes vl for 32-bit elements at LMUL=8, the accumulator group
    // size after two widening steps from m2. This is the maximum legal
    // LMUL, so this trait uses the full register file width available
    // for grouped operations. A different VLEN changes vl proportionally
    // without requiring any LMUL change.
    static size_t setvl_e32(size_t n) { return __riscv_vsetvl_e32m8(n); }

    // Loads vl 8-bit pixels into an m2 register group — same role as the
    // LMUL=1 version but processing twice the elements per call. Chosen
    // here to amortize instruction overhead across more pixels when more
    // register file budget can be spent. VLEN scaling: more pixels per
    // load as VLEN grows, same as any LMUL.
    static u8  vle8 (const uint8_t* p, size_t vl)  { return __riscv_vle8_v_u8m2(p, vl);     }
    // Zero-initializes the m8 accumulator, forced by the widening chain
    // (m2 -> m4 -> m8) rather than chosen independently. VLEN affects lane
    // count, not the LMUL relationship.
    static u32 vmv0 (size_t vl)                    { return __riscv_vmv_v_x_u32m8(0, vl);   }
    // Zero-extends 8-bit (m2) to 16-bit (m4), the first widening step.
    // LMUL doubles for the same reason as in the LMUL=1 trait: doubled
    // element width needs doubled register-group width at constant
    // element count. VLEN scaling is unaffected by this relationship.
    static u16 vzext(u8 v, size_t vl)              { return __riscv_vzext_vf2_u16m4(v, vl); }

    // Widening multiply-accumulate from 16-bit (m4) pixels times an 8-bit
    // scalar tap into the 32-bit (m8) accumulator. This is the workhorse
    // op of the convolution; LMUL=8 here means each call covers the
    // maximum number of output lanes the ISA allows in one register
    // group, maximizing throughput per instruction issued. Larger VLEN
    // multiplies this benefit further since vl also scales with VLEN.
    static u32 vwmaccu(u32 acc, uint8_t s, u16 v, size_t vl)
    { return __riscv_vwmaccu_vx_u32m8(acc, s, v, vl); }

    // Non-widening multiply-accumulate on already-32-bit (m8) intermediate
    // sums, used in the separable vertical pass. Same LMUL as the
    // accumulator group; VLEN scaling identical to vwmaccu above.
    static u32 vmacc(u32 acc, uint8_t s, u32 v, size_t vl)
    { return __riscv_vmacc_vx_u32m8(acc, s, v, vl); }

    // Fixed-point multiply by 240 (part of the /273 approximation) on the
    // m8 accumulator. VLEN only changes how many sums are scaled at once.
    static u32 vmul(u32 v, uint32_t s, size_t vl) { return __riscv_vmul_vx_u32m8(v, s, vl); }
    // Logical right-shift by 16 to finish the fixed-point division
    // approximation, on the m8 accumulator. VLEN affects lane count only.
    static u32 vsrl(u32 v, uint32_t s, size_t vl) { return __riscv_vsrl_vx_u32m8(v, s, vl); }

    // Clamps to [0,255] then narrows m8 -> m4 -> m2, mirroring the
    // widening chain in reverse. Each narrowing step halves LMUL because
    // each output element is half the byte-width of its input at the
    // same element count. VLEN changes lane count per call only.
    static u8 narrow_u32_to_u8(u32 v, size_t vl)
    {
        v = __riscv_vminu_vx_u32m8(v, 255, vl);
        return __riscv_vncvt_x_x_w_u8m2(__riscv_vncvt_x_x_w_u16m4(v, vl), vl);
    }

    // Stores vl 8-bit results (m2) — final output write, matching the
    // pixel LMUL used for loads in this trait. VLEN scales lanes per store.
    static void vse8 (uint8_t*  p, u8  v, size_t vl) { __riscv_vse8_v_u8m2(p, v, vl);   }
    // Stores vl 32-bit intermediate sums (m8) for the separable path's
    // horizontal-pass spill. Matches accumulator LMUL. VLEN scales lanes.
    static void vse32(uint32_t* p, u32 v, size_t vl) { __riscv_vse32_v_u32m8(p, v, vl); }

    // Loads vl 32-bit intermediates (m8) back for the vertical pass,
    // matching the accumulator LMUL so it feeds vmacc without conversion.
    // VLEN scaling identical to the other 32-bit ops in this trait.
    static u32 vle32(const uint32_t* p, size_t vl) { return __riscv_vle32_v_u32m8(p, vl); }
};

template <> struct gaussian_rvv_traits<4>
{
    // u8m4 would normally widen to u16m8 -> u32m16, but m16 exceeds the
    // maximum legal LMUL (8). The accumulator is therefore capped at m8
    // by keeping the pixel/intermediate types at the same width as the
    // LMUL=2 specialization; LMUL=4 in this trait's name refers to the
    // caller-facing tuning knob, not every intermediate register group.
    using u8  = vuint8m2_t;
    using u16 = vuint16m4_t;
    using u32 = vuint32m8_t;

    // Computes vl for 8-bit elements at the m2 group width (capped, see
    // note above — true LMUL=4 pixel groups aren't usable here without
    // exceeding m16 on the widened accumulator). VLEN scaling: linear,
    // same as any LMUL — larger VLEN yields proportionally larger vl.
    static size_t setvl_e8 (size_t n) { return __riscv_vsetvl_e8m2(n);  }
    // Computes vl for 32-bit elements at m8, the maximum legal accumulator
    // group, reached via the capped widening chain described above.
    static size_t setvl_e32(size_t n) { return __riscv_vsetvl_e32m8(n); }

    // Loads vl 8-bit pixels (m2, capped). Same role as the other traits'
    // pixel load; VLEN scales lanes per call identically.
    static u8  vle8 (const uint8_t* p, size_t vl)  { return __riscv_vle8_v_u8m2(p, vl);     }
    // Zero-initializes the m8 accumulator — the ceiling LMUL for 32-bit
    // groups, forced by the capped widening chain above, not chosen
    // independently. VLEN affects lane count, not this LMUL ceiling.
    static u32 vmv0 (size_t vl)                    { return __riscv_vmv_v_x_u32m8(0, vl);   }
    // Zero-extends 8-bit (m2) to 16-bit (m4) pixels, first widening step,
    // identical reasoning to the LMUL=2 trait (doubling element width
    // doubles register-group width). VLEN scaling unaffected.
    static u16 vzext(u8 v, size_t vl)              { return __riscv_vzext_vf2_u16m4(v, vl); }

    // Widening multiply-accumulate from 16-bit (m4) pixels into the 32-bit
    // (m8) accumulator — same operation as LMUL=2's vwmaccu since the
    // widening chain is capped to the same register-group sizes; the
    // LMUL=4 label on this trait does not change this particular
    // instruction's group width. VLEN scaling: larger VLEN still yields
    // more lanes processed per call, exactly as in the other traits.
    static u32 vwmaccu(u32 acc, uint8_t s, u16 v, size_t vl)
    { return __riscv_vwmaccu_vx_u32m8(acc, s, v, vl); }

    // Non-widening multiply-accumulate on 32-bit (m8) intermediates for
    // the separable vertical pass — identical group width to LMUL=2's
    // version for the reason above. VLEN scaling unchanged.
    static u32 vmacc(u32 acc, uint8_t s, u32 v, size_t vl)
    { return __riscv_vmacc_vx_u32m8(acc, s, v, vl); }

    // Fixed-point multiply by 240 on the m8 accumulator, part of the
    // /273 approximation. VLEN affects lane count only.
    static u32 vmul(u32 v, uint32_t s, size_t vl) { return __riscv_vmul_vx_u32m8(v, s, vl); }
    // Logical right-shift by 16 to finish the division approximation,
    // m8 accumulator. VLEN affects lane count only.
    static u32 vsrl(u32 v, uint32_t s, size_t vl) { return __riscv_vsrl_vx_u32m8(v, s, vl); }

    // Clamps to [0,255] then narrows m8 -> m4 -> m2, mirroring the capped
    // widening chain in reverse, identical group widths to the LMUL=2
    // trait. VLEN changes lane count per call only.
    static u8 narrow_u32_to_u8(u32 v, size_t vl)
    {
        v = __riscv_vminu_vx_u32m8(v, 255, vl);
        return __riscv_vncvt_x_x_w_u8m2(__riscv_vncvt_x_x_w_u16m4(v, vl), vl);
    }

    // Stores vl 8-bit results (m2, capped pixel width). VLEN scales lanes.
    static void vse8 (uint8_t*  p, u8  v, size_t vl) { __riscv_vse8_v_u8m2(p, v, vl);   }
    // Stores vl 32-bit intermediates (m8) for the separable path's
    // horizontal-pass spill. VLEN scales lanes per store.
    static void vse32(uint32_t* p, u32 v, size_t vl) { __riscv_vse32_v_u32m8(p, v, vl); }

    // Loads vl 32-bit intermediates (m8) back for the vertical pass.
    // VLEN scaling identical to the other 32-bit ops in this trait.
    static u32 vle32(const uint32_t* p, size_t vl) { return __riscv_vle32_v_u32m8(p, vl); }
};

#endif // __riscv_v

/**
 * @brief   Applies a 5x5 Gaussian blur to the input image using direct convolution.
 * @param   image The input image metadata, which will be modified in-place with the blurred output.
 * @return  Status indicating success or failure of the operation.
 */
template <typename PixelT = uint8_t, typename AccumulatorT = uint32_t, int LMUL = 2>
[[nodiscard]] Status spatial_5x5(image::io::metadata_t<PixelT>& image)
{
    if (!image.height || !image.width || !image.buffer)
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;

    const int32_t width  = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);
    const int32_t kernel_radius = 2;

    const uint32_t pw = width  + 2 * kernel_radius;
    const uint32_t ph = height + 2 * kernel_radius;

    // Zero-pad the input so the kernel never reads out of bounds
    auto padded_input = std::make_unique<PixelT[]>(pw * ph);
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), PixelT{0});
    for (int32_t y = 0; y < height; ++y)
        std::copy_n(&image.buffer[y * width], width,
                    &padded_input[(y + kernel_radius) * pw + kernel_radius]);

    auto raw_out = static_cast<PixelT*>(
        utils::memory::aligned_alloc(64, image.aligned_buffer_size));
    if (!raw_out) return Status::E_ALLOC_FAIL;
    std::unique_ptr<PixelT[], utils::memory::deleter> output_buffer(raw_out);

    // Fixed-point approximation of division by 273 (kernel normalization factor)
    constexpr uint32_t shift      = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; )
        {
#if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                // Vectorized path: accumulate the full 5x5 window per output lane.
                // T::setvl_e8 returns the runtime vl: how many output pixels this
                // iteration will produce, derived from remaining row width and
                // VLEN/LMUL. A larger VLEN means more pixels processed per outer
                // loop pass and fewer total iterations to cover the row; the
                // template parameter LMUL controls how many register groups are
                // used per pixel batch (see trait comments above for the
                // throughput/register-pressure tradeoff at each LMUL value).
                using T = gaussian_rvv_traits<LMUL>;
                const size_t vl  = T::setvl_e8(width - x);
                auto sum = T::vmv0(vl);

                for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky)
                {
                    const uint32_t row_off    = (y + kernel_radius + ky) * pw;
                    const uint32_t kernel_off = (ky + kernel_radius) * 5;
                    for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx)
                    {
                        // Load vl contiguous pixels for this tap offset, widen
                        // to 16-bit, then widening multiply-accumulate against
                        // the scalar kernel coefficient — see trait comments
                        // for per-LMUL/VLEN behavior of each of these three ops.
                        auto px  = T::vle8(&padded_input[row_off + (x + kernel_radius + kx)], vl);
                        auto px16 = T::vzext(px, vl);
                        sum = T::vwmaccu(sum,
                                         GAUSSIAN_5x5_DATA[kernel_off + (kx + kernel_radius)],
                                         px16, vl);
                    }
                }

                // sum * 240 / 65536 ≈ sum / 273 (fixed-point normalization)
                sum = T::vmul(sum, 240, vl);
                sum = T::vsrl(sum,  16, vl);
                T::vse8(&output_buffer[y * width + x], T::narrow_u32_to_u8(sum, vl), vl);
                x += static_cast<int32_t>(vl);
                continue;
            }
#endif
            // Scalar fallback: same 5x5 accumulation, one pixel at a time
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
template <typename PixelT = uint8_t, typename AccumulatorT = uint32_t, int LMUL = 2>
[[nodiscard]] Status separable_5x5(image::io::metadata_t<PixelT>& image)
{
    if (!image.height || !image.width || !image.buffer)
        return image.buffer ? Status::E_NOK : Status::E_INVAL_PTR;

    const int32_t width  = static_cast<int32_t>(image.width);
    const int32_t height = static_cast<int32_t>(image.height);
    const int32_t kernel_radius = 2;

    const uint32_t pw = width  + 2 * kernel_radius;
    const uint32_t ph = height + 2 * kernel_radius;

    // Zero-pad the input so the kernel never reads out of bounds
    auto padded_input = std::make_unique<PixelT[]>(pw * ph);
    std::fill(padded_input.get(), padded_input.get() + (pw * ph), PixelT{0});
    for (int32_t y = 0; y < height; ++y)
        std::copy_n(&image.buffer[y * width], width,
                    &padded_input[(y + kernel_radius) * pw + kernel_radius]);

    // Pass 1: horizontal 1D convolution into an intermediate accumulator buffer
    auto inter_buffer = std::make_unique<AccumulatorT[]>(pw * ph);

    for (uint32_t y = 0; y < ph; ++y)
    {
        for (uint32_t x = kernel_radius; x < pw - kernel_radius; )
        {
            const uint32_t row_off = y * pw;

#if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                // vl pixels processed this iteration; scales with VLEN and the
                // chosen LMUL exactly as in spatial_5x5 (see trait comments).
                using T = gaussian_rvv_traits<LMUL>;
                const size_t vl  = T::setvl_e8(pw - kernel_radius - x);
                auto sum = T::vmv0(vl);

                for (int32_t kx = -kernel_radius; kx <= kernel_radius; ++kx)
                {
                    // Load, widen, and widening-multiply-accumulate against the
                    // 1D kernel tap — same three ops and LMUL/VLEN behavior as
                    // the inner tap loop in spatial_5x5, just over one row only.
                    auto px   = T::vle8(&padded_input[row_off + (x + kx)], vl);
                    auto px16 = T::vzext(px, vl);
                    sum = T::vwmaccu(sum, GAUSSIAN_5x5_1D_DATA[kx + kernel_radius], px16, vl);
                }

                // Spill the unnormalized 32-bit horizontal sum for pass 2 to consume
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

    // Fixed-point approximation of division by 273 (kernel normalization factor)
    constexpr uint32_t shift      = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    // Pass 2: vertical 1D convolution over the intermediate buffer, normalize, write out
    for (int32_t y = 0; y < height; ++y)
    {
        for (int32_t x = 0; x < width; )
        {
            const uint32_t col_x = x + kernel_radius;

#if defined(__riscv_v) && __riscv_v >= 1000000
            if constexpr (std::is_same_v<uint8_t, PixelT>)
            {
                // Note: vl here is computed via setvl_e32 (32-bit element vl),
                // not setvl_e8, since pass 2 operates directly on the 32-bit
                // intermediate sums from pass 1 — no widening needed. vl still
                // scales with VLEN and LMUL the same way as in the trait
                // comments; using the e32 variant just reflects the element
                // width already in play at this stage.
                using T = gaussian_rvv_traits<LMUL>;
                const size_t vl  = T::setvl_e32(width - x);
                auto sum = T::vmv0(vl);

                for (int32_t ky = -kernel_radius; ky <= kernel_radius; ++ky)
                {
                    // Load vl 32-bit intermediate sums for this row offset and
                    // multiply-accumulate (non-widening, since both operands
                    // are already 32-bit) against the 1D kernel tap.
                    auto iv = T::vle32(reinterpret_cast<uint32_t*>(
                        &inter_buffer[(y + kernel_radius + ky) * pw + col_x]), vl);
                    sum = T::vmacc(sum, GAUSSIAN_5x5_1D_DATA[ky + kernel_radius], iv, vl);
                }

                // sum * 240 / 65536 ≈ sum / 273 (fixed-point normalization)
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

} // namespace processing::gaussian

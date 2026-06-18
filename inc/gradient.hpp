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
/** @brief  Compute the L1 gradient magnitude of an image.
 *  @param  image The input image metadata.
 *  @param  Gx The x-component of the gradient.
 *  @param  Gy The y-component of the gradient.
 *  @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t, typename GradientT = int16_t, typename MagT = uint16_t>
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
    {
        size_t vl      = __riscv_vsetvl_e16m2(count);
        auto   vmax_acc = __riscv_vmv_v_x_u16m2(0, vl);

        uint32_t i = 0;
        while (i < count)
        {
            vl = __riscv_vsetvl_e16m2(count - i);

            vint16m2_t vgx = __riscv_vle16_v_i16m2(Gx + i, vl);
            vint16m2_t vgy = __riscv_vle16_v_i16m2(Gy + i, vl);

            vint16m2_t vgx_abs = __riscv_vmax_vv_i16m2(vgx, __riscv_vneg_v_i16m2(vgx, vl), vl);
            vint16m2_t vgy_abs = __riscv_vmax_vv_i16m2(vgy, __riscv_vneg_v_i16m2(vgy, vl), vl);

            vint16m2_t  vmag_s = __riscv_vadd_vv_i16m2(vgx_abs, vgy_abs, vl);
            vuint16m2_t vmag   = __riscv_vreinterpret_v_i16m2_u16m2(vmag_s);

            __riscv_vse16_v_u16m2(pointer_buffer.get() + i, vmag, vl);

            vuint16m1_t vmax_scalar = __riscv_vmv_v_x_u16m1(0, 1);
            vmax_scalar = __riscv_vredmax_vs_u16m2_u16m1(vmag, vmax_scalar, vl);

            MagT chunk_max = static_cast<MagT>(__riscv_vmv_x_s_u16m1_u16(vmax_scalar));
            if (chunk_max > max_value) max_value = chunk_max;

            i += static_cast<uint32_t>(vl);
        }
    }

    {
        float scale = (max_value > 0) ? 255.0f / static_cast<float>(max_value) : 0.0f;

        uint32_t i = 0;
        while (i < count)
        {
            size_t vl = __riscv_vsetvl_e16m2(count - i);

            vuint16m2_t vmag = __riscv_vle16_v_u16m2(pointer_buffer.get() + i, vl);

            vuint32m4_t vmag32  = __riscv_vzext_vf2_u32m4(vmag, vl);
            vfloat32m4_t vmagf  = __riscv_vfcvt_f_xu_v_f32m4(vmag32, vl);
            vfloat32m4_t vscaled = __riscv_vfmul_vf_f32m4(vmagf, scale, vl);
            vuint32m4_t  vout32 = __riscv_vfcvt_xu_f_v_u32m4(vscaled, vl);

            vuint16m2_t vout16 = __riscv_vnclipu_wx_u16m2(vout32, 0, __RISCV_VXRM_RNU, vl);
            vuint8m1_t  vout8  = __riscv_vnclipu_wx_u8m1(vout16, 0, __RISCV_VXRM_RNU, vl);

            __riscv_vse8_v_u8m1(reinterpret_cast<uint8_t*>(image.buffer.get()) + i, vout8, vl);

            i += static_cast<uint32_t>(vl);
        }
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

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
/** @brief Compute the L1 gradient magnitude of an image.
 * @param image The input image metadata.
 * @param Gx The x-component of the gradient.
 * @param Gy The y-component of the gradient.
 * @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t, typename GradientT = int16_t, typename MagT = uint16_t>
[[nodiscard]] Status l1(const image::io::metadata_t<PixelT>& image,
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

    MagT max_value = 0;
    for(uint32_t i = 0; i < image.pixel_count; ++i)
    {
        MagT value = static_cast<MagT>(std::abs(Gx[i]) + std::abs(Gy[i]));
        pointer_buffer[i] = value;
        if(value > max_value)
        {
            max_value = value;
        }
    }

    float scale = (max_value > 0) ? 255.0f / static_cast<float>(max_value) : 0.0f;

    for(uint32_t i = 0; i < image.pixel_count; ++i) 
    {
        image.buffer[i] = static_cast<PixelT>(pointer_buffer[i] * scale);
    }

    return Status::E_OK;
}

/** @brief Compute the L2 gradient magnitude of an image.
 * @param image The input image metadata.
 * @param Gx The x-component of the gradient.
 * @param Gy The y-component of the gradient.
 * @return Status indicating success or failure.
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

/** @brief Compute the direction of the gradient of an image.
 * @param image The input image metadata.
 * @param Gx The x-component of the gradient.
 * @param Gy The y-component of the gradient.
 * @return Status indicating success or failure.
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

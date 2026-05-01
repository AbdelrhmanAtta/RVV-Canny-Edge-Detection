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
template <typename PixelT = uint8_t, typename GradientT = int16_t, typename MagT = uint16_t>
[[nodiscard]] Status l1(const image::io::metadata_t<PixelT>& image,
                        const GradientT* __restrict Gx,
                        const int16_t* __restrict Gy)
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

    GradientT x;
    GradientT y;
    uint16_t abs_x;
    uint16_t abs_y;
    uint8_t angle;
    for(uint32_t i = 0; i < image.pixel_count; ++i)
    {
        x = std::abs(Gx[i]);
        y = std::abs(Gy[i]);
        abs_x = static_cast<uint16_t>(x);
        abs_y = static_cast<uint16_t>(y);

        if((abs_y * 5) > (abs_x * 12))
        {
            angle = 90;
        }
        else if((abs_y * 5) > (abs_x * 2))
        {
            angle = 0;
        }
        else
        {
            if(((x > 0) && (y > 0)) || ((x < 0) && (y < 0)))
            {
                angle = 45;
            }
            else
            {
                angle = 135;
            }
        }
        image.buffer[i] = angle;
    }

    return Status::E_OK;
}

} // namespace processing

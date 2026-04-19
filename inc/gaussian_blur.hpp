#pragma once

#include "std_types.hpp"
#include <iostream>
#include <string>
#include <cstdint>
#include <cstdlib>

/**
 * @file gaussian_blur.hpp
 * @brief Image processing utilities for applying Gaussian blur.
 *
 * This module provides functions for blurring images using a Gaussian filter,
 * typically used for smoothing and noise reduction.
 */

inline constexpr uint16_t GAUSSIAN_KERNEL_2D[5][5] = 
{
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};

template <typename PixelT = uint8_t, typename KernelT = uint16_t, typename AccumulatorT = uint32_t>
Status gaussian_blur_convolution(PixelT* imageInput, PixelT* imageOutput, uint32_t imageWidth, uint32_t imageHeight)
{
    Status stat = Status::E_NOK;

    if(!imageInput || !imageOutput) return Status::E_INVAL_PTR;
    
    const int32_t kRadius = 2;
    AccumulatorT kSum = 273;

    for(uint32_t y=0; y<imageHeight; ++y)
    {
        for(uint32_t x=0; x<imageWidth; ++x)
        {
            AccumulatorT convolutionSum = 0;

            for(int32_t ky=-kRadius; ky<=kRadius; ++ky)
            {
                for(int32_t kx=-kRadius; kx<=kRadius; ++kx)
                {
                    int32_t nx = kx + x;
                    int32_t ny = ky + y;
                    if((nx>=0 && nx<imageWidth) && (ny>=0 && ny<imageHeight)) 
                        convolutionSum += static_cast<AccumulatorT>(imageInput[ny*imageWidth+nx])*GAUSSIAN_KERNEL_2D[ky+kRadius][kx+kRadius];
                }
            }
            convolutionSum /= kSum;

            /* Clamp */
            if(convolutionSum<0) convolutionSum=0;
            if(convolutionSum>255) convolutionSum=255;

            imageOutput[y*imageWidth+x] = static_cast<PixelT>(convolutionSum);
        }
    }

    stat = Status::E_OK;
    return stat;
}

/**
 * @file    kernels.hpp
 * @brief   Kernels for image processing operations.
 * 
 * This module defines various convolution kernels of different sizes, such Gaussian blur kernels
 * and Sobel Gradient kernels.
 */

#pragma once

#include "std_types.hpp"

namespace processing::convolution
{
/**
 * @brief   3x3 Gaussian blur kernel and its 1D separable counterpart.
 * The 3x3 kernel is normalized by a factor of 16 and sigma approximately 0.85.
 */
inline constexpr uint8_t GAUSSIAN_3x3_DATA[] =
{
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
};
inline constexpr uint8_t GAUSSIAN_3x3_1D_DATA[3] = {1, 2, 1};
inline constexpr kernel_t<uint8_t, uint16_t> GAUSSIAN_3x3 = 
{
    .data = GAUSSIAN_3x3_DATA,
    .width = 3,
    .height = 3,
    .sum = 16
};
inline constexpr kernel_t<uint8_t, uint16_t> GAUSSIAN_3x3_1D = 
{
    .data = GAUSSIAN_3x3_1D_DATA,
    .width = 3,
    .height = 1,
    .sum = 4
};

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
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_5x5 = 
{
    .data = GAUSSIAN_5x5_DATA,
    .width = 5,
    .height = 5,
    .sum = 273
};  
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_5x5_1D = 
{
    .data = GAUSSIAN_5x5_1D_DATA,
    .width = 5,
    .height = 1,
    .sum = 17
};

/**
 * @brief   7x7 Gaussian blur kernel and its 1D separable counterpart.
 * The 7x7 kernel is normalized by a factor of 1111 and sigma approximately 1.4.
 */
inline constexpr uint8_t GAUSSIAN_7x7_DATA[] = 
{
    0,  0,  1,   2,  1,  0, 0,
    0,  3, 13,  22, 13,  3, 0,
    1, 13, 59,  97, 59, 13, 1,
    2, 22, 97, 159, 97, 22, 2,
    1, 13, 59,  97, 59, 13, 1,
    0,  3, 13,  22, 13,  3, 0,
    0,  0,  1,   2,  1,  0, 0
};
inline constexpr uint8_t GAUSSIAN_7x7_1D_DATA[] = {1, 3, 7, 11, 7, 3, 1};
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_7x7 = 
{
    .data = GAUSSIAN_7x7_DATA,
    .width = 7,
    .height = 7,
    .sum = 1111
};
inline constexpr kernel_t<uint8_t, uint32_t> GAUSSIAN_7x7_1D = 
{
    .data = GAUSSIAN_7x7_1D_DATA,
    .width = 7,
    .height = 1,
    .sum = 33
};

} // namespace processing::convolution

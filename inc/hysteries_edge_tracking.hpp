/**
 * @file    hysteresis_edge_tracking.hpp
 * @brief   RVV-optimized Hysteresis Edge Tracking for Canny Edge Detection.
 */

#pragma once

#include "std_types.hpp"
#include <vector>
#include <riscv_vector.h>

namespace processing::canny
{

/** * @brief Perform Hysteresis Double Thresholding and Edge Tracking.
 * @param image The input image metadata (usually output of Non-Maximum Suppression).
 * @param low_threshold The lower bound threshold for weak edges.
 * @param high_threshold The upper bound threshold for strong edges.
 * @return Status indicating success or failure.
 */
template <typename PixelT = uint8_t>
[[nodiscard]] Status hysteresis(const image::io::metadata_t<PixelT>& image, 
                                PixelT low_threshold, 
                                PixelT high_threshold)
{
    if(!image.height || !image.width || !image.aligned_buffer_size || !image.buffer)
    {
        return Status::E_INVAL_PTR;
    }

    const size_t total_pixels = image.pixel_count;
    PixelT* __restrict ptr = image.buffer.get();

    // ========================================================================
    // PASS 1: Vectorized Double Thresholding (RVV)
    // Map pixels >= high_threshold -> 255 (Strong)
    // Map pixels >= low_threshold && < high_threshold -> 128 (Weak)
    // Map pixels < low_threshold -> 0 (Suppressed)
    // ========================================================================
    size_t vl;
    for (size_t i = 0; i < total_pixels; i += vl) 
    {
        // Request max vector length for 8-register grouping (LMUL=8)
        vl = __riscv_vsetvl_e8m8(total_pixels - i);
        
        // Load chunk of pixels
        vuint8m8_t v_src = __riscv_vle8_v_u8m8(ptr + i, vl);

        // Create masks for weak and strong thresholds
        vbool1_t m_strong = __riscv_vmsgeu_vx_u8m8_b1(v_src, high_threshold, vl);
        vbool1_t m_weak   = __riscv_vmsgeu_vx_u8m8_b1(v_src, low_threshold, vl);

        // Initialize destination vector with 0
        vuint8m8_t v_dst = __riscv_vmv_v_x_u8m8(0, vl);
        
        // Merge weak pixels (128) using m_weak mask
        v_dst = __riscv_vmerge_vxm_u8m8(v_dst, 128, m_weak, vl);
        // Overwrite strong pixels (255) using m_strong mask
        v_dst = __riscv_vmerge_vxm_u8m8(v_dst, 255, m_strong, vl);

        // Store back to buffer
        __riscv_vse8_v_u8m8(ptr + i, v_dst, vl);
    }

    // ========================================================================
    // PASS 2: Scalar Edge Tracking (Depth First Search)
    // ========================================================================
    std::vector<uint32_t> stack;
    // Pre-allocate to avoid reallocations (heuristic: ~5% of image might be strong edges)
    stack.reserve(total_pixels / 20); 

    // Find initial strong pixels (ignoring boundaries to simplify 8-neighbor checks)
    for (uint32_t y = 1; y < image.height - 1; ++y) 
    {
        for (uint32_t x = 1; x < image.width - 1; ++x) 
        {
            uint32_t idx = y * image.width + x;
            if (ptr[idx] == 255) 
            {
                stack.push_back(idx);
            }
        }
    }

    // 8-connected neighborhood offsets
    const int dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    // Track edges
    while (!stack.empty()) 
    {
        uint32_t idx = stack.back();
        stack.pop_back();

        uint32_t x = idx % image.width;
        uint32_t y = idx / image.width;

        // Check the 8 surrounding pixels
        for (int i = 0; i < 8; ++i) 
        {
            uint32_t nx = x + dx[i];
            uint32_t ny = y + dy[i];

            // Prevent out-of-bounds mapping
            if (nx < image.width && ny < image.height) 
            {
                uint32_t n_idx = ny * image.width + nx;
                
                // If we find a weak edge connected to our strong edge, promote it
                if (ptr[n_idx] == 128) 
                {
                    ptr[n_idx] = 255;
                    stack.push_back(n_idx);
                }
            }
        }
    }

    // ========================================================================
    // PASS 3: Vectorized Cleanup (RVV)
    // Suppress any remaining weak edges (128) that were not connected
    // ========================================================================
    for (size_t i = 0; i < total_pixels; i += vl) 
    {
        vl = __riscv_vsetvl_e8m8(total_pixels - i);
        
        vuint8m8_t v_src = __riscv_vle8_v_u8m8(ptr + i, vl);
        
        // Find pixels that are exactly 128
        vbool1_t m_weak = __riscv_vmseq_vx_u8m8_b1(v_src, 128, vl);
        
        // Replace 128 with 0
        vuint8m8_t v_dst = __riscv_vmerge_vxm_u8m8(v_src, 0, m_weak, vl);
        
        __riscv_vse8_v_u8m8(ptr + i, v_dst, vl);
    }

    return Status::E_OK;
}

} // namespace processing::canny
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <utility>

#include "../inc/utils.hpp"
#include "../inc/std_types.hpp"
#include "../inc/io.hpp"
#include "../inc/gaussian.hpp"
#include "../inc/sobel.hpp"
#include "../inc/gradient.hpp"
// Added new pipeline headers
#include "../inc/nonmaximum_suppression.hpp"
#include "../inc/double_thresholding.hpp"
#include "../inc/hysteresis_tracking.hpp"

// Utility to allocate aligned images
template <typename T>
image::io::metadata_t<T> allocate_image(uint32_t w, uint32_t h) {
    image::io::metadata_t<T> img;
    img.width = w;
    img.height = h;
    img.pixel_count = static_cast<size_t>(w) * h;
    img.aligned_buffer_size = utils::memory::align_64(img.pixel_count * sizeof(T));
    img.buffer.reset(static_cast<T*>(
        utils::memory::aligned_alloc(64, img.aligned_buffer_size)));
    return img;
}

// Utility to dump raw buffers to disk for visual inspection
template <typename T>
void dump_raw_buffer(const std::string& prefix, const std::string& base_name, const std::string& stage, const T* data, size_t count) {
    // Ensured path drops files into tests_assets instead of root
    std::string filename = "./tests_assets/" + prefix + "_" + base_name + "_" + stage + ".raw";
    std::ofstream out(filename, std::ios::binary);
    if (out) {
        out.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    } else {
        std::cerr << "Failed to open " << filename << " for writing.\n";
    }
}

namespace reference {

// Scalar Gaussian 5x5 (Spatial)
void spatial_5x5(const image::io::metadata_t<uint8_t>& in, uint8_t* out) {
    int32_t width = in.width, height = in.height;
    int32_t kr = 2;
    int32_t pw = width + 2 * kr, ph = height + 2 * kr;
    
    std::vector<uint8_t> padded(pw * ph, 0);
    for (int32_t y = 0; y < height; ++y) {
        std::copy_n(&in.buffer[y * width], width, &padded[(y + kr) * pw + kr]);
    }

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            uint32_t sum = 0;
            for (int32_t ky = -kr; ky <= kr; ++ky) {
                for (int32_t kx = -kr; kx <= kr; ++kx) {
                    sum += padded[(y + kr + ky) * pw + (x + kr + kx)] * processing::GAUSSIAN_5x5_DATA[(ky + kr) * 5 + (kx + kr)];
                }
            }
            sum = static_cast<uint32_t>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            out[y * width + x] = (sum > 255) ? 255 : sum;
        }
    }
}

// Scalar Gaussian 5x5 (Separable)
void separable_5x5(const image::io::metadata_t<uint8_t>& in, uint8_t* out) {
    int32_t width = in.width, height = in.height;
    int32_t kr = 2;
    int32_t pw = width + 2 * kr, ph = height + 2 * kr;

    std::vector<uint8_t> padded(pw * ph, 0);
    for (int32_t y = 0; y < height; ++y) {
        std::copy_n(&in.buffer[y * width], width, &padded[(y + kr) * pw + kr]);
    }

    std::vector<uint32_t> inter(pw * ph, 0);
    for (int32_t y = 0; y < ph; ++y) {
        for (int32_t x = kr; x < pw - kr; ++x) {
            uint32_t sum = 0;
            for (int32_t kx = -kr; kx <= kr; ++kx) {
                sum += padded[y * pw + (x + kx)] * processing::GAUSSIAN_5x5_1D_DATA[kx + kr];
            }
            inter[y * pw + x] = sum;
        }
    }

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            uint32_t sum = 0;
            for (int32_t ky = -kr; ky <= kr; ++ky) {
                sum += inter[(y + kr + ky) * pw + (x + kr)] * processing::GAUSSIAN_5x5_1D_DATA[ky + kr];
            }
            sum = static_cast<uint32_t>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            out[y * width + x] = (sum > 255) ? 255 : sum;
        }
    }
}

// Scalar Sobel 3x3
void sobel_3x3(const image::io::metadata_t<uint8_t>& in, int16_t* gx, int16_t* gy) {
    int32_t width = in.width, height = in.height;
    std::fill_n(gx, width * height, 0);
    std::fill_n(gy, width * height, 0);

    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            int16_t t0 = in.buffer[(y - 1) * width + (x - 1)];
            int16_t t1 = in.buffer[(y - 1) * width + x];
            int16_t t2 = in.buffer[(y - 1) * width + (x + 1)];
            int16_t m0 = in.buffer[y * width + (x - 1)];
            int16_t m2 = in.buffer[y * width + (x + 1)];
            int16_t b0 = in.buffer[(y + 1) * width + (x - 1)];
            int16_t b1 = in.buffer[(y + 1) * width + x];
            int16_t b2 = in.buffer[(y + 1) * width + (x + 1)];

            gx[y * width + x] = (t2 - t0) + ((m2 - m0) << 1) + (b2 - b0);
            gy[y * width + x] = (b0 - t0) + ((b1 - t1) << 1) + (b2 - t2);
        }
    }
}

// Scalar L1 Gradient Magnitude
void l1(const int16_t* gx, const int16_t* gy, uint8_t* out, size_t count) {
    std::vector<uint16_t> mags(count);
    uint16_t max_val = 0;

    for (size_t i = 0; i < count; ++i) {
        mags[i] = std::abs(gx[i]) + std::abs(gy[i]);
        if (mags[i] > max_val) max_val = mags[i];
    }

    float scale = (max_val > 0) ? 255.0f / static_cast<float>(max_val) : 0.0f;
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint8_t>(mags[i] * scale);
    }
}

// Scalar L2 Gradient Magnitude
void l2(const int16_t* gx, const int16_t* gy, uint8_t* out, size_t count) {
    std::vector<float> mags(count);
    float max_val = 0.0f;

    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(gx[i]);
        float y = static_cast<float>(gy[i]);
        mags[i] = std::sqrt(x * x + y * y);
        if (mags[i] > max_val) max_val = mags[i];
    }

    float scale = (max_val > 0.0f) ? 255.0f / max_val : 0.0f;
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<uint8_t>(mags[i] * scale);
    }
}

// Scalar Gradient Direction
void direction(const int16_t* gx, const int16_t* gy, uint8_t* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int16_t cur_x = gx[i];
        int16_t cur_y = gy[i];
        uint16_t abs_x = static_cast<uint16_t>(std::abs(cur_x));
        uint16_t abs_y = static_cast<uint16_t>(std::abs(cur_y));

        if ((abs_y * 5) > (abs_x * 12)) out[i] = 90;
        else if ((abs_y * 2) > abs_x) {
            if (((cur_x > 0) && (cur_y > 0)) || ((cur_x < 0) && (cur_y < 0))) out[i] = 135;
            else out[i] = 45;
        } else {
            out[i] = 0;
        }
    }
}

// Scalar Non-Maximum Suppression Baseline
void nms(const uint8_t* mag_ptr, const uint8_t* dir_ptr, uint8_t* out_ptr, uint32_t width, uint32_t height) {
    std::memset(out_ptr, 0, width * height);
    for(uint32_t y = 1; y < height - 1; ++y) {
        for(uint32_t x = 1; x < width - 1; ++x) {
            const uint32_t idx = y * width + x;
            const uint8_t mag = mag_ptr[idx];
            const uint8_t angle = dir_ptr[idx];

            uint8_t q = 255;
            uint8_t r = 255;

            if(angle == 0) {
                q = mag_ptr[idx + 1];
                r = mag_ptr[idx - 1];
            } else if(angle == 45) {
                q = mag_ptr[(y - 1) * width + (x + 1)];
                r = mag_ptr[(y + 1) * width + (x - 1)];
            } else if(angle == 90) {
                q = mag_ptr[(y - 1) * width + x];
                r = mag_ptr[(y + 1) * width + x];
            } else if(angle == 135) {
                q = mag_ptr[(y - 1) * width + (x - 1)];
                r = mag_ptr[(y + 1) * width + (x + 1)];
            }
            out_ptr[idx] = (mag >= q && mag >= r) ? mag : 0;
        }
    }
}

// Scalar Double Threshold Baseline
void double_thresholding(uint8_t* ptr, size_t count, uint8_t low_threshold, uint8_t high_threshold) {
    for(size_t i = 0; i < count; ++i) {
        if(ptr[i] >= high_threshold) {
            ptr[i] = 255;
        } else if(ptr[i] >= low_threshold) {
            ptr[i] = 128;
        } else {
            ptr[i] = 0;
        }
    }
}

// Scalar Hysteresis Baseline
void hysteresis(uint8_t* ptr, uint32_t width, uint32_t height, uint8_t low_threshold, uint8_t high_threshold) {
    size_t total_pixels = width * height;

    // PASS 1: Scalar Double Thresholding
    for (size_t i = 0; i < total_pixels; ++i) {
        if (ptr[i] >= high_threshold) ptr[i] = 255;
        else if (ptr[i] >= low_threshold) ptr[i] = 128;
        else ptr[i] = 0;
    }

    // PASS 2: Scalar Edge Tracking (DFS)
    std::vector<uint32_t> stack;
    stack.reserve(total_pixels / 20); 

    for (uint32_t y = 1; y < height - 1; ++y) {
        for (uint32_t x = 1; x < width - 1; ++x) {
            const uint32_t idx = y * width + x;
            if (ptr[idx] == 255) {
                stack.push_back(idx);
            }
        }
    }

    const int dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    while (!stack.empty()) {
        const uint32_t idx = stack.back();
        stack.pop_back();

        const uint32_t x = idx % width;
        const uint32_t y = idx / width;

        for (int i = 0; i < 8; ++i) {
            const uint32_t nx = x + dx[i];
            const uint32_t ny = y + dy[i];

            if (nx < width && ny < height) {
                const uint32_t n_idx = ny * width + nx;
                
                if (ptr[n_idx] == 128) {
                    ptr[n_idx] = 255;
                    stack.push_back(n_idx);
                }
            }
        }
    }

    // PASS 3: Final Suppression
    for (size_t i = 0; i < total_pixels; ++i) {
        if (ptr[i] == 128) {
            ptr[i] = 0;
        }
    }
}

} // namespace reference


TEST(CannyEquivalence, AssetFilesMatch) {
    std::vector<std::string> bases = {
        "circ", "diag", "diag_inv", "full_black", "full_white", 
        "half_bw", "horiz", "quad_bw", "rect", "vert"
    };
    std::vector<std::pair<uint32_t, uint32_t>> dims = {
        {312, 444}, {512, 512}, {600, 400}
    };

    for(const auto& base : bases) {
        for(const auto& [w, h] : dims) {
            std::string filename = base + std::to_string(w) + "x" + std::to_string(h) + ".raw";
            std::string base_name = base + std::to_string(w) + "x" + std::to_string(h);
            
            auto img_orig = allocate_image<uint8_t>(w, h);
            Status load_status = image::io::load_raw<uint8_t>(filename, img_orig);
            
            // If the file doesn't exist in this specific dim, skip it
            if(load_status != Status::E_OK) {
                continue; 
            }

            std::cout << "Testing Equivalence on: " << filename << std::endl;

            // ---------------------------------------------------------
            // 1. Gaussian Spatial
            // ---------------------------------------------------------
            auto img_rvv_spatial = allocate_image<uint8_t>(w, h);
            std::memcpy(img_rvv_spatial.buffer.get(), img_orig.buffer.get(), img_orig.pixel_count);
            std::vector<uint8_t> ref_spatial(img_orig.pixel_count);
            
            reference::spatial_5x5(img_orig, ref_spatial.data());
            ASSERT_EQ(processing::spatial_5x5(img_rvv_spatial), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "spatial", ref_spatial.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "spatial", img_rvv_spatial.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_NEAR(img_rvv_spatial.buffer.get()[i], ref_spatial[i], 1) 
                    << "Spatial Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 2. Gaussian Separable
            // ---------------------------------------------------------
            auto img_rvv_sep = allocate_image<uint8_t>(w, h);
            std::memcpy(img_rvv_sep.buffer.get(), img_orig.buffer.get(), img_orig.pixel_count);
            std::vector<uint8_t> ref_sep(img_orig.pixel_count);
            
            reference::separable_5x5(img_orig, ref_sep.data());
            ASSERT_EQ(processing::separable_5x5(img_rvv_sep), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "sep", ref_sep.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "sep", img_rvv_sep.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_NEAR(img_rvv_sep.buffer.get()[i], ref_sep[i], 1) 
                    << "Separable Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 3. Sobel 3x3
            // ---------------------------------------------------------
            auto gx_rvv = allocate_image<int16_t>(w, h);
            auto gy_rvv = allocate_image<int16_t>(w, h);
            std::vector<int16_t> gx_ref(img_orig.pixel_count);
            std::vector<int16_t> gy_ref(img_orig.pixel_count);

            reference::sobel_3x3(img_orig, gx_ref.data(), gy_ref.data());
            ASSERT_EQ(processing::spatial_3x3(img_orig, gx_rvv.buffer.get(), gy_rvv.buffer.get()), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "gx", gx_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "gx", gx_rvv.buffer.get(), img_orig.pixel_count);
            dump_raw_buffer("scalar", base_name, "gy", gy_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "gy", gy_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_EQ(gx_rvv.buffer.get()[i], gx_ref[i]) << "Gx Mismatch in " << filename << " at index " << i;
                ASSERT_EQ(gy_rvv.buffer.get()[i], gy_ref[i]) << "Gy Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 4. Magnitude L1
            // ---------------------------------------------------------
            auto mag1_rvv = allocate_image<uint8_t>(w, h);
            std::vector<uint8_t> mag1_ref(img_orig.pixel_count);

            reference::l1(gx_ref.data(), gy_ref.data(), mag1_ref.data(), img_orig.pixel_count);
            ASSERT_EQ(processing::l1(mag1_rvv, gx_ref.data(), gy_ref.data()), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "mag1", mag1_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "mag1", mag1_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_NEAR(mag1_rvv.buffer.get()[i], mag1_ref[i], 1) 
                    << "L1 Mag Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 5. Magnitude L2
            // ---------------------------------------------------------
            auto mag2_rvv = allocate_image<uint8_t>(w, h);
            std::vector<uint8_t> mag2_ref(img_orig.pixel_count);

            reference::l2(gx_ref.data(), gy_ref.data(), mag2_ref.data(), img_orig.pixel_count);
            ASSERT_EQ(processing::l2(mag2_rvv, gx_ref.data(), gy_ref.data()), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "mag2", mag2_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "mag2", mag2_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_NEAR(mag2_rvv.buffer.get()[i], mag2_ref[i], 1) 
                    << "L2 Mag Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 6. Direction
            // ---------------------------------------------------------
            auto dir_rvv = allocate_image<uint8_t>(w, h);
            std::vector<uint8_t> dir_ref(img_orig.pixel_count);

            reference::direction(gx_ref.data(), gy_ref.data(), dir_ref.data(), img_orig.pixel_count);
            ASSERT_EQ(processing::direction(dir_rvv, gx_ref.data(), gy_ref.data()), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "dir", dir_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "dir", dir_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_EQ(dir_rvv.buffer.get()[i], dir_ref[i]) 
                    << "Direction Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 7. Non-Maximum Suppression (NMS)
            // ---------------------------------------------------------
            // We use L2 Magnitude (mag2) and Direction (dir) as the standard inputs for NMS.
            auto nms_rvv = allocate_image<uint8_t>(w, h);
            std::vector<uint8_t> nms_ref(img_orig.pixel_count);

            reference::nms(mag2_ref.data(), dir_ref.data(), nms_ref.data(), w, h);
            ASSERT_EQ(processing::nms(mag2_rvv, dir_rvv, nms_rvv), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "nms", nms_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "nms", nms_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_EQ(nms_rvv.buffer.get()[i], nms_ref[i]) 
                    << "NMS Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 8. Double Thresholding
            // ---------------------------------------------------------
            // Standard values usually used: Low = 20, High = 60
            auto dt_rvv = allocate_image<uint8_t>(w, h);
            std::memcpy(dt_rvv.buffer.get(), nms_rvv.buffer.get(), img_orig.pixel_count);
            std::vector<uint8_t> dt_ref(nms_ref); 

            reference::double_thresholding(dt_ref.data(), img_orig.pixel_count, 20, 60);
            ASSERT_EQ(processing::double_thresholding(dt_rvv, static_cast<uint8_t>(20), static_cast<uint8_t>(60)), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "double_thresholding", dt_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "double_thresholding", dt_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_EQ(dt_rvv.buffer.get()[i], dt_ref[i]) 
                    << "Double Thresholding Mismatch in " << filename << " at index " << i;
            }

            // ---------------------------------------------------------
            // 9. Hysteresis Edge Tracking
            // ---------------------------------------------------------
            // Note: Hysteresis uses the NMS output directly as its starting point.
            auto hyst_rvv = allocate_image<uint8_t>(w, h);
            std::memcpy(hyst_rvv.buffer.get(), nms_rvv.buffer.get(), img_orig.pixel_count);
            std::vector<uint8_t> hyst_ref(nms_ref); 

            reference::hysteresis(hyst_ref.data(), w, h, 20, 60);
            ASSERT_EQ(processing::hysteresis(hyst_rvv, static_cast<uint8_t>(20), static_cast<uint8_t>(60)), Status::E_OK);

            // Dump buffers
            dump_raw_buffer("scalar", base_name, "hysteresis", hyst_ref.data(), img_orig.pixel_count);
            dump_raw_buffer("rvv", base_name, "hysteresis", hyst_rvv.buffer.get(), img_orig.pixel_count);

            for (size_t i = 0; i < img_orig.pixel_count; ++i) {
                ASSERT_EQ(hyst_rvv.buffer.get()[i], hyst_ref[i]) 
                    << "Hysteresis Mismatch in " << filename << " at index " << i;
            }
        }
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


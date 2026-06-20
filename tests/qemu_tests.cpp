#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "../inc/utils.hpp"
#include "../inc/std_types.hpp"
#include "../inc/io.hpp"
#include "../inc/gaussian.hpp"
#include "../inc/sobel.hpp"
#include "../inc/gradient.hpp"
#include "../inc/nms.hpp"
#include "../inc/double_thresholding.hpp"
#include "../inc/hysteresis.hpp"

// ----------------------------------------------------------------------------
// Helper: allocate an aligned image buffer
// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
// Scalar reference implementation (the "oracle")
// ----------------------------------------------------------------------------
namespace reference {

void spatial_5x5(const image::io::metadata_t<uint8_t>& in, uint8_t* out) {
    int32_t width = in.width, height = in.height;
    int32_t kr = 2;
    int32_t pw = width + 2 * kr, ph = height + 2 * kr;

    std::vector<uint8_t> padded(pw * ph, 0);
    for (int32_t y = 0; y < height; ++y)
        std::copy_n(&in.buffer[y * width], width, &padded[(y + kr) * pw + kr]);

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            uint32_t sum = 0;
            for (int32_t ky = -kr; ky <= kr; ++ky)
                for (int32_t kx = -kr; kx <= kr; ++kx)
                    sum += padded[(y + kr + ky) * pw + (x + kr + kx)]
                         * processing::gaussian::GAUSSIAN_5x5_DATA[(ky + kr) * 5 + (kx + kr)];
            sum = static_cast<uint32_t>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            out[y * width + x] = (sum > 255) ? 255 : static_cast<uint8_t>(sum);
        }
    }
}

void separable_5x5(const image::io::metadata_t<uint8_t>& in, uint8_t* out) {
    int32_t width = in.width, height = in.height;
    int32_t kr = 2;
    int32_t pw = width + 2 * kr, ph = height + 2 * kr;

    std::vector<uint8_t> padded(pw * ph, 0);
    for (int32_t y = 0; y < height; ++y)
        std::copy_n(&in.buffer[y * width], width, &padded[(y + kr) * pw + kr]);

    std::vector<uint32_t> inter(pw * ph, 0);
    for (int32_t y = 0; y < ph; ++y) {
        for (int32_t x = kr; x < pw - kr; ++x) {
            uint32_t sum = 0;
            for (int32_t kx = -kr; kx <= kr; ++kx)
                sum += padded[y * pw + (x + kx)]
                     * processing::gaussian::GAUSSIAN_5x5_1D_DATA[kx + kr];
            inter[y * pw + x] = sum;
        }
    }

    constexpr uint32_t shift = 16;
    constexpr uint64_t multiplier = (1ULL << shift) / 273;

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            uint32_t sum = 0;
            for (int32_t ky = -kr; ky <= kr; ++ky)
                sum += inter[(y + kr + ky) * pw + (x + kr)]
                     * processing::gaussian::GAUSSIAN_5x5_1D_DATA[ky + kr];
            sum = static_cast<uint32_t>((static_cast<uint64_t>(sum) * multiplier) >> shift);
            out[y * width + x] = (sum > 255) ? 255 : static_cast<uint8_t>(sum);
        }
    }
}

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

void l1(const int16_t* gx, const int16_t* gy, uint8_t* out, size_t count) {
    std::vector<uint16_t> mags(count);
    uint16_t max_val = 0;
    for (size_t i = 0; i < count; ++i) {
        mags[i] = std::abs(gx[i]) + std::abs(gy[i]);
        if (mags[i] > max_val) max_val = mags[i];
    }
    float scale = (max_val > 0) ? 255.0f / static_cast<float>(max_val) : 0.0f;
    for (size_t i = 0; i < count; ++i)
        out[i] = static_cast<uint8_t>(mags[i] * scale);
}

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
    for (size_t i = 0; i < count; ++i)
        out[i] = static_cast<uint8_t>(mags[i] * scale);
}

void direction(const int16_t* gx, const int16_t* gy, uint8_t* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        int16_t cur_x = gx[i];
        int16_t cur_y = gy[i];
        uint16_t abs_x = static_cast<uint16_t>(std::abs(cur_x));
        uint16_t abs_y = static_cast<uint16_t>(std::abs(cur_y));

        if ((abs_y * 5) > (abs_x * 12))
            out[i] = 90;
        else if ((abs_y * 2) > abs_x) {
            if (((cur_x > 0) && (cur_y > 0)) || ((cur_x < 0) && (cur_y < 0)))
                out[i] = 135;
            else
                out[i] = 45;
        } else {
            out[i] = 0;
        }
    }
}

void nms(const uint8_t* mag_ptr, const uint8_t* dir_ptr, uint8_t* out_ptr, uint32_t width, uint32_t height) {
    std::memset(out_ptr, 0, width * height);
    for (uint32_t y = 1; y < height - 1; ++y) {
        for (uint32_t x = 1; x < width - 1; ++x) {
            const uint32_t idx = y * width + x;
            const uint8_t mag = mag_ptr[idx];
            const uint8_t angle = dir_ptr[idx];

            uint8_t q = 255, r = 255;
            if (angle == 0) {
                q = mag_ptr[idx + 1];
                r = mag_ptr[idx - 1];
            } else if (angle == 45) {
                q = mag_ptr[(y - 1) * width + (x + 1)];
                r = mag_ptr[(y + 1) * width + (x - 1)];
            } else if (angle == 90) {
                q = mag_ptr[(y - 1) * width + x];
                r = mag_ptr[(y + 1) * width + x];
            } else if (angle == 135) {
                q = mag_ptr[(y - 1) * width + (x - 1)];
                r = mag_ptr[(y + 1) * width + (x + 1)];
            }
            out_ptr[idx] = (mag >= q && mag >= r) ? mag : 0;
        }
    }
}

void double_thresholding(uint8_t* ptr, size_t count, uint8_t low, uint8_t high) {
    for (size_t i = 0; i < count; ++i) {
        if (ptr[i] >= high)       ptr[i] = 255;
        else if (ptr[i] >= low)   ptr[i] = 128;
        else                      ptr[i] = 0;
    }
}

void hysteresis(uint8_t* ptr, uint32_t width, uint32_t height, uint8_t low, uint8_t high) {
    size_t total = width * height;
    for (size_t i = 0; i < total; ++i) {
        if (ptr[i] >= high)       ptr[i] = 255;
        else if (ptr[i] >= low)   ptr[i] = 128;
        else                      ptr[i] = 0;
    }

    std::vector<uint32_t> stack;
    stack.reserve(total / 20);
    for (uint32_t y = 1; y < height - 1; ++y)
        for (uint32_t x = 1; x < width - 1; ++x) {
            uint32_t idx = y * width + x;
            if (ptr[idx] == 255) stack.push_back(idx);
        }

    const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    while (!stack.empty()) {
        uint32_t idx = stack.back(); stack.pop_back();
        uint32_t x = idx % width, y = idx / width;
        for (int i = 0; i < 8; ++i) {
            uint32_t nx = x + dx[i], ny = y + dy[i];
            if (nx < width && ny < height) {
                uint32_t nidx = ny * width + nx;
                if (ptr[nidx] == 128) {
                    ptr[nidx] = 255;
                    stack.push_back(nidx);
                }
            }
        }
    }

    for (size_t i = 0; i < total; ++i)
        if (ptr[i] == 128) ptr[i] = 0;
}

} // namespace reference

// ----------------------------------------------------------------------------
// Test: compare RVV-optimised pipeline against scalar reference
// ----------------------------------------------------------------------------
TEST(CannyEquivalence, RVVMatchesScalar) {
    // Use two images: one power-of-two (tests alignment), one not (tests strip-mining)
    std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> test_cases = {
        {"rect", {512, 512}},
        {"circ", {312, 444}}
    };

    for (const auto& [base, dims] : test_cases) {
        auto [w, h] = dims;
        std::string filename = base + std::to_string(w) + "x" + std::to_string(h) + ".raw";

        auto img_orig = allocate_image<uint8_t>(w, h);
        ASSERT_EQ(image::io::load_raw<uint8_t>(filename, img_orig), Status::E_OK)
            << "Failed to load " << filename;

        std::cout << "Testing " << filename << " ..." << std::endl;

        // --- Gaussian (spatial) ---
        auto img_scalar = allocate_image<uint8_t>(w, h);
        auto img_rvv    = allocate_image<uint8_t>(w, h);
        std::memcpy(img_scalar.buffer.get(), img_orig.buffer.get(), img_orig.pixel_count);
        std::memcpy(img_rvv.buffer.get(),    img_orig.buffer.get(), img_orig.pixel_count);

        std::vector<uint8_t> ref_spatial(img_orig.pixel_count);
        reference::spatial_5x5(img_orig, ref_spatial.data());
        ASSERT_EQ(processing::gaussian::spatial_5x5(img_rvv), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_NEAR(img_rvv.buffer.get()[i], ref_spatial[i], 1)
                << "Spatial mismatch at index " << i;

        // --- Gaussian (separable) ---
        std::memcpy(img_scalar.buffer.get(), img_orig.buffer.get(), img_orig.pixel_count);
        std::memcpy(img_rvv.buffer.get(),    img_orig.buffer.get(), img_orig.pixel_count);
        std::vector<uint8_t> ref_sep(img_orig.pixel_count);
        reference::separable_5x5(img_orig, ref_sep.data());
        ASSERT_EQ(processing::gaussian::separable_5x5(img_rvv), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_NEAR(img_rvv.buffer.get()[i], ref_sep[i], 1)
                << "Separable mismatch at index " << i;

        // --- Sobel ---
        auto gx_scalar = allocate_image<int16_t>(w, h);
        auto gy_scalar = allocate_image<int16_t>(w, h);
        auto gx_rvv    = allocate_image<int16_t>(w, h);
        auto gy_rvv    = allocate_image<int16_t>(w, h);
        std::vector<int16_t> gx_ref(img_orig.pixel_count), gy_ref(img_orig.pixel_count);

        reference::sobel_3x3(img_orig, gx_ref.data(), gy_ref.data());
        ASSERT_EQ(processing::sobel::spatial_3x3(img_orig, gx_rvv.buffer.get(), gy_rvv.buffer.get()), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i) {
            ASSERT_EQ(gx_rvv.buffer.get()[i], gx_ref[i]) << "Gx mismatch at " << i;
            ASSERT_EQ(gy_rvv.buffer.get()[i], gy_ref[i]) << "Gy mismatch at " << i;
        }

        // --- Magnitude L1 ---
        auto mag1_rvv = allocate_image<uint8_t>(w, h);
        std::vector<uint8_t> mag1_ref(img_orig.pixel_count);
        reference::l1(gx_ref.data(), gy_ref.data(), mag1_ref.data(), img_orig.pixel_count);
        ASSERT_EQ(processing::gradient::l1(mag1_rvv, gx_ref.data(), gy_ref.data()), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_NEAR(mag1_rvv.buffer.get()[i], mag1_ref[i], 1)
                << "L1 mismatch at " << i;

        // --- Magnitude L2 ---
        auto mag2_rvv = allocate_image<uint8_t>(w, h);
        std::vector<uint8_t> mag2_ref(img_orig.pixel_count);
        reference::l2(gx_ref.data(), gy_ref.data(), mag2_ref.data(), img_orig.pixel_count);
        ASSERT_EQ(processing::gradient::l2(mag2_rvv, gx_ref.data(), gy_ref.data()), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_NEAR(mag2_rvv.buffer.get()[i], mag2_ref[i], 1)
                << "L2 mismatch at " << i;

        // --- Direction ---
        auto dir_rvv = allocate_image<uint8_t>(w, h);
        std::vector<uint8_t> dir_ref(img_orig.pixel_count);
        reference::direction(gx_ref.data(), gy_ref.data(), dir_ref.data(), img_orig.pixel_count);
        ASSERT_EQ(processing::gradient::direction(dir_rvv, gx_ref.data(), gy_ref.data()), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_EQ(dir_rvv.buffer.get()[i], dir_ref[i])
                << "Direction mismatch at " << i;

        // --- NMS ---
        auto nms_rvv = allocate_image<uint8_t>(w, h);
        std::vector<uint8_t> nms_ref(img_orig.pixel_count);
        reference::nms(mag2_ref.data(), dir_ref.data(), nms_ref.data(), w, h);
        ASSERT_EQ(processing::nms::mag(mag2_rvv, dir_rvv, nms_rvv), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_EQ(nms_rvv.buffer.get()[i], nms_ref[i])
                << "NMS mismatch at " << i;

        // --- Double Thresholding (cast thresholds to uint8_t) ---
        auto dt_rvv = allocate_image<uint8_t>(w, h);
        std::memcpy(dt_rvv.buffer.get(), nms_rvv.buffer.get(), img_orig.pixel_count);
        std::vector<uint8_t> dt_ref(nms_ref);
        reference::double_thresholding(dt_ref.data(), img_orig.pixel_count, 20, 60);
        ASSERT_EQ(processing::double_thresholding::mag(dt_rvv,
                    static_cast<uint8_t>(20),
                    static_cast<uint8_t>(60)), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_EQ(dt_rvv.buffer.get()[i], dt_ref[i])
                << "Double thresholding mismatch at " << i;

        // --- Hysteresis (cast thresholds to uint8_t) ---
        auto hyst_rvv = allocate_image<uint8_t>(w, h);
        std::memcpy(hyst_rvv.buffer.get(), nms_rvv.buffer.get(), img_orig.pixel_count);
        std::vector<uint8_t> hyst_ref(nms_ref);
        reference::hysteresis(hyst_ref.data(), w, h, 20, 60);
        ASSERT_EQ(processing::hysteresis::mag(hyst_rvv,
                    static_cast<uint8_t>(20),
                    static_cast<uint8_t>(60)), Status::E_OK);

        for (size_t i = 0; i < img_orig.pixel_count; ++i)
            ASSERT_EQ(hyst_rvv.buffer.get()[i], hyst_ref[i])
                << "Hysteresis mismatch at " << i;
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

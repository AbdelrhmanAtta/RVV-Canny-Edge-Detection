#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <cmath>
#include <utility>

#include "utils.hpp"
#include "std_types.hpp"
#include "io.hpp"
#include "gaussian.hpp"
#include "sobel.hpp"
#include "gradient.hpp"
#include "nms.hpp" 
#include "double_thresholding.hpp"
#include "hysteresis.hpp"

template <typename T>
image::io::metadata_t<T> allocate_image(uint32_t w, uint32_t h) 
{
    image::io::metadata_t<T> img;
    img.width = w;
    img.height = h;
    img.pixel_count = static_cast<size_t>(w) * h;
    img.aligned_buffer_size = utils::memory::align_64(img.pixel_count * sizeof(T));
    img.buffer.reset(static_cast<T*>(
        utils::memory::aligned_alloc(64, img.aligned_buffer_size)));
    return img;
}

TEST (CannyPipeline, ProcessAndSaveAllImages) 
{
    std::vector<std::string> bases = {
        "circ", "diag", "diag_inv", "full_black", "full_white", 
        "half_bw", "horiz", "quad_bw", "rect", "vert"
    };
    std::vector<std::pair<uint32_t, uint32_t>> dims = {
        {312, 444}, {512, 512}, {600, 400}
    };

    for(const auto& base : bases) {
        for(const auto& [w, h] : dims) {
            std::string base_name = base + std::to_string(w) + "x" + std::to_string(h);
            std::string src = base_name + ".raw";
            
            std::string prefix = "google_" + base_name; 

            auto img_orig = allocate_image<uint8_t>(w, h);
            if(image::io::load_raw<uint8_t>(src, img_orig) != Status::E_OK) {
                continue;
            }

            auto img_spatial = allocate_image<uint8_t>(w, h);
            std::memcpy(img_spatial.buffer.get(), img_orig.buffer.get(), img_orig.pixel_count);
            EXPECT_EQ(processing::gaussian::spatial_5x5(img_spatial), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_spatial.raw", img_spatial), Status::E_OK);

            auto img_separable = allocate_image<uint8_t>(w, h);
            std::memcpy(img_separable.buffer.get(), img_orig.buffer.get(), img_orig.pixel_count);
            EXPECT_EQ(processing::gaussian::separable_5x5(img_separable), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_sep.raw", img_separable), Status::E_OK);

            auto gx = allocate_image<int16_t>(w, h);
            auto gy = allocate_image<int16_t>(w, h);
            EXPECT_EQ(processing::sobel::spatial_3x3(img_separable, gx.buffer.get(), gy.buffer.get()), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<int16_t>(prefix + "_gx.raw", gx), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<int16_t>(prefix + "_gy.raw", gy), Status::E_OK);

            auto mag_l1 = allocate_image<uint8_t>(w, h);
            EXPECT_EQ(processing::gradient::l1(mag_l1, gx.buffer.get(), gy.buffer.get()), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_mag1.raw", mag_l1), Status::E_OK);

            auto mag_l2 = allocate_image<uint8_t>(w, h);
            EXPECT_EQ(processing::gradient::l2(mag_l2, gx.buffer.get(), gy.buffer.get()), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_mag2.raw", mag_l2), Status::E_OK);

            auto dir = allocate_image<uint8_t>(w, h);
            EXPECT_EQ(processing::gradient::direction(dir, gx.buffer.get(), gy.buffer.get()), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_dir.raw", dir), Status::E_OK);

            auto nms_out = allocate_image<uint8_t>(w, h);
            EXPECT_EQ(processing::nms::mag(mag_l2, dir, nms_out), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_nms.raw", nms_out), Status::E_OK);

            auto double_thresh_out = allocate_image<uint8_t>(w, h);
            std::memcpy(double_thresh_out.buffer.get(), nms_out.buffer.get(), nms_out.pixel_count);
            EXPECT_EQ(processing::double_thresholding::mag(double_thresh_out, static_cast<uint8_t>(20), static_cast<uint8_t>(60)), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_double_thresholding.raw", double_thresh_out), Status::E_OK);

            auto hysteresis_out = allocate_image<uint8_t>(w, h);
            std::memcpy(hysteresis_out.buffer.get(), nms_out.buffer.get(), nms_out.pixel_count);
            EXPECT_EQ(processing::hysteresis::mag(hysteresis_out, static_cast<uint8_t>(20), static_cast<uint8_t>(60)), Status::E_OK);
            EXPECT_EQ(image::io::save_raw<uint8_t>(prefix + "_hysteresis.raw", hysteresis_out), Status::E_OK);
        }
    }
}

TEST(CannyGaussian, UniformImage) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    std::memset(img.buffer.get(), 128, img.pixel_count);

    Status stat = processing::gaussian::separable_5x5(img);
    EXPECT_EQ(stat, Status::E_OK);

    const uint8_t expected_val = 135;

    for (uint32_t y = 2; y < dim - 2; ++y) {
        for (uint32_t x = 2; x < dim - 2; ++x) {
            uint8_t val = img.buffer.get()[y * dim + x];
            EXPECT_NEAR(val, expected_val, 1);
        }
    }
}

TEST(CannyGaussian, AllBlackImage) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    std::memset(img.buffer.get(), 0, img.pixel_count);

    Status stat = processing::gaussian::separable_5x5(img);
    EXPECT_EQ(stat, Status::E_OK);

    for (size_t i = 0; i < img.pixel_count; ++i) {
        EXPECT_EQ(img.buffer.get()[i], 0);
    }
}

TEST(CannyGaussian, ImpulseSymmetry) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    std::memset(img.buffer.get(), 0, img.pixel_count);

    const uint32_t cx = dim / 2;
    const uint32_t cy = dim / 2;
    img.buffer.get()[cy * dim + cx] = 255;

    Status stat = processing::gaussian::separable_5x5(img);
    EXPECT_EQ(stat, Status::E_OK);

    uint8_t* ptr = img.buffer.get();
    
    EXPECT_EQ(ptr[cy * dim + (cx - 1)], ptr[cy * dim + (cx + 1)]);
    EXPECT_EQ(ptr[cy * dim + (cx - 2)], ptr[cy * dim + (cx + 2)]);
    
    EXPECT_EQ(ptr[(cy - 1) * dim + cx], ptr[(cy + 1) * dim + cx]);
    EXPECT_EQ(ptr[(cy - 2) * dim + cx], ptr[(cy + 2) * dim + cx]);
}

TEST(CannySobel, UniformImage) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    auto gx  = allocate_image<int16_t>(dim, dim);
    auto gy  = allocate_image<int16_t>(dim, dim);
    
    std::memset(img.buffer.get(), 128, img.pixel_count);

    Status stat = processing::sobel::spatial_3x3(img, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat, Status::E_OK);

    for (uint32_t y = 1; y < dim - 1; ++y) {
        for (uint32_t x = 1; x < dim - 1; ++x) {
            size_t idx = y * dim + x;
            EXPECT_EQ(gx.buffer.get()[idx], 0);
            EXPECT_EQ(gy.buffer.get()[idx], 0);
        }
    }
}

TEST(CannySobel, VerticalEdge) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    auto gx  = allocate_image<int16_t>(dim, dim);
    auto gy  = allocate_image<int16_t>(dim, dim);

    for (uint32_t y = 0; y < dim; ++y) {
        for (uint32_t x = 0; x < dim; ++x) {
            img.buffer.get()[y * dim + x] = (x < dim / 2) ? 0 : 255;
        }
    }

    Status stat = processing::sobel::spatial_3x3(img, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat, Status::E_OK);

    const uint32_t edge_x = dim / 2;
    for (uint32_t y = 1; y < dim - 1; ++y) {
        size_t idx = y * dim + edge_x;
        EXPECT_GT(std::abs(gx.buffer.get()[idx]), 0);
        EXPECT_EQ(gy.buffer.get()[idx], 0);
    }
}

TEST(CannySobel, HorizontalEdge) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    auto gx  = allocate_image<int16_t>(dim, dim);
    auto gy  = allocate_image<int16_t>(dim, dim);

    for (uint32_t y = 0; y < dim; ++y) {
        for (uint32_t x = 0; x < dim; ++x) {
            img.buffer.get()[y * dim + x] = (y < dim / 2) ? 0 : 255;
        }
    }

    Status stat = processing::sobel::spatial_3x3(img, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat, Status::E_OK);

    const uint32_t edge_y = dim / 2;
    for (uint32_t x = 1; x < dim - 1; ++x) {
        size_t idx = edge_y * dim + x;
        EXPECT_EQ(gx.buffer.get()[idx], 0);
        EXPECT_GT(std::abs(gy.buffer.get()[idx]), 0);
    }
}

TEST(CannySobel, DiagonalEdge) {
    const uint32_t dim = 128;
    auto img = allocate_image<uint8_t>(dim, dim);
    auto gx  = allocate_image<int16_t>(dim, dim);
    auto gy  = allocate_image<int16_t>(dim, dim);

    for (uint32_t y = 0; y < dim; ++y) {
        for (uint32_t x = 0; x < dim; ++x) {
            img.buffer.get()[y * dim + x] = (x > y) ? 255 : 0;
        }
    }

    Status stat = processing::sobel::spatial_3x3(img, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat, Status::E_OK);

    for (uint32_t i = 1; i < dim - 1; ++i) {
        size_t idx = i * dim + i;
        EXPECT_GT(std::abs(gx.buffer.get()[idx]), 0);
        EXPECT_GT(std::abs(gy.buffer.get()[idx]), 0);
    }
}

TEST(CannyDirection, EdgeAngles) {
    const uint32_t dim = 128;
    auto gx  = allocate_image<int16_t>(dim, dim);
    auto gy  = allocate_image<int16_t>(dim, dim);
    auto dir = allocate_image<uint8_t>(dim, dim);

    gx.buffer.get()[0] = 255; 
    gy.buffer.get()[0] = 0;

    gx.buffer.get()[1] = 0; 
    gy.buffer.get()[1] = 255;

    gx.buffer.get()[2] = 255; 
    gy.buffer.get()[2] = 255;

    Status stat = processing::gradient::direction(dir, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat, Status::E_OK);

    EXPECT_EQ(dir.buffer.get()[0], 0);
    EXPECT_EQ(dir.buffer.get()[1], 90);
    EXPECT_EQ(dir.buffer.get()[2], 135);
}

TEST(CannyMagnitude, NonZeroOutput) {
    const uint32_t dim = 128;
    auto gx  = allocate_image<int16_t>(dim, dim);
    auto gy  = allocate_image<int16_t>(dim, dim);
    auto mag = allocate_image<uint8_t>(dim, dim);

    for (size_t i = 0; i < gx.pixel_count; ++i) {
        gx.buffer.get()[i] = (i % 512) - 256; 
        gy.buffer.get()[i] = ((i * 3) % 512) - 256;
    }

    Status stat_l1 = processing::gradient::l1(mag, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat_l1, Status::E_OK);
    
    bool has_nonzero_l1 = false;
    for (size_t i = 0; i < mag.pixel_count; ++i) {
        if (mag.buffer.get()[i] > 0) {
            has_nonzero_l1 = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_l1);

    std::memset(mag.buffer.get(), 0, mag.pixel_count);

    Status stat_l2 = processing::gradient::l2(mag, gx.buffer.get(), gy.buffer.get());
    EXPECT_EQ(stat_l2, Status::E_OK);
    
    bool has_nonzero_l2 = false;
    for (size_t i = 0; i < mag.pixel_count; ++i) {
        if (mag.buffer.get()[i] > 0) {
            has_nonzero_l2 = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_l2);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
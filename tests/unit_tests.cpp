#include <iostream>
#include <cstdint>
#include <string_view>
#include <vector>
#include "../inc/utils.hpp"
#include "../inc/std_types.hpp"
#include "../inc/io.hpp"
#include "../inc/gaussian.hpp"

// Macro to automatically construct the paths and parameters.
// #w and #h convert the integer dimensions directly into strings.
#define TEST_IMG(name, w, h) \
    { name #w "x" #h ".raw", \
      name #w "x" #h "_copy.raw", \
      w, h }

struct raw_image_t
{
    std::string_view input_file;
    std::string_view output_file;
    uint32_t image_width;
    uint32_t image_height;
};

#ifdef HOST_MODE
    #include <gtest/gtest.h>
#else

#endif

#ifdef HOST_MODE
    TEST(CannyInfrastructure, IO_AllImages) 
    {
        std::vector<raw_image_t> images = 
        {
            // --- 312x444 Images ---
            TEST_IMG("circ", 312, 444),
            TEST_IMG("diag", 312, 444),
            TEST_IMG("diag_inv", 312, 444),
            TEST_IMG("full_black", 312, 444),
            TEST_IMG("full_white", 312, 444),
            TEST_IMG("half_bw", 312, 444),
            TEST_IMG("horiz", 312, 444),
            TEST_IMG("quad_bw", 312, 444),
            TEST_IMG("rect", 312, 444),
            TEST_IMG("vert", 312, 444),

            // --- 512x512 Images ---
            TEST_IMG("circ", 512, 512),
            TEST_IMG("diag", 512, 512),
            TEST_IMG("diag_inv", 512, 512),
            TEST_IMG("full_black", 512, 512),
            TEST_IMG("full_white", 512, 512),
            TEST_IMG("half_bw", 512, 512),
            TEST_IMG("horiz", 512, 512),
            TEST_IMG("quad_bw", 512, 512),
            TEST_IMG("rect", 512, 512),
            TEST_IMG("vert", 512, 512),

            // --- 600x400 Images ---
            TEST_IMG("circ", 600, 400),
            TEST_IMG("diag", 600, 400),
            TEST_IMG("diag_inv", 600, 400),
            TEST_IMG("full_black", 600, 400),
            TEST_IMG("full_white", 600, 400),
            TEST_IMG("half_bw", 600, 400),
            TEST_IMG("horiz", 600, 400),
            TEST_IMG("quad_bw", 600, 400),
            TEST_IMG("rect", 600, 400),
            TEST_IMG("vert", 600, 400)
        };

        for(const auto& [src, dest, w, h] : images)
        {
            const size_t pixel_count = static_cast<size_t>(w) * h;
            const size_t aligned_size = utils::memory::align_64(pixel_count);

            image::io::metadata_t<uint8_t> raw_image;
            raw_image.width              = w;
            raw_image.height             = h;
            raw_image.pixel_count        = pixel_count;
            raw_image.aligned_buffer_size = aligned_size;
            raw_image.buffer.reset(static_cast<uint8_t*>(
                utils::memory::aligned_alloc(64, aligned_size)));

            Status load_status = image::io::load_raw<uint8_t>(src, raw_image);
            
            // Appending a message to EXPECT_EQ tells you exactly WHICH file failed if one breaks
            EXPECT_EQ(load_status, Status::E_OK) << "Failed loading: " << src;

            if (load_status == Status::E_OK)
            {
                Status save_status = image::io::save_raw<uint8_t>(dest, raw_image);
                EXPECT_EQ(save_status, Status::E_OK) << "Failed saving: " << dest;
            }
        }
    }

    TEST(CannyPerformance, GaussianBlurSpatialVsSeparable) 
    {
        const uint32_t width = 512;
        const uint32_t height = 512;
        const size_t pixel_count = width * height;
        const size_t aligned_size = utils::memory::align_64(pixel_count);

        image::io::metadata_t<uint8_t> img_spatial;
        img_spatial.width = width;
        img_spatial.height = height;
        img_spatial.pixel_count = pixel_count;
        img_spatial.aligned_buffer_size = aligned_size;

        image::io::metadata_t<uint8_t> img_separable;
        img_separable.width = width;
        img_separable.height = height;
        img_separable.pixel_count = pixel_count;
        img_separable.aligned_buffer_size = aligned_size;

        // Load the tiger image for both tests
        Status stat_spatial = image::io::load_raw<uint8_t>("tiger.raw", img_spatial);
        ASSERT_EQ(stat_spatial, Status::E_OK) << "Failed to load tiger.raw for spatial blur";

        Status stat_separable = image::io::load_raw<uint8_t>("tiger.raw", img_separable);
        ASSERT_EQ(stat_separable, Status::E_OK) << "Failed to load tiger.raw for separable blur";

        // --- Profile Spatial 5x5 ---
        auto start_spatial = std::chrono::high_resolution_clock::now();
        Status proc_spatial = processing::spatial_5x5(img_spatial);
        auto end_spatial = std::chrono::high_resolution_clock::now();
        
        EXPECT_EQ(proc_spatial, Status::E_OK) << "Spatial processing failed";
        if (proc_spatial == Status::E_OK) {
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_spatial - start_spatial);
            std::cout << "[ PERFORMANCE ] Spatial Gaussian Blur Time: " << dur.count() << " ms\n";
            
            Status save_stat = image::io::save_raw<uint8_t>("tiger_spatial.raw", img_spatial);
            EXPECT_EQ(save_stat, Status::E_OK) << "Failed to save tiger_spatial.raw";
        }

        // --- Profile Separable 5x5 ---
        auto start_separable = std::chrono::high_resolution_clock::now();
        Status proc_separable = processing::separable_5x5(img_separable);
        auto end_separable = std::chrono::high_resolution_clock::now();

        EXPECT_EQ(proc_separable, Status::E_OK) << "Separable processing failed";
        if (proc_separable == Status::E_OK) {
            auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_separable - start_separable);
            std::cout << "[ PERFORMANCE ] Separable Gaussian Blur Time: " << dur.count() << " ms\n";
            
            Status save_stat = image::io::save_raw<uint8_t>("tiger_separable.raw", img_separable);
            EXPECT_EQ(save_stat, Status::E_OK) << "Failed to save tiger_separable.raw";
        }
    }

    int main(int argc, char** argv) {
        #if GTEST_HAS_EXCEPTIONS
        #endif
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
#else

#endif

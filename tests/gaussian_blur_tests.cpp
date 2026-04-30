<<<<<<< Updated upstream
#include "io.hpp"
#include "gaussian_blur.hpp"
#include "std_types.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <chrono> // Added so we can time the performance!

struct rawImage_t
{
    std::string fileName;
    std::string dest2DName;
    std::string dest1DName;
    uint32_t imageWidth;
    uint32_t imageHeight;
};

int main()
{
    const uint32_t dim = 512;
    std::vector<rawImage_t> images = 
    {
        {"rect.raw", "rect_blur_2D.raw", "rect_blur_1D.raw", dim, dim},
        {"circ.raw", "circ_blur_2D.raw", "circ_blur_1D.raw", dim, dim},
        {"diag.raw", "diag_blur_2D.raw", "diag_blur_1D.raw", dim, dim},
    };

    std::cout << "--- Starting Canny Edge Pipeline (Phase 2.2) ---" << std::endl;

    for(const auto& [src, dest2D, dest1D, w, h] : images)
    {
        std::cout << "\nProcessing: " << src << "..." << std::endl;

        // 1. Load the raw image into the input buffer
        uint8_t* inputBuffer = io_loadRaw(src, w, h);
        if (inputBuffer == nullptr) 
        {
            std::cerr << "  [ERROR] Failed to load " << src << ". Skipping." << std::endl;
            continue;
        }

        // 2. Allocate the output buffer (Must be 64-byte aligned for Phase 6!)
        size_t alloc_size = ((w * h) + 63) & ~63; 
        uint8_t* outputBuffer = static_cast<uint8_t*>(std::aligned_alloc(64, alloc_size));
        
        if (outputBuffer == nullptr)
        {
            std::cerr << "  [ERROR] Output memory allocation failed for " << src << std::endl;
            std::free(inputBuffer);
            continue;
        }

        // =================================================================
        // RUN 1: Standard 2D Convolution
        // =================================================================
        auto start2D = std::chrono::high_resolution_clock::now();
        Status stat2D = gaussian_blur_convolution(inputBuffer, outputBuffer, w, h);
        auto end2D = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time2D = end2D - start2D;

        if (stat2D == Status::E_OK)
        {
            io_saveRaw(dest2D, outputBuffer, w, h);
            std::cout << "  [SUCCESS] 2D Convolution saved to " << dest2D 
                      << " (" << time2D.count() << " ms)" << std::endl;
        }
        else 
        {
            std::cerr << "  [ERROR] 2D Convolution failed for " << src << std::endl;
        }

        // =================================================================
        // RUN 2: Optimized 1D Separable Blur
        // =================================================================
        auto start1D = std::chrono::high_resolution_clock::now();
        Status stat1D = gaussian_blur_separable(inputBuffer, outputBuffer, w, h);
        auto end1D = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time1D = end1D - start1D;

        if (stat1D == Status::E_OK)
        {
            io_saveRaw(dest1D, outputBuffer, w, h);
            std::cout << "  [SUCCESS] 1D Separable saved to " << dest1D 
                      << " (" << time1D.count() << " ms)" << std::endl;
        }
        else 
        {
            std::cerr << "  [ERROR] 1D Separable failed for " << src << std::endl;
        }

        // 5. Clean up BOTH memory buffers
        std::free(inputBuffer);
        std::free(outputBuffer);
    }

    std::cout << "\n--- Processing Complete ---" << std::endl;
=======
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <memory>
#include "std_types.hpp"
#include "io.hpp"
#include "gaussian.hpp"
#include "utils.hpp"

int main()
{
    Status stat;
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

    stat = image::io::load_raw<uint8_t>("circ.raw", img_spatial);
    if (Status::E_OK != stat) return static_cast<int>(stat);

    stat = image::io::load_raw<uint8_t>("circ.raw", img_separable);
    if (Status::E_OK != stat) return static_cast<int>(stat);

    auto start = std::chrono::high_resolution_clock::now();
    stat = processing::gaussian::spatial_5x5(img_spatial);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (Status::E_OK == stat) {
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Spatial Gaussian Blur Time: " << dur.count() << " ms" << std::endl;
        
        stat = image::io::save_raw<uint8_t>("circ_spatial.raw", img_spatial);
        if (Status::E_OK != stat) return static_cast<int>(stat);
    } else {
        return static_cast<int>(stat);
    }

    start = std::chrono::high_resolution_clock::now();
    stat = processing::gaussian::separable_5x5(img_separable);
    end = std::chrono::high_resolution_clock::now();

    if (Status::E_OK == stat) {
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Separable Gaussian Blur Time: " << dur.count() << " ms" << std::endl;
        
        stat = image::io::save_raw<uint8_t>("circ_separable.raw", img_separable);
        if (Status::E_OK != stat) return static_cast<int>(stat);
    } else {
        return static_cast<int>(stat);
    }
>>>>>>> Stashed changes

    return 0;
}
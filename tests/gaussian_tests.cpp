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

    stat = image::io::load_raw<uint8_t>("rect.raw", img_spatial);
    if (Status::E_OK != stat) return static_cast<int>(stat);

    stat = image::io::load_raw<uint8_t>("rect.raw", img_separable);
    if (Status::E_OK != stat) return static_cast<int>(stat);

    auto start = std::chrono::high_resolution_clock::now();
    stat = processing::gaussian::spatial_5x5(img_spatial);
    auto end = std::chrono::high_resolution_clock::now();
    
    if (Status::E_OK == stat) {
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Spatial Gaussian Blur Time: " << dur.count() << " ms" << std::endl;
        
        stat = image::io::save_raw<uint8_t>("rect_spatial.raw", img_spatial);
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
        
        stat = image::io::save_raw<uint8_t>("rect_separable.raw", img_separable);
        if (Status::E_OK != stat) return static_cast<int>(stat);
    } else {
        return static_cast<int>(stat);
    }

    return 0;
}

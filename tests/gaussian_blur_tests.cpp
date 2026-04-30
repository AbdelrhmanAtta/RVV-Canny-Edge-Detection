#include <chrono>
#include <iostream>
#include "std_types.hpp"
#include "io.hpp"
#include "gaussian.hpp"

int main()
{
    Status stat;

    image::io::metadata_t<uint8_t> raw_image
    {
        .width = 512,
        .height = 512
    };
    
    stat = image::io::load_raw<uint8_t>("circ.raw", raw_image);
    if(Status::E_OK!=stat) return static_cast<int>(stat);
    
    
    image::io::metadata_t<uint8_t> blurred_image
    {
        .width = raw_image.width,
        .height = raw_image.height,
        .pixel_count = raw_image.pixel_count,
        .aligned_buffer_size = raw_image.aligned_buffer_size
    };

    void* output_ptr = std::aligned_alloc(64, blurred_image.aligned_buffer_size);
    if(nullptr==output_ptr) return static_cast<int>(Status::E_ALLOC_FAIL);
    
    blurred_image.buffer.reset(static_cast<uint8_t*>(output_ptr));

    auto start_time = std::chrono::high_resolution_clock::now();
    stat = processing::gaussian::spatial(raw_image, blurred_image.buffer.get(),
                                        processing::gaussian::GAUSSIAN_5x5);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Spatial Gaussian Blur Time: " << duration.count() << " ms\n";
    
    start_time = std::chrono::high_resolution_clock::now();
    stat = processing::gaussian::separable(raw_image, blurred_image.buffer.get(),
                                        processing::gaussian::GAUSSIAN_5x5_1D);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Separable Gaussian Blur Time: " << duration.count() << " ms\n";

    if(Status::E_OK!=stat) return static_cast<int>(stat);

    stat = image::io::save_raw<uint8_t>("circ_blurred.raw", blurred_image);
    if(Status::E_OK!=stat) return static_cast<int>(stat);

    return 0;
}

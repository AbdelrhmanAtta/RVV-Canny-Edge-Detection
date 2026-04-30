#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "std_types.hpp"
#include "io.hpp"
#include "sobel.hpp"

int main()
{
    Status stat;

    image::io::metadata_t<uint8_t> raw_image { .width = 512, .height = 512 };
    stat = image::io::load_raw<uint8_t>("rect.raw", raw_image);
    if(Status::E_OK != stat) return static_cast<int>(stat);
    
    image::io::metadata_t<uint8_t> output_image {
        .width = raw_image.width,
        .height = raw_image.height,
        .pixel_count = raw_image.pixel_count,
        .aligned_buffer_size = raw_image.aligned_buffer_size
    };
    output_image.buffer.reset(static_cast<uint8_t*>(std::aligned_alloc(64, output_image.aligned_buffer_size)));

    std::vector<int16_t> sobel_x(raw_image.pixel_count, 0);
    std::vector<int16_t> sobel_y(raw_image.pixel_count, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    stat = processing::sobel::spatial_3x3(raw_image, sobel_x.data(), sobel_y.data());
    if(Status::E_OK != stat) return static_cast<int>(stat);

    for(uint32_t i = 0; i < raw_image.pixel_count; ++i) 
    {
        int32_t magnitude = std::abs(sobel_x[i]) + std::abs(sobel_y[i]);
        output_image.buffer.get()[i] = static_cast<uint8_t>(std::min(magnitude, 255));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time: " << duration.count() << " ms\n";
    
    stat = image::io::save_raw<uint8_t>("rect_sobel_output.raw", output_image);
    if(Status::E_OK != stat) return static_cast<int>(stat);

    return 0;
}
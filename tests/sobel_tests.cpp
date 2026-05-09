#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include "std_types.hpp"
#include "io.hpp"
#include "sobel.hpp"

int main()
{
    Status stat;
    const uint32_t w = 512;
    const uint32_t h = 512;

    image::io::metadata_t<uint8_t> raw_image;
    raw_image.width = w;
    raw_image.height = h;
    raw_image.pixel_count = w * h;
    raw_image.aligned_buffer_size = utils::memory::align_64(w * h);
    raw_image.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, raw_image.aligned_buffer_size)));

    stat = image::io::load_raw<uint8_t>("tiger.raw", raw_image);
    if(Status::E_OK != stat) return -1;

    image::io::metadata_t<uint8_t> out_gx_img;
    out_gx_img.width = w;
    out_gx_img.height = h;
    out_gx_img.pixel_count = raw_image.pixel_count;
    out_gx_img.aligned_buffer_size = raw_image.aligned_buffer_size;
    out_gx_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, out_gx_img.aligned_buffer_size)));

    image::io::metadata_t<uint8_t> out_gy_img;
    out_gy_img.width = w;
    out_gy_img.height = h;
    out_gy_img.pixel_count = raw_image.pixel_count;
    out_gy_img.aligned_buffer_size = raw_image.aligned_buffer_size;
    out_gy_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, out_gy_img.aligned_buffer_size)));

    std::vector<int16_t> gx_raw(raw_image.pixel_count, 0);
    std::vector<int16_t> gy_raw(raw_image.pixel_count, 0);

    auto start_time = std::chrono::high_resolution_clock::now();

    stat = processing::sobel::spatial_3x3(raw_image, gx_raw.data(), gy_raw.data());
    if(Status::E_OK != stat) return -1;

    for(uint32_t i = 0; i < raw_image.pixel_count; ++i) 
    {
        // Map [-1020, 1020] to [0, 255]
        // 0 becomes 127 (Gray)
        out_gx_img.buffer.get()[i] = static_cast<uint8_t>(std::clamp((gx_raw[i] / 8) + 127, 0, 255));
        out_gy_img.buffer.get()[i] = static_cast<uint8_t>(std::clamp((gy_raw[i] / 8) + 127, 0, 255));
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Sobel Processing Time: " << duration.count() << " ms\n";
    
    stat = image::io::save_raw<uint8_t>("gx_output.raw", out_gx_img);
    stat = image::io::save_raw<uint8_t>("gy_output.raw", out_gy_img);

    return 0;
}
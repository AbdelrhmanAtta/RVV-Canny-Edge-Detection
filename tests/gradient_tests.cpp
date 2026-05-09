#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include "std_types.hpp"
#include "io.hpp"
#include "sobel.hpp"
#include "gradient.hpp"
#include "utils.hpp"

int main()
{
    Status stat;
    const uint32_t w = 512;
    const uint32_t h = 512;
    const size_t pixel_count = w * h;
    const size_t aligned_size = utils::memory::align_64(pixel_count);

    image::io::metadata_t<uint8_t> blurred_img;
    blurred_img.width = w;
    blurred_img.height = h;
    blurred_img.pixel_count = pixel_count;
    blurred_img.aligned_buffer_size = aligned_size;
    blurred_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    stat = image::io::load_raw<uint8_t>("tiger_spatial.raw", blurred_img);
    if(Status::E_OK != stat) return -1;

    image::io::metadata_t<uint8_t> out_l1;
    out_l1.width = w; out_l1.height = h; out_l1.pixel_count = pixel_count; out_l1.aligned_buffer_size = aligned_size;
    out_l1.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    image::io::metadata_t<uint8_t> out_l2;
    out_l2.width = w; out_l2.height = h; out_l2.pixel_count = pixel_count; out_l2.aligned_buffer_size = aligned_size;
    out_l2.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    image::io::metadata_t<uint8_t> out_dir;
    out_dir.width = w; out_dir.height = h; out_dir.pixel_count = pixel_count; out_dir.aligned_buffer_size = aligned_size;
    out_dir.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    std::vector<int16_t> gx(pixel_count);
    std::vector<int16_t> gy(pixel_count);

    stat = processing::sobel::spatial_3x3(blurred_img, gx.data(), gy.data());
    if(Status::E_OK != stat) return -1;

    auto t1 = std::chrono::high_resolution_clock::now();
    stat = processing::l1(out_l1, gx.data(), gy.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    if(Status::E_OK != stat) return -1;

    auto t3 = std::chrono::high_resolution_clock::now();
    stat = processing::l2(out_l2, gx.data(), gy.data());
    auto t4 = std::chrono::high_resolution_clock::now();
    if(Status::E_OK != stat) return -1;

    auto t5 = std::chrono::high_resolution_clock::now();
    stat = processing::direction(out_dir, gx.data(), gy.data());
    auto t6 = std::chrono::high_resolution_clock::now();
    if(Status::E_OK != stat) return -1;

    auto d_l1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    auto d_l2 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
    auto d_dir = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);

    std::cout << "L1 Magnitude Time:  " << d_l1.count() << " ms" << std::endl;
    std::cout << "L2 Magnitude Time:  " << d_l2.count() << " ms" << std::endl;
    std::cout << "Ditigerion Time:     " << d_dir.count() << " ms" << std::endl;

    if (image::io::save_raw<uint8_t>("tiger_l1.raw", out_l1) != Status::E_OK) return -1;
    if (image::io::save_raw<uint8_t>("tiger_l2.raw", out_l2) != Status::E_OK) return -1;
    if (image::io::save_raw<uint8_t>("tiger_dir.raw", out_dir) != Status::E_OK) return -1;

    return 0;
}
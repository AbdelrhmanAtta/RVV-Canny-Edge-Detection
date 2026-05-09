#include <chrono>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include "std_types.hpp"
#include "io.hpp"
#include "nonmaximum_suppression.hpp"
#include "utils.hpp"

int main()
{
    Status stat;
    const uint32_t w = 512;
    const uint32_t h = 512;
    const size_t pixel_count = w * h;
    const size_t aligned_size = utils::memory::align_64(pixel_count);

    image::io::metadata_t<uint8_t> mag_img;
    mag_img.width = w;
    mag_img.height = h;
    mag_img.pixel_count = pixel_count;
    mag_img.aligned_buffer_size = aligned_size;
    mag_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    stat = image::io::load_raw<uint8_t>("rect_l2.raw", mag_img);
    if(Status::E_OK != stat) return -1;

    image::io::metadata_t<uint8_t> dir_img;
    dir_img.width = w; 
    dir_img.height = h; 
    dir_img.pixel_count = pixel_count; 
    dir_img.aligned_buffer_size = aligned_size;
    dir_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    stat = image::io::load_raw<uint8_t>("rect_dir.raw", dir_img);
    if(Status::E_OK != stat) return -1;

    image::io::metadata_t<uint8_t> out_nms;
    out_nms.width = w; 
    out_nms.height = h; 
    out_nms.pixel_count = pixel_count; 
    out_nms.aligned_buffer_size = aligned_size;
    out_nms.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    auto t1 = std::chrono::high_resolution_clock::now();
    stat = processing::nms(mag_img, dir_img, out_nms);
    auto t2 = std::chrono::high_resolution_clock::now();
    if(Status::E_OK != stat) return -1;

    auto d_nms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

    std::cout << "Non-Maximum Suppression Time: " << d_nms.count() << " ms" << std::endl;

    if (image::io::save_raw<uint8_t>("rect_nms.raw", out_nms) != Status::E_OK) return -1;

    return 0;
}

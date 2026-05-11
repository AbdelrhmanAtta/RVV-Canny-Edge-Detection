#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include "std_types.hpp"
#include "io.hpp"
#include "double_thresholding.hpp"
#include "utils.hpp"

int main()
{
    Status stat;
    const uint32_t w = 512;
    const uint32_t h = 512;
    const size_t pixel_count = w * h;
    const size_t aligned_size = utils::memory::align_64(pixel_count);

    image::io::metadata_t<uint8_t> thresh_img;
    thresh_img.width = w;
    thresh_img.height = h;
    thresh_img.pixel_count = pixel_count;
    thresh_img.aligned_buffer_size = aligned_size;
    thresh_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    // Load the output from the Non-Maximum Suppression stage
    stat = image::io::load_raw<uint8_t>("tiger_nms.raw", thresh_img);
    if(Status::E_OK != stat) 
    {
        std::cerr << "Failed to load tiger_nms.raw" << std::endl;
        return -1;
    }

    // Typical Canny thresholds (can be tuned)
    const uint8_t low_threshold = 50;
    const uint8_t high_threshold = 100;

    auto t1 = std::chrono::high_resolution_clock::now();
    
    // Categorize pixels into Strong (255), Weak (128), and Suppressed (0)
    stat = processing::double_thresholding(thresh_img, low_threshold, high_threshold);
    
    auto t2 = std::chrono::high_resolution_clock::now();

    if(Status::E_OK != stat) 
    {
        std::cerr << "Double Thresholding failed!" << std::endl;
        return -1;
    }

    auto d_thresh = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Double Thresholding Time: " << d_thresh.count() << " ms" << std::endl;

    // Save intermediate result to verify classification
    if (image::io::save_raw<uint8_t>("tiger_thresholded.raw", thresh_img) != Status::E_OK) 
    {
        return -1;
    }

    return 0;
}

#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include "std_types.hpp"
#include "io.hpp"
#include "hysteresis_tracking.hpp"
#include "utils.hpp"

int main()
{
    Status stat;
    const uint32_t w = 512;
    const uint32_t h = 512;
    const size_t pixel_count = w * h;
    const size_t aligned_size = utils::memory::align_64(pixel_count);

    // Metadata setup for the image to be processed (output of NMS)
    image::io::metadata_t<uint8_t> hyst_img;
    hyst_img.width = w;
    hyst_img.height = h;
    hyst_img.pixel_count = pixel_count;
    hyst_img.aligned_buffer_size = aligned_size;
    hyst_img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));

    // Load the thinned edges from NMS stage
    stat = image::io::load_raw<uint8_t>("tiger_nms.raw", hyst_img);
    if(Status::E_OK != stat) return -1;

    // Threshold definitions
    const uint8_t low_threshold = 50;
    const uint8_t high_threshold = 100;

    // Time the Hysteresis and Edge Tracking pass
    auto t1 = std::chrono::high_resolution_clock::now();
    stat = processing::hysteresis(hyst_img, low_threshold, high_threshold);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    if(Status::E_OK != stat) 
    {
        std::cerr << "Hysteresis failed with status: " << static_cast<int>(stat) << std::endl;
        return -1;
    }

    auto d_hyst = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << "Hysteresis Edge Tracking Time: " << d_hyst.count() << " ms" << std::endl;

    // Save final binary edge map
    if (image::io::save_raw<uint8_t>("tiger_final_edges.raw", hyst_img) != Status::E_OK) return -1;

    return 0;
}

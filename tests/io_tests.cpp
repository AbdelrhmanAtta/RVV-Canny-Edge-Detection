#include "io.hpp"
#include <iostream>
#include <string_view>
#include <vector>
#include <cstdint>

struct raw_image_t
{
    std::string_view input_file;
    std::string_view output_file;
    uint32_t image_width;
    uint32_t image_height;
};

int main() {
    const uint32_t dim = 512;
    std::vector<raw_image_t> images = 
    {
        {"rect.raw", "rect_copy.raw", dim, dim},
        {"circ.raw", "circ_copy.raw", dim, dim},
        {"diag.raw", "diag_copy.raw", dim, dim},
    };
    
    for(const auto& [src, dest, w, h] : images)
    {
        auto load_result = image::io::load_raw<uint8_t>(src, w, h);

        if(!load_result)
        {
            std::cerr << "[ERROR " << static_cast<int>(load_result.error()) << "]: " 
                      << "Failed to load image: " << src << std::endl;
            return 1;
        }

        auto& image = *load_result;

        Status save_status = image::io::save_raw<uint8_t>(dest, image.image_buffer.get(), w, h);
        
        if(Status::E_OK != save_status)
        {
            std::cerr << "Failed to save " << dest << " | Error: " 
                      << static_cast<int>(save_status) << "\n";
            return 1;
        }
    }

    return 0;
}
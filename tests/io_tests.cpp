#include "io.hpp"
#include <iostream>
#include <string_view>
#include <vector>
#include <cstdint>

image::io::metadata_t raw_image
{
    .image_width = 512,
    .image_height = 512,
};

int main() {
    Status load_status = image::io::load_raw<uint8_t>("rect.raw", raw_image);
    std::cout << "Load Status: " << static_cast<int>(load_status) << std::endl;

    Status save_status = image::io::save_raw<uint8_t>("rect_copyy.raw", raw_image);
    std::cout << "Save Status: " << static_cast<int>(save_status) << std::endl;

    return 0;
}

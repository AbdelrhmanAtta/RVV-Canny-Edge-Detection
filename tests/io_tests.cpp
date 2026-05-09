#include "io.hpp"
#include "utils.hpp"
#include <iostream>
#include <string_view>
#include <cstdint>

int main()
{
    const uint32_t w = 512;
    const uint32_t h = 512;
    const size_t pixel_count  = static_cast<size_t>(w) * h;
    const size_t aligned_size = utils::memory::align_64(pixel_count);

    image::io::metadata_t<uint8_t> raw_image;
    raw_image.width              = w;
    raw_image.height             = h;
    raw_image.pixel_count        = pixel_count;
    raw_image.aligned_buffer_size = aligned_size;
    raw_image.buffer.reset(static_cast<uint8_t*>(
        utils::memory::aligned_alloc(64, aligned_size)));

    Status load_status = image::io::load_raw<uint8_t>("rect.raw", raw_image);
    std::cout << "Load Status: " << static_cast<int>(load_status) << std::endl;

    Status save_status = image::io::save_raw<uint8_t>("rect_copy.raw", raw_image);
    std::cout << "Save Status: " << static_cast<int>(save_status) << std::endl;

    return 0;
}

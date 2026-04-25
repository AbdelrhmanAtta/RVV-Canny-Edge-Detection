#include "std_types.hpp"
#include "io.hpp"
#include "kernels.hpp"
#include "math.hpp"
#include <iostream>

int main()
{
    image::io::metadata_t<uint8_t> raw_image
    {
        .image_width = 512,
        .image_height = 512,
    };

    Status load_status = image::io::load_raw<uint8_t>("rect.raw", raw_image);
    std::cout << "Load Status: " << static_cast<int>(load_status) << std::endl;

    if (load_status != Status::E_OK) {
        return -1;
    }

    image::io::metadata_t<uint8_t> blurred_image
    {
        .image_width = raw_image.image_width,
        .image_height = raw_image.image_height,
        .pixel_count = raw_image.pixel_count,
        .aligned_buffer_size = raw_image.aligned_buffer_size
    };

    void* out_ptr = std::aligned_alloc(64, blurred_image.aligned_buffer_size);
    if (!out_ptr) {
        std::cerr << "Output allocation failed!" << std::endl;
        return -1;
    }
    blurred_image.image_buffer.reset(static_cast<uint8_t*>(out_ptr));

    auto conv_result = processing::convolution::spatial<false>(
        raw_image, 
        blurred_image.image_buffer.get(), 
        processing::convolution::GAUSSIAN_5x5, // Deducted as kernel_t<uint8_t, uint32_t>
        processing::convolution::clamp_t<uint8_t>{.clamp_min=0, .clamp_max=255}
    );

    if (!conv_result) {
        std::cerr << "Convolution failed!" << std::endl;
        return -1;
    }

    Status save_status = image::io::save_raw<uint8_t>("rect_blurred.raw", blurred_image);
    std::cout << "Save Status: " << static_cast<int>(save_status) << std::endl;

    return 0;
}
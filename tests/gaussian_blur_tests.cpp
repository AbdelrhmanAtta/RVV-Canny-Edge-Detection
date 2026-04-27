#include "std_types.hpp"
#include "io.hpp"
#include "filters.hpp"
#include "math.hpp"
#include "kernels.hpp"

int main()
{
    image::io::metadata_t<uint8_t> raw_image
    {
        .width = 512,
        .height = 512
    };
    
    Status status = image::io::load_raw<uint8_t>("circ.raw", raw_image);
    if(Status::E_OK!=status) return -1;
    
    
    image::io::metadata_t<uint8_t> blurred_image
    {
        .width = raw_image.width,
        .height = raw_image.height,
        .pixel_count = raw_image.pixel_count,
        .aligned_buffer_size = raw_image.aligned_buffer_size
    };

    void* output_ptr = std::aligned_alloc(64, blurred_image.aligned_buffer_size);
    if(nullptr==output_ptr) return -1;
    
    blurred_image.buffer.reset(static_cast<uint8_t*>(output_ptr));

    Status gaussian_status = processing::filters::gaussian_blur(raw_image, blurred_image.buffer.get(),
                                                                        processing::convolution::GAUSSIAN_5x5_1D);

    if(Status::E_OK!=gaussian_status) return -1;

    Status save_status = image::io::save_raw<uint8_t>("circ_blurred.raw", blurred_image);

    return 0;
}

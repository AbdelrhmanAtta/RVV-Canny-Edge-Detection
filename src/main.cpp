#include <iostream>
#include <vector>
#include <string>

#include "std_types.hpp"
#include "utils.hpp"
#include "io.hpp"
#include "gaussian.hpp"
#include "sobel.hpp"
#include "gradient.hpp"
#include "nms.hpp"
#include "double_thresholding.hpp"
#include "hysteresis.hpp"

struct ImageTask 
{
    std::string filename;
    uint32_t width;
    uint32_t height;
};

template <typename T>
image::io::metadata_t<T> allocate_image(uint32_t w, uint32_t h) 
{
    image::io::metadata_t<T> img;
    img.width = w;
    img.height = h;
    img.pixel_count = static_cast<size_t>(w) * h;
    img.aligned_buffer_size = utils::memory::align_64(img.pixel_count * sizeof(T));
    img.buffer.reset(static_cast<T*>(utils::memory::aligned_alloc(64, img.aligned_buffer_size)));
    return img;
}

void save_normalized(const std::string& name, const int16_t* data, uint32_t w, uint32_t h) 
{
    size_t count = static_cast<size_t>(w) * h;
    auto out = allocate_image<uint8_t>(w, h);
    int16_t min_v = 32767, max_v = -32768;
    for (size_t i = 0; i < count; ++i) {
        if (data[i] < min_v) min_v = data[i];
        if (data[i] > max_v) max_v = data[i];
    }
    float range = (max_v - min_v == 0) ? 1.0f : static_cast<float>(max_v - min_v);
    for (size_t i = 0; i < count; ++i)
        out.buffer[i] = static_cast<uint8_t>(255.0f * (static_cast<float>(data[i] - min_v) / range));

    (void)image::io::save_raw<uint8_t>(name, out);
}

int main(int argc, char** argv) 
{
    std::vector<ImageTask> tasks = 
    {
        {"Aquarium_1200x1600_gray.raw", 1200, 1600},
        {"Cars_2006_250x370_gray.raw", 250, 370},
        {"Drone_638x480_gray.raw", 638, 480},
        {"Me_1280x720_gray.raw", 1280, 720}
    };

    const uint8_t LOW = 30;
    const uint8_t HIGH = 90;

    for (const auto& task : tasks) 
    {
        std::cout << "[Pipeline] Starting " << task.filename << "...\n";

        auto img = allocate_image<uint8_t>(task.width, task.height);
        if (image::io::load_raw<uint8_t>(task.filename, img) != Status::E_OK) continue;

        (void)processing::gaussian::separable_5x5(img);
        (void)image::io::save_raw<uint8_t>("blur_" + task.filename, img);

        auto gx = allocate_image<int16_t>(task.width, task.height);
        auto gy = allocate_image<int16_t>(task.width, task.height);
        (void)processing::sobel::spatial_3x3(img, gx.buffer.get(), gy.buffer.get());
        save_normalized("gx_" + task.filename, gx.buffer.get(), task.width, task.height);
        save_normalized("gy_" + task.filename, gy.buffer.get(), task.width, task.height);

        auto mag = allocate_image<uint8_t>(task.width, task.height);
        auto dir = allocate_image<uint8_t>(task.width, task.height);
        (void)processing::gradient::l1(mag, gx.buffer.get(), gy.buffer.get());
        (void)processing::gradient::direction(dir, gx.buffer.get(), gy.buffer.get());
        (void)image::io::save_raw<uint8_t>("mag_" + task.filename, mag);
        (void)image::io::save_raw<uint8_t>("dir_" + task.filename, dir);

        auto nms = allocate_image<uint8_t>(task.width, task.height);
        (void)processing::nms::mag(mag, dir, nms);
        (void)image::io::save_raw<uint8_t>("nms_" + task.filename, nms);

        (void)processing::double_thresholding::mag(nms, LOW, HIGH);
        (void)image::io::save_raw<uint8_t>("thresh_" + task.filename, nms);

        (void)processing::hysteresis::mag(nms, LOW, HIGH);
        (void)image::io::save_raw<uint8_t>("canny_" + task.filename, nms);

        std::cout << "  -> Success: All stages saved for " << task.filename << "\n";
    }
    return 0;
}

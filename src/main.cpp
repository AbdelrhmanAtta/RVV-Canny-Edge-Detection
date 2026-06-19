    #include <iostream>
    #include <vector>
    #include <string>
    #include <chrono>
    #include <cmath>
    #include <algorithm>

    #include "../inc/std_types.hpp"
    #include "../inc/utils.hpp"
    #include "../inc/io.hpp"
    #include "../inc/gaussian.hpp"
    #include "../inc/sobel.hpp"
    #include "../inc/gradient.hpp"
    #include "../inc/nonmaximum_suppression.hpp" 
    #include "../inc/double_thresholding.hpp"
    #include "../inc/hysteresis_tracking.hpp"    

    struct ImageTask { std::string filename; uint32_t width; uint32_t height; };

    template <typename T>
    [[nodiscard]] image::io::metadata_t<T> allocate_image(uint32_t w, uint32_t h) {
        image::io::metadata_t<T> img;
        img.width = w; img.height = h;
        img.pixel_count = static_cast<size_t>(w) * h;
        img.aligned_buffer_size = utils::memory::align_64(img.pixel_count * sizeof(T));
        img.buffer.reset(static_cast<T*>(utils::memory::aligned_alloc(64, img.aligned_buffer_size)));
        return img;
    }

    void save_normalized(const std::string& name, const int16_t* data, uint32_t w, uint32_t h) {
        size_t count = static_cast<size_t>(w) * h;
        auto out = allocate_image<uint8_t>(w, h);
        int16_t min_v = 32767, max_v = -32768;
        for(size_t i = 0; i < count; ++i) {
            if(data[i] < min_v) min_v = data[i];
            if(data[i] > max_v) max_v = data[i];
        }
        float range = (max_v - min_v == 0) ? 1.0f : static_cast<float>(max_v - min_v);
        for(size_t i = 0; i < count; ++i) 
            out.buffer[i] = static_cast<uint8_t>(255.0f * (static_cast<float>(data[i] - min_v) / range));
        image::io::save_raw<uint8_t>(name, out);
    }

    int main() {
        std::vector<ImageTask> tasks = {
            {"Aquarium_1200x1600_gray.raw", 1200, 1600},
            {"Cars_2006_250x370_gray.raw", 250, 370},
            {"Drone_638x480_gray.raw", 638, 480},
            {"Me_1280x720_gray.raw", 1280, 720}
        };

        const uint8_t LOW = 30, HIGH = 90;

        for (const auto& task : tasks) {
            std::cout << "[Pipeline] Starting " << task.filename << "...\n";

            auto img = allocate_image<uint8_t>(task.width, task.height);
            if (image::io::load_raw<uint8_t>(task.filename, img) != Status::E_OK) continue;

            // 1. Gaussian Blur
            processing::separable_5x5(img);
            image::io::save_raw<uint8_t>("blur_" + task.filename, img);

            // 2. Sobel Gradients
            auto gx = allocate_image<int16_t>(task.width, task.height);
            auto gy = allocate_image<int16_t>(task.width, task.height);
            processing::spatial_3x3(img, gx.buffer.get(), gy.buffer.get());
            save_normalized("gx_" + task.filename, gx.buffer.get(), task.width, task.height);
            save_normalized("gy_" + task.filename, gy.buffer.get(), task.width, task.height);

            // 3. Magnitude & Direction
            auto mag = allocate_image<uint8_t>(task.width, task.height);
            auto dir = allocate_image<uint8_t>(task.width, task.height);
            processing::l1(mag, gx.buffer.get(), gy.buffer.get());
            processing::direction(dir, gx.buffer.get(), gy.buffer.get());
            image::io::save_raw<uint8_t>("mag_" + task.filename, mag);
            image::io::save_raw<uint8_t>("dir_" + task.filename, dir);

            // 4. Non-Maximum Suppression
            auto nms = allocate_image<uint8_t>(task.width, task.height);
            processing::nms(mag, dir, nms);
            image::io::save_raw<uint8_t>("nms_" + task.filename, nms);

            // 5. Double Thresholding
            processing::double_thresholding(nms, LOW, HIGH);
            image::io::save_raw<uint8_t>("thresh_" + task.filename, nms);

            // 6. Hysteresis Tracking
            processing::hysteresis(nms, LOW, HIGH);
            image::io::save_raw<uint8_t>("canny_" + task.filename, nms);
            
            std::cout << "  -> Success: All stages saved for " << task.filename << "\n";
        }
        return 0;
    }

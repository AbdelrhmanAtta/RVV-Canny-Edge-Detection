#include <iostream>
#include <cstring>
#include "std_types.hpp"
#include "nonmaximum_suppression.hpp"
#include "utils.hpp"

#define ASSERT_TEST(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << std::endl; \
            return 1; \
        } else { \
            std::cout << "[PASS] " << msg << std::endl; \
        } \
    } while(0)

/**
 * @brief Helper to allocate and initialise a metadata_t image of size w x h,
 *        with all pixels set to 0.
 */
static image::io::metadata_t<uint8_t> make_image(uint32_t w, uint32_t h)
{
    const size_t pixel_count   = static_cast<size_t>(w) * h;
    const size_t aligned_size  = utils::memory::align_64(pixel_count);
    image::io::metadata_t<uint8_t> img;
    img.width              = w;
    img.height             = h;
    img.pixel_count        = pixel_count;
    img.aligned_buffer_size = aligned_size;
    img.buffer.reset(static_cast<uint8_t*>(utils::memory::aligned_alloc(64, aligned_size)));
    std::memset(img.buffer.get(), 0, aligned_size);
    return img;
}

int main()
{
    std::cout << "--- NMS Unit Tests ---" << std::endl;

    // -------------------------------------------------------
    // Test 1: Null pointer guard
    // -------------------------------------------------------
    {
        image::io::metadata_t<uint8_t> mag, dir, out;  // all buffers null
        Status s = processing::nms(mag, dir, out);
        ASSERT_TEST(s == Status::E_INVAL_PTR, "Null pointer check");
    }

    // -------------------------------------------------------
    // Test 2: Horizontal direction (angle = 0)
    //   Grid (5x5), single horizontal ridge at row y=2:
    //      col:  0    1    2    3    4
    //   y=2:  [  0, 100, 200, 100,   0 ]   dir=0 for all
    //   Expected after NMS: only (y=2, x=2) survives.
    // -------------------------------------------------------
    {
        const uint32_t W = 5, H = 5;
        auto mag = make_image(W, H);
        auto dir = make_image(W, H);
        auto out = make_image(W, H);

        mag.buffer[2 * W + 1] = 100;
        mag.buffer[2 * W + 2] = 200;  // peak
        mag.buffer[2 * W + 3] = 100;

        // direction = 0 for the whole ridge
        dir.buffer[2 * W + 1] = 0;
        dir.buffer[2 * W + 2] = 0;
        dir.buffer[2 * W + 3] = 0;

        Status s = processing::nms(mag, dir, out);
        ASSERT_TEST(s == Status::E_OK, "Horizontal NMS status");
        ASSERT_TEST(out.buffer[2 * W + 2] == 200, "Horizontal NMS: peak preserved");
        ASSERT_TEST(out.buffer[2 * W + 1] == 0,   "Horizontal NMS: left suppressed");
        ASSERT_TEST(out.buffer[2 * W + 3] == 0,   "Horizontal NMS: right suppressed");
    }

    // -------------------------------------------------------
    // Test 3: Vertical direction (angle = 90)
    //   Grid (5x5), single vertical ridge at col x=2:
    //      row:  0    1    2    3    4
    //   x=2:  [  0, 100, 200, 100,   0 ]   dir=90
    //   Expected after NMS: only (y=2, x=2) survives.
    // -------------------------------------------------------
    {
        const uint32_t W = 5, H = 5;
        auto mag = make_image(W, H);
        auto dir = make_image(W, H);
        auto out = make_image(W, H);

        mag.buffer[1 * W + 2] = 100;
        mag.buffer[2 * W + 2] = 200;  // peak
        mag.buffer[3 * W + 2] = 100;

        dir.buffer[1 * W + 2] = 90;
        dir.buffer[2 * W + 2] = 90;
        dir.buffer[3 * W + 2] = 90;

        Status s = processing::nms(mag, dir, out);
        ASSERT_TEST(s == Status::E_OK, "Vertical NMS status");
        ASSERT_TEST(out.buffer[2 * W + 2] == 200, "Vertical NMS: peak preserved");
        ASSERT_TEST(out.buffer[1 * W + 2] == 0,   "Vertical NMS: top suppressed");
        ASSERT_TEST(out.buffer[3 * W + 2] == 0,   "Vertical NMS: bottom suppressed");
    }

    // -------------------------------------------------------
    // Test 4: Diagonal / direction (angle = 45)
    //   Grid (5x5), diagonal ridge (top-right to bottom-left):
    //     (1,3)=100  (2,2)=200  (3,1)=100  -- dir=45 for all
    //   Neighbors of peak (2,2) are (1,3)=100 and (3,1)=100.
    //   Expected: (2,2) preserved, (1,3) and (3,1) suppressed.
    // -------------------------------------------------------
    {
        const uint32_t W = 5, H = 5;
        auto mag = make_image(W, H);
        auto dir = make_image(W, H);
        auto out = make_image(W, H);

        // Diagonal /: q = (y-1, x+1), r = (y+1, x-1)
        // For peak at (y=2,x=2): q=(1,3)=100, r=(3,1)=100
        mag.buffer[1 * W + 3] = 100;   // q for peak
        mag.buffer[2 * W + 2] = 200;   // peak
        mag.buffer[3 * W + 1] = 100;   // r for peak

        dir.buffer[1 * W + 3] = 45;
        dir.buffer[2 * W + 2] = 45;
        dir.buffer[3 * W + 1] = 45;

        Status s = processing::nms(mag, dir, out);
        ASSERT_TEST(s == Status::E_OK, "Diagonal-/ NMS status");
        ASSERT_TEST(out.buffer[2 * W + 2] == 200, "Diagonal-/ NMS: peak preserved");
        ASSERT_TEST(out.buffer[1 * W + 3] == 0,   "Diagonal-/ NMS: top-right suppressed");
        ASSERT_TEST(out.buffer[3 * W + 1] == 0,   "Diagonal-/ NMS: bottom-left suppressed");
    }

    // -------------------------------------------------------
    // Test 5: Diagonal \ direction (angle = 135)
    //   Grid (5x5), diagonal ridge (top-left to bottom-right):
    //     (1,1)=100  (2,2)=200  (3,3)=100  -- dir=135 for all
    //   Neighbors of peak (2,2) are (1,1)=100 and (3,3)=100.
    //   Expected: (2,2) preserved, (1,1) and (3,3) suppressed.
    // -------------------------------------------------------
    {
        const uint32_t W = 5, H = 5;
        auto mag = make_image(W, H);
        auto dir = make_image(W, H);
        auto out = make_image(W, H);

        // Diagonal \: q = (y-1, x-1), r = (y+1, x+1)
        // For peak at (y=2,x=2): q=(1,1)=100, r=(3,3)=100
        mag.buffer[1 * W + 1] = 100;   // q for peak
        mag.buffer[2 * W + 2] = 200;   // peak
        mag.buffer[3 * W + 3] = 100;   // r for peak

        dir.buffer[1 * W + 1] = 135;
        dir.buffer[2 * W + 2] = 135;
        dir.buffer[3 * W + 3] = 135;

        Status s = processing::nms(mag, dir, out);
        ASSERT_TEST(s == Status::E_OK, "Diagonal-\\ NMS status");
        ASSERT_TEST(out.buffer[2 * W + 2] == 200, "Diagonal-\\ NMS: peak preserved");
        ASSERT_TEST(out.buffer[1 * W + 1] == 0,   "Diagonal-\\ NMS: top-left suppressed");
        ASSERT_TEST(out.buffer[3 * W + 3] == 0,   "Diagonal-\\ NMS: bottom-right suppressed");
    }

    // -------------------------------------------------------
    // Test 6: Pixel at border is untouched (stays 0)
    // -------------------------------------------------------
    {
        const uint32_t W = 5, H = 5;
        auto mag = make_image(W, H);
        auto dir = make_image(W, H);
        auto out = make_image(W, H);

        mag.buffer[0 * W + 0] = 255;   // corner pixel, should be left as 0 in output

        Status s = processing::nms(mag, dir, out);
        ASSERT_TEST(s == Status::E_OK,            "Border pixel NMS status");
        ASSERT_TEST(out.buffer[0 * W + 0] == 0,   "Border pixel stays 0");
    }

    std::cout << "--- All NMS tests passed ---" << std::endl;
    return 0;
}

#include "io.hpp"
#include "gaussian.hpp"
#include "sobel.hpp"
#include "gradient.hpp"
#include "std_types.hpp"
#include "utils.hpp"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <time.h>

// ── Timing helper ─────────────────────────────────────────────
static double elapsed_ms(struct timespec s, struct timespec e)
{
    return (e.tv_sec  - s.tv_sec)  * 1000.0
         + (e.tv_nsec - s.tv_nsec) / 1e6;
}

int main()
{
    constexpr uint32_t WIDTH  = 512;
    constexpr uint32_t HEIGHT = 512;
    constexpr int      ITERS  = 100;

    // ── 1. Load image ──────────────────────────────────────────
    image::io::metadata_t<uint8_t> image;
    image.width  = WIDTH;
    image.height = HEIGHT;

    Status st = image::io::load_raw("tiger.raw", image);
    if (st != Status::E_OK)
    {
        printf("ERROR: could not load assets/tiger.raw (status=%d)\n",
               static_cast<int>(st));
        return 1;
    }

    // ── 2. Allocate Sobel output buffers ───────────────────────
    const size_t n = image.pixel_count;

    auto gx = static_cast<int16_t*>(utils::memory::aligned_alloc(64, n * sizeof(int16_t)));
    auto gy = static_cast<int16_t*>(utils::memory::aligned_alloc(64, n * sizeof(int16_t)));
    if (!gx || !gy) { printf("ERROR: alloc failed\n"); return 1; }

    // ── 3. Benchmark helpers ───────────────────────────────────
    struct timespec t0, t1;

#define BENCH(label, code)                                      \
    clock_gettime(CLOCK_MONOTONIC, &t0);                        \
    for (int _i = 0; _i < ITERS; ++_i) { code; }               \
    clock_gettime(CLOCK_MONOTONIC, &t1);                        \
    printf("%-20s %.3f ms/iter\n", label,                       \
           elapsed_ms(t0, t1) / ITERS);

    // ── 4. Gaussian spatial ────────────────────────────────────
    {
        image::io::metadata_t<uint8_t> img_copy;
        img_copy.width              = image.width;
        img_copy.height             = image.height;
        img_copy.pixel_count        = image.pixel_count;
        img_copy.aligned_buffer_size = image.aligned_buffer_size;
        auto* raw = static_cast<uint8_t*>(
            utils::memory::aligned_alloc(64, image.aligned_buffer_size));
        img_copy.buffer.reset(raw);

        BENCH("Gaussian spatial",
            std::copy_n(image.buffer.get(), n, img_copy.buffer.get());
            processing::spatial_5x5(img_copy);
        )
    }

    // ── 5. Gaussian separable ──────────────────────────────────
    {
        image::io::metadata_t<uint8_t> img_copy;
        img_copy.width              = image.width;
        img_copy.height             = image.height;
        img_copy.pixel_count        = image.pixel_count;
        img_copy.aligned_buffer_size = image.aligned_buffer_size;
        auto* raw = static_cast<uint8_t*>(
            utils::memory::aligned_alloc(64, image.aligned_buffer_size));
        img_copy.buffer.reset(raw);

        BENCH("Gaussian separable",
            std::copy_n(image.buffer.get(), n, img_copy.buffer.get());
            processing::separable_5x5(img_copy);
        )
    }

    // ── 6. Sobel ───────────────────────────────────────────────
    BENCH("Sobel",
        processing::spatial_3x3(image, gx, gy);
    )

    // ── 7. Magnitude L1 ───────────────────────────────────────
    {
        image::io::metadata_t<uint8_t> mag;
        mag.width              = image.width;
        mag.height             = image.height;
        mag.pixel_count        = image.pixel_count;
        mag.aligned_buffer_size = image.aligned_buffer_size;
        auto* raw = static_cast<uint8_t*>(
            utils::memory::aligned_alloc(64, image.aligned_buffer_size));
        mag.buffer.reset(raw);

        BENCH("Magnitude L1",
            processing::l1(mag, gx, gy);
        )
    }

    // ── 8. Magnitude L2 ───────────────────────────────────────
    {
        image::io::metadata_t<uint8_t> mag;
        mag.width              = image.width;
        mag.height             = image.height;
        mag.pixel_count        = image.pixel_count;
        mag.aligned_buffer_size = image.aligned_buffer_size;
        auto* raw = static_cast<uint8_t*>(
            utils::memory::aligned_alloc(64, image.aligned_buffer_size));
        mag.buffer.reset(raw);

        BENCH("Magnitude L2",
            processing::l2(mag, gx, gy);
        )
    }

    // ── 9. Direction ───────────────────────────────────────────
    {
        image::io::metadata_t<uint8_t> dir;
        dir.width              = image.width;
        dir.height             = image.height;
        dir.pixel_count        = image.pixel_count;
        dir.aligned_buffer_size = image.aligned_buffer_size;
        auto* raw = static_cast<uint8_t*>(
            utils::memory::aligned_alloc(64, image.aligned_buffer_size));
        dir.buffer.reset(raw);

        // direction takes int32_t Gx/Gy — upcast
        auto gx32 = static_cast<int32_t*>(
            utils::memory::aligned_alloc(64, n * sizeof(int32_t)));
        auto gy32 = static_cast<int32_t*>(
            utils::memory::aligned_alloc(64, n * sizeof(int32_t)));
        for (size_t i = 0; i < n; ++i) { gx32[i] = gx[i]; gy32[i] = gy[i]; }

        BENCH("Direction",
            processing::direction(dir, gx32, gy32);
        )

        std::free(gx32);
        std::free(gy32);
    }

    std::free(gx);
    std::free(gy);

    printf("\nDone. Image: %ux%u, %d iterations each.\n", WIDTH, HEIGHT, ITERS);
    return 0;
}
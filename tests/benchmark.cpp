#include "io.hpp"
#include "gaussian.hpp"
#include "sobel.hpp"
#include "gradient.hpp"
#include "std_types.hpp"
#include "utils.hpp"

#include <cstdio>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <time.h>

/** 
 * CHANGE THESE TO CONFIGURE THE RUN
 *
 * LMUL_SWEEP   RVV LMUL factor  (1, 2, 4)
 *
 * PIPELINE_SEL
 *              0 -> Gaussian spatial only
 *              1 -> Gaussian separable only
 *              2 -> Sobel only
 *              3 -> Magnitude L1 only
 *              4 -> Magnitude L2 only
 *              5 -> Direction only   
 *              6 -> Full pipeline    
 *
 * Scalar vs RVV is controlled by the Makefile -march flag
 *              bench-scalar-O3  ->  -march=rv64gc   (__riscv_v undefined)
 *              bench-rvv-O3     ->  -march=rv64gcv  (__riscv_v defined)
 *
**/
#define LMUL_SWEEP   1
#define PIPELINE_SEL 3

/* GEM5 Setup */
#ifdef GEM5_MODE
#include <gem5/m5ops.h>
#define GEM5_RESET_STATS() m5_reset_stats(0, 0)
#define GEM5_DUMP_STATS()  m5_dump_stats(0, 0)
#endif

static double elapsed_ms(struct timespec s, struct timespec e)
{
    return (e.tv_sec  - s.tv_sec)  * 1000.0
         + (e.tv_nsec - s.tv_nsec) / 1e6;
}

#define BENCH(label, ...)                                         \
    clock_gettime(CLOCK_MONOTONIC, &t0);                          \
    GEM5_RESET_STATS();                                           \
    { __VA_ARGS__ }                                               \
    GEM5_DUMP_STATS();                                            \
    clock_gettime(CLOCK_MONOTONIC, &t1);                          \
    printf("[TIME     ]: %-24s %.3f ms\n", label, elapsed_ms(t0, t1));


/* Allocate a fresh metadata copy of an image */
static image::io::metadata_t<uint8_t> make_copy(const image::io::metadata_t<uint8_t>& src)
{
    image::io::metadata_t<uint8_t> dst;
    dst.width               = src.width;
    dst.height              = src.height;
    dst.pixel_count         = src.pixel_count;
    dst.aligned_buffer_size = src.aligned_buffer_size;
    dst.buffer.reset(static_cast<uint8_t*>(
        utils::memory::aligned_alloc(64, src.aligned_buffer_size)));
    return dst;
}

int main()
{
    constexpr uint32_t WIDTH  = 512;
    constexpr uint32_t HEIGHT = 512;

    image::io::metadata_t<uint8_t> image;
    image.width  = WIDTH;
    image.height = HEIGHT;

    if (image::io::load_raw("tiger.raw", image) != Status::E_OK)
    {
        printf("[ERROR    ]: Could not Load Tiger.raw\n");
        return 1;
    }

    const size_t n = image.pixel_count;

    auto* gx = static_cast<int16_t*>(utils::memory::aligned_alloc(64, n * sizeof(int16_t)));
    auto* gy = static_cast<int16_t*>(utils::memory::aligned_alloc(64, n * sizeof(int16_t)));
    if (!gx || !gy) 
    { 
        printf("[ERROR    ]: alloc failed\n"); 
        return 1;
    }

    struct timespec t0, t1;

    printf("[Benchmark]: LMUL=%d, %s\n",
           LMUL_SWEEP,
#if defined(__riscv_v)
           "RVV"
#else
           "Scalar"
#endif
    );

    /* Pre-compute required images */
#if PIPELINE_SEL == 2 || PIPELINE_SEL == 3 || \
    PIPELINE_SEL == 4 || PIPELINE_SEL == 5
    {
        auto blurred = make_copy(image);
        std::copy_n(image.buffer.get(), n, blurred.buffer.get());
        (void)processing::separable_5x5<uint8_t, int32_t, LMUL_SWEEP>(blurred);

#if PIPELINE_SEL == 3 || PIPELINE_SEL == 4 || PIPELINE_SEL == 5
        (void)processing::spatial_3x3<LMUL_SWEEP>(blurred, gx, gy);
#endif

#if PIPELINE_SEL == 2
        BENCH("Sobel Filter",
            (void)processing::spatial_3x3<LMUL_SWEEP>(blurred, gx, gy);
        )
#endif

#if PIPELINE_SEL == 3
        {
            auto mag = make_copy(image);
            BENCH("Magnitude L1",
            (void)processing::l1<uint8_t, int16_t, uint16_t, LMUL_SWEEP>(mag, gx, gy);
            )
        }
#endif

#if PIPELINE_SEL == 4
        {
            auto mag = make_copy(image);
            BENCH("Magnitude L2",
                (void)processing::l2(mag, gx, gy);
            )
        }
#endif

#if PIPELINE_SEL == 5
        {
            auto dir = make_copy(image);
            auto* gx32 = static_cast<int32_t*>(utils::memory::aligned_alloc(64, n * sizeof(int32_t)));
            auto* gy32 = static_cast<int32_t*>(utils::memory::aligned_alloc(64, n * sizeof(int32_t)));
            for (size_t i = 0; i < n; ++i) { gx32[i] = gx[i]; gy32[i] = gy[i]; }
            BENCH("Direction",
                (void)processing::direction(dir, gx32, gy32);
            )
            std::free(gx32);
            std::free(gy32);
        }
#endif
    }
#endif /* stages 2-5 */

#if PIPELINE_SEL == 0
    {
        auto img_copy = make_copy(image);
        BENCH("Gaussian spatial",
            std::copy_n(image.buffer.get(), n, img_copy.buffer.get());
            (void)processing::spatial_5x5(img_copy);
        )
    }
#endif

#if PIPELINE_SEL == 1
    {
        auto img_copy = make_copy(image);
        BENCH("Gaussian separable",
            std::copy_n(image.buffer.get(), n, img_copy.buffer.get());
            (void)processing::separable_5x5<uint8_t, int32_t, LMUL_SWEEP>(img_copy);
        )
    }
#endif

#if PIPELINE_SEL == 6
    {
        auto blurred = make_copy(image);
        auto mag     = make_copy(image);
        auto dir     = make_copy(image);
        auto* gx32 = static_cast<int32_t*>(utils::memory::aligned_alloc(64, n * sizeof(int32_t)));
        auto* gy32 = static_cast<int32_t*>(utils::memory::aligned_alloc(64, n * sizeof(int32_t)));

        BENCH("Full pipeline",
            std::copy_n(image.buffer.get(), n, blurred.buffer.get());
            (void)processing::separable_5x5<uint8_t, int32_t, LMUL_SWEEP>(blurred);
            (void)processing::spatial_3x3<LMUL_SWEEP>(blurred, gx, gy);
            (void)processing::l1(mag, gx, gy);
            for (size_t i = 0; i < n; ++i) { gx32[i] = gx[i]; gy32[i] = gy[i]; }
            (void)processing::direction(dir, gx32, gy32);
        )

        std::free(gx32);
        std::free(gy32);
    }
#endif

    std::free(gx);
    std::free(gy);
    return 0;
}


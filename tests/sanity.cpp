#include <iostream>
#include <vector>
#include <cstdint>
#include <riscv_vector.h>

int main() {
    std::cout << "=== RVV Sanity Test & Version Check ===" << std::endl;

    // ── 1. ISA version ────────────────────────────────────────────────────────
#ifdef __riscv_v
    std::cout << "[ISA]       __riscv_v = " << __riscv_v;
    if (__riscv_v >= 1000000)
        std::cout << "  →  RVV 1.0 ISA confirmed" << std::endl;
    else
        std::cout << "  →  WARNING: pre-1.0 RVV ISA" << std::endl;
#else
    std::cout << "[ISA]       __riscv_v NOT DEFINED — no vector extension" << std::endl;
    return 1;
#endif

    // ── 2. Intrinsics API version ─────────────────────────────────────────────
#ifdef __riscv_v_intrinsic
    std::cout << "[Intrinsics] __riscv_v_intrinsic = " << __riscv_v_intrinsic;
    if (__riscv_v_intrinsic >= 1000000)
        std::cout << "  →  header API v1.0+" << std::endl;
    else
        std::cout << "  →  header API pre-1.0 (GCC 15 known issue; ISA still valid)"
                  << std::endl;
#else
    std::cout << "[Intrinsics] __riscv_v_intrinsic NOT DEFINED" << std::endl;
#endif

    // ── 3. Compile-time HW info ───────────────────────────────────────────────
#ifdef __riscv_v_min_vlen
    std::cout << "[CT]        compile-time min VLEN = " << __riscv_v_min_vlen << " bits" << std::endl;
#endif
#ifdef __riscv_v_elen
    std::cout << "[CT]        ELEN     = " << __riscv_v_elen << " bits" << std::endl;
#endif
#ifdef __riscv_v_elen_fp
    std::cout << "[CT]        ELEN_FP  = " << __riscv_v_elen_fp << " bits" << std::endl;
#endif

    // ── 4. Runtime VLEN detection ─────────────────────────────────────────────
    {
        size_t vl = __riscv_vsetvl_e32m1(1024);
        size_t runtime_vlen = vl * 32;
        std::cout << "[RT]        runtime vl (e32m1, avl=1024) = " << vl
                  << "  →  VLEN = " << runtime_vlen << " bits";
        if (runtime_vlen == __riscv_v_min_vlen)
            std::cout << "  (matches compile-time)" << std::endl;
        else
            std::cout << "  (differs from compile-time " << __riscv_v_min_vlen
                      << " — runtime is authoritative)" << std::endl;
    }

    std::cout << std::endl;

    // ── 5. Functional test: integer add ──────────────────────────────────────
    // Non-power-of-two size forces tail processing to execute.
    constexpr size_t N = 105;
    std::vector<int32_t> a(N), b(N), c(N, 0);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<int32_t>(i);
        b[i] = static_cast<int32_t>(i * 2);
    }

    {
        const int32_t* pA = a.data();
        const int32_t* pB = b.data();
        int32_t*       pC = c.data();
        size_t avl = N;
        while (avl > 0) {
            size_t vl      = __riscv_vsetvl_e32m1(avl);
            vint32m1_t va  = __riscv_vle32_v_i32m1(pA, vl);
            vint32m1_t vb  = __riscv_vle32_v_i32m1(pB, vl);
            vint32m1_t vc  = __riscv_vadd_vv_i32m1(va, vb, vl);
            __riscv_vse32_v_i32m1(pC, vc, vl);
            pA += vl; pB += vl; pC += vl; avl -= vl;
        }
    }

    bool ok1 = true;
    for (size_t i = 0; i < N && ok1; ++i)
        if (c[i] != a[i] + b[i]) {
            std::cerr << "[FAIL] int-add mismatch at " << i
                      << ": expected " << a[i] + b[i] << ", got " << c[i] << std::endl;
            ok1 = false;
        }
    std::cout << "[Test 1/3]  int32 vadd          " << (ok1 ? "PASS" : "FAIL") << std::endl;

    // ── 6. Functional test: float multiply ────────────────────────────────────
    std::vector<float> fa(N), fb(N), fc(N, 0.f);
    for (size_t i = 0; i < N; ++i) {
        fa[i] = static_cast<float>(i) * 0.5f;
        fb[i] = static_cast<float>(i) * 1.5f;
    }

    {
        const float* pA = fa.data();
        const float* pB = fb.data();
        float*       pC = fc.data();
        size_t avl = N;
        while (avl > 0) {
            size_t vl        = __riscv_vsetvl_e32m1(avl);
            vfloat32m1_t va  = __riscv_vle32_v_f32m1(pA, vl);
            vfloat32m1_t vb  = __riscv_vle32_v_f32m1(pB, vl);
            vfloat32m1_t vc  = __riscv_vfmul_vv_f32m1(va, vb, vl);
            __riscv_vse32_v_f32m1(pC, vc, vl);
            pA += vl; pB += vl; pC += vl; avl -= vl;
        }
    }

    bool ok2 = true;
    for (size_t i = 0; i < N && ok2; ++i) {
        float expected = fa[i] * fb[i];
        float diff     = fc[i] - expected;
        if (diff < -1e-4f || diff > 1e-4f) {
            std::cerr << "[FAIL] fp-mul mismatch at " << i
                      << ": expected " << expected << ", got " << fc[i] << std::endl;
            ok2 = false;
        }
    }
    std::cout << "[Test 2/3]  float32 vfmul       " << (ok2 ? "PASS" : "FAIL") << std::endl;

    // ── 7. Functional test: masked store ─────────────────────────────────────
    // Zero out every odd-indexed element using a mask.
    std::vector<int32_t> m(N);
    for (size_t i = 0; i < N; ++i) m[i] = static_cast<int32_t>(i);

    {
        int32_t* pM  = m.data();
        size_t avl   = N;
        size_t base  = 0;
        while (avl > 0) {
            size_t vl = __riscv_vsetvl_e32m1(avl);

            // vid produces unsigned indices; add scalar base
            vuint32m1_t vidx  = __riscv_vid_v_u32m1(vl);
            vuint32m1_t vbase = __riscv_vmv_v_x_u32m1(static_cast<uint32_t>(base), vl);
            vuint32m1_t vabs  = __riscv_vadd_vv_u32m1(vidx, vbase, vl);

            // mask: element index is odd (LSB set)
            vbool32_t odd = __riscv_vmsne_vx_u32m1_b32(
                __riscv_vand_vx_u32m1(vabs, 1u, vl), 0u, vl);

            // store 0 into odd positions, leave even positions untouched
            __riscv_vse32_v_i32m1_m(odd, pM, __riscv_vmv_v_x_i32m1(0, vl), vl);

            pM += vl; avl -= vl; base += vl;
        }
    }

    bool ok3 = true;
    for (size_t i = 0; i < N && ok3; ++i) {
        int32_t expected = (i % 2 == 0) ? static_cast<int32_t>(i) : 0;
        if (m[i] != expected) {
            std::cerr << "[FAIL] masked-store mismatch at " << i
                      << ": expected " << expected << ", got " << m[i] << std::endl;
            ok3 = false;
        }
    }
    std::cout << "[Test 3/3]  masked vse32        " << (ok3 ? "PASS" : "FAIL") << std::endl;

    std::cout << std::endl;
    bool all = ok1 && ok2 && ok3;
    std::cout << "=== Overall: " << (all ? "ALL PASS" : "SOME TESTS FAILED") << " ===" << std::endl;
    return all ? 0 : 1;
}
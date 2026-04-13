#include <iostream>
#include <cstdint>

#ifdef HOST_MODE
    #include <gtest/gtest.h>
#else
    #include <cstdio>
    #include <riscv_vector.h>

    #define ASSERT_RVV(cond, msg) \
        if (!(cond)) { \
            printf("[FAIL] %s\n", msg); \
            return 1; \
        } else { \
            printf("[PASS] %s\n", msg); \
        }
#endif


#ifdef HOST_MODE
TEST(CannyInfrastructure, HostSanity) {
    int a = 1;
    EXPECT_EQ(a, 1);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#else
int main() {
    printf("--- Running QEMU-side Assertions ---\n");

    // Placeholder for your future RVV logic tests
    int rvv_ready = 1; 
    ASSERT_RVV(rvv_ready == 1, "RVV Environment Initialization");

    printf("All QEMU tests passed.\n");
    return 0;
}
#endif
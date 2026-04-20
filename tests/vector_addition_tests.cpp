#include <iostream>
#include <vector>
#include <cstdint>
#include <riscv_vector.h>

void vector_add(const int32_t* a, const int32_t* b, int32_t* c, size_t n);


int main()
{
    bool success = true;
    
    size_t n = 10;
    const int32_t a[n] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    const int32_t b[n] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int32_t c[n] = {0};

    vector_add(a, b, c, n);

    for (size_t i = 0; i < n; i++)
    {
        std::cout << "element " << i << "=" << c[i];
        if (c[i] != b[i] + a[i])
        {
            success = false;
            std::cout << "[FAIL]";
        }
        std::cout << std::endl;
    }

    return success ? 0 : 1;
}


void vector_add(const int32_t* a, const int32_t* b, int32_t* c, size_t n)
{
    for (size_t vl; n > 0; n-=vl, a+=vl, b+=vl, c+=vl)
    {
        vl = __riscv_vsetvl_e32m1(n);

        vint32m1_t va = __riscv_vle32_v_i32m1(a, vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(b, vl);
        vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, vl);

        __riscv_vse32_v_i32m1(c, vc, vl);
    }
}
module;
#include <stdint.h>
#include <math.h>
export module ggml:cpu.to_float;
import :ds;
import :types;

export
{
    template <typename T>
    void to_float(const T* x, float* y, int64_t n)
    {
        for (int64_t i = 0; i < n; i++) {
            y[i] = toFloat32<T>(x[i]);
        }
    }
}
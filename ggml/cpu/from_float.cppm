module;
#include <stdint.h>
#include <math.h>
export module ggml:cpu.from_float;
import :ds;
import :types;

export
{
    template <typename T>
    void from_float(const float* x, T* y, int64_t n)
    {
        for (int64_t i = 0; i < n; i++) {
            y[i] = fromFloat32<T>(x[i]);
        }
    }

    template <typename T>
    void from_float_wrapper(const float* x, void* y, int64_t n)
    {
        from_float<T>(x, reinterpret_cast<T*>(y), n);
    }
}
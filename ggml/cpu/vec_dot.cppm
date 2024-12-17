module;
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

export module ggml:cpu.vec_dot;
import :ds;
import :types;

using ggml_float = double;

export
{
    template <typename T>
    float ggml_vec_dot(int n, const T* x, const T* y, int nrc)
    {
        assert(nrc == 1);
        // Waiting for C++26 SIMD
        ggml_float sumf = 0.0;
        for (int i = 0; i < n; ++i) {
            sumf += ggml_float(toFloat32(x[i])) * ggml_float(toFloat32(y[i]));
        }
        return sumf;
    }

    template <typename T>
    void ggml_vec_dot_wrapper(int n, float* out, size_t, const void* x, size_t, const void* y, size_t, int nrc)
    {
        *out = ggml_vec_dot<T>(n, reinterpret_cast<const T*>(x), reinterpret_cast<const T*>(y), nrc);
    }
}

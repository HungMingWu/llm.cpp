#include "cuda_func.h"
#include "internal_ds.h"

void argsort_f32_i32_cuda(ggml_cuda_pool&,
    const float* x, int* dst,
    const int ncols, const int nrows,
    ggml_sort_order order, cudaStream_t stream)
{
    argsort_f32_i32_cuda_bitonic(x, dst, ncols, nrows, order, stream);
}
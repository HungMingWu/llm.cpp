#include "cuda_func.h"

void mean_cuda(const mean_context& ctx, cudaStream_t stream) {
    mean_fallback(ctx, stream);
}
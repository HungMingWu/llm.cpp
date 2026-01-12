#include "cuda_func.h"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        include <cuda/iterator>
#        define CUB_TOP_K_AVAILABLE
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool& pool,
    const float* src,
    int* dst,
    const int        ncols,
    const int        k,
    cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
        cuda::execution::output_ordering::unsorted);
    auto stream_env = cuda::stream_ref{ stream };
    auto env = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
        env);

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void* d_temp_storage = temp_storage_alloc.get();

    DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
        ncols, k, env);
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

void top_k_cuda(const top_k_context& ctx, cudaStream_t  stream) {

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < ctx.nrows; i++) {
        top_k_cub(ctx.pool, ctx.src0_d + i * ctx.ncols, ctx.dst_d + i * ctx.k, ctx.ncols, ctx.k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad = next_power_of_2(ctx.ncols);
    const size_t shared_mem = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;

    ggml_cuda_pool_alloc<int> temp_dst_alloc(ctx.pool, ctx.ncols * ctx.nrows);
    int* tmp_dst = temp_dst_alloc.get();

    if (shared_mem > max_shared_mem || ctx.ncols > 1024) {
        argsort_f32_i32_cuda(ctx.pool, ctx.src0_d, tmp_dst, ctx.ncols, ctx.nrows, internal::GGML_SORT_ORDER_DESC, stream);
    }
    else {
        argsort_f32_i32_cuda_bitonic(ctx.src0_d, tmp_dst, ctx.ncols, ctx.nrows, internal::GGML_SORT_ORDER_DESC, stream);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(ctx.dst_d, ctx.k * sizeof(int), tmp_dst, ctx.ncols * sizeof(int), ctx.k * sizeof(int), ctx.nrows,
        cudaMemcpyDeviceToDevice, stream));
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(ctx.pool, ctx.ncols * ctx.nrows);
    int* tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(ctx.src0_d, tmp_dst, ctx.ncols, ctx.nrows, internal::GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(ctx.dst_d, ctx.k * sizeof(int), tmp_dst, ctx.ncols * sizeof(int), ctx.k * sizeof(int), ctx.nrows,
        cudaMemcpyDeviceToDevice, stream));
#endif
}

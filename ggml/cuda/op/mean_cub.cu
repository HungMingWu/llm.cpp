#include <cub/cub.cuh>
#include "cuda_func.h"
using namespace cub;

template <typename T> __global__ void divide_by_count(T* result, size_t count) {
    *result /= static_cast<T>(count);
}

void mean_cuda(ggml_cuda_pool& pool, const float* src0_d, float* dst_d, const int64_t ncols, const int64_t nrows, cudaStream_t stream) {
    // Special case for reducing vectors
#ifdef USE_CUDA_GRAPH
    cudaStreamCaptureStatus iscapturing;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &iscapturing));
#endif // USE_CUDA_GRAPH
    if ((nrows == 1) &&
#ifdef USE_CUDA_GRAPH
        // CUDA_GRAPHS_DISABLED
        ((ncols > 65536) &&
            ((ctx.cuda_graph->instance == nullptr) && (iscapturing == cudaStreamCaptureStatusNone) ||
                ctx.cuda_graph->disable_due_to_gpu_arch || ctx.cuda_graph->disable_due_to_too_many_updates ||
                ctx.cuda_graph->disable_due_to_failed_graph_capture)) ||
        // CUDA_GRAPHS ENABLED
        ((ncols > 32768) &&
            !((ctx.cuda_graph->instance == nullptr) && (iscapturing == cudaStreamCaptureStatusNone) ||
                ctx.cuda_graph->disable_due_to_gpu_arch || ctx.cuda_graph->disable_due_to_too_many_updates ||
                ctx.cuda_graph->disable_due_to_failed_graph_capture))) {
#else
        (ncols > 65536)) {
#endif // USE_CUDA_GRAPH
        // Single row - use device-wide reduction
        size_t           tmp_size = 0;

        DeviceReduce::Sum(nullptr, tmp_size, src0_d, dst_d, ncols, stream);

        ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);
        DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, src0_d, dst_d, ncols, stream);

        // Divide by ncols
        divide_by_count<float> << <1, 1, 0, stream >> > (dst_d, ncols);
        return;
    }
    mean_fallback(src0_d, dst_d, ncols, nrows, stream);
}
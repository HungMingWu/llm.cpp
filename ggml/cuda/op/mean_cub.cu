#include <cub/cub.cuh>
#include "cuda_func.h"
using namespace cub;

template <typename T> __global__ void divide_by_count(T* result, size_t count) {
    *result /= static_cast<T>(count);
}

void mean_cuda(const mean_context& ctx, cudaStream_t stream) {
    // Special case for reducing vectors
    const bool is_special_case = [=]() {
        if (ctx.nrows != 1) return false;
        if constexpr (use_cuda_graph_v) {
            cudaStreamCaptureStatus iscapturing;
            CUDA_CHECK(cudaStreamIsCapturing(stream, &iscapturing));

            // Determine if CUDA graphs are effectively disabled for this context
            // (no graph instance exists and we're not capturing, OR graphs are explicitly enabled)
            const bool cuda_graph_enabled = (!ctx.any_cuda_graph_has_instance && iscapturing == cudaStreamCaptureStatusNone)
                || ctx.any_cuda_graph_enabled;

            if (ctx.ncols > 65536 && cuda_graph_enabled) return true;
            // CUDA graphs are enabled - use lower threshold
            if (ctx.ncols > 32768 && !cuda_graph_enabled) return true;
            return false;
        }
        else {
            return ctx.ncols > 65536;
        }
    }();
    if (is_special_case) {
        // Single row - use device-wide reduction
        size_t           tmp_size = 0;

        DeviceReduce::Sum(nullptr, tmp_size, ctx.src0_d, ctx.dst_d, ctx.ncols, stream);

        ggml_cuda_pool_alloc<uint8_t> tmp_alloc(ctx.pool, tmp_size);
        DeviceReduce::Sum(tmp_alloc.ptr, tmp_size, ctx.src0_d, ctx.dst_d, ctx.ncols, stream);

        // Divide by ncols
        divide_by_count<float> << <1, 1, 0, stream >> > (ctx.dst_d, ctx.ncols);
        return;
    }
    mean_fallback(ctx, stream);
}
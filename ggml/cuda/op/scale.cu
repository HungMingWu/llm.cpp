#include <span>
#include "launch.cuh"

void scale_f32_cuda(const float* x, float* dst, const float scale, 
    const float bias, const size_t nelements, cudaStream_t stream) {
    std::span x_span{ x, nelements };
    std::span dst_span{ dst, nelements };
    launch_functor(stream, std::make_tuple(nelements),
        [=] __device__(int64_t idx) {
            dst_span[idx] = scale * x_span[idx] + bias;
        }
    );
}
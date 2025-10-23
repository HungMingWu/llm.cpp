#include <span>
#include "launch.cuh"

void arange_f32_cuda(float *dst, size_t dst_size, const float start, const float step, cudaStream_t stream) {
    std::span dst_span{ dst, dst_size };
    launch_functor(stream, std::make_tuple(dst_size),
        [=] __device__ (int64_t nidx) {
            dst_span[nidx] = start + step * nidx;
	    }
    );
}

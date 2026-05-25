#include "common.cuh"

static __global__ void softcap_f32(const float* x, float* dst, const float scale, const float softcap, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    ggml_cuda_pdl_sync();
    dst[i] = tanhf(scale * x[i]) * softcap;
}

void softcap_f32_cuda(const float* x, float* dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_SOFTCAP_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(softcap_f32, launch_params, x, dst, scale, softcap, k);
}
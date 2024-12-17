#include "internal_ds.h"
#include "convert.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template <typename src_t, typename dst_t>
static __global__ void convert_unary(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k) {
    const int64_t i = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }
    const src_t* x = (src_t*)vx;
    y[i] = x[i];
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void* __restrict__ vx, dst_t* __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE;
    convert_unary<src_t> << <num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream >> > (vx, y, k);
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type)
{
    switch (type) {
#if 0
    case GGML_TYPE_Q4_0:
        return dequantize_row_q4_0_cuda;
    case GGML_TYPE_Q4_1:
        return dequantize_row_q4_1_cuda;
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
    case GGML_TYPE_Q8_0:
        return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
    case GGML_TYPE_Q2_K:
        return dequantize_row_q2_K_cuda;
    case GGML_TYPE_Q3_K:
        return dequantize_row_q3_K_cuda;
    case GGML_TYPE_Q4_K:
        return dequantize_row_q4_K_cuda;
    case GGML_TYPE_Q5_K:
        return dequantize_row_q5_K_cuda;
    case GGML_TYPE_Q6_K:
        return dequantize_row_q6_K_cuda;
    case GGML_TYPE_IQ2_XXS:
        return dequantize_row_iq2_xxs_cuda;
    case GGML_TYPE_IQ2_XS:
        return dequantize_row_iq2_xs_cuda;
    case GGML_TYPE_IQ2_S:
        return dequantize_row_iq2_s_cuda;
    case GGML_TYPE_IQ3_XXS:
        return dequantize_row_iq3_xxs_cuda;
    case GGML_TYPE_IQ1_S:
        return dequantize_row_iq1_s_cuda;
    case GGML_TYPE_IQ1_M:
        return dequantize_row_iq1_m_cuda;
    case GGML_TYPE_IQ4_NL:
        return dequantize_row_iq4_nl_cuda;
    case GGML_TYPE_IQ4_XS:
        return dequantize_row_iq4_xs_cuda;
    case GGML_TYPE_IQ3_S:
        return dequantize_row_iq3_s_cuda;
#endif
    case GGML_TYPE_F16:
        return convert_unary_cuda<half>;
    default:
        return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type)
{
    switch (type) {
#if 0
    case GGML_TYPE_Q4_0:
        return dequantize_row_q4_0_cuda;
    case GGML_TYPE_Q4_1:
        return dequantize_row_q4_1_cuda;
    case GGML_TYPE_Q5_0:
        return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
    case GGML_TYPE_Q5_1:
        return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
    case GGML_TYPE_Q8_0:
        if (ggml_cuda_info().devices[ggml_cuda_get_device()].cc >= GGML_CUDA_CC_PASCAL) {
            return dequantize_block_q8_0_f16_cuda;
        }
        return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
    case GGML_TYPE_Q2_K:
        return dequantize_row_q2_K_cuda;
    case GGML_TYPE_Q3_K:
        return dequantize_row_q3_K_cuda;
    case GGML_TYPE_Q4_K:
        return dequantize_row_q4_K_cuda;
    case GGML_TYPE_Q5_K:
        return dequantize_row_q5_K_cuda;
    case GGML_TYPE_Q6_K:
        return dequantize_row_q6_K_cuda;
    case GGML_TYPE_IQ2_XXS:
        return dequantize_row_iq2_xxs_cuda;
    case GGML_TYPE_IQ2_XS:
        return dequantize_row_iq2_xs_cuda;
    case GGML_TYPE_IQ2_S:
        return dequantize_row_iq2_s_cuda;
    case GGML_TYPE_IQ3_XXS:
        return dequantize_row_iq3_xxs_cuda;
    case GGML_TYPE_IQ1_S:
        return dequantize_row_iq1_s_cuda;
    case GGML_TYPE_IQ1_M:
        return dequantize_row_iq1_m_cuda;
    case GGML_TYPE_IQ4_NL:
        return dequantize_row_iq4_nl_cuda;
    case GGML_TYPE_IQ4_XS:
        return dequantize_row_iq4_xs_cuda;
    case GGML_TYPE_IQ3_S:
        return dequantize_row_iq3_s_cuda;
    case GGML_TYPE_F32:
        return convert_unary_cuda<float>;
#endif
    default:
        return nullptr;
    }
}
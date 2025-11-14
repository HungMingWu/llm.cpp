#include <assert.h>
#include "cuda_func.h"
#include "mmq.cuh"

void ggml_cuda_mul_mat_q_switch_type(ggml_cuda_pool& pool, const mmq_args& args, cudaStream_t stream) {
    switch (args.type_x) {
    case internal::GGML_TYPE_Q4_0:
        mul_mat_q_case<internal::GGML_TYPE_Q4_0, block_q4_0>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q4_1:
        mul_mat_q_case<internal::GGML_TYPE_Q4_1, block_q4_1>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q5_0:
        mul_mat_q_case<internal::GGML_TYPE_Q5_0, block_q5_0>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q5_1:
        mul_mat_q_case<internal::GGML_TYPE_Q5_1, block_q5_1>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q8_0:
        mul_mat_q_case<internal::GGML_TYPE_Q8_0, block_q8_0>(pool, args, stream);
        break;
    case internal::GGML_TYPE_MXFP4:
        mul_mat_q_case<internal::GGML_TYPE_MXFP4, block_mxfp4>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q2_K:
        mul_mat_q_case<internal::GGML_TYPE_Q2_K, block_q2_K>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q3_K:
        mul_mat_q_case<internal::GGML_TYPE_Q3_K, block_q3_K>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q4_K:
        mul_mat_q_case<internal::GGML_TYPE_Q4_K, block_q4_K>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q5_K:
        mul_mat_q_case<internal::GGML_TYPE_Q5_K, block_q5_K>(pool, args, stream);
        break;
    case internal::GGML_TYPE_Q6_K:
        mul_mat_q_case<internal::GGML_TYPE_Q6_K, block_q6_K>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ2_XXS:
        mul_mat_q_case<internal::GGML_TYPE_IQ2_XXS, block_iq2_xxs>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ2_XS:
        mul_mat_q_case<internal::GGML_TYPE_IQ2_XS, block_iq2_xs>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ2_S:
        mul_mat_q_case<internal::GGML_TYPE_IQ2_S, block_iq2_s>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ3_XXS:
        mul_mat_q_case<internal::GGML_TYPE_IQ3_XXS, block_iq3_xxs>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ3_S:
        mul_mat_q_case<internal::GGML_TYPE_IQ3_S, block_iq3_s>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ1_S:
        mul_mat_q_case<internal::GGML_TYPE_IQ1_S, block_iq1_s>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ4_XS:
        mul_mat_q_case<internal::GGML_TYPE_IQ4_XS, block_iq4_xs>(pool, args, stream);
        break;
    case internal::GGML_TYPE_IQ4_NL:
        mul_mat_q_case<internal::GGML_TYPE_IQ4_NL, block_iq4_nl>(pool, args, stream);
        break;
    default:
        GGML_ABORT("fatal error");
        break;
    }
}

module;
#include <stdint.h>
#include <unordered_map>

export module ggml:cpu.traits;
import :ds;
import :utility;

export
{
    struct ggml_type_traits_cpu {
        enum ggml_type           vec_dot_type;
        int64_t                  nrows; // number of rows to process simultaneously
    };

    std::unordered_map<ggml_type, ggml_type_traits_cpu> type_traits_cpu{
        {GGML_TYPE_F32, {
            .vec_dot_type = GGML_TYPE_F32,
            .nrows = 1,
        }},
        {GGML_TYPE_F16,  {
            .vec_dot_type = GGML_TYPE_F16,
            .nrows = 1,
        }},
        {GGML_TYPE_Q4_0, {
            .vec_dot_type = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        }},
        {GGML_TYPE_Q4_1, {
            .vec_dot_type = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        }},
        {GGML_TYPE_Q5_0, {
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        }},
        {GGML_TYPE_Q5_1, {
            .vec_dot_type = GGML_TYPE_Q8_1,
            .nrows = 1,
        }},
        {GGML_TYPE_Q8_0, {
            .vec_dot_type = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        }},
        {GGML_TYPE_Q8_1, {
            .vec_dot_type = GGML_TYPE_Q8_1,
            .nrows = 1,
        }},
        {GGML_TYPE_Q2_K, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_Q3_K, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_Q4_K, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_Q5_K, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_Q6_K, {
            .vec_dot_type = GGML_TYPE_Q8_K,
#if defined (__ARM_FEATURE_MATMUL_INT8)
            .nrows = 2,
#else
            .nrows = 1,
#endif
        }},
        {GGML_TYPE_IQ2_XXS, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ2_XS, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ3_XXS, {
            // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ3_S, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ2_S, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ1_S, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ1_M, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ4_NL, {
            .vec_dot_type = GGML_TYPE_Q8_0,
            .nrows = 1,
        }},
        {GGML_TYPE_IQ4_XS, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_Q8_K, {
        }},
        {GGML_TYPE_BF16, {
            .vec_dot_type = GGML_TYPE_BF16,
            .nrows = 1,
        }},
        {GGML_TYPE_TQ1_0, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }},
        {GGML_TYPE_TQ2_0, {
            .vec_dot_type = GGML_TYPE_Q8_K,
            .nrows = 1,
        }}
    };

	const ggml_type_traits_cpu* ggml_get_type_traits_cpu(enum ggml_type type) {
        auto it = type_traits_cpu.find(type);
        if (it == type_traits_cpu.end()) return nullptr;
        return &it->second;
    }

	bool ggml_exist_cpu_from_float(enum ggml_type type) {
        return is_one_of(type, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
            GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ1_S,
            GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_K,
            GGML_TYPE_BF16, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0);
	}
}

#pragma once
// CUDA Compiler doesn't support module
// Need a workaround


// NOTE: always add types at the end of the enum to keep backward compatibility
enum ggml_type : int {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q4_2 = 4, // support has been removed
    GGML_TYPE_Q4_3 = 5, // support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_Q4_0_4_4 = 31, // support has been removed
    GGML_TYPE_Q4_0_4_8 = 32, // support has been removed
    GGML_TYPE_Q4_0_8_8 = 33, // support has been removed
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    GGML_TYPE_IQ4_NL_4_4 = 36, // support has been removed
    GGML_TYPE_IQ4_NL_4_8 = 37, // support has been removed
    GGML_TYPE_IQ4_NL_8_8 = 38, // support has been removed
    GGML_TYPE_MXFP4 = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT = 40,
};

// precision
enum ggml_prec : int {
    GGML_PREC_DEFAULT = 0,
    GGML_PREC_F32 = 10,
};

enum ggml_op_pool : int {
    GGML_OP_POOL_MAX,
    GGML_OP_POOL_AVG,
    GGML_OP_POOL_COUNT,
};

enum ggml_sort_order : int {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

enum ggml_scale_mode : int {
    GGML_SCALE_MODE_NEAREST = 0,
    GGML_SCALE_MODE_BILINEAR = 1,

    GGML_SCALE_MODE_COUNT
};

enum ggml_scale_flag : int {
    GGML_SCALE_FLAG_ALIGN_CORNERS = (1 << 8)
};
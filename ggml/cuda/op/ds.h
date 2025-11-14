#pragma once

// CUDA Compiler doesn't support module
// Need a workaround

// Copy from outer's definition
namespace internal {
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

    enum ggml_glu_op : int {
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_GEGLU,
        GGML_GLU_OP_SWIGLU,
        GGML_GLU_OP_SWIGLU_OAI,
        GGML_GLU_OP_GEGLU_ERF,
        GGML_GLU_OP_GEGLU_QUICK,

        GGML_GLU_OP_COUNT,
    };
}

enum mmq_q8_1_ds_layout {
    MMQ_Q8_1_DS_LAYOUT_D4,
    MMQ_Q8_1_DS_LAYOUT_DS4,
    MMQ_Q8_1_DS_LAYOUT_D2S6,
};

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4 * QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4 * QK8_1 + 4 * sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4 * sizeof(block_q8_1), "Unexpected block_q8_1_mmq size");

struct ggml_cuda_mm_fusion_args_device {
    const void* x_bias = nullptr;
    const void* gate = nullptr;
    const void* gate_bias = nullptr;
    internal::ggml_glu_op glu_op;
};
module;
#include <assert.h>
#include <string_view>
#include <unordered_map>
#include "block.h"

export module ggml:traits;
import :ds;
import :types;
import :utility;

using ggml_from_float_t = void (*)(const float*, void*, int64_t);

struct ggml_type_traits {
    std::string_view type_name;
    int64_t                  blck_size;
    int64_t                  blck_size_interleave; // interleave elements in blocks
    size_t                   type_size;
    bool                     is_quantized;
    ggml_from_float_t        from_float_ref;
};

static std::unordered_map<ggml_type, ggml_type_traits> type_traits {
    {
        GGML_TYPE_I8,
        {
            .type_name = "i8",
            .blck_size = 1,
            .type_size = sizeof(int8_t),
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_I16,
        {
            .type_name = "i16",
            .blck_size = 1,
            .type_size = sizeof(int16_t),
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_I32,
        {
            .type_name = "i32",
            .blck_size = 1,
            .type_size = sizeof(int32_t),
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_I64,
        {
            .type_name = "i64",
            .blck_size = 1,
            .type_size = sizeof(int64_t),
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_F64,
        {
            .type_name = "f64",
            .blck_size = 1,
            .type_size = sizeof(double),
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_F32,
        {
            .type_name = "f32",
            .blck_size = 1,
            .type_size = sizeof(float),
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_Q4_2,
        {
            .type_name = "DEPRECATED",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_Q4_3,
        {
            .type_name = "DEPRECATED",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_Q4_0_4_4,
        {
            .type_name = "TYPE_Q4_0_4_4 REMOVED, use Q4_0 with runtime repacking",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_Q4_0_4_8,
        {
            .type_name = "TYPE_Q4_0_4_8 REMOVED, use Q4_0 with runtime repacking",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_Q4_0_8_8,
        {
            .type_name = "TYPE_Q4_0_8_8 REMOVED, use Q4_0 with runtime repacking",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_IQ4_NL_4_4,
        {
            .type_name = "TYPE_IQ4_NL_4_4 REMOVED, use IQ4_NL with runtime repacking",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_IQ4_NL_4_8,
        {
            .type_name = "TYPE_IQ4_NL_4_8 REMOVED, use IQ4_NL with runtime repacking",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_IQ4_NL_8_8,
        {
            .type_name = "TYPE_IQ4_NL_8_8 REMOVED, use IQ4_NL with runtime repacking",
            .blck_size = 0,
            .type_size = 0,
            .is_quantized = false,
        }
    },
    {
        GGML_TYPE_F16,
        {
            .type_name = "f16",
            .blck_size = 1,
            .type_size = sizeof(ggml_fp16_t),
            .is_quantized = false,
            //.from_float_ref = (ggml_from_float_t)ggml_fp32_to_fp16_row,
        }
    },
    {
        GGML_TYPE_Q4_0,
        {
            .type_name = "q4_0",
            .blck_size = block_q4_0::block_size,
            .type_size = sizeof(block_q4_0),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q4_0_ref,
        }
    },
    {
        GGML_TYPE_Q4_1,
        {
            .type_name = "q4_1",
            .blck_size = block_q4_1::block_size,
            .type_size = sizeof(block_q4_1),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q4_1_ref,
        }
    },
    {
        GGML_TYPE_Q5_0,
        {
            .type_name = "q5_0",
            .blck_size = block_q5_0::block_size,
            .type_size = sizeof(block_q5_0),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q5_0_ref,
        }
    },
    {
        GGML_TYPE_Q5_1,
        {
            .type_name = "q5_1",
            .blck_size = block_q5_1::block_size,
            .type_size = sizeof(block_q5_1),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q5_1_ref,
        }
    },
    {
        GGML_TYPE_Q8_0,
        {
            .type_name = "q8_0",
            .blck_size = block_q8_0::block_size,
            .type_size = sizeof(block_q8_0),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q8_0_ref,
        }
    },
    {
        GGML_TYPE_Q8_1,
        {
            .type_name = "q8_1",
            .blck_size = QK8_1,
            .type_size = sizeof(block_q8_1),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q8_1_ref,
        }
    },
    {
        GGML_TYPE_MXFP4,
        {
            .type_name = "mxfp4",
            .blck_size = block_mxfp4::block_size,
            .type_size = sizeof(block_mxfp4),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q8_1_ref,
        }
    },
    {
        GGML_TYPE_Q2_K,
        {
            .type_name = "q2_K",
            .blck_size = QK_K,
            .type_size = sizeof(block_q2_K),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q2_K_ref,
        }
    },
    {
        GGML_TYPE_Q3_K,
        {
            .type_name = "q3_K",
            .blck_size = QK_K,
            .type_size = sizeof(block_q3_K),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q3_K_ref,
        }
    },
    {
        GGML_TYPE_Q4_K,
        {
            .type_name = "q4_K",
            .blck_size = QK_K,
            .type_size = sizeof(block_q4_K),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q4_K_ref,
        }
    },
    {
        GGML_TYPE_Q5_K,
        {
            .type_name = "q5_K",
            .blck_size = QK_K,
            .type_size = sizeof(block_q5_K),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q5_K_ref,
        }
    },
    {
        GGML_TYPE_Q6_K,
        {
            .type_name = "q6_K",
            .blck_size = QK_K,
            .type_size = sizeof(block_q6_K),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_q6_K_ref,
        }
    },
    {
        GGML_TYPE_IQ2_XXS,
        {
            .type_name = "iq2_xxs",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq2_xxs),
            .is_quantized = true,
            //.from_float_ref = NULL,
        }
    },
    {
        GGML_TYPE_IQ2_XS,
        {
            .type_name = "iq2_xs",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq2_xs),
            .is_quantized = true,
            //.from_float_ref = NULL,
        }
    },
    {
        GGML_TYPE_IQ3_XXS,
        {
            .type_name = "iq3_xxs",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq3_xxs),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
        }
    },
    {
        GGML_TYPE_IQ3_S,
        {
            .type_name = "iq3_s",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq3_s),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_iq3_s_ref
        }
    },
    {
        GGML_TYPE_IQ2_S,
        {
            .type_name = "iq2_s",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq2_s),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_iq2_s_ref,
        }
    },
    {
        GGML_TYPE_IQ1_S,
        {
            .type_name = "iq1_s",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq1_s),
            .is_quantized = true,
            //.from_float_ref = NULL,
        }
    },
    {
        GGML_TYPE_IQ1_M,
        {
            .type_name = "iq1_m",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq1_m),
            .is_quantized = true,
            //.from_float_ref = NULL,
        }
    },
    {
        GGML_TYPE_IQ4_NL,
        {
            .type_name = "iq4_nl",
            .blck_size = block_iq4_nl::block_size,
            .type_size = sizeof(block_iq4_nl),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_iq4_nl_ref,
        }
    },
    {
        GGML_TYPE_IQ4_XS,
        {
            .type_name = "iq4_xs",
            .blck_size = QK_K,
            .type_size = sizeof(block_iq4_xs),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_iq4_xs_ref,
        }
    },
    {
        GGML_TYPE_Q8_K,
        {
            .type_name = "q8_K",
            .blck_size = QK_K,
            .type_size = sizeof(block_q8_K),
            .is_quantized = true,
        }
    },
    {
        GGML_TYPE_BF16,
        {
            .type_name = "bf16",
            .blck_size = 1,
            .type_size = sizeof(ggml_bf16_t),
            .is_quantized = false,
            //.from_float_ref = (ggml_from_float_t)ggml_fp32_to_bf16_row_ref,
        }
    },
    {
        GGML_TYPE_TQ1_0,
        {
            .type_name = "tq1_0",
            .blck_size = QK_K,
            .type_size = sizeof(block_tq1_0),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_tq1_0_ref,
        }
    },
    {
        GGML_TYPE_TQ2_0,
        {
            .type_name = "tq2_0",
            .blck_size = QK_K,
            .type_size = sizeof(block_tq2_0),
            .is_quantized = true,
            //.from_float_ref = (ggml_from_float_t)quantize_row_tq2_0_ref,
        }
    },
};

static const char* GGML_OP_NAME[GGML_OP_COUNT] = {
    "NONE",

    "DUP",
    "ADD",
    "ADD_ID",
    "ADD1",
    "ACC",
    "SUB",
    "MUL",
    "DIV",
    "SQR",
    "SQRT",
    "LOG",
    "SIN",
    "COS",
    "SUM",
    "SUM_ROWS",
    "MEAN",
    "ARGMAX",
    "COUNT_EQUAL",
    "REPEAT",
    "REPEAT_BACK",
    "CONCAT",
    "SILU_BACK",
    "NORM",
    "RMS_NORM",
    "RMS_NORM_BACK",
    "GROUP_NORM",
    "L2_NORM",

    "MUL_MAT",
    "MUL_MAT_ID",
    "OUT_PROD",

    "SCALE",
    "SET",
    "CPY",
    "CONT",
    "RESHAPE",
    "VIEW",
    "PERMUTE",
    "TRANSPOSE",
    "GET_ROWS",
    "GET_ROWS_BACK",
    "SET_ROWS",
    "DIAG",
    "DIAG_MASK_INF",
    "DIAG_MASK_ZERO",
    "SOFT_MAX",
    "SOFT_MAX_BACK",
    "ROPE",
    "ROPE_BACK",
    "CLAMP",
    "CONV_TRANSPOSE_1D",
    "IM2COL",
    "IM2COL_BACK",
    "CONV_2D",
    "CONV_2D_DW",
    "CONV_TRANSPOSE_2D",
    "POOL_1D",
    "POOL_2D",
    "POOL_2D_BACK",
    "UPSCALE",
    "PAD",
    "PAD_REFLECT_1D",
    "ROLL",
    "ARANGE",
    "TIMESTEP_EMBEDDING",
    "ARGSORT",
    "LEAKY_RELU",

    "FLASH_ATTN_EXT",
    "FLASH_ATTN_BACK",
    "SSM_CONV",
    "SSM_SCAN",
    "WIN_PART",
    "WIN_UNPART",
    "GET_REL_POS",
    "ADD_REL_POS",
    "RWKV_WKV6",
    "GATED_LINEAR_ATTN",
    "RWKV_WKV7",

    "UNARY",

    "CUSTOM",

    "CROSS_ENTROPY_LOSS",
    "CROSS_ENTROPY_LOSS_BACK",
    "OPT_STEP_ADAMW",
    "OPT_STEP_SGD",

    "GLU",
};

static const char* GGML_OP_SYMBOL[GGML_OP_COUNT] = {
    "none",

    "x",
    "x+y",
    "x[i]+y",
    "x+y",
    "view(x,nb,offset)+=y->x",
    "x-y",
    "x*y",
    "x/y",
    "x^2",
    "√x",
    "log(x)",
    "sin(x)",
    "cos(x)",
    "Σx",
    "Σx_k",
    "Σx/n",
    "argmax(x)",
    "count_equal(x)",
    "repeat(x)",
    "repeat_back(x)",
    "concat(x, y)",
    "silu_back(x)",
    "norm(x)",
    "rms_norm(x)",
    "rms_norm_back(x)",
    "group_norm(x)",
    "l2_norm(x)",

    "X*Y",
    "X[i]*Y",
    "X*Y",

    "x*v",
    "y-\\>view(x)",
    "x-\\>y",
    "cont(x)",
    "reshape(x)",
    "view(x)",
    "permute(x)",
    "transpose(x)",
    "get_rows(x)",
    "get_rows_back(x)",
    "set_rows(x)",
    "diag(x)",
    "diag_mask_inf(x)",
    "diag_mask_zero(x)",
    "soft_max(x)",
    "soft_max_back(x)",
    "rope(x)",
    "rope_back(x)",
    "clamp(x)",
    "conv_transpose_1d(x)",
    "im2col(x)",
    "im2col_back(x)",
    "conv_2d(x)",
    "conv_2d_dw(x)",
    "conv_transpose_2d(x)",
    "pool_1d(x)",
    "pool_2d(x)",
    "pool_2d_back(x)",
    "upscale(x)",
    "pad(x)",
    "pad_reflect_1d(x)",
    "roll(x)",
    "arange(start, stop, step)",
    "timestep_embedding(timesteps, dim, max_period)",
    "argsort(x)",
    "leaky_relu(x)",

    "flash_attn_ext(x)",
    "flash_attn_back(x)",
    "ssm_conv(x)",
    "ssm_scan(x)",
    "win_part(x)",
    "win_unpart(x)",
    "get_rel_pos(x)",
    "add_rel_pos(x)",
    "rwkv_wkv6(k, v, r, tf, td, s)",
    "gated_linear_attn(k, v, q, gate, s)",
    "rwkv_wkv7(r, w, k, v, a, b, s)",

    "unary(x)",

    "custom(x)",

    "cross_entropy_loss(x,y)",
    "cross_entropy_loss_back(x,y)",
    "adamw(x)",
    "sgd(x)",

    "glu(x)",
};

static const char* GGML_UNARY_OP_NAME[GGML_UNARY_OP_COUNT] = {
    "ABS",
    "SGN",
    "NEG",
    "STEP",
    "TANH",
    "ELU",
    "RELU",
    "SIGMOID",
    "GELU",
    "GELU_QUICK",
    "SILU",
    "HARDSWISH",
    "HARDSIGMOID",
    "EXP",
    "GELU_ERF",
};

static const char* GGML_GLU_OP_NAME[GGML_GLU_OP_COUNT] = {
    "REGLU",
    "GEGLU",
    "SWIGLU",
    "SWIGLU_OAI",
    "GEGLU_ERF",
    "GEGLU_QUICK",
};

export
{
    size_t ggml_blck_size(enum ggml_type type) {
        return type_traits[type].blck_size;
    }

    std::string_view ggml_type_name(enum ggml_type type) {
        return type < GGML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
    }

    size_t ggml_type_size(enum ggml_type type) {
        return type_traits[type].type_size;
    }

    size_t ggml_row_size(enum ggml_type type, int64_t ne) {
        assert(ne % ggml_blck_size(type) == 0);
        return ggml_type_size(type) * ne / ggml_blck_size(type);
    }

    bool ggml_is_quantized(enum ggml_type type) {
        return type_traits[type].is_quantized;
    }

    const char* ggml_op_name(enum ggml_op op) {
        return GGML_OP_NAME[op];
    }

    const char* ggml_op_symbol(enum ggml_op op) {
        return GGML_OP_SYMBOL[op];
    }

    bool ggml_exist_to_float(enum ggml_type type) {
        return is_one_of(type, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ4_XS, GGML_TYPE_BF16, GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0);
    }

    const char* ggml_unary_op_name(enum ggml_unary_op op) {
        return GGML_UNARY_OP_NAME[op];
    }

    const char* ggml_glu_op_name(enum ggml_glu_op op) {
        return GGML_GLU_OP_NAME[op];
    }

    const struct ggml_type_traits* ggml_get_type_traits(enum ggml_type type) {
        //GGML_ASSERT(type < GGML_TYPE_COUNT);
        return &type_traits[type];
    }

    template <typename type>
    struct vec_dot_trait;

    template <>
    struct vec_dot_trait<ggml_fp32_t> {
        using type = ggml_fp32_t;
    };

    template <>
    struct vec_dot_trait<ggml_fp16_t> {
        using type = ggml_fp16_t;
    };

    template <>
    struct vec_dot_trait<ggml_bf16_t> {
        using type = ggml_bf16_t;
    };

    template <>
    struct vec_dot_trait<block_q4_0> {
        using type = block_q8_0;
    };

    template <>
    struct vec_dot_trait<block_q4_1> {
        using type = block_q8_1;
    };

    template <>
    struct vec_dot_trait<block_mxfp4> {
        using type = block_q8_0;
    };

    template <>
    struct vec_dot_trait<block_q5_0> {
        using type = block_q8_0;
    };

    template <>
    struct vec_dot_trait<block_q5_1> {
        using type = block_q8_1;
    };

    template <>
    struct vec_dot_trait<block_q8_0> {
        using type = block_q8_0;
    };

    template <>
    struct vec_dot_trait<block_q8_1> {
        using type = block_q8_1;
    };

    template <>
    struct vec_dot_trait<block_q2_K> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_q3_K> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_q4_K> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_q5_K> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_q6_K> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq1_s> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq1_m> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq2_xxs> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq2_xs> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq2_s> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq3_xxs> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq3_s> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_iq4_nl> {
        using type = block_q8_0;
    };

    template <>
    struct vec_dot_trait<block_iq4_xs> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_tq1_0> {
        using type = block_q8_K;
    };

    template <>
    struct vec_dot_trait<block_tq2_0> {
        using type = block_q8_K;
    };
}

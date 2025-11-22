module;
#include "common.h"
#include <bit>
#include <span>
#include "op/cuda_func.h"

static constexpr int64_t MMF_ROWS_PER_BLOCK = 32;

module ggml;
import :cuda.utils;

namespace utils
{
    bin_bcast_context create_bcast_context(const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst)
    {
        bin_bcast_context ctx{
            .dst_d = dst->data,
            .src0_type = std::bit_cast<internal::ggml_type>(src0->type),
            .src1_type = std::bit_cast<internal::ggml_type>(src1->type),
            .dst_type = std::bit_cast<internal::ggml_type>(dst->type),
			.src0_ne = {src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]},
            .src0_nb = {src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]},
            .src1_ne = {src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]},
            .src1_nb = {src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]},
            .dst_ne = { dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]},
            .dst_nb = { dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]}
        };
        for (size_t i = 0; i < dst->src.size(); i++)
            ctx.src_data[i] = dst->src[i]->data;
        return ctx;
    }

    bool should_use_mmf(ggml_type type, int cc, int warp_size, std::span<const int64_t> src0_ne,
        std::span<const size_t> src0_nb, int64_t src1_ncols, bool mul_mat_id)
    {
        if (ggml_is_quantized(type)) return false;

        const size_t ts = ggml_type_size(type);

        if (src0_ne[0] % (warp_size * (4 / ts)) != 0) {
            return false;
        }

        if (src0_nb[0] != ts) {
            return false;
        }

        // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
        for (auto nb : src0_nb.subspan(1)) {
            if (nb % (2 * ts) != 0) {
                return false;
            }
        }
        if (src0_ne[1] % MMF_ROWS_PER_BLOCK != 0) {
            return false;
        }

        if (mul_mat_id) {
            if (src0_ne[1] <= 1024 && src1_ncols > 512) {
                return false;
            }
            else if (src0_ne[1] > 1024 && src1_ncols > 128) {
                return false;
            }
        }
        else {
            if (src1_ncols > 16) {
                return false;
            }
        }

        switch (type) {
        case GGML_TYPE_F32:
            return ampere_mma_available(cc);
        case GGML_TYPE_F16:
            return volta_mma_available(cc) || turing_mma_available(cc);
        case GGML_TYPE_BF16:
            return ampere_mma_available(cc);
        default:
            return false;
        }
    }

    bool should_use_mmq(ggml_type type, int cc, int64_t ne11) {
        if constexpr (force_enable_cuda_blas_v) {
            return false;
        }

        bool mmq_supported;

        switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
        }

        if (!mmq_supported) {
            return false;
        }

        if (turing_mma_available(cc)) {
            return true;
        }

        if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
            return false;
        }

#ifdef GGML_CUDA_FORCE_MMQ
        return true;
#endif //GGML_CUDA_FORCE_MMQ

        if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
            return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
        }

        if (amd_mfma_available(cc)) {
            // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
            // performs better but is currently suffering from a crash on this architecture.
            // TODO: Revisit when hipblaslt is fixed on CDNA3
            if (GGML_CUDA_CC_IS_CDNA3(cc)) {
                return true;
            }
            if (ne11 <= 128 || type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
                return true;
            }
            if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
                return true;
            }
            return false;
        }

        return (!GGML_CUDA_CC_IS_RDNA4(cc) && !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    bool should_use_mmvf(ggml_type type, int cc, std::span<const int64_t> src0_ne, std::span<const size_t> src0_nb, int64_t ne11) {
        if (src0_ne[0] % 2 != 0) {
            return false;
        }

        const size_t ts = ggml_type_size(type);
        if (src0_nb[0] != ts) {
            return false;
        }

        // Pointers not aligned to the size of half2/nv_bfloat162/float2 would result in a crash:
        for (auto nb : src0_nb.subspan(1)) {
            if (nb % (2 * ts) != 0) {
                return false;
            }
        }

        switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (ampere_mma_available(cc)) {
                    return ne11 <= 3;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
                if (ampere_mma_available(cc)) {
                    return src0_small && ne11 == 1;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        default:
            return false;
        }
    }

    bool should_use_mmv(ggml_type type, int cc, std::span<const int64_t> src0_ne, int64_t ne11) {
        if (src0_ne[0] % 2 != 0) {
            return false;
        }
        switch (type) {
        case GGML_TYPE_F32:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return ne11 <= 8;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    return ne11 <= 4;
                }
                return ne11 <= 3;
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp32_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_F16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (fp16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (fp16_mma_hardware_available(cc)) {
                    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
                        return ne11 <= 5;
                    }
                    return ne11 <= 2;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        case GGML_TYPE_BF16:
            if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
                const bool src0_small = (src0_ne[1] <= 512 || src0_ne[2] * src0_ne[3] == 1);
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    return src0_small && ne11 <= 4;
                }
                if (bf16_mma_hardware_available(cc)) {
                    return src0_small && ne11 <= 3;
                }
                return ne11 <= 8;
            }
            else if (GGML_CUDA_CC_IS_AMD(cc)) {
                if (bf16_mma_hardware_available(cc)) {
                    return ne11 <= 3;
                }
                return ne11 <= 8;
            }
            return ne11 <= 8;
        default:
            return false;
        }
    }
}
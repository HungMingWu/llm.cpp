module;
#include <stdint.h>
#include <span>
#include "op/cuda_func.h"

module ggml:cuda.utils;
import :ds;

namespace utils
{
	// Best FlashAttention kernel for a specific GPU:
	enum best_fattn_kernel {
		BEST_FATTN_KERNEL_NONE = 0,
		BEST_FATTN_KERNEL_TILE = 200,
		BEST_FATTN_KERNEL_VEC = 100,
		BEST_FATTN_KERNEL_WMMA_F16 = 300,
		BEST_FATTN_KERNEL_MMA_F16 = 400,
	};

	best_fattn_kernel ggml_cuda_get_best_fattn_kernel([[maybe_unused]] const int device, [[maybe_unused]] const ggml_tensor* dst);
	bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor* dst);

	bin_bcast_context create_bcast_context(const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst);

	bool should_use_mmf(ggml_type type, int cc, int warp_size, std::span<const int64_t> src0, std::span<const size_t> src0_nb, int64_t src1_ncols, bool mul_mat_id);
	bool should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);
	bool should_use_mmvf(ggml_type type, int cc, std::span<const int64_t> src0_ne, std::span<const size_t> src0_nb, int64_t ne11);
	bool should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

	size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor* dst);
	// maybe it cna be removed
	bool should_use_mmv(ggml_type type, int cc, std::span<const int64_t> src0_ne, int64_t ne11);
}
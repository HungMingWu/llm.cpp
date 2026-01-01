module;
#include <stdint.h>
#include <span>
#include "op/cuda_func.h"

module ggml:cuda.utils;
import :ds;

namespace utils
{
	bin_bcast_context create_bcast_context(const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst);

	bool should_use_mmf(ggml_type type, int cc, int warp_size, std::span<const int64_t> src0, std::span<const size_t> src0_nb, int64_t src1_ncols, bool mul_mat_id);
	bool should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);
	bool should_use_mmvf(ggml_type type, int cc, std::span<const int64_t> src0_ne, std::span<const size_t> src0_nb, int64_t ne11);

	// maybe it cna be removed
	bool should_use_mmv(ggml_type type, int cc, std::span<const int64_t> src0_ne, int64_t ne11);
}
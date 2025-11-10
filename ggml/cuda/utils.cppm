module;
#include <stdint.h>
#include <span>
#include "internal_ds.h"

module ggml:cuda.utils;

namespace utils
{
	bool should_use_mmf(enum ggml_type type, int cc, int warp_size, std::span<const int64_t> src0, std::span<const size_t> src0_nb, int64_t src1_ncols, bool mul_mat_id);
	bool should_use_mmq(enum ggml_type type, int cc, int64_t ne11);
	bool should_use_mmvf(enum ggml_type type, int cc, std::span<const int64_t> src0_ne, std::span<const size_t> src0_nb, int64_t ne11);
}
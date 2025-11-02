module;
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include <barrier>
#include <cstdlib>
#include <new>
#include <numbers>
#include <numeric>
#include <iostream>
#include <exec/static_thread_pool.hpp>
#include <exec/async_scope.hpp>
#include "block.h"
#include "mdspan.hpp"
#include "helper.h"

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :ds;
import :quants;
import :tensor;
import :utility;
import :cpu.ds;
import :cpu.from_float;
import :cpu.to_float;
import :cpu.traits;
import :cpu.vec_dot;
import :cpu.op;

#ifdef GGML_USE_LLAMAFILE
import :cpu.llamafile.sgemm;
#endif

static int register_ok = []() {
	get_reg().register_backend(ggml_backend_cpu_reg());
	return 0;
}();

// ggml_compute_forward_mul_mat

template <typename T, typename U>
constexpr T* cast_with_offset_impl(U* ptr, size_t offset)
{
	using BytePtr = std::conditional_t<std::is_const_v<U>, const std::byte*, std::byte*>;
	return reinterpret_cast<T*>(reinterpret_cast<BytePtr>(ptr) + offset);
}

template <typename T>
constexpr T* cast_with_offset(void* ptr, size_t offset)
{
	return cast_with_offset_impl<T>(ptr, offset);
}

template <typename T>
constexpr const T* cast_with_offset(const void* ptr, size_t offset)
{
	return cast_with_offset_impl<const T>(ptr, offset);
}

template <typename T>
void fromFloat(const float* x, T* y, int64_t n)
{
	if constexpr (is_quant_type_v<T>) {
		quantize_row(x, y, n);
	}
	else {
		from_float(x, y, n);
	}
}

template <typename src0_type, typename src1_type>
static void ggml_compute_forward_mul_mat_one_chunk(
	src0_type src0_data,
	src1_type src1_data,
	ggml_tensor* dst,
	const int64_t num_rows_per_vec_dot,
	const int64_t ir0_start,
	const int64_t ir0_end,
	const int64_t ir1_start,
	const int64_t ir1_end) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	const bool src1_cont = ggml_is_contiguous(src1);

	enum ggml_type const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;

	// broadcast factors
	const int64_t r2 = src1->ne[2] / src0->ne[2];
	const int64_t r3 = src1->ne[3] / src0->ne[3];

	//printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

	// threads with no work simply yield (not sure if it helps)
	if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
		return;
	}

	const size_t row_size = ggml_row_size(vec_dot_type, src1->ne[0]);

	assert(src1->ne[2] % src0->ne[2] == 0);
	assert(src1->ne[3] % src0->ne[3] == 0);

	// block-tiling attempt
	static constexpr int64_t blck_0 = 16;
	static constexpr int64_t blck_1 = 16;

	const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : src1->nb[1];

	// attempt to reduce false-sharing (does not seem to make a difference)
	// 16 * 2, accounting for mmla kernels
	float tmp[32];
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);

	for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
		for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
			for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
				const int64_t i13 = (ir1 / (src1->ne[2] * dst->ne[1]));
				const int64_t i12 = (ir1 - i13 * src1->ne[2] * dst->ne[1]) / dst->ne[1];
				const int64_t i11 = (ir1 - i13 * src1->ne[2] * dst->ne[1] - i12 * dst->ne[1]);

				// broadcast src0 into src1
				const int64_t i03 = i13 / r3;
				const int64_t i02 = i12 / r2;

				const size_t bs = (num_rows_per_vec_dot > 1 ? 16 : 0);
				const size_t bx = (num_rows_per_vec_dot > 1 ? src0->nb[1] : 0);
				const size_t by = (num_rows_per_vec_dot > 1 ? src1_col_stride : 0);
				for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
					ggml_vec_dot(src0->ne[0], &tmp[ir0 - iir0], bs,
						&src0_data[i03, i02, ir0, 0], bx,
						&src1_data[i13, i12, i11, 0], by, num_rows_per_vec_dot);
				}

				for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
					memcpy(&dst_data[i13, i12, i11, iir0 + cn * dst->nb[1] / dst->nb[0]], tmp + (cn * 16), (std::min(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
				}
			}
		}
	}
}

template <typename src0_t>
static void ggml_compute_forward_mul_mat(
	exec::static_thread_pool &pool,
	exec::async_scope &scope,
	ggml_tensor* dst) {
	using vec_dot_t = typename vec_dot_trait<src0_t>::type;
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	const int nth = pool.available_parallelism();

	enum ggml_type           const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;

	GGML_ASSERT(dst->ne[0] == src0->ne[1]);
	GGML_ASSERT(dst->ne[1] == src1->ne[1]);
	GGML_ASSERT(dst->ne[2] == src1->ne[2]);
	GGML_ASSERT(dst->ne[3] == src1->ne[3]);

	// we don't support permuted src0 or src1
	GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
	GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));

	// dst cannot be transposed or permuted
	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
	GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
	GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

	std::vector<vec_dot_t> wdata;
	// nb01 >= nb00 - src0 is not transposed
	//   compute by src0 rows

	// TODO: extract to "extra_op"
#if GGML_USE_LLAMAFILE
	// broadcast factors
	const int64_t r2 = ne12 / ne02;
	const int64_t r3 = ne13 / ne03;

	const bool src1_cont = ggml_is_contiguous(src1);

	if (src1_cont) {
		for (int64_t i13 = 0; i13 < ne13; i13++)
			for (int64_t i12 = 0; i12 < ne12; i12++)
				if (!llamafile_sgemm(params,
					ne01, ne11, ne00 / ggml_blck_size(src0->type),
					(const char*)src0->data + i12 / r2 * nb02 + i13 / r3 * nb03,
					nb01 / ggml_type_size(src0->type),
					(const char*)src1->data + i12 * nb12 + i13 * nb13,
					nb11 / ggml_type_size(src1->type),
					(char*)dst->data + i12 * nb2 + i13 * nb3,
					nb1 / ggml_type_size(dst->type),
					src0->type,
					src1->type,
					dst->type))
					goto UseGgmlGemm1;
		return;
	}
UseGgmlGemm1:;
#endif

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	if (src1->type != vec_dot_type) {
		const size_t we0 = src1->ne[0] / ggml_blck_size(vec_dot_type);
		GGML_ASSERT(src1->type == GGML_TYPE_F32);
		wdata.resize(src1->ne[3] * src1->ne[2] * src1->ne[1] * we0);
		std::experimental::mdspan wdata_span(wdata.data(), src1->ne[3], src1->ne[2], src1->ne[1], we0);
		auto src1_data = make_strided_mdspan(static_cast<const float*>(src1->data), src1->ne, src1->nb);

		for (int64_t i11 = 0; i11 < src1->ne[1]; i11++) {
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &wdata] {
				for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
					for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
						fromFloat(
							&src1_data[i13, i12, i11, 0],
							&wdata_span[i13, i12, i11, 0],
							src1->ne[0]);
					}
				}
			});
			scope.spawn(std::move(sender));
		}
		stdexec::sync_wait(scope.on_empty());
	}

#if GGML_USE_LLAMAFILE
	if (src1->type != vec_dot_type) {
		const void* wdata = (src1->type == vec_dot_type) ? src1->data : &wdata[0];
		const size_t row_size = ggml_row_size(vec_dot_type, ne10);

		for (int64_t i13 = 0; i13 < ne13; i13++)
			for (int64_t i12 = 0; i12 < ne12; i12++)
				if (!llamafile_sgemm(params,
					ne01, ne11, ne00 / ggml_blck_size(src0->type),
					(const char*)src0->data + i12 / r2 * nb02 + i13 / r3 * nb03,
					nb01 / ggml_type_size(src0->type),
					(const char*)wdata + (i12 * ne11 + i13 * ne12 * ne11) * row_size,
					row_size / ggml_type_size(vec_dot_type),
					(char*)dst->data + i12 * nb2 + i13 * nb3,
					nb1 / ggml_type_size(dst->type),
					src0->type,
					vec_dot_type,
					dst->type))
					goto UseGgmlGemm2;
		return;
	}
UseGgmlGemm2:;
#endif

	// This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
	const int64_t nr0 = dst->ne[0];

	// This is the size of the rest of the dimensions of the result
	const int64_t nr1 = dst->ne[1] * dst->ne[2] * dst->ne[3];

	// Now select a reasonable chunk size.
	const int chunk_size = [&] {
		// We need to step up the size if it's small
		if (nr0 == 1 || nr1 == 1) {
			return 64;
		}
		return 16;
	}();

	// distribute the work across the inner or outer loop based on which one is larger
	// The number of chunks in the 0/1 dim.
	// CEIL(nr0/chunk_size)
	int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
	int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

	// If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
	//   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggerganov/llama.cpp/pull/6915
	//   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
	if (nchunk0* nchunk1 < nth * 4 || ggml_is_numa()) {
		// distribute the thread work across the inner or outer loop based on which one is larger
		nchunk0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
		nchunk1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows
	}

	// The number of elements in each chunk
	const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
	const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

	for (int64_t current_chunk = 0; current_chunk < nchunk0 * nchunk1; current_chunk++) {
		const int64_t ith0 = current_chunk % nchunk0;
		const int64_t ith1 = current_chunk / nchunk0;

		const int64_t ir0_start = dr0 * ith0;
		const int64_t ir0_end = std::min(ir0_start + dr0, nr0);

		const int64_t ir1_start = dr1 * ith1;
		const int64_t ir1_end = std::min(ir1_start + dr1, nr1);

		// dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
		const int64_t num_rows_per_vec_dot = [&]() -> int64_t {
			// these checks are needed to avoid crossing dim1 boundaries
			// can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
			if ((nr0 % 2 != 0) || (src1->ne[1] % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
				return 1;
			}
			return type_traits_cpu[src0->type].nrows;
		}();

		auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(src0->data), src0->ne, src0->nb);
		auto src1_data = make_strided_mdspan(static_cast<const vec_dot_t*>(src1->data), src1->ne, src1->nb);
		const size_t we0 = src1->ne[0] / ggml_blck_size(vec_dot_type);
		std::experimental::mdspan wdata_span(wdata.data(), src1->ne[3], src1->ne[2], src1->ne[1], we0);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &wdata] {
			if (src1->type == vec_dot_type) {
				ggml_compute_forward_mul_mat_one_chunk(
					src0_data,
					src1_data,
					dst, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
			}
			else {
				ggml_compute_forward_mul_mat_one_chunk(
					src0_data,
					wdata_span,
					dst, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
			}
		});
		scope.spawn(std::move(sender));
	}
	stdexec::sync_wait(scope.on_empty());
}

static void ggml_compute_forward_mul_mat(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	switch (src0->type) {
	case GGML_TYPE_F32: {
		ggml_compute_forward_mul_mat<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16: {
		ggml_compute_forward_mul_mat<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16: {
		ggml_compute_forward_mul_mat<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_mul_mat<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_mul_mat<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4: {
		ggml_compute_forward_mul_mat<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_mul_mat<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_mul_mat<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_mul_mat<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_mul_mat<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_mul_mat<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_mul_mat<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_mul_mat<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_mul_mat<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_mul_mat<block_iq1_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_mul_mat<block_iq1_m>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_mul_mat<block_iq2_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_mul_mat<block_iq2_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_mul_mat<block_iq2_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_mul_mat<block_iq3_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_mul_mat<block_iq3_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_mul_mat<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_mul_mat<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_mul_mat<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_mul_mat<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_1:
	default:
		assert(false);
	}
}

static void ggml_compute_forward_arange_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	GGML_ASSERT(dst->nb[0] == sizeof(float));

	const float start = std::bit_cast<float>(dst->op_params[0]);
	const float stop = std::bit_cast<float>(dst->op_params[1]);
	const float step = std::bit_cast<float>(dst->op_params[2]);

	const int64_t steps = (int64_t)ceilf((stop - start) / step);

	GGML_ASSERT(dst->nelements() == steps);

	constexpr int64_t slices = 65536;
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t begin = 0; begin < steps; begin += slices) {
		const int64_t end = std::min(begin + slices, steps);
		stdexec::sender auto sender = stdexec::schedule(scheduler) |
			stdexec::then([=] {
				for (int64_t i = begin; i < end; i++) {
					float value = start + step * i;
					((float*)dst->data)[i] = value;
				}
			});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_arange(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->type) {
		case GGML_TYPE_F32:
		{
			ggml_compute_forward_arange_f32(pool, scope, dst);
		} break;
		default:
		{
			GGML_ABORT("fatal error");
		}
	}
}

template <typename T>
static void ggml_compute_forward_conv_transpose_1d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F32);
	GGML_ASSERT(src0->nb[0] == sizeof(T));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src0->ne[2] == src1->ne[1]);
	GGML_ASSERT(dst->ne[1] == src0->ne[1]);

	const int64_t CIn = src0->ne[2];
	const int64_t COut = src0->ne[1];
	const int64_t K = src0->ne[0];
	const int64_t L = src1->ne[0];
	const int64_t LOut = dst->ne[0];
	const int32_t stride = ((const int32_t*)(dst->op_params))[0];
	const int32_t padding = ((const int32_t*)(dst->op_params))[1];
	const int32_t dilation = ((const int32_t*)(dst->op_params))[2];
	GGML_ASSERT(LOut == (L - 1) * stride - 2 * padding + dilation * (K - 1) + 1);

	std::experimental::mdspan src0_data(static_cast<const T*>(src0->data), CIn, COut, K);
	std::experimental::mdspan src1_data(static_cast<const float*>(src1->data), CIn, L);
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), COut, LOut);

	for (int64_t cout = 0; cout < COut; cout++) {
		for (int64_t lout = 0; lout < LOut; lout++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				float accumulator = 0.0;
				for (int64_t cin = 0; cin < CIn; cin++) {
					for (int64_t k = 0; k < K; k++) {
						int64_t lin = lout + padding - k * dilation;
						if (lin % stride != 0) continue;
						lin /= stride;
						if (lin < 0 || lin >= L) continue;
						accumulator += toFloat32(src0_data[cin, cout, k]) * src1_data[cin, lin];
					}
				}
				dst_data[cout, lout] = accumulator;
			});
			scope.spawn(std::move(sender));
		}
	}
}

template <typename T>
static void ggml_compute_forward_conv_transpose_2d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* kernel = dst->src[0];
	const ggml_tensor* input = dst->src[1];

	GGML_ASSERT(input->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F32);

	GGML_ASSERT(kernel->nb[0] == sizeof(T));
	GGML_ASSERT(input->nb[0] == sizeof(float));
	GGML_ASSERT(kernel->ne[3] == input->ne[2]);
	GGML_ASSERT(dst->ne[2] == kernel->ne[2]);
	GGML_ASSERT(dst->ne[3] == input->ne[3]);

	const int64_t CIn = kernel->ne[3];
	const int64_t COut = kernel->ne[2];
	const int64_t Kh = kernel->ne[1];
	const int64_t Kw = kernel->ne[0];
	const int64_t N = dst->ne[3];
	const int64_t HOut = dst->ne[1];
	const int64_t WOut = dst->ne[0];
	const int64_t HIn = input->ne[1];
	const int64_t WIn = input->ne[0];
	std::experimental::mdspan kernel_data(static_cast<const T*>(kernel->data), CIn, COut, Kh, Kw);
	std::experimental::mdspan input_kernel(static_cast<const float*>(input->data), N, CIn, HIn, WIn);
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), N, COut, HOut, WOut);

	const int32_t stride_w = std::bit_cast<int32_t>(dst->op_params[0]);
	const int32_t stride_h = std::bit_cast<int32_t>(dst->op_params[1]);
	const int32_t padding_w = std::bit_cast<int32_t>(dst->op_params[2]);
	const int32_t padding_h = std::bit_cast<int32_t>(dst->op_params[3]);
	const int32_t dilation_w = std::bit_cast<int32_t>(dst->op_params[4]);
	const int32_t dilation_h = std::bit_cast<int32_t>(dst->op_params[5]);

	for (int64_t n = 0; n < N; n++) {
		for (int64_t cout = 0; cout < COut; cout++) {
			for (int64_t hout = 0; hout < HOut; hout++) {
				for (int64_t wout = 0; wout < WOut; wout++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						float accumulator = 0;

						for (int64_t cin = 0; cin < CIn; cin++) {
							for (int64_t kh = 0; kh < Kh; ++kh) {
								int64_t hin = hout + padding_h  - kh * dilation_h;
								if (hin < 0 || hin % stride_h) continue;
								hin /= stride_h;
								if (hin >= HIn) continue;

								for (int64_t kw = 0; kw < Kw; ++kw) {
									int64_t win = wout + padding_w - kw * dilation_w;
									if (win < 0 || win % stride_w) continue;
									win /= stride_w;
									if (win >= WIn) continue;

									accumulator += input_kernel[n, cin, hin, win] *
										toFloat32(kernel_data[cin, cout, kh, kw]);
								}
							}
						}

						dst_data[n, cout, hout, wout] = accumulator;
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
}

static void ggml_compute_forward_conv_transpose_2d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_conv_transpose_2d<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_conv_transpose_2d<ggml_fp32_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_conv_transpose_1d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_conv_transpose_1d<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_conv_transpose_1d<ggml_fp32_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

// ggml_compute_forward_dup

static void ggml_compute_forward_dup_same_cont(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());
	GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
	GGML_ASSERT(src0->type == dst->type);

	const size_t nb0 = ggml_type_size(src0->type);

	const int nth = pool.available_parallelism(); // number of threads

	// parallelize by blocks
	const int nk = src0->nelements() / ggml_blck_size(src0->type);
	const int dr = (nk + nth - 1) / nth;
	for (int k0 = 0; k0 < nk; k0 += dr) {
		const int k1 = std::min(k0 + dr, nk);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |	stdexec::then([=] {
			memcpy(
				cast_with_offset<char>(dst->data, k0 * nb0),
				cast_with_offset<char>(src0->data, k0 * nb0),
				(k1 - k0) * nb0);
		});
		scope.spawn(std::move(sender));
	}
}

// A simplified version of ggml_compute_forward_dup that doesn't do float upcasting, and just plain old memcpy.
static void ggml_compute_forward_dup_bytes(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());
	GGML_ASSERT(src0->type == dst->type);

	if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
		ggml_compute_forward_dup_same_cont(pool, scope, dst);
		return;
	}

	const size_t type_size = ggml_type_size(src0->type);

	if (src0->type == dst->type &&
		ggml_are_same_shape(src0, dst) &&
		src0->nb[0] == type_size && dst->nb[0] == type_size) {
		// copy by rows
		const size_t rs = ggml_row_size(src0->type, src0->ne[0]);
		for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
					for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
						memcpy(
							((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]),
							((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]),
							rs);
					}
				}
			});
			scope.spawn(std::move(sender));
		}
		return;
	}

	if (ggml_is_contiguous(dst)) {
		char* dst_ptr = (char*)dst->data;
		const size_t rs = src0->ne[0] * type_size;

		if (src0->nb[0] == type_size) {
			// src0 is contigous on first dimension, copy by rows
			for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
				for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						size_t id = ((i03 * (src0->ne[2] * src0->ne[1] * src0->ne[0]) + i02 * (src0->ne[1]) * src0->ne[0])) * type_size;
						for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
							const char* src0_ptr = (char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3];
							memcpy(dst_ptr + id, src0_ptr, rs);
							id += rs;
						}
					});
					scope.spawn(std::move(sender));
				}
			}
		}
		else {
			//printf("%s: this is not optimal - fix me\n", __func__);

			for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
				for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						size_t id = ((i03 * (src0->ne[2] * src0->ne[1] * src0->ne[0]) + i02 * (src0->ne[1]) * src0->ne[0]))* type_size;
						for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
							for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
								const char* src0_ptr = (char*)src0->data + i00 * src0->nb[0] + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3];
								memcpy(dst_ptr + id, src0_ptr, type_size);

								id += type_size;
							}
						}
					});
					scope.spawn(std::move(sender));
				}
			}
		}

		return;
	}

	// number of blocks in a row
	const int64_t nk00 = src0->ne[0] / ggml_blck_size(src0->type);
	const int64_t nk0 = dst->ne[0] / ggml_blck_size(dst->type);

	for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
		for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				// dst counters
				int64_t k10 = i03 * (src0->ne[2] * src0->ne[1] * src0->ne[0]) + i02 * (src0->ne[1] * src0->ne[0]);
				int64_t i11 = 0;
				int64_t i12 = 0;
				int64_t i13 = 0;
				while (k10 >= nk0) {
					k10 -= nk0;
					if (++i11 == dst->ne[1]) {
						i11 = 0;
						if (++i12 == dst->ne[2]) {
							i12 = 0;
							if (++i13 == dst->ne[3]) {
								i13 = 0;
							}
						}
					}
				}
				for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
					for (int64_t k00 = 0; k00 < nk00; k00++) {
						const char* src0_ptr = ((char*)src0->data + k00 * src0->nb[0] + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);
						char* dst_ptr = ((char*)dst->data + k10 * dst->nb[0] + i11 * dst->nb[1] + i12 * dst->nb[2] + i13 * dst->nb[3]);

						memcpy(dst_ptr, src0_ptr, type_size);

						if (++k10 == nk0) {
							k10 = 0;
							if (++i11 == dst->ne[1]) {
								i11 = 0;
								if (++i12 == dst->ne[2]) {
									i12 = 0;
									if (++i13 == dst->ne[3]) {
										i13 = 0;
									}
								}
							}
						}
					}
				}
			});
			scope.spawn(std::move(sender));
		}
	}
}

template<typename src_t, typename dst_t>
	requires (not std::is_same_v<src_t, dst_t>)
static void ggml_compute_forward_dup_flt(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());
	GGML_ASSERT(!ggml_is_quantized(src0->type) && !ggml_is_quantized(dst->type));

	// case: type & row size equal
	if (src0->type == dst->type &&
		src0->ne[0] == dst->ne[0] &&
		src0->nb[0] == ggml_type_size(src0->type) && dst->nb[0] == ggml_type_size(dst->type)) {
		// copy by rows
		const size_t rs = src0->ne[0] * src0->nb[0];
		for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
			for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
						memcpy(
							((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]),
							((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]),
							rs);
					}
				});
				scope.spawn(std::move(sender));
			}
		}
		return;
	}

	auto src0_data = make_strided_mdspan(static_cast<const src_t*>(src0->data), src0->ne, src0->nb);
	auto dst_data = make_strided_mdspan(static_cast<dst_t*>(dst->data), dst->ne, dst->nb);
	// case: dst tensor is contiguous
	if (ggml_is_contiguous(dst)) {
		if (src0->nb[0] == sizeof(src_t)) {
			for (int i03 = 0; i03 < src0_data.extent(0); i03++) {
				for (int i02 = 0; i02 < src0_data.extent(1); i02++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						for (int i01 = 0; i01 < src0_data.extent(2); i01++)
							for (int i00 = 0; i00 < src0_data.extent(3); i00++)
								dst_data[i03, i02, i01, i00] = fromFloat32<dst_t>(toFloat32(src0_data[i03, i02, i01, i00]));
					});
					scope.spawn(std::move(sender));
				}
			}
		}
		else {
			//printf("%s: this is not optimal - fix me\n", __func__);

			for (int i03 = 0; i03 < src0_data.extent(0); i03++) {
				for (int i02 = 0; i02 < src0_data.extent(1); i02++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						dst_t* dst_ptr = (dst_t*)dst->data + i03 * (src0->ne[2] * src0->ne[1] * src0->ne[0]) + i02 * (src0->ne[1] * src0->ne[0]);
						for (int i01 = 0; i01 < src0_data.extent(2); i01++)
							for (int i00 = 0; i00 < src0_data.extent(3); i00++)
								*dst_ptr++ = fromFloat32<dst_t>(toFloat32(src0_data[i03, i02, i01, i00]));
					});
					scope.spawn(std::move(sender));
				}
			}
		}
		return;
	}

	for (int64_t i03 = 0; i03 < src0_data.extent(0); i03++) {
		for (int64_t i02 = 0; i02 < src0_data.extent(1); i02++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				// dst counters
				int64_t i10 = i03 * (src0->ne[2] * src0->ne[1] * src0->ne[0]) + i02 * (src0->ne[1] * src0->ne[0]);
				int64_t i11 = 0;
				int64_t i12 = 0;
				int64_t i13 = 0;
				while (i10 >= dst->ne[0]) {
					i10 -= dst->ne[0];
					if (++i11 == dst->ne[1]) {
						i11 = 0;
						if (++i12 == dst->ne[2]) {
							i12 = 0;
							if (++i13 == dst->ne[3]) {
								i13 = 0;
							}
						}
					}
				}
				for (int64_t i01 = 0; i01 < src0_data.extent(2); i01++) {
					for (int64_t i00 = 0; i00 < src0_data.extent(3); i00++) {
						dst_data[i13, i12, i11, i10] = fromFloat32<dst_t>(toFloat32(src0_data[i03, i02, i01, i00]));

						if (++i10 == dst->ne[0]) {
							i10 = 0;
							if (++i11 == dst->ne[1]) {
								i11 = 0;
								if (++i12 == dst->ne[2]) {
									i12 = 0;
									if (++i13 == dst->ne[3]) {
										i13 = 0;
									}
								}
							}
						}
					}
				}
			});
			scope.spawn(std::move(sender));
		}
	}
}

template <typename src_t, typename dst_t>
static void ggml_compute_forward_dup_to_q(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());
	GGML_ASSERT(!ggml_is_quantized(src0->type));

	auto src0_data = make_strided_mdspan(static_cast<const src_t*>(src0->data), src0->ne, src0->nb);
	auto dst_data = make_strided_mdspan(static_cast<dst_t*>(dst->data), dst->ne, dst->nb);
	if (ggml_is_contiguous(dst) && src0->nb[0] == sizeof(src_t)) {
		// casting non-quantized types --> intermediate f32 --> quantized
		for (int64_t i03 = 0; i03 < src0_data.extent(0); i03++) {
			for (int64_t i02 = 0; i02 < src0_data.extent(1); i02++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					std::vector<float> src0_f32(src0->ne[0]);
					for (int64_t i01 = 0; i01 < src0_data.extent(2); i01++) {
						dst_t* dst_ptr = &dst_data[i03, i02, i01, 0];
						for (int64_t i00 = 0; i00 < src0_data.extent(3); i00++)
							src0_f32[i00] = toFloat32(src0_data[i03, i02, i01, i00]);
						quantize_row(src0_f32.data(), dst_ptr, src0->ne[0]);
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
	else {
		// printf("%s %s\n", ggml_type_name(src0->type), ggml_type_name(dst->type));
		GGML_ABORT("not implemented");
	}
}

template <typename src_t>
static void ggml_compute_forward_dup_from_q(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	const ggml_type type = src0->type;

	const size_t qk = ggml_blck_size(type);

	// destination must be contiguous in the first dimension
	GGML_ASSERT(src1->nb[0] == ggml_type_size(dst->type));
	// must either have first dimension large enough to hold a row, or fully contiguous
	GGML_ASSERT((src1->ne[0] % qk) == 0 || ggml_is_contiguous(dst));

	for (int64_t i = 0; i < src1->nelements(); i += qk) {
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			const int64_t i03 = i / (src0->ne[0] * src0->ne[1] * src0->ne[2]);
			const int64_t i02 = (i - i03 * src0->ne[0] * src0->ne[1] * src0->ne[2]) / (src0->ne[0] * src0->ne[1]);
			const int64_t i01 = (i - i03 * src0->ne[0] * src0->ne[1] * src0->ne[2] - i02 * src0->ne[1] * src0->ne[0]) / src0->ne[0];
			const int64_t i00 = i - i03 * src0->ne[0] * src0->ne[1] * src0->ne[2] - i02 * src0->ne[1] * src0->ne[0] - i01 * src0->ne[0];
			const int64_t x_offset = (i00 / qk) * src0->nb[0] + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3];

			const int64_t i13 = i / (src1->ne[0] * src1->ne[1] * src1->ne[2]);
			const int64_t i12 = (i - i13 * src1->ne[0] * src1->ne[1] * src1->ne[2]) / (src1->ne[0] * src1->ne[1]);
			const int64_t i11 = (i - i13 * src1->ne[0] * src1->ne[1] * src1->ne[2] - i12 * src1->ne[0] * src1->ne[1]) / src1->ne[0];
			const int64_t i10 = i - i13 * src1->ne[0] * src1->ne[1] * src1->ne[2] - i12 * src1->ne[0] * src1->ne[1] - i11 * src1->ne[0];
			const int64_t dst_offset = i10 * src1->nb[0] + i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3];

			dequantize_row(
				(const src_t*)((char*)src0->data + x_offset),
				(float*)((char*)dst->data + dst_offset), qk);
		});
		scope.spawn(std::move(sender));
	}
}

template <typename src_t>
static void ggml_compute_forward_dup_to_q(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	GGML_ASSERT(ggml_is_quantized(dst->type));
	switch (dst->type) {
	case GGML_TYPE_Q4_0:
		return ggml_compute_forward_dup_to_q<src_t, block_q4_0>(pool, scope, dst);
	case GGML_TYPE_Q4_1:
		return ggml_compute_forward_dup_to_q<src_t, block_q4_1>(pool, scope, dst);
	case GGML_TYPE_Q5_0:
		return ggml_compute_forward_dup_to_q<src_t, block_q5_0>(pool, scope, dst);
	case GGML_TYPE_Q5_1:
		return ggml_compute_forward_dup_to_q<src_t, block_q5_1>(pool, scope, dst);
	case GGML_TYPE_Q8_0:
		return ggml_compute_forward_dup_to_q<src_t, block_q8_0>(pool, scope, dst);
	case GGML_TYPE_MXFP4:
		return ggml_compute_forward_dup_to_q<src_t, block_mxfp4>(pool, scope, dst);
	case GGML_TYPE_Q2_K:
		return ggml_compute_forward_dup_to_q<src_t, block_q2_K>(pool, scope, dst);
	case GGML_TYPE_Q3_K:
		return ggml_compute_forward_dup_to_q<src_t, block_q3_K>(pool, scope, dst);
	case GGML_TYPE_Q4_K:
		return ggml_compute_forward_dup_to_q<src_t, block_q4_K>(pool, scope, dst);
	case GGML_TYPE_Q5_K:
		return ggml_compute_forward_dup_to_q<src_t, block_q5_K>(pool, scope, dst);
	case GGML_TYPE_Q6_K:
		return ggml_compute_forward_dup_to_q<src_t, block_q6_K>(pool, scope, dst);
	case GGML_TYPE_IQ4_NL:
		return ggml_compute_forward_dup_to_q<src_t, block_iq4_nl>(pool, scope, dst);
	case GGML_TYPE_IQ4_XS:
		return ggml_compute_forward_dup_to_q<src_t, block_iq4_xs>(pool, scope, dst);
	default:
		assert(false);
		break;
	}
}

static void ggml_compute_forward_dup_from_q(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst)
{
	const ggml_tensor* src0 = dst->src[0];
	switch (src0->type) {
	case GGML_TYPE_Q4_0:
		return ggml_compute_forward_dup_from_q<block_q4_0>(pool, scope, dst);
	case GGML_TYPE_Q4_1:
		return ggml_compute_forward_dup_from_q<block_q4_1>(pool, scope, dst);
	case GGML_TYPE_Q5_0:
		return ggml_compute_forward_dup_from_q<block_q5_0>(pool, scope, dst);
	case GGML_TYPE_Q5_1:
		return ggml_compute_forward_dup_from_q<block_q5_1>(pool, scope, dst);
	case GGML_TYPE_Q8_0:
		return ggml_compute_forward_dup_from_q<block_q8_0>(pool, scope, dst);
	case GGML_TYPE_MXFP4:
		return ggml_compute_forward_dup_from_q<block_mxfp4>(pool, scope, dst);
	case GGML_TYPE_Q2_K:
		return ggml_compute_forward_dup_from_q<block_q2_K>(pool, scope, dst);
	case GGML_TYPE_Q3_K:
		return ggml_compute_forward_dup_from_q<block_q3_K>(pool, scope, dst);
	case GGML_TYPE_Q4_K:
		return ggml_compute_forward_dup_from_q<block_q4_K>(pool, scope, dst);
	case GGML_TYPE_Q5_K:
		return ggml_compute_forward_dup_from_q<block_q5_K>(pool, scope, dst);
	case GGML_TYPE_Q6_K:
		return ggml_compute_forward_dup_from_q<block_q6_K>(pool, scope, dst);
	case GGML_TYPE_IQ1_S:
		return ggml_compute_forward_dup_from_q<block_iq1_s>(pool, scope, dst);
	case GGML_TYPE_IQ1_M:
		return ggml_compute_forward_dup_from_q<block_iq1_m>(pool, scope, dst);
	case GGML_TYPE_IQ2_S:
		return ggml_compute_forward_dup_from_q<block_iq2_s>(pool, scope, dst);
	case GGML_TYPE_IQ2_XS:
		return ggml_compute_forward_dup_from_q<block_iq2_xs>(pool, scope, dst);
	case GGML_TYPE_IQ2_XXS:
		return ggml_compute_forward_dup_from_q<block_iq2_xxs>(pool, scope, dst);
	case GGML_TYPE_IQ3_S:
		return ggml_compute_forward_dup_from_q<block_iq3_s>(pool, scope, dst);
	case GGML_TYPE_IQ3_XXS:
		return ggml_compute_forward_dup_from_q<block_iq3_xxs>(pool, scope, dst);
	case GGML_TYPE_IQ4_NL:
		return ggml_compute_forward_dup_from_q<block_iq4_nl>(pool, scope, dst);
	case GGML_TYPE_IQ4_XS:
		return ggml_compute_forward_dup_from_q<block_iq4_xs>(pool, scope, dst);
	default:
		assert(false);
	}
}

static void ggml_compute_forward_dup(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	if (src0->type == dst->type) {
		ggml_compute_forward_dup_bytes(pool, scope, dst);
		return;
	}

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		/**/ if (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<ggml_fp16_t, ggml_bf16_t>(pool, scope, dst);
		else if (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_fp16_t, float      >(pool, scope, dst);
		else ggml_compute_forward_dup_to_q<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		/**/ if (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<ggml_bf16_t, ggml_fp16_t>(pool, scope, dst);
		else if (dst->type == GGML_TYPE_F32)  ggml_compute_forward_dup_flt<ggml_bf16_t, float      >(pool, scope, dst);
		else ggml_compute_forward_dup_to_q<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		/**/ if (dst->type == GGML_TYPE_F16)  ggml_compute_forward_dup_flt<float, ggml_fp16_t>(pool, scope, dst);
		else if (dst->type == GGML_TYPE_BF16) ggml_compute_forward_dup_flt<float, ggml_bf16_t>(pool, scope, dst);
		else if (dst->type == GGML_TYPE_I32)  ggml_compute_forward_dup_flt<float, int32_t    >(pool, scope, dst);
		else ggml_compute_forward_dup_to_q<float>(pool, scope, dst);
	} break;
	case GGML_TYPE_I32:
	{
		if (dst->type == GGML_TYPE_F32) ggml_compute_forward_dup_flt<int32_t, float>(pool, scope, dst);
		else GGML_ABORT("not implemented");
	} break;
	default:
	{
		if (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
			ggml_compute_forward_dup_from_q(pool, scope, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	}
	}
}

static void ggml_compute_forward_pad_reflect_1d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F32);

	const int32_t* opts = (const int32_t*)dst->op_params;
	const int64_t p0 = opts[0];
	const int64_t p1 = opts[1];

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);

	for (int64_t i1 = 0; i1 < src0_data.extent(2); i1++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t i3 = 0; i3 < src0_data.extent(0); i3++) {
				for (int64_t i2 = 0; i2 < src0_data.extent(1); i2++) {
					for (int64_t i0 = 0; i0 < src0_data.extent(3); i0++)
						dst_data[i3, i2, i1, i0 + p0] = src0_data[i3, i2, i1, i0];	
					for (int64_t i0 = 1; i0 <= p0; i0++) 
						dst_data[i3, i2, i1, p0 - i0] = dst_data[i3, i2, i1, p0 + i0];
					for (int64_t i0 = 1; i0 <= p1; i0++) 
						dst_data[i3, i2, i1, (dst_data.extent(3) - p1 - 1) + i0] = dst_data[i3, i2, i1, (dst_data.extent(3) - p1 - 1) - i0];
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

// ggml_compute_forward_im2col
// src0: kernel [OC, IC, KH, KW]
// src1: image [N, IC, IH, IW]
// dst:  result [N, OH, OW, IC, KH, KW]
template <typename T>
static void ggml_compute_forward_im2col_impl(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	if constexpr (std::is_same_v<T, ggml_fp16_t>) {
		GGML_ASSERT(src0->type == GGML_TYPE_F16);
		GGML_ASSERT(src1->type == GGML_TYPE_F32);
		GGML_ASSERT(dst->type == GGML_TYPE_F16);
	}
	else {
		GGML_ASSERT(src1->type == GGML_TYPE_F32);
		GGML_ASSERT(dst->type == GGML_TYPE_F32);
	}

	const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
	const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
	const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
	const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
	const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
	const int32_t d1 = ((const int32_t*)(dst->op_params))[5];
	const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

	const int64_t N = is_2D ? src1->ne[3] : src1->ne[2];
	const int64_t IC = is_2D ? src1->ne[2] : src1->ne[1];
	const int64_t IH = is_2D ? src1->ne[1] : 1;
	const int64_t IW = src1->ne[0];

	const int64_t KH = is_2D ? src0->ne[1] : 1;
	const int64_t KW = src0->ne[0];

	const int64_t OH = is_2D ? dst->ne[2] : 1;
	const int64_t OW = dst->ne[1];

	if constexpr (std::is_same_v<T, ggml_fp16_t>) {
		GGML_ASSERT(src0->nb[0] == sizeof(T));
	}
	GGML_ASSERT(src1->nb[0] == sizeof(float));

	// im2col: [N, IC, IH, IW] => [N, OH, OW, IC, KH, KW]
	std::experimental::mdspan src_data(static_cast<ggml_fp32_t*>(src1->data), N, IC, IH, IW);
	std::experimental::mdspan dst_data(static_cast<T*>(dst->data), N, OH, OW, IC, KH, KW);

	for (int64_t in = 0; in < N; in++) {
		for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
			for (int64_t iow = 0; iow < OW; iow++) {
				for (int64_t iic = 0; iic < IC; iic++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
						stdexec::then([=] {
							for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
								for (int64_t ikw = 0; ikw < KW; ikw++) {
									const int64_t iiw = iow * s0 + ikw * d0 - p0;
									const int64_t iih = ioh * s1 + ikh * d1 - p1;

									if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
										dst_data[in, ioh, iow, iic, ikh, ikw] = 0;
									}
									else {
										dst_data[in, ioh, iow, iic, ikh, ikw] = fromFloat32<T>(src_data[in, iic, iih, iiw]);
									}
								}
							}
						});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
}

static void ggml_compute_forward_im2col(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_im2col_impl<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_im2col_impl<ggml_fp32_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename T>
static void ggml_compute_forward_out_prod(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(dst->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	GGML_ASSERT(dst->ne[0] == src0->ne[0]);
	GGML_ASSERT(dst->ne[1] == src1->ne[0]);
	GGML_ASSERT(dst->ne[2] == src1->ne[2]);
	GGML_ASSERT(dst->ne[3] == src1->ne[3]);

	GGML_ASSERT(dst->ne[2] % src0->ne[2] == 0);
	GGML_ASSERT(dst->ne[3] % src0->ne[3] == 0);

	// we don't support permuted src0 or src1
	GGML_ASSERT(src0->nb[0] == sizeof(T));

	// dst cannot be transposed or permuted
	GGML_ASSERT(dst->nb[0] == sizeof(float));
	// GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
	// GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
	// GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

	// src0->nb[1] >= src0->nb[0] - src0 is not transposed
	//   compute by src0 rows

	// dst[:,:,:,:] = 0
	// for i3
	//  for i2
	//   for i1:
	//    for i01:
	//     for i0:
	//         dst[i3,i2,i1,i0] += src0[i03,i02,i01,i0] * src1[i13,i12,i11,i1]

	// dps == dst per src0, used for group query attention
	const int64_t dps2 = dst->ne[2] / src0->ne[2];
	const int64_t dps3 = dst->ne[3] / src0->ne[3];

	stdexec::scheduler auto scheduler = pool.get_scheduler();
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan(static_cast<const float*>(src1->data), src1->ne, src1->nb);

	// dst indices
	for (int64_t i3 = 0; i3 < dst_data.extent(0); i3++) {
		for (int64_t i2 = 0; i2 < dst_data.extent(1); i2++) {
			for (int64_t i1 = 0; i1 < dst_data.extent(2); i1++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					std::vector<float> wdata(dst_data.extent(3));
					for (int64_t i0 = 0; i0 < dst_data.extent(3); i0++) dst_data[i3, i2, i1, i0] = 0;

					for (int64_t i01 = 0; i01 < src0_data.extent(2); i01++) {
						const int64_t i02 = i2 / dps2;
						const int64_t i03 = i3 / dps3;

						//const int64_t i10 = i1;
						const int64_t i12 = i2;
						const int64_t i13 = i3;

						const int64_t i11 = i01;
						if constexpr (is_quant_type_v<T>) {
							dequantize_row(&src0_data[i03, i02, i01, 0], wdata.data(), wdata.size());
						}
						else {
							for (int64_t i0 = 0; i0 < wdata.size(); i0++) {
								wdata[i0] = toFloat32(src0_data[i03, i02, i01, i0]);
							}
						}
						for (int64_t i0 = 0; i0 < dst_data.extent(3); i0++) {
							dst_data[i3, i2, i1, i0] += wdata[i0] * src1_data[i13, i12, i11, i1];
						}
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

static void ggml_compute_forward_out_prod(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_out_prod<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_out_prod<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4: {
		ggml_compute_forward_out_prod<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_out_prod<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_out_prod<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_out_prod<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_out_prod<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_out_prod<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_out_prod<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_out_prod<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_out_prod<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_out_prod<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_out_prod<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_out_prod<block_iq2_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_out_prod<block_iq2_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_out_prod<block_iq3_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_out_prod<block_iq1_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_out_prod<block_iq1_m>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_out_prod<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_out_prod<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_out_prod<block_iq3_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_out_prod<block_iq2_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		GGML_ABORT("fatal error"); // todo
		// ggml_compute_forward_out_prod_f16_f32(params, dst);
	}
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_out_prod<ggml_fp32_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename T>
static void ggml_compute_forward_get_rows(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	const int64_t nc = src0->ne[0];
	const int64_t nr = src1->nelements();

	assert(dst->ne[0] == nc);
	assert(src0->ne[2] == src1->ne[1]);
	assert(src0->nb[0] == sizeof(T));
	assert(ggml_nrows(dst) == nr);

	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan(static_cast<const int32_t*>(src1->data), src1->ne, src1->nb);

	GGML_ASSERT(src1_data.extent(0) == 1);

	for (int64_t i12 = 0; i12 < src1_data.extent(1); i12++) {
		for (int64_t i11 = 0; i11 < src1_data.extent(2); i11++) {
			for (int64_t i10 = 0; i10 < src1_data.extent(3); i10++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					const int32_t i01 = src1_data[0, i12, i11, i10];

					GGML_ASSERT(i01 >= 0 && i01 < src0_data.extent(2));

					if constexpr (is_quant_type_v<T>) {
						dequantize_row(&src0_data[i12, i11, i01, 0], &dst_data[i12, i11, i10, 0], src0_data.extent(3));
					}
					else {
						for (int64_t i0 = 0; i0 < src0_data.extent(3); i0++) {
							dst_data[i12, i11, i10, i0] = toFloat32(src0_data[i12, i11, i01, i0]);
						}
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

static void ggml_compute_forward_get_rows(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_get_rows<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_get_rows<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_get_rows<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_get_rows<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_get_rows<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4: {
		ggml_compute_forward_get_rows<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_get_rows<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_get_rows<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_get_rows<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_get_rows<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_get_rows<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_get_rows<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_get_rows<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_get_rows<block_iq2_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_get_rows<block_iq2_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_get_rows<block_iq3_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_get_rows<block_iq1_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_get_rows<block_iq1_m>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_get_rows<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_get_rows<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_get_rows<block_iq3_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_get_rows<block_iq2_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_get_rows<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_get_rows<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_get_rows<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_1:
	default:
	{
		GGML_ABORT("fatal error");
	}
	}

	//static bool first = true;
	//printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
	//if (first) {
	//    first = false;
	//} else {
	//    for (int k = 0; k < dst->ne[1]; ++k) {
	//        for (int j = 0; j < dst->ne[0]/16; ++j) {
	//            for (int i = 0; i < 16; ++i) {
	//                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
	//            }
	//            printf("\n");
	//        }
	//        printf("\n");
	//    }
	//    printf("\n");
	//    exit(0);
	//}
}

static void ggml_compute_forward_count_equal_i32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(src0->type == GGML_TYPE_I32);
	GGML_ASSERT(src1->type == GGML_TYPE_I32);
	GGML_ASSERT(ggml_are_same_shape(src0, src1));
	GGML_ASSERT(ggml_is_scalar(dst));
	GGML_ASSERT(dst->type == GGML_TYPE_I64);

	std::atomic<int64_t> sum_thread{ 0 };
	auto src0_data = make_strided_mdspan(static_cast<const int32_t*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan(static_cast<const int32_t*>(src1->data), src1->ne, src1->nb);

	for (int64_t i03 = 0; i03 < src0_data.extent(0); ++i03) {
		for (int64_t i02 = 0; i02 < src0_data.extent(1); ++i02) {
			for (int64_t i01 = 0; i01 < src0_data.extent(2); ++i01) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=, &sum_thread] {
					int64_t sum = 0;
					for (int64_t i00 = 0; i00 < src0_data.extent(3); ++i00) {
						sum += src0_data[i03, i02, i01, i00] == src1_data[i03, i02, i01, i00];
					}
					sum_thread += sum;
				});
				scope.spawn(std::move(sender));
			}
		}
	}

	stdexec::sync_wait(scope.on_empty());
	*((int64_t*)dst->data) = sum_thread.load();
}

static void ggml_compute_forward_count_equal(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_count_equal_i32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename T>
static void ggml_compute_forward_set(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

	// view src0 and dst with these strides and data offset inbytes during set
	// nb0 is implicitly element_size because src0 and dst are contiguous
	size_t nb1 = ((int32_t*)dst->op_params)[0];
	size_t nb2 = ((int32_t*)dst->op_params)[1];
	size_t nb3 = ((int32_t*)dst->op_params)[2];
	size_t offset = ((int32_t*)dst->op_params)[3];
	bool   inplace = (bool)((int32_t*)dst->op_params)[4];

	if (!inplace) {
		// memcpy needs to be synchronized across threads to avoid race conditions.
		// => do it in INIT phase
		memcpy(
			((char*)dst->data),
			((char*)src0->data),
			dst->nbytes());
	}

	// src0 and dst as viewed during set
	const size_t nb0 = ggml_element_size(src0);
	GGML_ASSERT(src1->nb[0] == sizeof(T));

	std::array<size_t, 4> dst_nb = { ggml_element_size(dst), nb1, nb2, nb3 };
	auto dst_data = make_strided_mdspan(
		static_cast<T*>(static_cast<void*>((static_cast<char *>(dst->data) + offset))), 
		dst->ne, dst_nb);
	auto src1_data = make_strided_mdspan(static_cast<const T*>(src1->data), src1->ne, src1->nb);

	for (int64_t i3 = 0; i3 < src1_data.extent(0); i3++) {
		for (int64_t i2 = 0; i2 < src1_data.extent(1); i2++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				for (int64_t i1 = 0; i1 < src1_data.extent(2); i1++) {
					// src0 and dst are viewed with shape of src1 and offset
					// => same indice
					for (int64_t i0 = 0; i0 < src1_data.extent(3); i0++)
						dst_data[i3, i2, i1, i0] = src1_data[i3, i2, i1, i0];
				}
			});
			scope.spawn(std::move(sender));
		}
	}
}

static void ggml_compute_forward_set(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_set<float>(pool, scope, dst);
	} break;
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_set<int32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_1:
	case GGML_TYPE_MXFP4:
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_Q3_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_TQ1_0:
	case GGML_TYPE_TQ2_0:
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ1_M:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ2_S:
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename T>
static void ggml_compute_forward_add_q_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	const int64_t nr = ggml_nrows(src0);

	const int nth = pool.available_parallelism();

	GGML_ASSERT(src0->type == dst->type);

	// we don't support permuted src0 or src1
	GGML_ASSERT(src0->nb[0] == sizeof(T));
	GGML_ASSERT(src1->nb[0] == sizeof(float));

	// dst cannot be transposed or permuted
	GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
	GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
	GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

	GGML_ASSERT(ggml_is_quantized(src0->type));
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> wdata(src0->ne[0]);
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0 indices
				const int i03 = ir / (src0->ne[2] * src0->ne[1]);
				const int i02 = (ir - i03 * src0->ne[2] * src0->ne[1]) / src0->ne[1];
				const int i01 = (ir - i03 * src0->ne[2] * src0->ne[1] - i02 * src0->ne[1]);

				// src1 and dst are same shape as src0 => same indices
				const int i13 = i03;
				const int i12 = i02;
				const int i11 = i01;

				const int i3 = i03;
				const int i2 = i02;
				const int i1 = i01;

				void* src0_row = (void*)((char*)src0->data + (i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]));
				float* src1_row = (float*)((char*)src1->data + (i11 * src1->nb[1] + i12 * src1->nb[2] + i13 * src1->nb[3]));
				void* dst_row = (void*)((char*)dst->data + (i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]));

				assert(src0->ne[0] % 32 == 0);

				// unquantize row from src0 to temp buffer
				dequantize_row(static_cast<const T*>(src0_row), wdata.data(), src0->ne[0]);
				// add src1
				std::span<const float> src1_row_span{ src1_row, static_cast<size_t>(src0->ne[0]) };
				std::ranges::transform(src1_row_span, wdata, wdata.begin(), std::plus<>());

				// quantize row to dst
				if constexpr (requires { quantize_row(wdata.data(), static_cast<T*>(dst_row), src0->ne[0]); }) {
					quantize_row(wdata.data(), static_cast<T*>(dst_row), src0->ne[0]);
				}
				else {
					memcpy(dst_row, wdata.data(), dst->ne[0] * dst->nb[0]);
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float, float)>
static void binary_op(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	/*  */ if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) { // all f32
		apply_binary_op<op, float, float, float>(pool, scope, dst);
	}
	else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) { // all f16
		apply_binary_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(pool, scope, dst);
	}
	else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
		apply_binary_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(pool, scope, dst);
	}
	else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
		apply_binary_op<op, ggml_bf16_t, float, ggml_bf16_t>(pool, scope, dst);
	}
	else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
		apply_binary_op<op, ggml_bf16_t, float, float>(pool, scope, dst);
	}
	else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
		apply_binary_op<op, ggml_fp16_t, float, ggml_fp16_t>(pool, scope, dst);
	}
	else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
		apply_binary_op<op, ggml_fp16_t, float, float>(pool, scope, dst);
	}
	else {
		GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
			ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
	}
}

static inline float op_add(float a, float b) {
	return a + b;
}

static inline float op_sub(float a, float b) {
	return a - b;
}

static inline float op_mul(float a, float b) {
	return a * b;
}

static inline float op_div(float a, float b) {
	return a / b;
}

static void ggml_compute_forward_add_non_quantized(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	binary_op<op_add>(pool, scope, dst);
}

static void ggml_compute_forward_add(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_add_non_quantized(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_add_q_f32<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_add_q_f32<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4: {
		ggml_compute_forward_add_q_f32<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_add_q_f32<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_add_q_f32<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_add_q_f32<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_add_q_f32<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_add_q_f32<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_add_q_f32<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_add_q_f32<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_add_q_f32<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_add_q_f32<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_add_q_f32<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_add_q_f32<block_iq2_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_add_q_f32<block_iq2_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_add_q_f32<block_iq3_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_add_q_f32<block_iq1_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_add_q_f32<block_iq1_m>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_add_q_f32<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_add_q_f32<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_add_q_f32<block_iq3_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_add_q_f32<block_iq2_s>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_mul(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	struct ggml_tensor* dst) {
	binary_op<op_mul>(pool, scope, dst);
}

static void ggml_compute_forward_div(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	struct ggml_tensor* dst) {
	binary_op<op_div>(pool, scope, dst);
}

template <typename src0_t, typename src1_t>
static void ggml_compute_forward_add1(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = toFloat32(*(const src1_t*)src1->data);

	GGML_ASSERT(dst->nb[0] == sizeof(src0_t));
	GGML_ASSERT(src0->nb[0] == sizeof(src0_t));

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	auto dst_data = make_strided_mdspan(static_cast<src0_t*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(src0->data), src0->ne, src0->nb);

	for (int64_t i3 = 0; i3 < dst_data.extent(0); i3++) {
		for (int64_t i2 = 0; i2 < dst_data.extent(1); i2++) {
			for (int64_t i1 = 0; i1 < dst_data.extent(2); i1++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					// src0 and dst are same shape => same indices
					for (int64_t i0 = 0; i0 < dst_data.extent(3); i0++)
						dst_data[i3, i2, i1, i0] = fromFloat32<src0_t>(toFloat32(src0_data[i3, i2, i1, i0]) + v);
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

static void ggml_vec_acc1_f32(const int n, float* y, const float   v) { for (int i = 0; i < n; ++i) y[i] += v; }

template <typename T>
static void ggml_compute_forward_add1_q_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = *(float*)src1->data;

	// we don't support permuted src0
	GGML_ASSERT(src0->nb[0] == sizeof(T));

	// dst cannot be transposed or permuted
	GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
	GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
	GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

	GGML_ASSERT(ggml_is_quantized(src0->type));
	GGML_ASSERT(dst->type == src0->type);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
		for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
			for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
				//stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					std::vector<float> wdata(dst->ne[0]);
					// src0 and dst are same shape => same indices

					void* src0_row = (void*)((char*)src0->data + (i1 * src0->nb[1] + i2 * src0->nb[2] + i3 * src0->nb[3]));
					void* dst_row = (void*)((char*)dst->data + (i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]));

					assert(dst->ne[0] % 32 == 0);

					// unquantize row from src0 to temp buffer
					dequantize_row(static_cast<const T*>(src0_row), wdata.data(), dst->ne[0]);
					// add src1
					ggml_vec_acc1_f32(dst->ne[0], wdata.data(), v);
					// quantize row to dst
					quantize_row(wdata.data(), static_cast<T*>(dst_row), dst->ne[0]);
				//});
				//scope.spawn(std::move(sender));	
			}
		}
	}
}

static void ggml_compute_forward_add1(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(dst->type == src0->type);

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		GGML_ASSERT(src1->type == GGML_TYPE_F32);
		ggml_compute_forward_add1<ggml_fp32_t, ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		if (src1->type == GGML_TYPE_F16) {
			ggml_compute_forward_add1<ggml_fp16_t, ggml_fp16_t>(pool, scope, dst);
		}
		else if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add1<ggml_fp16_t, ggml_fp32_t>(pool, scope, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
	case GGML_TYPE_BF16:
	{
		if (src1->type == GGML_TYPE_BF16) {
			ggml_compute_forward_add1<ggml_bf16_t, ggml_bf16_t>(pool, scope, dst);
		}
		else if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add1<ggml_bf16_t, ggml_fp32_t>(pool, scope, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_add1_q_f32<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_add1_q_f32<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4: {
		ggml_compute_forward_add1_q_f32<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_add1_q_f32<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_add1_q_f32<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_add1_q_f32<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_add1_q_f32<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_add1_q_f32<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_add1_q_f32<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_add1_q_f32<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_add1_q_f32<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_add1_q_f32<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_add1_q_f32<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_add1_q_f32<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_add1_q_f32<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ1_M:
	case GGML_TYPE_IQ2_S:
	{
		// These types are lack of qutization row functionity
		[[fallthrough]];
	}
	case GGML_TYPE_Q8_1:
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_ssm_conv_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0]; // conv_x
	const ggml_tensor* src1 = dst->src[1]; // conv1d.weight

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int d_conv = src1->ne[0];
	const int d_inner = src0->ne[1];
	const int n_t = dst->ne[1]; // tokens per sequence
	const int n_s = dst->ne[2]; // number of sequences in the batch

	GGML_ASSERT(dst->ne[0] == d_inner);
	GGML_ASSERT(src0->ne[1] == d_inner);
	GGML_ASSERT(src0->nb[0] == sizeof(float));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
	GGML_ASSERT(dst->ne[2] == src0->ne[2]);

	// { n_s, n_t, d_inner }
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), n_s, n_t, d_inner);
	// { n_s, d_inner, d_conv - 1 + n_t }
	std::experimental::mdspan conv_x(static_cast<const float*>(src0->data), n_s, d_inner, src0->ne[0]);
	// { d_inner, d_conv }
	std::experimental::mdspan conv1d_weight(static_cast<const float*>(src1->data), d_inner, d_conv);

	for (int64_t i1 = 0; i1 < d_inner; i1++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int i3 = 0; i3 < n_s; ++i3) {
				for (int i2 = 0; i2 < n_t; ++i2) {
					// sliding window
					// TODO: transpose the output for smaller strides for big batches?
					// rowwise dot product
					float sumf = 0.0f;

					for (int64_t i0 = 0; i0 < d_conv; ++i0) {
						sumf += conv_x[i3, i1, i2 + i0] * conv1d_weight[i1, i0];
					}
					dst_data[i3, i2, i1] = sumf;
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_ssm_conv(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->src[0]->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_ssm_conv_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_ssm_scan_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0]; // s  {d_state, dim, n_head, n_seqs+}
	const ggml_tensor* src1 = dst->src[1]; // x  {dim, n_head, n_seq_tokens, n_seqs}
	const ggml_tensor* src2 = dst->src[2]; // dt {n_head, n_seq_tokens, n_seqs}
	const ggml_tensor* src3 = dst->src[3]; // A  {d_state, n_head} or {1, n_head}
	const ggml_tensor* src4 = dst->src[4]; // B  {d_state, n_group, n_seq_tokens, n_seqs}
	const ggml_tensor* src5 = dst->src[5]; // C  {d_state, n_group, n_seq_tokens, n_seqs}
	const ggml_tensor* src6 = dst->src[6]; // ids {n_seqs}

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int64_t nc = src0->ne[0]; // d_state
	const int64_t nr = src0->ne[1]; // dim
	const int64_t nh = src1->ne[1]; // n_head
	const int64_t ng = src4->ne[1];
	const int64_t nt = src1->ne[2]; // number of tokens per sequence
	const int64_t ns = src1->ne[3]; // number of sequences in the batch

	// can't use ggml_nbytes because src1 is not necessarily contiguous
	const int64_t s_off = src1->nelements() * ggml_element_size(src1);

	GGML_ASSERT(src1->nelements() + nc * nr * nh * ns == dst->nelements());
	GGML_ASSERT(src0->nb[0] == sizeof(float));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src2->nb[0] == sizeof(float));
	GGML_ASSERT(src3->nb[0] == sizeof(float));
	GGML_ASSERT(src4->nb[0] == sizeof(float));
	GGML_ASSERT(src5->nb[0] == sizeof(float));
	GGML_ASSERT(src6->nb[0] == sizeof(int32_t));
	GGML_ASSERT(nh % ng == 0);
	std::experimental::mdspan y(static_cast<float*>(dst->data), ns, nt, nh, nr);  // { ns, nt, nh, dim }
	auto s = make_strided_mdspan((float*)((char*)dst->data + s_off), src0->ne, src0->nb); 	// {ns, nh, dim, d_state }
	auto s0 = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb); // { ns, nh, dim, d_state }
	auto x = make_strided_mdspan(static_cast<const float*>(src1->data), src1->ne, src1->nb); // { ns, nt, nh, dim }
	auto dt = make_strided_mdspan<3>(static_cast<const float*>(src2->data), src2->ne, src2->nb); // { ns, nt, nh }
	auto A = make_strided_mdspan<2>(static_cast<const float*>(src3->data), src3->ne, src3->nb); // { nh, d_state } or { nh, 1 }
	auto B = make_strided_mdspan(static_cast<const float*>(src4->data), src4->ne, src4->nb); // { ns, nt, ng, d_state }
	auto C = make_strided_mdspan(static_cast<const float*>(src5->data), src5->ne, src5->nb); // { ns, nt, ng, d_state }

	std::span<const int32_t> ids((const int32_t*)src6->data, ns);

	if (src3->ne[0] == 1) {
		// n_head
		for (int64_t h = 0; h < nh; ++h) {
			// ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
				for (int64_t i3 = 0; i3 < ns; ++i3) {
					std::experimental::mdspan sub_s0(&s0[ids[i3], 0, 0, 0], nh, nr, nc);
					std::experimental::mdspan sub_s(&s[i3, 0, 0, 0], nh, nr, nc);
					for (int64_t i2 = 0; i2 < nt; ++i2) {
						// Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

						const float dt_soft_plus = ggml_softplus(dt[i3, i2, h]);
						const float dA = expf(dt_soft_plus * A[h, 0]);
						const int g = h / (nh / ng); // repeat_interleave

						// dim
						for (int64_t i1 = 0; i1 < nr; ++i1) {
							const float x_dt = x[i3, i2, h, i1] * dt_soft_plus;
							float sumf = 0.0f;
							// d_state
							for (int64_t i0 = 0; i0 < nc; ++i0) {
								// state = prev_state * dA + dB * x
								const float state = (sub_s0[h, i1, i0] * dA) + (B[i3, i2, g, i0] * x_dt);
								// y = rowwise_dotprod(state, C)
								sumf += state * C[i3, i2, g, i0];
								s[i3, h, i1, i0] = state;
							}
							y[i3, i2, h, i1] = sumf;
						}
						// use the output as the source when it's not the first token-wise iteration
						sub_s0 = sub_s;
					}
				}
			});
			scope.spawn(std::move(sender));
		}
	}
	else {
		// n_head
		for (int64_t h = 0; h < nh; ++h) {
			// ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
				for (int64_t i3 = 0; i3 < ns; ++i3) {
					std::experimental::mdspan sub_s0(&s0[ids[i3], 0, 0, 0], nh, nr, nc);
					std::experimental::mdspan sub_s(&s[i3, 0, 0, 0], nh, nr, nc);
					for (int64_t i2 = 0; i2 < nt; ++i2) {
						// Mamba-1 has an element-wise decay factor for the states

						const float dt_soft_plus = ggml_softplus(dt[i3, i2, h]);
						const int g = h / (nh / ng); // repeat_interleave

						// dim
						for (int64_t i1 = 0; i1 < nr; ++i1) {
							const float x_dt = x[i3, i2, h, i1] * dt_soft_plus;
							float sumf = 0.0f;
							// NOTE: can't really use GGML_SIMD here because d_state is usually 16
							//       and also because expf is used within the loop.
							// d_state
							for (int64_t i0 = 0; i0 < nc; ++i0) {
								// state = prev_state * dA + dB * x
								const float state = (sub_s0[h, i1, i0] * expf(dt_soft_plus * A[h, i0])) + (B[i3, i2, g, i0] * x_dt);
								// y = rowwise_dotprod(state, C)
								sumf += state * C[i3, i2, g, i0];
								s[i3, h, i1, i0] = state;
							}
							y[i3, i2, h, i1] = sumf;
						}
						// use the output as the source when it's not the first token-wise iteration
						sub_s0 = sub_s;
					}
				}
			});
			scope.spawn(std::move(sender));
		}
	}
}

static void ggml_compute_forward_ssm_scan(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->src[0]->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_ssm_scan_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_rwkv_wkv6_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const int64_t T = dst->src[1]->ne[2];
	const int64_t C = dst->ne[0];
	const int64_t HEADS = dst->src[1]->ne[1];
	const int64_t n_seqs = dst->src[5]->ne[1];
	const int64_t head_size = C / HEADS;

	float* state = ((float*)dst->data) + C * T;

	stdexec::scheduler auto scheduler = pool.get_scheduler();
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), T, HEADS, head_size);
	std::experimental::mdspan k(static_cast<const float*>(dst->src[0]->data), T, HEADS, head_size);
	std::experimental::mdspan v(static_cast<const float*>(dst->src[1]->data), T, HEADS, head_size);
	std::experimental::mdspan r(static_cast<const float*>(dst->src[2]->data), T, HEADS, head_size);
	std::experimental::mdspan time_faaaa(static_cast<const float*>(dst->src[3]->data), HEADS, head_size);
	std::experimental::mdspan time_decay(static_cast<const float*>(dst->src[4]->data), T, HEADS, head_size);

	GGML_ASSERT(HEADS * head_size == C);
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS

	memset(dst->data, 0, T * C * sizeof(float));

	for (int64_t h = 0; h < HEADS; h++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			// basically fused operations:
			// dst = r @ (time_faaaa * (k @ v) + state),
			// state = time_decay * state + (k @ v),
			// recursive through each token
			for (int64_t t = 0; t < T; t++) {
				size_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				std::experimental::mdspan state_cur1(static_cast<float*>(state_cur), HEADS, head_size, head_size);
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;
				std::experimental::mdspan state_prev1(static_cast<float*>(state_prev), HEADS, head_size, head_size);

				for (int64_t i = 0; i < head_size; i++) {
					float k_val = k[t, h, i];
					float r_val = r[t, h, i];
					float time_faaaa_val = time_faaaa[h, i];
					// RWKV v6: different time_decay for each token.
					float time_decay_val = time_decay[t, h, i];

					for (int64_t j = 0; j < head_size; j++) {
						float v_val = v[t, h, j];
						float kv_val = v_val * k_val;
						float prev_state_val = state_prev1[h, i, j];
						float temp_val = kv_val * time_faaaa_val + prev_state_val;
						dst_data[t, h, j] += temp_val * r_val;
						state_cur1[h, i, j] = prev_state_val * time_decay_val + kv_val;
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_rwkv_wkv6(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rwkv_wkv6_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_gla_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const int64_t T = dst->src[1]->ne[2];
	const int64_t C = dst->ne[0];
	const int64_t HEADS = dst->src[1]->ne[1];
	const int64_t n_seqs = dst->src[4]->ne[1];
	const int64_t head_size = C / HEADS;
	const float scale = std::bit_cast<float>(dst->op_params[0]);

	float* state = ((float*)dst->data) + C * T;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), T, HEADS, head_size);
	std::experimental::mdspan k(static_cast<const float*>(dst->src[0]->data), T, HEADS, head_size);
	std::experimental::mdspan v(static_cast<const float*>(dst->src[1]->data), T, HEADS, head_size);
	std::experimental::mdspan q(static_cast<const float*>(dst->src[2]->data), T, HEADS, head_size);
	std::experimental::mdspan g(static_cast<const float*>(dst->src[3]->data), T, HEADS, head_size);

	size_t t_stride = HEADS * head_size; // Same to C

	size_t h_stride = C / HEADS;
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
	size_t h_stride_2d = head_size * head_size;

	memset(dst->data, 0, T * C * sizeof(float));

	for (int64_t h = 0; h < HEADS; h++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t t = 0; t < T; t++) {
				size_t t_offset = t * t_stride;
				size_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				std::experimental::mdspan state_cur1(static_cast<float*>(state_cur), HEADS, head_size, head_size);
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;
				std::experimental::mdspan state_prev1(static_cast<float*>(state_prev), HEADS, head_size, head_size);


				size_t h_offset = h * h_stride;
				size_t t_h_offset = t_offset + h_offset;
				size_t h_2d_offset = h * h_stride_2d;

				for (int64_t i = 0; i < head_size; i++) {
					size_t h_2d_i_offset = h_2d_offset + i * h_stride;

					float k_val = k[t, h, i];
					float q_val = q[t, h, i] * scale;
					float g_val = g[t, h, i];

					for (int64_t j = 0; j < head_size; j++) {
						size_t t_h_j_offset = t_h_offset + j;
						size_t h_2d_i_j_offset = h_2d_i_offset + j;

						float v_val = v[t, h, j];
						float kv_val = v_val * k_val;
						float prev_state_val = state_prev1[h, i, j];
						float temp_val = prev_state_val * g_val + kv_val;
						dst_data[t, h, j] += temp_val * q_val;
						state_cur1[h, i, j] = temp_val;
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_gla(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_gla_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

struct mmid_row_mapping {
	int32_t i1;
	int32_t i2;
};

template <typename src0_t>
static void ggml_compute_forward_mul_mat_id_one_chunk(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst,
	const int64_t cur_a,
	const int64_t ir0_start,
	const int64_t ir0_end,
	const int64_t ir1_start,
	const int64_t ir1_end,
	const char* src0_cur,
	const std::experimental::mdspan<mmid_row_mapping, std::experimental::dims<2>> matrix_rows,
	const size_t row_size,
	const bool src1_cont,
	const void* wdata) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	const ggml_tensor* ids = dst->src[2];

	const enum ggml_type type = src0->type;

	using vec_dot_t = typename vec_dot_trait<src0_t>::type;
	enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;
	const int64_t blck_0 = 16;
	const int64_t blck_1 = 16;

	stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
		float tmp[16];

		for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
			for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
				for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ++ir1) {
					const int64_t _i12 = ir1; // logical row index for this expert

					struct mmid_row_mapping row_mapping = matrix_rows[cur_a, _i12];
					const int id = row_mapping.i1; // selected expert index

					const int64_t  i11 = id % src1->ne[1];
					const int64_t  i12 = row_mapping.i2; // row index in src1

					const int64_t  i1 = id;  // selected expert index
					const int64_t  i2 = i12; // row

					// desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
					//       if it is, then we have either copied the data to wdata and made it contiguous or we are using
					//       the original src1 data pointer, so we should index using the indices directly
					// TODO: this is a bit of a hack, we should probably have a better way to handle this
					const char* src1_col = (const char*)wdata +
						(src1_cont || src1->type != vec_dot_type
							? (i11 + i12 * src1->ne[1]) * row_size
							: (i11 * src1->nb[1] + i12 * src1->nb[2]));

					float* dst_col = (float*)((char*)dst->data + (i1 * dst->nb[1] + i2 * dst->nb[2]));

					for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
						ggml_vec_dot<src0_t, vec_dot_t>(src0->ne[0], &tmp[ir0 - iir0], 0,
							cast_with_offset<src0_t>(src0_cur, ir0 * src0->nb[1]), 0,
							cast_with_offset<vec_dot_t>(src1_col, 0), 0, 1);
					}

					memcpy(&dst_col[iir0], tmp, (std::min(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
				}
			}
		}
	});
	scope.spawn(std::move(sender));
}

template <typename src0_t>
static void ggml_compute_forward_mul_mat_id(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	using vec_dot_t = typename vec_dot_trait<src0_t>::type;
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	const ggml_tensor* ids = dst->src[2];

	const int nth = pool.available_parallelism();

	const ggml_type type = src0->type;

	const bool src1_cont = ggml_is_contiguous(src1);

	enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

	// we don't support permuted src0 or src1
	GGML_ASSERT(src0->nb[0] == ggml_type_size(type));
	GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));

	// dst cannot be transposed or permuted
	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
	GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
	GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

	// row groups
	const int n_ids = ids->ne[0]; // n_expert_used
	const int n_as = src0->ne[2];       // n_expert

	// initialize matrix_row_counts
	std::vector<int64_t> matrix_row_counts(n_as, 0); // [n_as]
	std::vector<mmid_row_mapping> matrix_rows(n_as * ids->ne[0] * ids->ne[1]); // [n_as][ids->ne[0]*ids->ne[1]]
	std::experimental::mdspan matrix_rows_view(matrix_rows.data(), n_as, ids->ne[0] * ids->ne[1]);

	std::vector<uint8_t> wdata_1;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	if (src1->type != vec_dot_type) {
		wdata_1.resize(ggml_row_size(vec_dot_type, src1->nelements()));

		const size_t nbw1 = ggml_row_size(vec_dot_type, src1->ne[0]);
		const size_t nbw2 = nbw1 * src1->ne[1];
		const size_t nbw3 = nbw2 * src1->ne[2];

		GGML_ASSERT(src1->type == GGML_TYPE_F32);
		for (int64_t ir0 = 0; ir0 < src1->ne[1]; ir0 += nth) {
			const int64_t ir1 = std::min(ir0 + nth, src1->ne[1]);
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &wdata_1] {
				for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
					for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
						for (int64_t i11 = ir0; i11 < ir1; i11++) {
							fromFloat(
								cast_with_offset<float>(src1->data, i13 * src1->nb[3] + i12 * src1->nb[2] + i11 * src1->nb[1]),
								cast_with_offset<vec_dot_t>(wdata_1.data(), i13 * nbw3 + i12 * nbw2 + i11 * nbw1),
								src1->ne[0]);
						}
					}
				}
			});
			scope.spawn(std::move(sender));
		}
		stdexec::sync_wait(scope.on_empty());
	}

	// group rows by src0 matrix
	for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
		for (int id = 0; id < n_ids; ++id) {
			const int32_t i02 = *(const int32_t*)((const char*)ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

			assert(i02 >= 0 && i02 < n_as);

			mmid_row_mapping new_map{ id, static_cast<int32_t>(iid1) };
			//MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = new_map;
			matrix_rows_view[i02, matrix_row_counts[i02]++] = new_map;
		}
	}

	// compute each matrix multiplication in sequence
	for (int cur_a = 0; cur_a < n_as; ++cur_a) {
		const int64_t cne1 = matrix_row_counts[cur_a];

		if (cne1 == 0) {
			continue;
		}

		const char* src0_cur = (const char*)src0->data + cur_a * src0->nb[2];
		const void* wdata = (src1->type == vec_dot_type) ? src1->data : (const void*)wdata_1.data();
		const size_t row_size = ggml_row_size(vec_dot_type, src1->ne[0]);

		const int64_t nr0 = src0->ne[1];
		const int64_t nr1 = cne1;

		int chunk_size = 16;
		if (nr0 == 1 || nr1 == 1) {
			chunk_size = 64;
		}

		// disable for NUMA
		const bool disable_chunking = ggml_is_numa();

		int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
		int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

		if (nchunk0* nchunk1 < nth * 4 || disable_chunking) {
			nchunk0 = nr0 > nr1 ? nth : 1;
			nchunk1 = nr0 > nr1 ? 1 : nth;
		}

		const int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
		const int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

		for (int64_t current_chunk = 0; current_chunk < nchunk0 * nchunk1; current_chunk++) {
			const int64_t ith0 = current_chunk % nchunk0;
			const int64_t ith1 = current_chunk / nchunk0;

			const int64_t ir0_start = dr0 * ith0;
			const int64_t ir0_end = std::min(ir0_start + dr0, nr0);

			const int64_t ir1_start = dr1 * ith1;
			const int64_t ir1_end = std::min(ir1_start + dr1, nr1);

			ggml_compute_forward_mul_mat_id_one_chunk<src0_t>(
				pool, scope, dst, cur_a,
				ir0_start, ir0_end, ir1_start, ir1_end,
				src0_cur, matrix_rows_view, row_size, src1_cont, wdata
			);
		}
	}
	stdexec::sync_wait(scope.on_empty());
}

static void ggml_compute_forward_mul_mat_id(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	switch (src0->type) {
	case GGML_TYPE_F32:	{
		ggml_compute_forward_mul_mat_id<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16: {
		ggml_compute_forward_mul_mat_id<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16: {
		ggml_compute_forward_mul_mat_id<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_mul_mat_id<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_mul_mat_id<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_mul_mat_id<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_mul_mat_id<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_mul_mat_id<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_mul_mat_id<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_mul_mat_id<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_mul_mat_id<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_mul_mat_id<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_mul_mat_id<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_mul_mat_id<block_iq1_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_mul_mat_id<block_iq1_m>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_mul_mat_id<block_iq2_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_mul_mat_id<block_iq2_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_mul_mat_id<block_iq2_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_mul_mat_id<block_iq3_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_mul_mat_id<block_iq3_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_mul_mat_id<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_mul_mat_id<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_mul_mat_id<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_mul_mat_id<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4: {
		ggml_compute_forward_mul_mat_id<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_1:
	default:
		assert(false);
	}
}

template <typename T>
static void ggml_compute_forward_clamp(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	float min = std::bit_cast<float>(dst->op_params[0]);
	float max = std::bit_cast<float>(dst->op_params[1]);
	GGML_ASSERT(min <= max);

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int nc = src0->ne[0];

	const size_t nb00 = src0->nb[0];
	const size_t nb01 = src0->nb[1];

	const size_t nb0 = dst->nb[0];
	const size_t nb1 = dst->nb[1];

	GGML_ASSERT(nb0 == sizeof(T));
	GGML_ASSERT(nb00 == sizeof(T));

	auto dst_data = make_strided_mdspan(static_cast<T*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);

	for (int64_t i = 0; i < src0_data.extent(0); i++) {
		for (int64_t j = 0; j < src0_data.extent(1); j++) {
			for (int64_t k = 0; k < src0_data.extent(2); k++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					for (int64_t l = 0; l < src0_data.extent(3); l++) {
						dst_data[i, j, k, l] = fromFloat32<T>(std::clamp(toFloat32(src0_data[i, j, k, l]), min, max));
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

static void ggml_compute_forward_clamp(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_clamp<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_clamp<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16:
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_1:
	case GGML_TYPE_MXFP4:
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_Q3_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_TQ1_0:
	case GGML_TYPE_TQ2_0:
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ1_M:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ2_S:
	case GGML_TYPE_Q8_K:
	case GGML_TYPE_I8:
	case GGML_TYPE_I16:
	case GGML_TYPE_I32:
	case GGML_TYPE_I64:
	case GGML_TYPE_F64:
	case GGML_TYPE_COUNT:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_diag_mask_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst,
	const float value) {

	const ggml_tensor* src0 = dst->src[0];

	const int  n_past = ((int32_t*)dst->op_params)[0];
	const bool inplace = src0->data == dst->data;

	GGML_ASSERT(n_past >= 0);

	if (!inplace) {
		// memcpy needs to be synchronized across threads to avoid race conditions.
		// => do it in INIT phase
		GGML_ASSERT(dst->nelements() == src0->nelements());
		GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
		memcpy(
			((char*)dst->data),
			((char*)src0->data),
			dst->nbytes());
	}

	// TODO: handle transposed/permuted matrices
	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[0] == sizeof(float));

	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	for (int64_t i3 = 0; i3 < src0->ne[3]; i3++)
		for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				for (int64_t i1 = 0; i1 < src0->ne[1]; i1++)
					for (int64_t i0 = n_past; i0 < src0->ne[0]; i0++) {
						if (i0 > n_past + i1) {
							dst_data[i3, i2, i1, i0] = value;
						}
					}
			});
			scope.spawn(std::move(sender));
		}
}

static void ggml_compute_forward_diag_mask(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		const float value = (dst->op == GGML_OP_DIAG_MASK_INF) ? -INFINITY : 0.0f;
		ggml_compute_forward_diag_mask_f32(pool, scope, dst, value);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename src1_t>
static void ggml_compute_forward_soft_max_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	const ggml_tensor* src2 = dst->src[2];

	assert(ggml_is_contiguous(dst));
	assert(ggml_are_same_shape(src0, dst));

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);

	// TODO: is this supposed to be ceil instead of floor?
	//       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
	const uint32_t n_head = src0->ne[2];
	const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

	const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
	const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	const int nc = src0->ne[0];

	const int64_t nh = src0->ne[1];

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);
	auto src1_data = [=]() {
		using type_t = decltype(make_strided_mdspan(static_cast<const src1_t*>(src1 ? src1->data : nullptr), src1->ne, src1->nb));
		if (!src1) return type_t{};
		return make_strided_mdspan(static_cast<const src1_t*>(src1->data), src1->ne, src1->nb);
	}();

	// sinks
	const float* sk = src2 ? (float*)((char*)src2->data) : nullptr;

	for (int64_t i01 = 0; i01 < nh; i01 ++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> wp(nc);
			for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
				for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
					// ALiBi
					const uint32_t h = i02; // head
					const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

					// broadcast the mask across rows

					for (size_t i00 = 0; i00 < src0->ne[0]; i00++) {
						wp[i00] = src0_data[i03, i02, i01, i00] * scale;
						if (!src1_data.empty()) {
							if constexpr (std::is_same_v<src1_t, ggml_fp16_t> || std::is_same_v<src1_t, ggml_fp32_t>) {
								wp[i00] += slope * toFloat32(src1_data[i03 % src1_data.extent(0), i02 % src1_data.extent(1), i01, i00]);
							}
						}
#ifndef NDEBUG
						//printf("p[%d] = %f\n", i, p[i]);
						assert(!isnan(wp[i00]));
#endif
					}

					float max = *std::max_element(wp.begin(), wp.end());

					// if we have sinks, make a correction as if they were included in the softmax
					if (sk) {
						max = std::max(max, sk[i02]);
					}

					float sum = 0;
					for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
						float val = expf(wp[i00] - max);
						sum += val;
						wp[i00] = val;
					}
					assert(sum > 0.0);

					if (sk) {
						sum += (ggml_float)expf(sk[i02] - max);
					}

					for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
						dst_data[i03, i02, i01, i00] = wp[i00] / sum;
#ifndef NDEBUG
						assert(!isnan(dst_data[i03, i02, i01, i00]));
						assert(!isinf(dst_data[i03, i02, i01, i00]));
#endif
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

template <typename src1_t>
static void ggml_compute_forward_soft_max(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_soft_max_f32<src1_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_soft_max(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src1 = dst->src[1];
	if (!src1) {
		ggml_compute_forward_soft_max_f32<std::nullptr_t>(pool, scope, dst);
		return;
	}
	switch (src1->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_soft_max_f32<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_soft_max_f32<ggml_fp16_t>(pool, scope, dst);
	} break;
	default:
	{
		assert(false);
	}
	}
}

static void ggml_compute_forward_soft_max_ext_back_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_is_contiguous(src0));
	GGML_ASSERT(ggml_is_contiguous(src1));
	GGML_ASSERT(ggml_is_contiguous(dst));
	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_are_same_shape(src1, dst));

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);

	GGML_ASSERT(max_bias == 0.0f);

	// TODO: handle transposed/permuted matrices

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	std::experimental::mdspan dx(static_cast<float*>(dst->data), dst->ne[3] * dst->ne[2] * dst->ne[1], dst->ne[0]);
	std::experimental::mdspan dy(static_cast<const float*>(src0->data), src0->ne[3] * src0->ne[2] * src0->ne[1], src0->ne[0]);
	std::experimental::mdspan y(static_cast<const float*>(src1->data), src1->ne[3] * src1->ne[2] * src1->ne[1], src1->ne[0]);

	// row range for this thread
	for (int64_t i1 = 0; i1 < dx.extent(0); i1++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			// Jii = yi - yi*yi
			// Jij = -yi*yj
			// J = diag(y)-y.T*y
			// dx = J * dy
			// dxk = sum_i(Jki * dyi)
			// dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
			// dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
			// dxk = sum_i(-yk*yi * dyi) + yk*dyk
			// dxk = -yk * sum_i(yi * dyi) + yk*dyk
			// dxk = -yk * dot(y, dy) + yk*dyk
			// dxk = yk * (- dot(y, dy) + dyk)
			// dxk = yk * (dyk - dot(y, dy))
			//
			// post-order:
			// dot_y_dy := dot(y, dy)
			// dx := dy
			// dx := dx - dot_y_dy
			// dx := dx * y

			// linear runtime, no additional memory
			float dot_y_dy = 0;
			for (int64_t i0 = 0; i0 < dx.extent(1); i0++) {
#ifndef NDEBUG
				assert(!isnan(dy[i1, i0]));
				assert(!isnan(y[i1, i0]));
#endif
				dot_y_dy += y[i1, i0] * dy[i1, i0];
			}
			for (int64_t i0 = 0; i0 < dx.extent(1); i0++) {
				dx[i1, i0] = (dy[i1, i0] - dot_y_dy) * y[i1, i0] * scale;
#ifndef NDEBUG
				assert(!isnan(dx[i1, i0]));
				assert(!isinf(dx[i1, i0]));
#endif
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_soft_max_ext_back(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_soft_max_ext_back_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_dim(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
static float ggml_rope_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float base) {
	return n_dims * logf(n_ctx_orig / (n_rot * 2 * std::numbers::pi_v<float>)) / (2 * logf(base));
}

static void ggml_rope_yarn_corr_dims(
	int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]
) {
	// start and end correction dims
	float start = floorf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
	float end = ceilf(ggml_rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
	dims[0] = std::max<float>(0, start);
	dims[1] = std::min<float>(n_dims - 1, end);
}

static float rope_yarn_ramp(const float low, const float high, const int i0) {
	const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
	return 1 - std::min(1.0f, std::max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
	float theta_extrap, float freq_scale, float corr_dims[2], int64_t i0, float ext_factor, float mscale,
	float* cos_theta, float* sin_theta) {
	// Get n-d rotational scaling corrected for extrapolation
	float theta_interp = freq_scale * theta_extrap;
	float theta = theta_interp;
	if (ext_factor != 0.0f) {
		float ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
		theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

		// Get n-d magnitude scaling corrected for interpolation
		mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
	}
	*cos_theta = cosf(theta) * mscale;
	*sin_theta = sinf(theta) * mscale;
}

static void ggml_rope_cache_init(
	float theta_base, float freq_scale, const float* freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
	float* cache, float sin_sign, float theta_scale) {
	// ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
	float theta = theta_base;
	for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
		const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
		rope_yarn(
			theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
		);
		cache[i0 + 1] *= sin_sign;

		theta *= theta_scale;
	}
}

static void ggml_mrope_cache_init(
	float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool is_imrope, bool indep_sects,
	float freq_scale, const float* freq_factors, float corr_dims[2], int64_t ne0, float ext_factor, float mscale,
	float* cache, float sin_sign, float theta_scale) {
	// ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
	float theta_t = theta_base_t;
	float theta_h = theta_base_h;
	float theta_w = theta_base_w;
	float theta_e = theta_base_e;  // extra position id for vision encoder
	int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
	int sec_w = sections[1] + sections[0];
	int sec_e = sections[2] + sec_w;
	GGML_ASSERT(sect_dims <= ne0);

	for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
		const float ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;

		int sector = (i0 / 2) % sect_dims;
		if (indep_sects) {
			// compute theta independently for each dim sections
			// (i.e. reset corresponding theta when `i0` go from one section to another)
			if (sector == 0) {
				theta_t = theta_base_t;
			}
			else if (sector == sections[0]) {
				theta_h = theta_base_h;;
			}
			else if (sector == sec_w) {
				theta_w = theta_base_w;
			}
			else if (sector == sec_e) {
				theta_e = theta_base_e;
			}
		}

		float theta = theta_t;
		if (is_imrope) { // qwen3vl apply interleaved mrope
			if (sector % 3 == 1 && sector < 3 * sections[1]) {
				theta = theta_h;
			}
			else if (sector % 3 == 2 && sector < 3 * sections[2]) {
				theta = theta_w;
			}
			else if (sector % 3 == 0 && sector < 3 * sections[0]) {
				theta = theta_t;
			}
			else {
				theta = theta_e;
			}
		}
		else {
			if (sector >= sections[0] && sector < sec_w) {
				theta = theta_h;
			}
			else if (sector >= sec_w && sector < sec_w + sections[2]) {
				theta = theta_w;
			}
			else if (sector >= sec_w + sections[2]) {
				theta = theta_e;
			}
		}

		rope_yarn(
			theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, &cache[i0 + 0], &cache[i0 + 1]
		);
		cache[i0 + 1] *= sin_sign;

		theta_t *= theta_scale;
		theta_w *= theta_scale;
		theta_h *= theta_scale;
		theta_e *= theta_scale;
	}
}

template <typename T, bool forward>
static void ggml_compute_forward_rope(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	const ggml_tensor* src2 = dst->src[2];

	int sections[4];

	//const int n_past     = std::bit_cast<int>(dst->op_params[0]);
	const int n_dims = std::bit_cast<int>(dst->op_params[1]);
	const int mode = std::bit_cast<int>(dst->op_params[2]);
	//const int n_ctx      = std::bit_cast<int>(dst->op_params[3]);
	const int n_ctx_orig = std::bit_cast<int>(dst->op_params[4]);
	const float freq_base = std::bit_cast<float>(dst->op_params[5]);
	const float freq_scale = std::bit_cast<float> (dst->op_params[6]);
	const float ext_factor = std::bit_cast<float> (dst->op_params[7]);
	const float attn_factor = std::bit_cast<float> (dst->op_params[8]);
	const float beta_fast = std::bit_cast<float> (dst->op_params[9]);
	const float beta_slow = std::bit_cast<float> (dst->op_params[10]);
	memcpy(&sections, (int32_t*)dst->op_params + 11, sizeof(int) * 4);

	//printf("dst->ne[0]: %d, dst->ne[1]: %d, dst->ne[2]: %d, dst->ne[3]: %d\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
	//printf("n_past = %d, dst->ne[2] = %d\n", n_past, dst->ne[2]);

	GGML_ASSERT(dst->nb[0] == sizeof(T));

	GGML_ASSERT(n_dims <= dst->ne[0]);
	GGML_ASSERT(n_dims % 2 == 0);

	const float theta_scale = powf(freq_base, -2.0f / n_dims);

	float corr_dims[2];
	ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

	const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
	const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
	const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE; // qwen3vl apply interleaved mrope
	const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

	if (is_mrope) {
		GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
	}

	if (is_vision) {
		GGML_ASSERT(n_dims == dst->ne[0] / 2);
	}

	const float* freq_factors = NULL;
	if (src2 != NULL) {
		GGML_ASSERT(src2->type == GGML_TYPE_F32);
		GGML_ASSERT(src2->ne[0] >= n_dims / 2);
		freq_factors = (const float*)src2->data;
	}

	// backward process uses inverse rotation by cos and sin.
	// cos and sin build a rotation matrix, where the inverse is the transpose.
	// this essentially just switches the sign of sin.
	constexpr float sin_sign = forward ? 1.0f : -1.0f;

	const int32_t* pos = (const int32_t*)src1->data;

	auto src0_data = make_strided_mdspan(static_cast<const T*>(src0->data), src0->ne, src0->nb);
	auto dst_data = make_strided_mdspan(static_cast<T*>(dst->data), dst->ne, dst->nb);

	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
		for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
			for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &corr_dims, &sections] {
					std::vector<float> cache(dst->ne[0]);

					if (!is_mrope) {
						const int64_t p = pos[i2];
						ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, dst->ne[0], ext_factor, attn_factor, &cache[0], sin_sign, theta_scale);
					}
					else {
						const int64_t p_t = pos[i2];
						const int64_t p_h = pos[i2 + dst->ne[2]];
						const int64_t p_w = pos[i2 + dst->ne[2] * 2];
						const int64_t p_e = pos[i2 + dst->ne[2] * 3];
						ggml_mrope_cache_init(
							p_t, p_h, p_w, p_e, sections, is_imrope, is_vision,
							freq_scale, freq_factors, corr_dims, dst->ne[0], ext_factor, attn_factor, &cache[0], sin_sign, theta_scale);
					}

					if (is_neox || is_mrope) {
						if (is_vision) {
							for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
								const int64_t ic = i0 / 2;
								
								const float cos_theta = cache[i0 + 0];
								const float sin_theta = cache[i0 + 1];

								const float x0 = toFloat32(src0_data[i3, i2, i1, ic + 0]);
								const float x1 = toFloat32(src0_data[i3, i2, i1, ic + n_dims]);

								dst_data[i3, i2, i1, ic + 0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
								dst_data[i3, i2, i1, ic + n_dims] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
							}
						}
						else {
							for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
								const int64_t ic = i0 / 2;

								const float cos_theta = cache[i0 + 0];
								const float sin_theta = cache[i0 + 1];

								const float x0 = toFloat32(src0_data[i3, i2, i1, ic + 0]);
								const float x1 = toFloat32(src0_data[i3, i2, i1, ic + n_dims / 2]);

								dst_data[i3, i2, i1, ic + 0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
								dst_data[i3, i2, i1, ic + n_dims / 2] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
							}
						}
					}
					else {
						for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
							const float cos_theta = cache[i0 + 0];
							const float sin_theta = cache[i0 + 1];

							const float x0 = toFloat32(src0_data[i3, i2, i1, i0]);
							const float x1 = toFloat32(src0_data[i3, i2, i1, i0 + 1]);

							dst_data[i3, i2, i1, i0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
							dst_data[i3, i2, i1, i0 + 1] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
						}
					}

					if (is_vision) {
						for (int64_t i0 = n_dims; i0 < dst->ne[0]; i0 += 2) {
							const int64_t ic = i0 / 2;

							const float cos_theta = cache[i0 + 0];
							const float sin_theta = cache[i0 + 1];

							const float x0 = toFloat32(src0_data[i3, i2, i1, ic + 0]);
							const float x1 = toFloat32(src0_data[i3, i2, i1, ic + n_dims]);

							dst_data[i3, i2, i1, ic + 0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
							dst_data[i3, i2, i1, ic + n_dims] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
						}
					}
					else {
						for (int64_t i0 = n_dims; i0 < dst->ne[0]; i0 += 2) {
							dst_data[i3, i2, i1, i0] = src0_data[i3, i2, i1, i0];
							dst_data[i3, i2, i1, i0 + 1] = src0_data[i3, i2, i1, i0 + 1];
						}
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
	stdexec::sync_wait(scope.on_empty());
}

static void ggml_compute_forward_rope(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_rope<ggml_fp16_t, true>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rope<ggml_fp32_t, true>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_rope_back(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_rope<ggml_fp16_t, false>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rope<ggml_fp32_t, false>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_acc_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

	// view src0 and dst with these strides and data offset inbytes during acc
	// nb0 is implicitly element_size because src0 and dst are contiguous
	size_t nb1 = ((int32_t*)dst->op_params)[0];
	size_t nb2 = ((int32_t*)dst->op_params)[1];
	size_t nb3 = ((int32_t*)dst->op_params)[2];
	size_t offset = ((int32_t*)dst->op_params)[3];
	bool   inplace = (bool)((int32_t*)dst->op_params)[4];

	if (!inplace) {
		// memcpy needs to be synchronized across threads to avoid race conditions.
		// => do it in INIT phase
		memcpy(
			((char*)dst->data),
			((char*)src0->data),
			dst->nbytes());
	}

	// src0 and dst as viewed during acc
	const size_t nb0 = ggml_element_size(src0);

	const size_t nb00 = nb0;
	const size_t nb01 = nb1;
	const size_t nb02 = nb2;
	const size_t nb03 = nb3;

	GGML_ASSERT(offset + (src1->ne[0] == 0 ? 0 : src1->ne[0] - 1) * nb0 + (src1->ne[1] == 0 ? 0 : src1->ne[1] - 1) * nb1 + (src1->ne[2] == 0 ? 0 : src1->ne[2] - 1) * nb2 + (src1->ne[3] == 0 ? 0 : src1->ne[3] - 1) * nb3 < dst->nbytes());
	GGML_ASSERT(offset + (src1->ne[0] == 0 ? 0 : src1->ne[0] - 1) * nb00 + (src1->ne[1] == 0 ? 0 : src1->ne[1] - 1) * nb01 + (src1->ne[2] == 0 ? 0 : src1->ne[2] - 1) * nb02 + (src1->ne[3] == 0 ? 0 : src1->ne[3] - 1) * nb03 < src0->nbytes());

	GGML_ASSERT(src1->nb[0] == sizeof(float));

	int offset3 = 0, offset2 = 0, offset1 = 0, offset0 = 0;
	while (offset >= dst->nb[3]) {
		offset -= dst->nb[3];
		offset3++;
	}
	while (offset >= dst->nb[2]) {
		offset -= dst->nb[2];
		offset2++;
	}
	while (offset >= dst->nb[1]) {
		offset -= dst->nb[1];
		offset1++;
	}
	while (offset >= dst->nb[0]) {
		offset -= dst->nb[0];
		offset0++;
	}

	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan(static_cast<const float*>(src1->data), src1->ne, src1->nb);
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);

	for (int64_t i3 = 0; i3 < src1_data.extent(0); i3++) {
		for (int64_t i2 = 0; i2 < src1_data.extent(1); i2++) {
			for (int64_t i1 = 0; i1 < src1_data.extent(2); i1++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					// src0 and dst are viewed with shape of src1 and offset
					// => same indices
					for (int64_t i0 = 0; i0 < src1_data.extent(3); i0++)
						dst_data[i3 + offset3, i2 + offset2, i1 + offset1, i0 + offset0] =
						src0_data[i3 + offset3, i2 + offset2, i1 + offset1, i0 + offset0] +
						src1_data[i3, i2, i1, i0];
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

static void ggml_compute_forward_acc(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_acc_f32(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
	case GGML_TYPE_Q8_1:
	case GGML_TYPE_Q2_K:
	case GGML_TYPE_Q3_K:
	case GGML_TYPE_Q4_K:
	case GGML_TYPE_Q5_K:
	case GGML_TYPE_Q6_K:
	case GGML_TYPE_TQ1_0:
	case GGML_TYPE_TQ2_0:
	case GGML_TYPE_IQ2_XXS:
	case GGML_TYPE_IQ2_XS:
	case GGML_TYPE_IQ3_XXS:
	case GGML_TYPE_IQ1_S:
	case GGML_TYPE_IQ1_M:
	case GGML_TYPE_IQ4_NL:
	case GGML_TYPE_IQ4_XS:
	case GGML_TYPE_IQ3_S:
	case GGML_TYPE_IQ2_S:
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_pad_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(src0->nb[0] == sizeof(float));
	GGML_ASSERT(dst->nb[0] == sizeof(float));

	const int32_t lp0 = ggml_get_op_params_i32(dst, 0);
	const int32_t rp0 = ggml_get_op_params_i32(dst, 1);
	const int32_t lp1 = ggml_get_op_params_i32(dst, 2);
	const int32_t rp1 = ggml_get_op_params_i32(dst, 3);
	const int32_t lp2 = ggml_get_op_params_i32(dst, 4);
	const int32_t rp2 = ggml_get_op_params_i32(dst, 5);
	const int32_t lp3 = ggml_get_op_params_i32(dst, 6);
	const int32_t rp3 = ggml_get_op_params_i32(dst, 7);

	auto src_data = make_strided_mdspan(static_cast<float*>(src0->data), src0->ne, src0->nb);
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);

	// TODO: optimize

	for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
		for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
					for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
						if ((i0 >= lp0 && i0 < dst->ne[0] - rp0)
							&& (i1 >= lp1 && i1 < dst->ne[1] - rp1)
							&& (i2 >= lp2 && i2 < dst->ne[2] - rp2)
							&& (i3 >= lp3 && i3 < dst->ne[3] - rp3)) {
							dst_data[i3, i2, i1, i0] = src_data[i3 - lp3, i2 - lp2, i1 - lp1, i0 - lp0];
						}
						else {
							dst_data[i3, i2, i1, i0] = 0;
						}
					}
				}
			});
			scope.spawn(std::move(sender));
		}
	}
}

static void ggml_compute_forward_pad(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_pad_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_timestep_embedding_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(src0->nb[0] == sizeof(float));

	const int dim = std::bit_cast<int>(dst->op_params[0]);
	const int max_period = std::bit_cast<int>(dst->op_params[1]);

	int half = dim / 2;
	std::experimental::mdspan embed_data(static_cast<float*>(dst->data), dst->ne[1], dst->ne[0]);
	for (int64_t i = 0; i < src0->ne[0]; i++) {
		if (dim % 2 != 0) {
			embed_data[i, 2 * half] = 0.f;
		}
		for (int64_t j = 0; j < half; j++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				float timestep = ((float*)src0->data)[i];
				float freq = (float)expf(-logf(max_period) * j / half);
				float arg = timestep * freq;
				embed_data[i, j] = cosf(arg);
				embed_data[i, j + half] = sinf(arg);
			});
			scope.spawn(std::move(sender));
		}
	}
}

static void ggml_compute_forward_timestep_embedding(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_timestep_embedding_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename V_TYPE, typename K_TYPE> 
static void ggml_compute_forward_flash_attn_ext(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* q = dst->src[0];
	const ggml_tensor* k = dst->src[1];
	const ggml_tensor* v = dst->src[2];
	const ggml_tensor* mask = dst->src[3];
	const ggml_tensor* sinks = dst->src[4];

	using q_to_vec_dot_t = typename vec_dot_trait<K_TYPE>::type;

	const int64_t DK = k->ne[0];
	const int64_t DV = v->ne[0];
	const int64_t N = q->ne[1];

	GGML_ASSERT(dst->ne[0] == DV);
	GGML_ASSERT(dst->ne[2] == N);

	// input tensor rows must be contiguous
	GGML_ASSERT(q->nb[0] == ggml_type_size(q->type));
	GGML_ASSERT(k->nb[0] == ggml_type_size(k->type));
	GGML_ASSERT(v->nb[0] == ggml_type_size(v->type));

	GGML_ASSERT(q->ne[0] == DK);
	GGML_ASSERT(q->ne[3] == dst->ne[3]);

	// dst cannot be transposed or permuted
	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(dst->nb[0] <= dst->nb[1]);
	GGML_ASSERT(dst->nb[1] <= dst->nb[2]);
	GGML_ASSERT(dst->nb[2] <= dst->nb[3]);

	// broadcast factors
	const int64_t rk2 = q->ne[2] / k->ne[2];
	const int64_t rk3 = q->ne[3] / k->ne[3];

	const int64_t rv2 = q->ne[2] / v->ne[2];
	const int64_t rv3 = q->ne[3] / v->ne[3];

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);
	float logit_softcap = std::bit_cast<float>(dst->op_params[2]);

	if (logit_softcap != 0) {
		scale /= logit_softcap;
	}

	const uint32_t n_head = q->ne[2];
	GGML_ASSERT(dst->ne[1] == n_head);

	const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

	const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
	const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	stdexec::scheduler auto scheduler = pool.get_scheduler();
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto v_data = make_strided_mdspan(static_cast<const V_TYPE*>(v->data), v->ne, v->nb);
	auto k_data = make_strided_mdspan(static_cast<const K_TYPE*>(k->data), k->ne, k->nb);
	auto q_data = make_strided_mdspan(static_cast<const float*>(q->data), q->ne, q->nb);

	// loop over n_batch and n_head
	for (int64_t iq3 = 0; iq3 < q->ne[3]; iq3++) {
		for (int64_t iq2 = 0; iq2 < n_head; iq2++) {
			for (int64_t iq1 = 0; iq1 < N; iq1++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					std::vector<float> VKQ(DV); // FP32 VKQ accumulator
					std::vector<float> V32(DV); // (temporary) FP32 V buffer
					std::vector<q_to_vec_dot_t> Q_q(DK); // (temporary) buffer for Q converted to quantized/FP16
					const uint32_t h = iq2; // head index
					const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

					float S = 0.0f;      // sum
					float M = -INFINITY; // maximum KQ value

					const ggml_fp16_t* mp = mask ? (ggml_fp16_t*)((char*)mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]) : NULL;

					// k indices
					const int ik3 = iq3 / rk3;
					const int ik2 = iq2 / rk2;

					// v indices
					const int iv3 = iq3 / rv3;
					const int iv2 = iq2 / rv2;

					if constexpr (is_quant_type_v<q_to_vec_dot_t>) {
						quantize_row(&q_data[iq3, iq2, iq1, 0], Q_q.data(), DK);
					}
					else {
						from_float(&q_data[iq3, iq2, iq1, 0], Q_q.data(), DK);
					}

					// online softmax / attention
					// loop over n_kv and n_head_kv
					// ref: https://arxiv.org/pdf/2112.05682.pdf
					for (int64_t ic = 0; ic < k->ne[1]; ++ic) {
						const float mv = mp ? slope * toFloat32(mp[ic]) : 0.0f;
						if (mv == -INFINITY) {
							continue;
						}

						float s; // KQ value

						ggml_vec_dot(DK, &s, 0, &k_data[ik3, ik2, ic, 0], 0, Q_q.data(), 0, 1);

						s = s * scale; // scale KQ value

						if (logit_softcap != 0.0f) {
							s = logit_softcap * tanhf(s);
						}

						s += mv; // apply mask

						const float Mold = M;

						float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
						float vs = 1.0f; // post-softmax KQ value, expf(s - M)

						if (s > M) {
							// s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
							M = s;
							ms = expf(Mold - M);

							// V = V*expf(Mold - M)
							for (int i = 0; i < DV; ++i) {
								VKQ[i] = VKQ[i] * ms;
							}
						}
						else {
							// no new maximum, ms == 1.0f, vs != 1.0f
							vs = expf(s - M);
						}

						if constexpr (is_quant_type_v<V_TYPE>) {
							dequantize_row(&v_data[iv3, iv2, ic, 0], V32.data(), DV);
						}
						else {
							to_float(&v_data[iv3, iv2, ic, 0], V32.data(), DV);
						}

						// V += v*expf(s - M)
						for (int i = 0; i < DV; ++i) {
							VKQ[i] += V32[i] * vs;
						}

						S = S * ms + vs; // scale and increment sum with partial sum
					}

					// sinks
					if (sinks) {
						const float s = ((float*)((char*)sinks->data))[h];

						float ms = 1.0f;
						float vs = 1.0f;

						if (s > M) {
							ms = expf(M - s);
							for (auto& v : VKQ) v *= ms;
						}
						else {
							vs = expf(s - M);
						}

						S = S * ms + vs;
					}

					// V /= S
					const float S_inv = S == 0.0f ? 0.0f : 1.0f / S;
					for (int i = 0; i < DV; ++i) {
						VKQ[i] *= S_inv;
					}

					// dst indices
					const int i1 = iq1;
					const int i2 = iq2;
					const int i3 = iq3;

					// permute(3, 1, 2, 0)
					for (int64_t i0 = 0; i0 < DV; i0++) {
						dst_data[i3, i1, i2, i0] = VKQ[i0];
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

template <typename V_TYPE>
static void ggml_compute_forward_flash_attn_ext(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* k = dst->src[1];
	switch (k->type) {
	case GGML_TYPE_F32: {
		ggml_compute_forward_flash_attn_ext<V_TYPE, ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16: {
		ggml_compute_forward_flash_attn_ext<V_TYPE, ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16: {
		ggml_compute_forward_flash_attn_ext<V_TYPE, ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_flash_attn_ext<V_TYPE, block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_flash_attn_ext<V_TYPE, block_q8_0>(pool, scope, dst);
	} break;
	default:
		assert(false);
	}
}

static void ggml_compute_forward_flash_attn_ext_inner(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* v = dst->src[2];
	switch (v->type) {
	case GGML_TYPE_F32: {
		ggml_compute_forward_flash_attn_ext<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16: {
		ggml_compute_forward_flash_attn_ext<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16: {
		ggml_compute_forward_flash_attn_ext<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_flash_attn_ext<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_flash_attn_ext<block_q8_0>(pool, scope, dst);
	} break;
	default:
		GGML_ASSERT(false && "fattn: unsupported V-type");
	}
}

static void ggml_compute_forward_flash_attn_ext(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->op_params[3]) {
	case GGML_PREC_DEFAULT:
	case GGML_PREC_F32:
	{
		// uses F32 accumulators
		ggml_compute_forward_flash_attn_ext_inner(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_cross_entropy_loss_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
	GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
	GGML_ASSERT(ggml_are_same_shape(src0, src1));
	GGML_ASSERT(ggml_is_scalar(dst));
	GGML_ASSERT(dst->type == GGML_TYPE_F32);

	// TODO: handle transposed/permuted matrices
	const int64_t nc = src0->ne[0];
	const int64_t nr = ggml_nrows(src0);

	std::atomic<float> sums;
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	std::experimental::mdspan src0_data(static_cast<float*>(src0->data), src0->ne[1], src0->ne[0]);
	std::experimental::mdspan src1_data(static_cast<float*>(src1->data), src1->ne[1], src1->ne[0]);

	for (int64_t i1 = 0; i1 < nr; i1++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &sums] {
			std::vector<float> st(nc);
			float sum_thread = 0.0f;

#ifndef NDEBUG
			for (int64_t i0 = 0; i0 < nc; i0++) {
				assert(!isnan(src0_data[i1, i0]));
				assert(!isnan(src1_data[i1, i0]));
			}
#endif

			float max = -INFINITY;
			for (int64_t i0 = 0; i0 < nc; i0++) {
				max = std::max(max, src0_data[i1, i0]);
			}

			// log soft max
			float sum_softmax = 0.0;
			for (int64_t i0 = 0; i0 < nc; i0++) {
				float val = src0_data[i1, i0] - max;
				st[i0] = val;
				sum_softmax += expf(val);
			}
			sum_softmax = logf(sum_softmax);
			assert(sum_softmax >= 0.0);

			for (int64_t i0 = 0; i0 < nc; i0++) {
				st[i0] = (st[i0] - sum_softmax) * src1_data[i1, i0];
#ifndef NDEBUG
				assert(!isnan(st[i0]));
				assert(!isinf(st[i0]));
#endif
				sum_thread += st[i0];
			}
			sums += sum_thread;
		});
		scope.spawn(std::move(sender));
	}

	stdexec::sync_wait(scope.on_empty());
	float* dp = (float*)dst->data;
	dp[0] = -1.0f * sums.load() / (float)nr;
}

static void ggml_compute_forward_cross_entropy_loss(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_cross_entropy_loss_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_cross_entropy_loss_back_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* grad = dst->src[0]; // gradient of forward pass output
	const ggml_tensor* src0f = dst->src[1]; // src0 of forward pass
	const ggml_tensor* src1f = dst->src[2]; // src1 of forward pass

	GGML_ASSERT(ggml_is_contiguous(dst));
	GGML_ASSERT(ggml_is_contiguous(src0f));
	GGML_ASSERT(ggml_is_contiguous(src1f));
	GGML_ASSERT(ggml_is_contiguous(grad));
	GGML_ASSERT(ggml_are_same_shape(src0f, src1f) && ggml_are_same_shape(src0f, dst));

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// TODO: handle transposed/permuted matrices
	const int64_t nc = src0f->ne[0];
	const int64_t nr = ggml_nrows(src0f);

	std::experimental::mdspan ds0(static_cast<float*>(dst->data), nr, nc);
	std::experimental::mdspan s0(static_cast<const float*>(src0f->data), nr, nc);
	std::experimental::mdspan s1(static_cast<const float*>(src1f->data), nr, nc);
	const float d_by_nr = ((const float*)grad->data)[0] / (float)nr;

	// row range for this thread
	for (int64_t i1 = 0; i1 < nr; i1++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
#ifndef NDEBUG
			for (int64_t i0 = 0; i0 < nc; i0++) {
				assert(!isnan(s0[i1, i0]));
				assert(!isnan(s1[i1, i0]));
			}
#endif

			// soft_max
			float max = -INFINITY;
			for (int64_t i0 = 0; i0 < nc; i0++) {
				if (s0[i1, i0] > max) {
					max = s0[i1, i0];
				}
			}
			float sum = 0.0f;
			for (int64_t i0 = 0; i0 < nc; i0++) {
				float val = expf(s0[i1, i0] - max);
				sum += val;
				ds0[i1, i0] = val;
			}
			assert(sum > 0.0);

			// grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
			for (int64_t i0 = 0; i0 < nc; i0++) {
				ds0[i1, i0] = ((ds0[i1, i0] / sum) - s1[i1, i0]) * d_by_nr;
#ifndef NDEBUG
				assert(!isnan(ds0[i1, i0]));
				assert(!isinf(ds0[i1, i0]));
#endif
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_cross_entropy_loss_back(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_cross_entropy_loss_back_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_opt_step_adamw_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src0_grad = dst->src[1];
	const ggml_tensor* src0_grad_m = dst->src[2];
	const ggml_tensor* src0_grad_v = dst->src[3];
	const ggml_tensor* adamw_params = dst->src[4];

	GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
	GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
	GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
	GGML_ASSERT(adamw_params->nelements() == 7);

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	GGML_ASSERT(src0->nb[0] == sizeof(float));

	// row range for this thread
	const float* adamw_params_ptr = ggml_get_data_f32(adamw_params);
	const float alpha = adamw_params_ptr[0];
	const float beta1 = adamw_params_ptr[1];
	const float beta2 = adamw_params_ptr[2];
	const float eps = adamw_params_ptr[3];
	const float wd = adamw_params_ptr[4];
	const float beta1h = adamw_params_ptr[5];
	const float beta2h = adamw_params_ptr[6];

	const float keep = 1.f - alpha * wd;

	auto w = make_strided_mdspan(static_cast<float*>(src0->data), src0->ne, src0->nb); // weight
	auto g = make_strided_mdspan(static_cast<const float*>(src0_grad->data), src0->ne, src0->nb); // grad
	auto m = make_strided_mdspan(static_cast<float*>(src0_grad_m->data), src0->ne, src0->nb);
	auto v = make_strided_mdspan(static_cast<float*>(src0_grad_v->data), src0->ne, src0->nb);

	for (int64_t i03 = 0; i03 < w.extent(0); i03++) {
		for (int64_t i02 = 0; i02 < w.extent(1); i02++) {
			for (int64_t i01 = 0; i01 < w.extent(2); i01++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					for (int i00 = 0; i00 < w.extent(3); ++i00) {
						m[i03, i02, i01, i00] = m[i03, i02, i01, i00] * beta1 + g[i03, i02, i01, i00] * (1.0f - beta1);
						v[i03, i02, i01, i00] = v[i03, i02, i01, i00] * beta2 + g[i03, i02, i01, i00] * g[i03, i02, i01, i00] * (1.0f - beta2);
						const float mh = m[i03, i02, i01, i00] * beta1h;
						const float vh = sqrtf(v[i03, i02, i01, i00] * beta2h) + eps;
						// The weight decay is applied independently of the Adam momenta m and v.
						// This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
						// See: https://arxiv.org/pdf/1711.05101v3.pdf
						w[i03, i02, i01, i00] = w[i03, i02, i01, i00] * keep - alpha * mh / vh;
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

static void ggml_compute_forward_opt_step_adamw(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_opt_step_adamw_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_l2_norm_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));

	GGML_ASSERT(src0->nb[0] == sizeof(float));

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	float eps = std::bit_cast<float>(dst->op_params[0]);
	GGML_ASSERT(eps >= 0.0f);

	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

	// TODO: optimize
	for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
				for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
					double sum = 0.0;
					for (int64_t i00 = 0; i00 < src0->ne[0]; i00++)
						sum += (double)(src0_data[i03, i02, i01, i00] * src0_data[i03, i02, i01, i00]);
					const float scale = fmaxf(sqrtf(sum), eps);
					for (int64_t i00 = 0; i00 < src0->ne[0]; i00++)
						dst_data[i03, i02, i01, i00] = src0_data[i03, i02, i01, i00] / scale;
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_l2_norm(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_l2_norm_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_rwkv_wkv7_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const int64_t T = dst->src[1]->ne[2];
	const int64_t C = dst->ne[0];
	const int64_t HEADS = dst->src[1]->ne[1];
	const int64_t n_seqs = dst->src[6]->ne[1];
	const int64_t head_size = C / HEADS;

	float* state = ((float*)dst->data) + C * T;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), T, HEADS, head_size);
	std::experimental::mdspan r(static_cast<const float*>(dst->src[0]->data), T, HEADS, head_size);
	std::experimental::mdspan w(static_cast<const float*>(dst->src[1]->data), T, HEADS, head_size);
	std::experimental::mdspan k(static_cast<const float*>(dst->src[2]->data), T, HEADS, head_size);
	std::experimental::mdspan v(static_cast<const float*>(dst->src[3]->data), T, HEADS, head_size);
	std::experimental::mdspan a(static_cast<const float*>(dst->src[4]->data), T, HEADS, head_size);
	std::experimental::mdspan b(static_cast<const float*>(dst->src[5]->data), T, HEADS, head_size);

	GGML_ASSERT(HEADS * head_size == C);
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS

	for (int64_t h = 0; h < HEADS; h++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t t = 0; t < T; t++) {
				int64_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				std::experimental::mdspan state_cur1(static_cast<float*>(state_cur), HEADS, head_size, head_size);
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;
				std::experimental::mdspan state_prev1(static_cast<float*>(state_prev), HEADS, head_size, head_size);

				for (int64_t i = 0; i < head_size; i++) {
					float v_val = v[t, h, i];

					float sa = 0, result = 0;
					for (int64_t j = 0; j < head_size; j++) {
						sa += a[t, h, j] * state_prev1[h, i, j];
					}

					for (int64_t j = 0; j < head_size; j++) {
						float r_val = r[t, h, j];
						float w_val = w[t, h, j];
						float k_val = k[t, h, j];
						float b_val = b[t, h, j];
						float kv_val = v_val * k_val;
						float prev_state_val = state_prev1[h, i, j];
						state_cur1[h, i, j] = prev_state_val * w_val + kv_val + sa * b_val;
						result += state_cur1[h, i, j] * r_val;
					}
					dst_data[t, h, i] = result;
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_rwkv_wkv7(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rwkv_wkv7_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

	GGML_ASSERT(dst->nb[0] == sizeof(dst_t));
	GGML_ASSERT(src0->nb[0] == sizeof(src0_t));

	const bool is_src1_contiguous = (src1->nb[0] == sizeof(src1_t));

	if (!is_src1_contiguous) { // broadcast not implemented yet for non-contiguous
		GGML_ASSERT(ggml_are_same_shape(src0, src1));
	}

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	auto dst_data = make_strided_mdspan(static_cast<dst_t*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const src0_t*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan(static_cast<const src1_t*>(src1->data), src1->ne, src1->nb);

	if (is_src1_contiguous) {
		for (int64_t i03 = 0; i03 < src0_data.extent(0); i03++) {
			for (int64_t i02 = 0; i02 < src0_data.extent(1); i02++) {
				for (int64_t i01 = 0; i01 < src0_data.extent(2); i01++) {
					stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
						const int64_t i13 = i03 % src1_data.extent(0);
						const int64_t i12 = i02 % src1_data.extent(1);
						const int64_t i11 = i01 % src1_data.extent(2);

						// src1 is broadcastable across src0 and dst in i1, i2, i3
						const int64_t nr0 = src0_data.extent(3) / src1_data.extent(3);

						for (int64_t r = 0; r < nr0; ++r) {
							for (int64_t i = 0; i < src1_data.extent(3); i++) {
								dst_data[i03, i02, i01, r * src1_data.extent(3) + i] =
									fromFloat32<dst_t>(
										op(
											toFloat32(src0_data[i03, i02, i01, r * src1_data.extent(3) + i]),
											toFloat32(src1_data[i13, i12, i11, i])));
							}
						}
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
	else {
		for (int64_t i03 = 0; i03 < src0_data.extent(0); i03++) {
			for (int64_t i02 = 0; i02 < src0_data.extent(1); i02++) {
				for (int64_t i01 = 0; i01 < src0_data.extent(2); i01++) {
					stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
						const int64_t i13 = i03 % src1_data.extent(0);
						const int64_t i12 = i02 % src1_data.extent(1);
						const int64_t i11 = i01 % src1_data.extent(2);

						for (int64_t i = 0; i < dst_data.extent(3); i++) {
							int64_t i10 = i % src1_data.extent(3);
							dst_data[i03, i02, i01, i] = fromFloat32<dst_t>(
								op(
									toFloat32(src0_data[i03, i02, i01, i]),
									toFloat32(src0_data[i13, i12, i11, i10 + i])));
						}
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
}

void ggml_compute_forward_sub(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	binary_op<op_sub>(pool, scope, dst);
}

struct ggml_conv_2d_dw_params {
	int64_t channels;
	int64_t batches;
	int64_t in_w;
	int64_t in_h;
	int64_t out_w;
	int64_t out_h;
	int64_t kernel_w;
	int64_t kernel_h;
	int stride_w;
	int stride_h;
	int padding_w;
	int padding_h;
	int dilation_w;
	int dilation_h;
};

static int64_t calculate_input_coord (int64_t out_coord, int64_t kern_coord, int64_t stride, int64_t dilation, int64_t padding)  {
	return out_coord * stride + kern_coord * dilation - padding;
}

static void ggml_compute_forward_conv_2d_dw_nchw(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* src,
	const ggml_tensor* kernel,
	ggml_tensor* dst,
	const ggml_conv_2d_dw_params& ctx) {
	// [N, C, H, W] layout
	std::experimental::mdspan input_data(static_cast<const float*>(src->data), ctx.batches, ctx.channels, ctx.in_h, ctx.in_w);
	std::experimental::mdspan kernel_data(static_cast<const float*>(kernel->data), ctx.channels, ctx.kernel_h, ctx.kernel_w);
	std::experimental::mdspan output_data(static_cast<float*>(dst->data), ctx.batches, ctx.channels, ctx.out_h, ctx.out_w);

	for (int64_t batch = 0; batch < ctx.batches; batch++) {
		for (int64_t channel = 0; channel < ctx.channels; channel++) {
			for (int64_t oh = 0; oh < ctx.out_h; oh++) {
				for (int64_t ow = 0; ow < ctx.out_w; ow++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						const int64_t kh_min = std::max(int64_t{ 0 }, (ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
						const int64_t kh_max = std::min(ctx.kernel_h, (ctx.in_h + ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
						const int64_t kw_min = std::max(int64_t{ 0 }, (ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);
						const int64_t kw_max = std::min(ctx.kernel_w, (ctx.in_w + ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);

						float accumulator = 0;
						for (int64_t kh = kh_min; kh < kh_max; kh++) {
							int64_t ih = calculate_input_coord(oh, kh, ctx.stride_h, ctx.dilation_h, ctx.padding_h);

							for (int64_t kw = kw_min; kw < kw_max; kw++) {
								int64_t iw = calculate_input_coord(ow, kw, ctx.stride_w, ctx.dilation_w, ctx.padding_w);

								const float input_val = input_data[batch, channel, ih, iw];
								const float kernel_val = kernel_data[channel, kh, kw];

								accumulator += input_val * kernel_val;
							}
						}

						output_data[batch, channel, oh, ow] = accumulator;
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
}

static void ggml_compute_forward_conv_2d_dw_nhwc(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* src,
	const ggml_tensor* kernel,
	ggml_tensor* dst,
	const ggml_conv_2d_dw_params& ctx) {

	// [N, H, W, C] layout
	std::experimental::mdspan input_data(static_cast<const float*>(src->data), ctx.batches, ctx.in_h, ctx.in_w, ctx.channels);
	std::experimental::mdspan kernel_data(static_cast<const float*>(kernel->data), ctx.kernel_h, ctx.kernel_w, ctx.channels);
	std::experimental::mdspan output_data(static_cast<float*>(dst->data), ctx.batches, ctx.out_h, ctx.out_w, ctx.channels);

	for (int64_t batch = 0; batch < ctx.batches; batch++) {
		for (int64_t oh = 0; oh < ctx.out_h; oh++) {
			for (int64_t ow = 0; ow < ctx.out_w; ow++) {
				for (int64_t channel = 0; channel < ctx.channels; channel++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						const int64_t kh_min = std::max(int64_t{ 0 }, (ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
						const int64_t kh_max = std::min(ctx.kernel_h, (ctx.in_h + ctx.padding_h - oh * ctx.stride_h + ctx.dilation_h - 1) / ctx.dilation_h);
						const int64_t kw_min = std::max(int64_t{ 0 }, (ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);
						const int64_t kw_max = std::min(ctx.kernel_w, (ctx.in_w + ctx.padding_w - ow * ctx.stride_w + ctx.dilation_w - 1) / ctx.dilation_w);

						float accumulator = 0;
						for (int64_t kh = kh_min; kh < kh_max; kh++) {
							int64_t ih = calculate_input_coord(oh, kh, ctx.stride_h, ctx.dilation_h, ctx.padding_h);

							for (int64_t kw = kw_min; kw < kw_max; kw++) {
								int64_t iw = calculate_input_coord(ow, kw, ctx.stride_w, ctx.dilation_w, ctx.padding_w);

								const float input_val = input_data[batch, ih, iw, channel];
								const float kernel_val = kernel_data[kh, kw, channel];

								accumulator += input_val * kernel_val;
							}
						}

						output_data[batch, oh, ow, channel] = accumulator;
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
}

// TODO
// NEED SIMD
void ggml_compute_forward_conv_2d_dw(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* kernel = dst->src[0];
	const ggml_tensor* src = dst->src[1];
	ggml_conv_2d_dw_params ctx {
		.channels = src->ne[2],
		.batches = src->ne[3],
		.in_w = src->ne[0],
		.in_h = src->ne[1],
		.out_w = dst->ne[0],
		.out_h = dst->ne[1],
		.kernel_w = kernel->ne[0],
		.kernel_h = kernel->ne[1],
		.stride_w = dst->op_params[0],
		.stride_h = dst->op_params[1],
		.padding_w = dst->op_params[2],
		.padding_h = dst->op_params[3],
		.dilation_w = dst->op_params[4],
		.dilation_h = dst->op_params[5]
	};

	GGML_ASSERT(kernel->ne[3] == ctx.channels);
	GGML_ASSERT(dst->ne[3] == ctx.batches);

	if (ggml_is_contiguous(src)) {
		ggml_compute_forward_conv_2d_dw_nchw(pool, scope, src, kernel, dst, ctx);
	}
	else if (ggml_is_contiguous_channels(src)) {
		// kernel should also have channels most contiguous in memory
		GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
		ggml_compute_forward_conv_2d_dw_nhwc(pool, scope, src, kernel, dst, ctx);
	}
	else {
		GGML_ABORT("non-contiguous memory layout not supported");
	}
}

// ggml_compute_forward_roll

static int64_t ggml_wrap_index(int64_t i, int64_t ne) {
	if (i < 0) {
		return i + ne;
	}
	else if (i >= ne) {
		return i - ne;
	}
	return i;
}

static void ggml_compute_forward_roll_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);

	const int s0 = ggml_get_op_params_i32(dst, 0);
	const int s1 = ggml_get_op_params_i32(dst, 1);
	const int s2 = ggml_get_op_params_i32(dst, 2);
	const int s3 = ggml_get_op_params_i32(dst, 3);

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t i3 = 0; i3 < dst_data.extent(0); i3++) {
		for (int64_t i2 = 0; i2 < dst_data.extent(1); i2++) {
			for (int64_t i1 = 0; i1 < dst_data.extent(2); i1++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					const int64_t i01 = ggml_wrap_index(i1 - s1, src0_data.extent(2));
					const int64_t i02 = ggml_wrap_index(i2 - s2, src0_data.extent(1));
					const int64_t i03 = ggml_wrap_index(i3 - s3, src0_data.extent(0));

					const int64_t s = ggml_wrap_index(-s0, src0_data.extent(3));
					const int64_t n = src0_data.extent(3) - s;
					for (int64_t i = 0; i < n; ++i) dst_data[i3, i2, i1, i] = src0_data[i03, i02, i01, s + i];
					for (int64_t i = 0; i < s; ++i) dst_data[i3, i2, i1, n + i] = src0_data[i03, i02, i01, i];
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

void ggml_compute_forward_roll(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_roll_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

inline static float ggml_relu_f32(float x) {
	return (x > 0.f) ? x : 0.f;
}

inline static float ggml_gelu_f32(float x) {
	static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
	static const float GELU_COEF_A = 0.044715f;
	return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

inline static float ggml_silu_f32(float x) {
	return x / (1.0f + expf(-x));
}

inline static float ggml_geglu_erf_f32(float x) {
	static const float SQRT_2_INV = 0.70710678118654752440084436210484f;
	return 0.5f * x * (1.0f + erff(x * SQRT_2_INV));
}

inline static float ggml_gelu_quick_f32(float x) {
	static const float GELU_QUICK_COEF = -1.702f;
	return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

template <auto op, typename T>
inline static void ggml_vec(const int n, T* y, const T* x, const T* g) {
	for (int i = 0; i < n; ++i) {
		y[i] = fromFloat32<T>(op(toFloat32(x[i])) * toFloat32(g[i]));
	}
}

template <auto op, typename T>
static void ggml_compute_forward_glu(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	char* src0_d = (char*)src0->data;
	char* src1_d = (char*)(src1 ? src1->data : src0->data);
	const size_t src0_o = src0->nb[1];
	const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

	GGML_ASSERT(ggml_is_contiguous_1(src0));
	GGML_ASSERT(ggml_is_contiguous_1(dst));

	if (src1) {
		GGML_ASSERT(ggml_is_contiguous_1(src1));
		GGML_ASSERT(src0->type == src1->type);
	}

	const int nth = pool.available_parallelism();

	const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
	const int64_t nr = ggml_nrows(src0);

	GGML_ASSERT(dst->ne[0] == nc);
	GGML_ASSERT(ggml_nrows(dst) == nr);

	const int32_t swapped = ggml_get_op_params_i32(dst, 1);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	for (int ir0 = 0; ir0 < nr; ir0 += dr) {
		const int ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
		stdexec::then([=] {
			for (int i1 = ir0; i1 < ir1; i1++) {
				const T* src0_p = (T*)(src0_d + i1 * src0_o);
				const T* src1_p = (T*)(src1_d + i1 * src1_o);

				if (!src1) {
					src0_p += swapped ? nc : 0;
					src1_p += swapped ? 0 : nc;
				}

				ggml_vec<op>(nc, (T*)((char*)dst->data + i1 * (dst->nb[1])), src0_p, src1_p);

#ifndef NDEBUG
				for (int k = 0; k < nc; k++) {
					const T x = ((T*)((char*)dst->data + i1 * (dst->nb[1])))[k];
					const float v = toFloat32(x);
					assert(!isnan(v));
					assert(!isinf(v));
				}
#endif
			}
		});
		scope.spawn(std::move(sender));
	}
}

template <auto op>
static void ggml_compute_forward_glu(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_glu<op, ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_glu<op, ggml_fp16_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_swiglu_oai_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	char* src0_d = (char*)src0->data;
	char* src1_d = (char*)(src1 ? src1->data : src0->data);
	const size_t src0_o = src0->nb[1];
	const size_t src1_o = src1 ? src1->nb[1] : src0->nb[1];

	GGML_ASSERT(ggml_is_contiguous_1(src0));
	GGML_ASSERT(ggml_is_contiguous_1(dst));

	if (src1) {
		GGML_ASSERT(ggml_is_contiguous_1(src1));
		GGML_ASSERT(src0->type == src1->type);
	}

	const int ith = 0;
	const int nth = 1;

	const int nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
	const int nr = ggml_nrows(src0);

	GGML_ASSERT(dst->ne[0] == nc);
	GGML_ASSERT(ggml_nrows(dst) == nr);

	const int32_t swapped = std::bit_cast<int32_t>(dst->op_params[1]);
	const float alpha = std::bit_cast<float>(dst->op_params[2]);
	const float limit = std::bit_cast<float>(dst->op_params[3]);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int i1 = ir0; i1 < ir1; i1++) {
		float* src0_p = (float*)(src0_d + i1 * src0_o);
		float* src1_p = (float*)(src1_d + i1 * src1_o);
		float* dst_p = (float*)((char*)dst->data + i1 * (dst->nb[1]));

		if (!src1) {
			src0_p += swapped ? nc : 0;
			src1_p += swapped ? 0 : nc;
		}

		for (int k = 0; k < nc; k++) {
			const float x = std::min(src0_p[k], limit);
			const float y = std::clamp(src1_p[k], -limit, limit);
			const float out_glu = x / (1.f + expf(alpha * (-x)));
			dst_p[k] = out_glu * (y + 1.f);
		}

#ifndef NDEBUG
		for (int k = 0; k < nc; k++) {
			const float x = dst_p[k];
			assert(!isnan(x));
			assert(!isinf(x));
		}
#endif
	}
}

static void ggml_compute_forward_swiglu_oai(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_swiglu_oai_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

void ggml_compute_forward_glu(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_glu_op op = ggml_get_glu_op(dst);

	switch (op) {
	case GGML_GLU_OP_REGLU:
	{
		ggml_compute_forward_glu<ggml_relu_f32>(pool, scope, dst);
	} break;
	case GGML_GLU_OP_GEGLU:
	{
		ggml_compute_forward_glu<ggml_gelu_f32>(pool, scope, dst);
	} break;
	case GGML_GLU_OP_SWIGLU:
	{
		ggml_compute_forward_glu<ggml_silu_f32>(pool, scope, dst);
	} break;
	case GGML_GLU_OP_SWIGLU_OAI:
	{
		ggml_compute_forward_swiglu_oai(pool, scope, dst);
	} break;
	case GGML_GLU_OP_GEGLU_ERF:
	{
		ggml_compute_forward_glu<ggml_geglu_erf_f32>(pool, scope, dst);
	} break;
	case GGML_GLU_OP_GEGLU_QUICK:
	{
		ggml_compute_forward_glu<ggml_gelu_quick_f32>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename idx_t, typename dst_t>
static void ggml_compute_forward_set_rows_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	const int64_t nc = src0->ne[0];
	const int64_t nr = src0->ne[1];

	assert(dst->ne[0] == nc);
	assert(dst->ne[2] == src0->ne[2]);
	assert(dst->ne[3] == src0->ne[3]);
	assert(src0->type == GGML_TYPE_F32);
	assert(src0->ne[2] % src1->ne[1] == 0);
	assert(src0->ne[3] % src1->ne[2] == 0);

	auto dst_data = make_strided_mdspan(static_cast<dst_t*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan<3>(static_cast<const idx_t*>(src1->data), src1->ne, src1->nb);

	for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			for (int64_t i03 = 0; i03 < src0->ne[3]; ++i03) {
				for (int64_t i02 = 0; i02 < src0->ne[2]; ++i02) {
					const int64_t i12 = i03 % src1->ne[2];
					const int64_t i11 = i02 % src1->ne[1];
					const int64_t i10 = i01;
					const int64_t i1 = src1_data[i12, i11, i10];

					GGML_ASSERT(i1 >= 0 && i1 < dst->ne[1]);
					fromFloat(
						&src0_data[i03, i02, i01, 0],
						&dst_data[i03, i02, i1, 0],
						nc);
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

template <typename dst_t>
void ggml_compute_forward_set_rows_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		if (src1->type == GGML_TYPE_I64) {
			ggml_compute_forward_set_rows_f32<int64_t, dst_t>(pool, scope, dst);
		}
		else if (src1->type == GGML_TYPE_I32) {
			ggml_compute_forward_set_rows_f32<int32_t, dst_t>(pool, scope, dst);
		}
		else {
			GGML_ABORT("src1->type = %d (%s) not supported", src1->type, ggml_type_name(src1->type));
		}
	} break;
	default:
	{
		GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
	}
	}
}

void ggml_compute_forward_set_rows(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_set_rows_f32<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_set_rows_f32<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_set_rows_f32<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0:
	{
		ggml_compute_forward_set_rows_f32<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1:
	{
		ggml_compute_forward_set_rows_f32<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0:
	{
		ggml_compute_forward_set_rows_f32<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1:
	{
		ggml_compute_forward_set_rows_f32<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0:
	{
		ggml_compute_forward_set_rows_f32<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_MXFP4:
	{
		ggml_compute_forward_set_rows_f32<block_mxfp4>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K:
	{
		ggml_compute_forward_set_rows_f32<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K:
	{
		ggml_compute_forward_set_rows_f32<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K:
	{
		ggml_compute_forward_set_rows_f32<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K:
	{
		ggml_compute_forward_set_rows_f32<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K:
	{
		ggml_compute_forward_set_rows_f32<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS:
	{
		ggml_compute_forward_set_rows_f32<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL:
	{
		ggml_compute_forward_set_rows_f32<block_iq4_nl>(pool, scope, dst);
	} break;
	default:
	{
		assert(false);
	}
	}
}

static void ggml_call_mul_mat(ggml_type type,
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	int64_t m, int64_t n, int64_t k,
	void* a, void* b, float* c) {
	const ggml_type_traits* traits = ggml_get_type_traits(type);
	struct ggml_tensor src1 = {};
	src1.type = type;
	src1.ne[0] = k;
	src1.ne[1] = m;
	src1.ne[2] = 1;
	src1.ne[3] = 1;
	src1.nb[0] = traits->type_size;
	src1.nb[1] = k * traits->type_size;
	src1.nb[2] = src1.nb[1];
	src1.nb[3] = src1.nb[2];
	src1.data = a;

	struct ggml_tensor src0 = {};
	src0.type = type;
	src0.ne[0] = k;
	src0.ne[1] = n;
	src0.ne[2] = 1;
	src0.ne[3] = 1;
	src0.nb[0] = traits->type_size;
	src0.nb[1] = k * traits->type_size;
	src0.nb[2] = src0.nb[1];
	src0.nb[3] = src0.nb[2];
	src0.data = b;

	struct ggml_tensor dst = {};
	dst.ne[0] = n;
	dst.ne[1] = m;
	dst.ne[2] = 1;
	dst.ne[3] = 1;
	dst.nb[0] = sizeof(float);
	dst.nb[1] = n * sizeof(float);
	dst.nb[2] = dst.nb[1];
	dst.nb[3] = dst.nb[2];
	dst.data = c;
	dst.src.push_back(&src0);
	dst.src.push_back(&src1);

	ggml_compute_forward_mul_mat(pool, scope, &dst);
}

// ggml_compute_forward_conv_2d
template <typename kernel_t>
static void ggml_compute_forward_conv_2d_impl(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst     // [N, COut, OH, OW]
) {
	const ggml_tensor* kernel = dst->src[0];  // [COut, CIn, KH, KW]
	const ggml_tensor* input = dst->src[1];     // [N, CIn, IH, IW]
	GGML_ASSERT(ggml_is_contiguous(kernel));
	GGML_ASSERT(input->ne[2] == kernel->ne[2]);
	GGML_ASSERT(kernel->ne[3] == dst->ne[2]);

	const int32_t stride_w = dst->op_params[0];
	const int32_t stride_h = dst->op_params[1];
	const int32_t pad_w = dst->op_params[2];
	const int32_t pad_h = dst->op_params[3];
	const int32_t dilation_w = dst->op_params[4];
	const int32_t dilation_h = dst->op_params[5];

	const int64_t N = input->ne[3];
	const int64_t COut = kernel->ne[3];
	const int64_t KH = kernel->ne[1];
	const int64_t KW = kernel->ne[0];
	const int64_t OH = dst->ne[1];
	const int64_t OW = dst->ne[0];
	const int64_t CIn = input->ne[2];
	const int64_t IH = input->ne[1];
	const int64_t IW = input->ne[0];
	std::experimental::mdspan input_data(static_cast<const float*>(input->data), N, CIn, IH, IW);
	std::experimental::mdspan kernel_data(static_cast<const kernel_t*>(kernel->data), COut, CIn, KH, KW);
	std::experimental::mdspan output_data(static_cast<float*>(dst->data), N, COut, OH, OW);

	for (int64_t n = 0; n < N; n++) {
		for (int64_t cout = 0; cout < COut; cout++) {
			for (int64_t oh = 0; oh < OH; oh++) {
				for (int64_t ow = 0; ow < OW; ow++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						const int64_t kh_min = std::max(int64_t{ 0 }, (pad_h - oh * stride_h + dilation_h - 1) / dilation_h);
						const int64_t kh_max = std::min(KH, (IH + pad_h - oh * stride_h + dilation_h - 1) / dilation_h);
						const int64_t kw_min = std::max(int64_t{ 0 }, (pad_w - ow * stride_w + dilation_w - 1) / dilation_w);
						const int64_t kw_max = std::min(KW, (IW + pad_w - ow * stride_w + dilation_w - 1) / dilation_w);
						float accumulator = 0.0f;

						for (int64_t cin = 0; cin < CIn; cin++) {
							for (int64_t kh = kh_min; kh < kh_max; kh++) {
								const int64_t ih = calculate_input_coord(oh, kh, stride_h, dilation_h, pad_h);

								for (int64_t kw = kw_min; kw < kw_max; kw++) {
									const int64_t iw = calculate_input_coord(ow, kw, stride_w, dilation_w, pad_w);

									accumulator += input_data[n, cin, ih, iw] *
										toFloat32(kernel_data[cout, cin, kh, kw]);
								}
							}
						}

						output_data[n, cout, oh, ow] = accumulator;
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}
}

void ggml_compute_forward_conv_2d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* kernel = dst->src[0];

	switch (kernel->type) {
	case GGML_TYPE_F32:
		ggml_compute_forward_conv_2d_impl<ggml_fp32_t>(pool, scope, dst);
		break;
	case GGML_TYPE_F16:
		ggml_compute_forward_conv_2d_impl<ggml_fp16_t>(pool, scope, dst);
		break;
	default:
		break;
	}
}

template <typename kernel_t>
static void ggml_compute_forward_conv_3d_impl(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* kernel,
	const ggml_tensor* src,
	ggml_tensor* dst) {
	GGML_ASSERT(ggml_is_contiguous(kernel));

	const int32_t s0 = dst->op_params[0];
	const int32_t s1 = dst->op_params[1];
	const int32_t s2 = dst->op_params[2];
	const int32_t p0 = dst->op_params[3];
	const int32_t p1 = dst->op_params[4];
	const int32_t p2 = dst->op_params[5];
	const int32_t d0 = dst->op_params[6];
	const int32_t d1 = dst->op_params[7];
	const int32_t d2 = dst->op_params[8];
	const int32_t c = dst->op_params[9];
	const int32_t n = dst->op_params[10];
	const int32_t oc = dst->op_params[11];

	const int64_t src_w = src->ne[0];
	const int64_t src_h = src->ne[1];
	const int64_t src_d = src->ne[2];
	const int64_t knl_w = kernel->ne[0];
	const int64_t knl_h = kernel->ne[1];
	const int64_t knl_d = kernel->ne[2];
	const int64_t dst_w = dst->ne[0];
	const int64_t dst_h = dst->ne[1];
	const int64_t dst_d = dst->ne[2];

	void* knl_data = kernel->data;

	const int64_t knl_n_per_channel = knl_w * knl_h * knl_d;
	const int64_t knl_n_total = knl_n_per_channel * c;
	const int64_t patch_total = n * dst_w * dst_h * dst_d;

	std::experimental::mdspan src_data(static_cast<const float*>(src->data), n, c, src_d, src_h, src_w);
	std::vector<kernel_t> tmp_data(knl_w * knl_h * knl_d * c * patch_total);

	std::array<int64_t, 8> element_ne = { knl_w, knl_h, knl_d, c, dst_w, dst_h, dst_d, n }; // reverse order
	std::array<size_t, 8> element_nb = {
		sizeof(kernel_t),
		sizeof(kernel_t) * knl_w,
		sizeof(kernel_t) * knl_w * knl_h,
		sizeof(kernel_t) * knl_w * knl_h * knl_d,
		sizeof(kernel_t) * knl_w * knl_h * knl_d * c,
		sizeof(kernel_t) * knl_w * knl_h * knl_d * c * dst_w,
		sizeof(kernel_t) * knl_w * knl_h * knl_d * c * dst_w * dst_h,
		sizeof(kernel_t) * knl_w * knl_h * knl_d * c * dst_w * dst_h * dst_d
	};

	auto element_data = make_strided_mdspan<8>(static_cast<kernel_t*>(tmp_data.data()), element_ne, element_nb);
	for (int64_t batch_idx = 0; batch_idx < n; batch_idx++) {
		for (int64_t dst_z = 0; dst_z < dst_d; dst_z++) {
			for (int64_t dst_y = 0; dst_y < dst_h; dst_y++) {
				for (int64_t dst_x = 0; dst_x < dst_w; dst_x++) {
					stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
						for (int64_t ic = 0; ic < c; ++ic) {
							for (int64_t kz = 0; kz < knl_d; ++kz) {
								for (int64_t ky = 0; ky < knl_h; ++ky) {
									for (int64_t kx = 0; kx < knl_w; ++kx) {
										const int64_t sz = dst_z * s2 + kz * d2 - p2;
										const int64_t sy = dst_y * s1 + ky * d1 - p1;
										const int64_t sx = dst_x * s0 + kx * d0 - p0;

										float src_val;
										if (sz < 0 || sz >= src_d || sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
											src_val = 0.0f;
										}
										else {
											src_val = src_data[batch_idx, ic, sz, sy, sx];
										}

										element_data[batch_idx, dst_z, dst_y, dst_x, ic, kz, ky, kx] = fromFloat32<kernel_t>(src_val);
									}
								}
							}
						}
					});
					scope.spawn(std::move(sender));
				}
			}
		}
	}

	stdexec::sync_wait(scope.on_empty());

	std::vector<float> gemm_output1(patch_total * oc);
	std::experimental::mdspan gemm_output(gemm_output1.data(), n, dst_d, dst_h, dst_w, oc);
	ggml_call_mul_mat(kernel->type, pool, scope, patch_total, oc, knl_n_total, tmp_data.data(), knl_data, gemm_output1.data());

	stdexec::sync_wait(scope.on_empty());

	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), n, oc, dst_d, dst_h, dst_w);

	for (int64_t batch_idx = 0; batch_idx < n; batch_idx++) {
		for (int64_t dst_z = 0; dst_z < dst_d; dst_z++) {
			for (int64_t dst_y = 0; dst_y < dst_h; dst_y++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					for (int64_t dst_x = 0; dst_x < dst_w; dst_x++) {
						for (int64_t ioc = 0; ioc < oc; ++ioc) {
							dst_data[batch_idx, ioc, dst_z, dst_y, dst_x] = gemm_output[batch_idx, dst_z, dst_y, dst_x, ioc];
						}
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}

	stdexec::sync_wait(scope.on_empty());
}

void ggml_compute_forward_conv_3d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst)
{
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	switch (src0->type) {
	case GGML_TYPE_F32:
		ggml_compute_forward_conv_3d_impl<ggml_fp32_t>(pool, scope, src0, src1, dst);
		break;
	case GGML_TYPE_F16:
		ggml_compute_forward_conv_3d_impl<ggml_fp16_t>(pool, scope, src0, src1, dst);
		break;
	default:
		break;
	}
}

static void ggml_compute_forward_add_rel_pos_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	const ggml_tensor* src2 = dst->src[2];

	const bool inplace = (bool)((int32_t*)dst->op_params)[0];
	if (!inplace) {
		memcpy((char*)dst->data, (char*)src0->data, dst->nbytes());
	}
	// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

	float* src1_data = (float*)src1->data;
	float* src2_data = (float*)src2->data;
	float* dst_data = (float*)dst->data;

	const int64_t ne10 = src1->ne[0];
	const int64_t ne11 = src1->ne[1];
	const int64_t ne12 = src1->ne[2];
	const int64_t ne13 = src1->ne[3];

	const int nth = pool.available_parallelism();

	// total patches in dst
	const int64_t np = ne13;

	// patches per thread
	const int64_t dp = (np + nth - 1) / nth;

	// patch range for this thread
	for (int64_t ip0 = 0; ip0 < np; ip0 += dp) {
		const int64_t ip1 = std::min(ip0 + dp, np);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			for (int64_t i13 = ip0; i13 < ip1; ++i13) {
				for (int64_t i12 = 0; i12 < ne12; ++i12) {
					for (int64_t i11 = 0; i11 < ne11; ++i11) {
						const int64_t jp1 = i13 * ne12 * ne11 * ne10 + i12 * ne11 * ne10 + i11 * ne10;
						for (int64_t i10 = 0; i10 < ne10; ++i10) {
							const int64_t jp0 = jp1 + i10;
							const float src1_e = src1_data[jp0];
							const float src2_e = src2_data[jp0];

							const int64_t jdh = jp0 * ne10;
							const int64_t jdw = jdh - (ne10 - 1) * i10;

							for (int64_t j = 0; j < ne10; ++j) {
								dst_data[jdh + j] += src2_e;
								dst_data[jdw + j * ne10] += src1_e;
							}
						}
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

void ggml_compute_forward_add_rel_pos(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_add_rel_pos_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_add_id_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];
	const ggml_tensor* src2 = dst->src[2];

	GGML_ASSERT(dst->type == GGML_TYPE_F32);
	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(src2->type == GGML_TYPE_I32);

	GGML_ASSERT(src0->nb[0] == sizeof(float));
	GGML_ASSERT(src1->nb[0] == sizeof(float));

	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(dst->ne[3] == 1);
	GGML_ASSERT(src0->ne[3] == 1);

	auto dst_data = make_strided_mdspan<3>(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan<3>(static_cast<const float*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan<2>(static_cast<const float*>(src1->data), src1->ne, src1->nb);
	auto src2_data = make_strided_mdspan<2>(static_cast<const int32_t*>(src2->data), src2->ne, src2->nb);

	for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
		for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
				// src1 indices
				const int i11 = src2_data[i2, i1];

				GGML_ASSERT(i11 >= 0 && i11 < src1->ne[1]);

				for (int64_t i0 = 0; i0 < dst->ne[0]; i0++)
					dst_data[i2, i1, i0] = src0_data[i2, i1, i0] + src1_data[i11, i0];
			});
			scope.spawn(std::move(sender));
		}
	}
}

void ggml_compute_forward_add_id(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_add_id_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("unsupported type for ggml_compute_forward_add_id: %s", ggml_type_name(src0->type));
	}
	}
}

void ggml_compute_forward_map_custom(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const int32_t n_threads = dst->hook.n_tasks.value();
	auto func = dst->hook.func;
	for (uint32_t i = 0; i < dst->hook.n_tasks.value(); i++) {
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			func(dst, i, n_threads);
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_opt_step_sgd_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src0_grad = dst->src[1];
	const ggml_tensor* sgd_params = dst->src[2];

	GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
	GGML_ASSERT(sgd_params->nelements() == 2);

	// using adamw param subset we care about - alpha, wd - could have a separate struct
	const float* sgd_params_ptr = (float*)(sgd_params->data);
	const float   alpha = sgd_params_ptr[0];
	const float   keep = 1.f - alpha * sgd_params_ptr[1];

	auto w = make_strided_mdspan(static_cast<float*>(src0->data), src0->ne, src0->nb); // weight
	auto g = make_strided_mdspan(static_cast<const float*>(src0_grad->data), src0_grad->ne, src0_grad->nb); // grad
	for (int64_t i03 = 0; i03 < src0->ne[3]; i03++)
		for (int64_t i02 = 0; i02 < src0->ne[2]; i02++)
			for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					for (int64_t i00 = 0; i00 < src0->ne[0]; ++i00) {
						w[i03, i02, i01, i00] = w[i03, i02, i01, i00] * keep - alpha * g[i03, i02, i01, i00];
					}
				});
				scope.spawn(std::move(sender));
			}
}

void ggml_compute_forward_opt_step_sgd(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_opt_step_sgd_f32(pool, scope, dst);
	}
	break;
	default:
	{
		GGML_ABORT("fatal error - sgd is F32 only");
	}
	}
}

// ggml_compute_forward_im2col_3d
// src0: kernel [OC*IC, KD, KH, KW]
// src1: image [N*IC, ID, IH, IW]
// dst:  result [N, OD, OH, OW, IC * KD * KH * KW]
template <typename dst_t>
static void ggml_compute_forward_im2col_3d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
	const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
	const int32_t s2 = ((const int32_t*)(dst->op_params))[2];
	const int32_t p0 = ((const int32_t*)(dst->op_params))[3];
	const int32_t p1 = ((const int32_t*)(dst->op_params))[4];
	const int32_t p2 = ((const int32_t*)(dst->op_params))[5];
	const int32_t d0 = ((const int32_t*)(dst->op_params))[6];
	const int32_t d1 = ((const int32_t*)(dst->op_params))[7];
	const int32_t d2 = ((const int32_t*)(dst->op_params))[8];
	const int32_t IC = ((const int32_t*)(dst->op_params))[9];

	const int64_t N = src1->ne[3] / IC;
	const int64_t ID = src1->ne[2];
	const int64_t IH = src1->ne[1];
	const int64_t IW = src1->ne[0];

	[[maybe_unused]] const int64_t OC = src0->ne[3] / IC;
	const int64_t KD = src0->ne[2];
	const int64_t KH = src0->ne[1];
	const int64_t KW = src0->ne[0];

	const int64_t OD = dst->ne[3] / N;
	const int64_t OH = dst->ne[2];
	const int64_t OW = dst->ne[1];

	GGML_ASSERT(src1->nb[0] == sizeof(float));

	std::array<int64_t, 5> new_src_ne = { IW, IH, ID, IC, N }; // reverse order
	std::array<size_t, 5> new_src_nb = { src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3], src1->nb[3]* IC};
	auto src_data = make_strided_mdspan<5>(static_cast<const float*>(src1->data), new_src_ne, new_src_nb);
	std::experimental::mdspan dst_data(static_cast<dst_t*>(dst->data), N, OD, OH, OW, IC, KD, KH, KW);
	// im2col: [N, IC, ID, IH, IW] => [N, OD, OH, OW, IC, KD, KH, KW]
	for (int64_t iic = 0; iic < IC; iic++) {
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			for (int64_t in = 0; in < N; in++) {
				for (int64_t iod = 0; iod < OD; iod++) {
					for (int64_t ioh = 0; ioh < OH; ioh++) {
						for (int64_t iow = 0; iow < OW; iow++) {
							// micro kernel
							for (int64_t ikd = 0; ikd < KD; ikd++) {
								for (int64_t ikh = 0; ikh < KH; ikh++) {
									for (int64_t ikw = 0; ikw < KW; ikw++) {
										const int64_t iiw = iow * s0 + ikw * d0 - p0;
										const int64_t iih = ioh * s1 + ikh * d1 - p1;
										const int64_t iid = iod * s2 + ikd * d2 - p2;

										if (iid < 0 || iid >= ID || iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
											dst_data[in, iod, ioh, iow, iic, ikd, ikh, ikw] = 0;
										}
										else {
											dst_data[in, iod, ioh, iow, iic, ikd, ikh, ikw] = fromFloat32<dst_t>(src_data[in, iic, iid, iih, iiw]);
										}
									}
								}
							}
						}
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

void ggml_compute_forward_im2col_3d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	switch (dst->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_im2col_3d<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_im2col_3d<ggml_fp32_t>(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

// ggml_compute_forward_im2col_back_f32

void ggml_compute_forward_im2col_back_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0]; // gradients of forward pass output
	const ggml_tensor* src1 = dst->src[1]; // convolution kernel

	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F32);

	const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
	const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
	const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
	const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
	const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
	const int32_t d1 = ((const int32_t*)(dst->op_params))[5];
	const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

	const int64_t N = is_2D ? dst->ne[3] : dst->ne[2];
	const int64_t IC = is_2D ? dst->ne[2] : dst->ne[1];
	const int64_t IH = is_2D ? dst->ne[1] : 1;
	const int64_t IW = dst->ne[0];

	const int64_t KH = is_2D ? src1->ne[1] : 1;
	const int64_t KW = src1->ne[0];

	const int64_t OH = is_2D ? src0->ne[2] : 1;
	const int64_t OW = src0->ne[1];

#if 0
	int ofs0 = is_2D ? dst->nb[3] : dst->nb[2];
	int ofs1 = is_2D ? dst->nb[2] : dst->nb[1];
#endif
	GGML_ASSERT(dst->nb[0] == sizeof(float));

	// im2col: [N, IC, IH, IW] => [N, OH, OW, IC, KH, KW]
	{
		std::experimental::mdspan grad_in(static_cast<const float*>(src0->data), N, OH, OW, IC, KH, KW);
#if 0
		float* const wdata = (float*)dst->data;
#else
		auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
#endif
		for (int64_t in = 0; in < N; in++) {
			for (int64_t iic = 0; iic < IC; iic++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					for (int64_t iih = 0; iih < IH; iih++) {
						for (int64_t iiw = 0; iiw < IW; iiw++) {

							// micro kernel
							float grad = 0.0f;
							for (int64_t ikh = 0; ikh < KH; ikh++) {
								for (int64_t ikw = 0; ikw < KW; ikw++) {
									// For s0 > 1 some values were skipped over in the forward pass.
									// These values have tmpw % s0 != 0 and need to be skipped in the backwards pass as well.
									const int64_t tmpw = (iiw + p0 - ikw * d0);
									if (tmpw % s0 != 0) {
										continue;
									}
									const int64_t iow = tmpw / s0;

									// Equivalent logic as above except for s1.
									int64_t ioh;
									if (is_2D) {
										const int64_t tmph = iih + p1 - ikh * d1;

										if (tmph % s1 != 0) {
											continue;
										}

										ioh = tmph / s1;
									}
									else {
										ioh = 0;
									}

									if (iow < 0 || iow >= OW || ioh < 0 || ioh >= OH) {
										continue;
									}

									grad += grad_in[in, ioh, iow, iic, ikh, ikw];
								}
							}
#if 0
							float* dst_data = (float*)((char*)wdata + (in * ofs0 + iic * ofs1)); // [IH, IW]
							dst_data[iih * IW + iiw] = grad;
#else
							dst_data[in, iic, iih, iiw] = grad;
#endif
						}
					}
				});
				scope.spawn(std::move(sender));
			}
		}
	}
}

void ggml_compute_forward(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_compute_params* params, 
	ggml_tensor* tensor) {
	GGML_ASSERT(params);

	if (is_one_of(tensor->op,
		GGML_OP_NONE,
		GGML_OP_VIEW,
		GGML_OP_RESHAPE,
		GGML_OP_PERMUTE,
		GGML_OP_TRANSPOSE) || ggml_is_empty(tensor)) {
		return;
	}

	// extra_buffer op?
	// TODO
#if 0
	if (ggml_cpu_extra_compute_forward(params, tensor)) return;
#endif
	switch (tensor->op) {
	case GGML_OP_DUP:
	case GGML_OP_CONT:
	case GGML_OP_CPY:
	{
		ggml_compute_forward_dup(pool, scope, params, tensor);
	} break;
	case GGML_OP_ADD:
	{
		ggml_compute_forward_add(pool, scope, tensor);
	} break;
	case GGML_OP_ADD_ID:
	{
		ggml_compute_forward_add_id(pool, scope, tensor);
	} break;
	case GGML_OP_ADD1:
	{
		ggml_compute_forward_add1(pool, scope, tensor);
	} break;
	case GGML_OP_ACC:
	{
		ggml_compute_forward_acc(pool, scope, tensor);
	} break;
	case GGML_OP_SUB:
	{
		ggml_compute_forward_sub(pool, scope, tensor);
	} break;
	case GGML_OP_MUL:
	{
		ggml_compute_forward_mul(pool, scope, tensor);
	} break;
	case GGML_OP_DIV:
	{
		ggml_compute_forward_div(pool, scope, tensor);
	} break;
	case GGML_OP_SQR:
	{
		ggml_compute_forward_sqr(tensor);
	} break;
	case GGML_OP_SQRT:
	{
		ggml_compute_forward_sqrt(tensor);
	} break;
	case GGML_OP_LOG:
	{
		ggml_compute_forward_log(tensor);
	} break;
	case GGML_OP_SIN:
	{
		ggml_compute_forward_sin(tensor);
	} break;
	case GGML_OP_COS:
	{
		ggml_compute_forward_cos(tensor);
	} break;
	case GGML_OP_SUM:
	{
		ggml_compute_forward_sum(tensor);
	} break;
	case GGML_OP_SUM_ROWS:
	{
		ggml_compute_forward_sum_rows(tensor);
	} break;
	case GGML_OP_MEAN:
	{
		ggml_compute_forward_mean(tensor);
	} break;
	case GGML_OP_ARGMAX:
	{
		ggml_compute_forward_argmax(tensor);
	} break;
	case GGML_OP_COUNT_EQUAL:
	{
		ggml_compute_forward_count_equal(pool, scope, tensor);
	} break;
	case GGML_OP_REPEAT:
	{
		ggml_compute_forward_repeat(tensor);
	} break;
	case GGML_OP_REPEAT_BACK:
	{
		ggml_compute_forward_repeat_back(tensor);
	} break;
	case GGML_OP_CONCAT:
	{
		ggml_compute_forward_concat(tensor);
	} break;
	case GGML_OP_SILU_BACK:
	{
		ggml_compute_forward_silu_back(tensor);
	} break;
	case GGML_OP_NORM:
	{
		ggml_compute_forward_norm(tensor);
	} break;
	case GGML_OP_RMS_NORM:
	{
		ggml_compute_forward_rms_norm(tensor);
	} break;
	case GGML_OP_RMS_NORM_BACK:
	{
		ggml_compute_forward_rms_norm_back(tensor);
	} break;
	case GGML_OP_GROUP_NORM:
	{
		ggml_compute_forward_group_norm(tensor);
	} break;
	case GGML_OP_L2_NORM:
	{
		ggml_compute_forward_l2_norm(pool, scope, tensor);
	} break;
	case GGML_OP_MUL_MAT:
	{
		ggml_compute_forward_mul_mat(pool, scope, tensor);
	} break;
	case GGML_OP_MUL_MAT_ID:
	{
		ggml_compute_forward_mul_mat_id(pool, scope, tensor);
	} break;
	case GGML_OP_OUT_PROD:
	{
		ggml_compute_forward_out_prod(pool, scope, tensor);
	} break;
	case GGML_OP_SCALE:
	{
		ggml_compute_forward_scale(tensor);
	} break;
	case GGML_OP_SET:
	{
		ggml_compute_forward_set(pool, scope, tensor);
	} break;
	case GGML_OP_GET_ROWS:
	{
		ggml_compute_forward_get_rows(pool, scope, tensor);
	} break;
	case GGML_OP_SET_ROWS:
	{
		ggml_compute_forward_set_rows(pool, scope, tensor);
	} break;
	case GGML_OP_GET_ROWS_BACK:
	{
		ggml_compute_forward_get_rows_back(tensor);
	} break;
	case GGML_OP_DIAG:
	{
		ggml_compute_forward_diag(tensor);
	} break;
	case GGML_OP_DIAG_MASK_INF:
	case GGML_OP_DIAG_MASK_ZERO:
	{
		ggml_compute_forward_diag_mask(pool, scope, tensor);
	} break;
	case GGML_OP_SOFT_MAX:
	{
		ggml_compute_forward_soft_max(pool, scope, tensor);
	} break;
	case GGML_OP_SOFT_MAX_BACK:
	{
		ggml_compute_forward_soft_max_ext_back(pool, scope, tensor);
	} break;
	case GGML_OP_ROPE:
	{
		ggml_compute_forward_rope(pool, scope, tensor);
	} break;
	case GGML_OP_ROPE_BACK:
	{
		ggml_compute_forward_rope_back(pool, scope, tensor);
	} break;
	case GGML_OP_CLAMP:
	{
		ggml_compute_forward_clamp(pool, scope, tensor);
	} break;
	case GGML_OP_CONV_TRANSPOSE_1D:
	{
		ggml_compute_forward_conv_transpose_1d(pool, scope, tensor);
	} break;
	case GGML_OP_IM2COL:
	{
		ggml_compute_forward_im2col(pool, scope, tensor);
	} break;
	case GGML_OP_IM2COL_BACK:
	{
		ggml_compute_forward_im2col_back_f32(pool, scope, tensor);
	} break;
	case GGML_OP_IM2COL_3D:
	{
		ggml_compute_forward_im2col_3d(pool, scope, tensor);
	} break;
	case GGML_OP_CONV_2D:
	{
		ggml_compute_forward_conv_2d(pool, scope, tensor);
	} break;
	case GGML_OP_CONV_3D:
	{
		ggml_compute_forward_conv_3d(pool, scope, tensor);
	} break;
	case GGML_OP_CONV_2D_DW:
	{
		ggml_compute_forward_conv_2d_dw(pool, scope, tensor);
	} break;
	case GGML_OP_CONV_TRANSPOSE_2D:
	{
		ggml_compute_forward_conv_transpose_2d(pool, scope, tensor);
	} break;
	case GGML_OP_POOL_1D:
	{
		ggml_compute_forward_pool_1d(tensor);
	} break;
	case GGML_OP_POOL_2D:
	{
		ggml_compute_forward_pool_2d(tensor);
	} break;
	case GGML_OP_POOL_2D_BACK:
	{
		ggml_compute_forward_pool_2d_back(tensor);
	} break;
	case GGML_OP_UPSCALE:
	{
		ggml_compute_forward_upscale(tensor);
	} break;
	case GGML_OP_PAD:
	{
		ggml_compute_forward_pad(pool, scope, tensor);
	} break;
	case GGML_OP_PAD_REFLECT_1D:
	{
		ggml_compute_forward_pad_reflect_1d(pool, scope, tensor);
	} break;
	case GGML_OP_ROLL:
	{
		ggml_compute_forward_roll(pool, scope, tensor);
	} break;
	case GGML_OP_ARANGE:
	{
		ggml_compute_forward_arange(pool, scope, tensor);
	} break;
	case GGML_OP_TIMESTEP_EMBEDDING:
	{
		ggml_compute_forward_timestep_embedding(pool, scope, tensor);
	} break;
	case GGML_OP_ARGSORT:
	{
		ggml_compute_forward_argsort(tensor);
	} break;
	case GGML_OP_LEAKY_RELU:
	{
		ggml_compute_forward_leaky_relu(tensor);
	} break;
	case GGML_OP_FLASH_ATTN_EXT:
	{
		ggml_compute_forward_flash_attn_ext(pool, scope, tensor);
	} break;
#if 0
	case GGML_OP_FLASH_ATTN_BACK:
	{
		int32_t t = ggml_get_op_params_i32(tensor, 0);
		GGML_ASSERT(t == 0 || t == 1);
		bool masked = t != 0;
		ggml_compute_forward_flash_attn_back(params, masked, tensor);
	} break;
#endif
	case GGML_OP_SSM_CONV:
	{
		ggml_compute_forward_ssm_conv(pool, scope, tensor);
	} break;
	case GGML_OP_SSM_SCAN:
	{
		ggml_compute_forward_ssm_scan(pool, scope, tensor);
	} break;
	case GGML_OP_WIN_PART:
	{
		ggml_compute_forward_win_part(tensor);
	} break;
	case GGML_OP_WIN_UNPART:
	{
		ggml_compute_forward_win_unpart(tensor);
	} break;
	case GGML_OP_UNARY:
	{
		ggml_compute_forward_unary(tensor);
	} break;
	case GGML_OP_GLU:
	{
		ggml_compute_forward_glu(pool, scope, tensor);
	} break;
	case GGML_OP_GET_REL_POS:
	{
		ggml_compute_forward_get_rel_pos(tensor);
	} break;
	case GGML_OP_ADD_REL_POS:
	{
		ggml_compute_forward_add_rel_pos(pool, scope, tensor);
	} break;
	case GGML_OP_RWKV_WKV6:
	{
		ggml_compute_forward_rwkv_wkv6(pool, scope, tensor);
	} break;
	case GGML_OP_RWKV_WKV7:
	{
		ggml_compute_forward_rwkv_wkv7(pool, scope, tensor);
	} break;
	case GGML_OP_GATED_LINEAR_ATTN:
	{
		ggml_compute_forward_gla(pool, scope, tensor);
	} break;
	case GGML_OP_CUSTOM:
	{
		uint32_t n_threads = pool.available_parallelism();
		if (!tensor->hook.n_tasks.has_value()) {
			tensor->hook.n_tasks = n_threads;
		}
		else {
			tensor->hook.n_tasks = std::min(tensor->hook.n_tasks.value(), n_threads);
		}
		ggml_compute_forward_map_custom(pool, scope, tensor);
	}
	break;
	case GGML_OP_CROSS_ENTROPY_LOSS:
	{
		ggml_compute_forward_cross_entropy_loss(pool, scope, tensor);
	}
	break;
	case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
	{
		ggml_compute_forward_cross_entropy_loss_back(pool, scope, tensor);
	}
	break;
	case GGML_OP_OPT_STEP_ADAMW:
	{
		ggml_compute_forward_opt_step_adamw(pool, scope, tensor);
	}
	break;
	case GGML_OP_OPT_STEP_SGD:
	{
		ggml_compute_forward_opt_step_sgd(pool, scope, tensor);
	}
	break;
	case GGML_OP_NONE:
	{
		// nop
	} break;
	case GGML_OP_COUNT:
	{
		GGML_ABORT("fatal error");
	}
	}
}
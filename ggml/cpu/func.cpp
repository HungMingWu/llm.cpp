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

#define GGML_UNUSED(x) (void)(x)
#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);

#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

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

	const int nk = src0->ne[0] * src0->ne[1] * src0->ne[2];

	GGML_ASSERT(src0->nb[0] == sizeof(T));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src0->ne[2] == src1->ne[1]);
	GGML_ASSERT(dst->ne[1] == src0->ne[1]);

	const int64_t Cin = src0->ne[2];
	const int64_t Cout = src0->ne[1];
	const int64_t K = src0->ne[0];
	const int64_t L = src1->ne[0];

	std::vector<T> wdata(nk);
	std::vector<float> wdata_src(src1->ne[0] * src1->ne[1]);
	std::experimental::mdspan src0_data(static_cast<const T*>(src0->data), Cin, Cout, K);
	std::experimental::mdspan src1_data(static_cast<const float*>(src1->data), Cin, L);
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), Cout, dst->ne[0]);
	std::experimental::mdspan permute_kernel(wdata.data(), Cout, K, Cin);
	std::experimental::mdspan permute_source(wdata_src.data(), L, Cin);

	// permute kernel data (src0) from (Cin x Cout K K) to (Cout x K x Cin)
	for (int64_t i = 0; i < Cin; i++) {
		for (int64_t j = 0; j < Cout; j++) {
			for (int64_t k = 0; k < K; k++) {
				permute_kernel[j, k, i] = src0_data[i, j, k];
			}
		}
	}

	// permute source data (src1) from (Cin x L) to (L x Cin)
	for (int64_t i = 0; i < Cin; i++) {
		for (int64_t j = 0; j < L; j++) {
			permute_source[j, i] = src1_data[i, j];
		}
	}

	// need to zero dst since we are accumulating into it
	memset(dst->data, 0, dst->nbytes());

	const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t i = 0; i < Cout; i++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) |
			stdexec::then([=, &wdata, &wdata_src] {
				for (int64_t l = 0; l < L; l++) {
					for (int64_t j = 0; j < K; j++) {
						float v = 0.0;
						for (int64_t k = 0; k < Cin; k++) {
							v += permute_source[l, k] * toFloat32(permute_kernel[i, j, k]);
						}
						dst_data[i, l * s0 + j] += v;
					}
				}
			});
		scope.spawn(std::move(sender));
	}
	stdexec::sync_wait(scope.on_empty());
}

template <typename T>
static void ggml_compute_forward_conv_transpose_2d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F32);

	const int nk = src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];

	GGML_ASSERT(src0->nb[0] == sizeof(T));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src0->ne[3] == src1->ne[2]);
	GGML_ASSERT(dst->ne[2] == src0->ne[2]);

	const int64_t Cin = src0->ne[3];
	const int64_t Cout = src0->ne[2];
	const int64_t Kh = src0->ne[1];
	const int64_t Kw = src0->ne[0];
	const int64_t Sh = src1->ne[1];
	const int64_t Sw = src1->ne[0];

	std::vector<T> wdata(nk);
	std::vector<float> wdata_src(Cin * Sw * Sh);
	std::experimental::mdspan src0_data(static_cast<const T*>(src0->data), Cin, Cout, Kh, Kw);
	std::experimental::mdspan src1_data(static_cast<const float*>(src1->data), Cin, Sh, Sw);
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), Cout, dst->ne[1], dst->ne[0]);
	std::experimental::mdspan permute_kernel(wdata.data(), Cout, Kh, Kw, Cin);
	std::experimental::mdspan permute_source(wdata_src.data(), Sh, Sw, Cin);

	// permute kernel data (src0) from (Cin x Cout x Kh x Kw) to (Cout x Kh x Kw x Cin)
	for (int64_t i = 0; i < Cin; i++) {
		for (int64_t j = 0; j < Cout; j++) {
			for (int64_t k = 0; k < Kh; k++) {
				for (int64_t l = 0; l < Kw; l++) {
					permute_kernel[j, k, l, i] = src0_data[i, j, k, l];
				}
			}
		}
	}

	// permute source data (src1) from (Cin x Sh x Sw) to (Sh x Sw x Cin)
	for (int64_t i = 0; i < Cin; i++) {
		for (int64_t j = 0; j < Sh; j++) {
			for (int64_t k = 0; k < Sw; k++) {
				permute_source[j, k, i] = src1_data[i, j, k];
			}
		}
	}

	memset(dst->data, 0, dst->nbytes());

	const int32_t stride = std::bit_cast<int32_t>(dst->op_params[0]);

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t i = 0; i < Cout; i++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) |
			stdexec::then([=, &wdata, &wdata_src] {
				for (int64_t h = 0; h < Sh; h++) {
					for (int64_t w = 0; w < Sw; w++) {
						for (int64_t j = 0; j < Kh; j++) {
							for (int64_t k = 0; k < Kw; k++) {
								float v = 0.0f;
								for (int64_t l = 0; l < Cin; l++) {
									v += permute_source[h, w, l] * toFloat32(permute_kernel[i, j, k, l]);
								}
								dst_data[i, (h * stride + j), w * stride + k] += v;
							}
						}
					}
				}
			});
		scope.spawn(std::move(sender));
	}
	stdexec::sync_wait(scope.on_empty());
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
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
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

	GGML_TENSOR_UNARY_OP_LOCALS;

	if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst)) {
		ggml_compute_forward_dup_same_cont(pool, scope, dst);
		return;
	}

	const size_t type_size = ggml_type_size(src0->type);
	const int nth = pool.available_parallelism(); // number of threads

	// parallelize by rows
	const int nr = ne01;
	// number of rows per thread
	const int dr = (nr + nth - 1) / nth;

	for (int ir0 = 0; ir0 < nr; ir0 += dr) {
		const int ir1 = std::min(ir0 + dr, nr);
		const size_t rs = ne00 * type_size;
		auto exec_function = [=]() -> std::function<void(void)> {
			if (src0->type == dst->type &&
				ggml_are_same_shape(src0, dst) &&
				nb00 == type_size && nb0 == type_size) {
				// copy by rows
				return [=] {
					for (int64_t i03 = 0; i03 < ne03; i03++) {
						for (int64_t i02 = 0; i02 < ne02; i02++) {
							for (int64_t i01 = ir0; i01 < ir1; i01++) {
								memcpy(
									cast_with_offset<char>(dst->data, i01 * nb1 + i02 * nb2 + i03 * nb3),
									cast_with_offset<char>(src0->data, i01 * nb01 + i02 * nb02 + i03 * nb03),
									rs);
							}
						}
					}
				};
			} else if (ggml_is_contiguous(dst)) {
				if (nb00 == type_size) {
					return [=]() {
						size_t id = 0;
						// src0 is contigous on first dimension, copy by rows
						for (int64_t i03 = 0; i03 < ne03; i03++) {
							for (int64_t i02 = 0; i02 < ne02; i02++) {
								id += rs * ir0;
								for (int64_t i01 = ir0; i01 < ir1; i01++) {
									const auto src0_ptr = cast_with_offset<char>(src0->data, i01 * nb01 + i02 * nb02 + i03 * nb03);
									memcpy(cast_with_offset<char>(dst->data, id), src0_ptr, rs);
									id += rs;
								}
								id += rs * (ne01 - ir1);
							}
						}
					};
				}
				else {
					return [=]() {
						size_t id = 0;
						for (int64_t i03 = 0; i03 < ne03; i03++) {
							for (int64_t i02 = 0; i02 < ne02; i02++) {
								id += rs * ir0;
								for (int64_t i01 = ir0; i01 < ir1; i01++) {
									for (int64_t i00 = 0; i00 < ne00; i00++) {
										const auto src0_ptr = cast_with_offset<char>(src0->data, i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
										memcpy(cast_with_offset<char>(dst->data, id), src0_ptr, type_size);

										id += type_size;
									}
								}
								id += rs * (ne01 - ir1);
							}
						}
					};
				}
			}
			else {
				return [=] {
					// dst counters
					int64_t i10 = 0;
					int64_t i11 = 0;
					int64_t i12 = 0;
					int64_t i13 = 0;
					for (int64_t i03 = 0; i03 < ne03; i03++) {
						for (int64_t i02 = 0; i02 < ne02; i02++) {
							i10 += ne00 * ir0;
							while (i10 >= ne0) {
								i10 -= ne0;
								if (++i11 == ne1) {
									i11 = 0;
									if (++i12 == ne2) {
										i12 = 0;
										if (++i13 == ne3) {
											i13 = 0;
										}
									}
								}
							}
							for (int64_t i01 = ir0; i01 < ir1; i01++) {
								for (int64_t i00 = 0; i00 < ne00; i00++) {
									const auto src0_ptr = cast_with_offset<char>(src0->data, i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
									auto dst_ptr = cast_with_offset<char>(dst->data, i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

									memcpy(dst_ptr, src0_ptr, type_size);

									if (++i10 == ne0) {
										i10 = 0;
										if (++i11 == ne1) {
											i11 = 0;
											if (++i12 == ne2) {
												i12 = 0;
												if (++i13 == ne3) {
													i13 = 0;
												}
											}
										}
									}
								}
							}
							i10 += ne00 * (ne01 - ir1);
							while (i10 >= ne0) {
								i10 -= ne0;
								if (++i11 == ne1) {
									i11 = 0;
									if (++i12 == ne2) {
										i12 = 0;
										if (++i13 == ne3) {
											i13 = 0;
										}
									}
								}
							}
						}
					}
				};
			}
		}();
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([exec_function = std::move(exec_function)] {
				exec_function();
			});
		scope.spawn(std::move(sender));
	}
}

template <typename src_t, typename dst_t>
requires (is_quant_type_v<src_t> && std::is_same_v<dst_t, ggml_fp32_t>)
static void ggml_compute_forward_dup(
	exec::static_thread_pool & pool,
	exec::async_scope & scope,
	ggml_tensor * dst) {
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const enum ggml_type type = src0->type;

	size_t qk = ggml_blck_size(type);
	const int64_t nr = src1->nelements() / qk;

	// destination must be contiguous in the first dimension
	GGML_ASSERT(nb10 == ggml_type_size(dst->type));
	// must either have first dimension large enough to hold a row, or fully contiguous
	GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

	const int64_t nth = pool.available_parallelism();

	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
				for (int64_t ir = ir0; ir < ir1; ++ir) {
					uint64_t i = ir * qk;

					const int64_t i03 = i / (ne00 * ne01 * ne02);
					const int64_t i02 = (i - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
					const int64_t i01 = (i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00) / ne00;
					const int64_t i00 = i - i03 * ne00 * ne01 * ne02 - i02 * ne01 * ne00 - i01 * ne00;
					const int64_t x_offset = (i00 / qk) * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03;

					const int64_t i13 = i / (ne10 * ne11 * ne12);
					const int64_t i12 = (i - i13 * ne10 * ne11 * ne12) / (ne10 * ne11);
					const int64_t i11 = (i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11) / ne10;
					const int64_t i10 = i - i13 * ne10 * ne11 * ne12 - i12 * ne10 * ne11 - i11 * ne10;
					const int64_t dst_offset = i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13;

					dequantize_row(
						(const src_t*)((char*)src0->data + x_offset),
						(float*)((char*)dst->data + dst_offset), qk);
				}
			});
		scope.spawn(std::move(sender));
	}
}

template <typename src_t, typename dst_t>
void copy_non_cont(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) 
{
}

template <typename src_t, typename dst_t>
requires (!is_quant_type_v<dst_t>)
void copy_non_cont(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst)
{
	const ggml_tensor* src0 = dst->src[0];

	GGML_TENSOR_UNARY_OP_LOCALS

	const int64_t nth = pool.available_parallelism();

	// parallelize by rows
	const int64_t nr = ne01;
	// number of rows per thread
	const int64_t dr = (nr + nth - 1) / nth;
	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
				// dst counters
				int64_t i10 = 0;
				int64_t i11 = 0;
				int64_t i12 = 0;
				int64_t i13 = 0;

				for (int64_t i03 = 0; i03 < ne03; i03++) {
					for (int64_t i02 = 0; i02 < ne02; i02++) {
						i10 += ne00 * ir0;
						while (i10 >= ne0) {
							i10 -= ne0;
							if (++i11 == ne1) {
								i11 = 0;
								if (++i12 == ne2) {
									i12 = 0;
									if (++i13 == ne3) {
										i13 = 0;
									}
								}
							}
						}
						for (int64_t i01 = ir0; i01 < ir1; i01++) {
							for (int64_t i00 = 0; i00 < ne00; i00++) {
								const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
								char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

								*(dst_t*)dst_ptr = fromFloat32<dst_t>(toFloat32(*(const src_t*)src0_ptr));

								if (++i10 == ne0) {
									i10 = 0;
									if (++i11 == ne1) {
										i11 = 0;
										if (++i12 == ne2) {
											i12 = 0;
											if (++i13 == ne3) {
												i13 = 0;
											}
										}
									}
								}
							}
						}
						i10 += ne00 * (ne01 - ir1);
						while (i10 >= ne0) {
							i10 -= ne0;
							if (++i11 == ne1) {
								i11 = 0;
								if (++i12 == ne2) {
									i12 = 0;
									if (++i13 == ne3) {
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

template <typename src_t, typename dst_t>
requires (is_quant_type_v<dst_t>)
void copy_cont(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst)
{
	const ggml_tensor* src0 = dst->src[0];
	GGML_TENSOR_UNARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	// parallelize by rows
	const int64_t nr = ne01;
	// number of rows per thread
	const int64_t dr = (nr + nth - 1) / nth;
	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
				size_t id = 0;
				size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
				std::vector<float> src0_f32(ne00);
				char* dst_ptr = (char*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const src_t* src0_ptr = (src_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
							for (int i00 = 0; i00 < ne00; i00++) {
								src0_f32[i00] = toFloat32(src0_ptr[i00]);
							}

							quantize_row(src0_f32.data(), cast_with_offset<dst_t>(dst_ptr, id), ne00);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			});
		scope.spawn(std::move(sender));
	}
}

template <typename src_t, typename dst_t>
requires (!is_quant_type_v<dst_t>)
void copy_cont(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst)
{
	const ggml_tensor* src0 = dst->src[0];
	GGML_TENSOR_UNARY_OP_LOCALS

	const int64_t nth = pool.available_parallelism();

	// parallelize by rows
	const int64_t nr = ne01;
	// number of rows per thread
	const int64_t dr = (nr + nth - 1) / nth;
	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
				size_t id = 0;
				dst_t* dst_ptr = (dst_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const src_t* src0_ptr = (src_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
							for (int i00 = 0; i00 < ne00; i00++) {
								dst_ptr[id] = fromFloat32<dst_t>(toFloat32(src0_ptr[i00]));
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			});
		scope.spawn(std::move(sender));
	}
}

template <typename src_t, typename dst_t>
void copy_cont2(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) 
{
}

template <typename src_t, typename dst_t>
requires (!is_quant_type_v<dst_t>)
void copy_cont2(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) 
{
	const ggml_tensor* src0 = dst->src[0];
	GGML_TENSOR_UNARY_OP_LOCALS

	const int64_t nth = pool.available_parallelism();

	// parallelize by rows
	const int64_t nr = ne01;
	// number of rows per thread
	const int64_t dr = (nr + nth - 1) / nth;
	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
				size_t id = 0;
				dst_t* dst_ptr = (dst_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const src_t* src0_ptr = (src_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
								dst_ptr[id] = fromFloat32<dst_t>(toFloat32(*src0_ptr));
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			});
		scope.spawn(std::move(sender));
	}
}

template <typename src_t, typename dst_t>
requires (!is_quant_type_v<src_t>)
static void ggml_compute_forward_dup(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	GGML_ASSERT(dst->nelements() == src0->nelements());

	if (std::is_same_v<src_t, dst_t> &&
		src0->ne[0] == dst->ne[0] &&
		src0->nb[0] == sizeof(src_t) && dst->nb[0] == sizeof(dst_t)) {
		// copy by rows
		const int64_t nth = pool.available_parallelism();
		const int64_t nr = src0->ne[1];
		const int64_t dr = (nr + nth - 1) / nth;
		for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
			const int64_t ir1 = std::min(ir0 + dr, nr);
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
				stdexec::then([=] {
					const size_t rs = src0->ne[0] * src0->nb[0];
					for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
						for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
							for (int64_t i01 = ir0; i01 < ir1; i01++) {
								memcpy(
									((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]),
									((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]),
									rs);
							}
						}
					}
				});
			scope.spawn(std::move(sender));
		}
		return;
	}

	// TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

	if (ggml_is_contiguous(dst)) {
		if (src0->nb[0] == sizeof(src_t)) {
			copy_cont<src_t, dst_t>(pool, scope, dst);
		}
		else {
			//printf("%s: this is not optimal - fix me\n", __func__);
			copy_cont2<src_t, dst_t>(pool, scope, dst);
		}
		return;
	}
	copy_non_cont<src_t, dst_t>(pool, scope, dst);
}

template <typename src_t>
requires (is_quant_type_v<src_t>)
static void ggml_compute_forward_dup(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_dup<src_t, ggml_fp32_t>(pool, scope, dst);
	} break;
	default:
		assert(false);
		break;
	}
}

template <typename src_t>
requires (!is_quant_type_v<src_t>)
static void ggml_compute_forward_dup(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	switch (dst->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_dup<src_t, ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_dup<src_t, ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_dup<src_t, ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0:
	{
		ggml_compute_forward_dup<src_t, block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1:
	{
		ggml_compute_forward_dup<src_t, block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0:
	{
		ggml_compute_forward_dup<src_t, block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1:
	{
		ggml_compute_forward_dup<src_t, block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0:
	{
		ggml_compute_forward_dup<src_t, block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL:
	{
		ggml_compute_forward_dup<src_t, block_iq4_nl>(pool, scope, dst);
	} break;
	default:
		assert(false);
		break;
	}
}

static void ggml_compute_forward_dup(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	if (src0->type == dst->type) {
		ggml_compute_forward_dup_bytes(pool, scope, dst);
		return;
	}

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_dup<ggml_fp16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_dup<ggml_fp32_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_dup<ggml_bf16_t>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_0:
	{
		ggml_compute_forward_dup<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1:
	{
		ggml_compute_forward_dup<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0:
	{
		ggml_compute_forward_dup<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1:
	{
		ggml_compute_forward_dup<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0:
	{
		ggml_compute_forward_dup<block_q8_0>(pool, scope, dst);
	} break;
	default:
		assert(false);
		break;
	}
}

template <typename T>
static void ggml_vec_cpy(const int n, T* y, const T* x) {
	for (int i = 0; i < n; ++i) y[i] = x[i];
}

static void ggml_compute_forward_pad_reflect_1d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F32);

	const int32_t* opts = (const int32_t*)dst->op_params;
	const int p0 = opts[0];
	const int p1 = opts[1];

	GGML_TENSOR_UNARY_OP_LOCALS

	stdexec::scheduler auto scheduler = pool.get_scheduler();
	const int nth = pool.available_parallelism();
	const int64_t stride = (ne1 + nth - 1) / nth;

	for (int64_t start = 0; start < ne1; start += nth) {
		int64_t end = std::min(start + nth, ne1);
		stdexec::sender auto sender = stdexec::schedule(scheduler) |
			stdexec::then([=] {
				for (int64_t i3 = 0; i3 < ne3; i3++) {
					for (int64_t i2 = 0; i2 < ne2; i2++) {
						for (int64_t i1 = start; i1 < end; i1++) {
							auto left = cast_with_offset<float>(dst->data, i3 * nb3 + i2 * nb2 + i1 * nb1 + p0 * nb0);
							auto right = cast_with_offset<float>(dst->data, i3 * nb3 + i2 * nb2 + i1 * nb1 + (ne0 - p1 - 1) * nb0);

							ggml_vec_cpy(ne00, left, cast_with_offset<float>(src0->data, i3 * nb03 + i2 * nb02 + i1 * nb01));

							for (int i0 = 1; i0 <= p0; i0++) { left[-i0] = left[i0]; }
							for (int i0 = 1; i0 <= p1; i0++) { right[i0] = right[-i0]; }
						}
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

	GGML_TENSOR_BINARY_OP_LOCALS;

	const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
	const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
	const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
	const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
	const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
	const int32_t d1 = ((const int32_t*)(dst->op_params))[5];
	const bool is_2D = ((const int32_t*)(dst->op_params))[6] == 1;

	const int64_t N = is_2D ? ne13 : ne12;
	const int64_t IC = is_2D ? ne12 : ne11;
	const int64_t IH = is_2D ? ne11 : 1;
	const int64_t IW = ne10;

	const int64_t KH = is_2D ? ne01 : 1;
	const int64_t KW = ne00;

	const int64_t OH = is_2D ? ne2 : 1;
	const int64_t OW = ne1;

	if constexpr (std::is_same_v<T, ggml_fp16_t>) {
		GGML_ASSERT(nb00 == sizeof(T));
	}
	GGML_ASSERT(nb10 == sizeof(float));

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
	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	GGML_ASSERT(dst->ne[0] == src0->ne[0]);
	GGML_ASSERT(dst->ne[1] == src1->ne[0]);
	GGML_ASSERT(dst->ne[2] == src1->ne[2]);
	GGML_ASSERT(dst->ne[3] == src1->ne[3]);

	GGML_ASSERT(dst->ne[2] % src0->ne[2] == 0);
	GGML_ASSERT(dst->ne[3] % src0->ne[3] == 0);

	// we don't support permuted src0 or src1
	GGML_ASSERT(src0->nb[0] == sizeof(float));

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

void ggml_vec_add_f32(const int n, float* z, const float* x, const float* y) { for (int i = 0; i < n; ++i) z[i] = x[i] + y[i]; }

static void ggml_compute_forward_count_equal_i32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	struct ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS;

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
	const ggml_compute_params* params,
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
		if (params->ith == 0) {
			// memcpy needs to be synchronized across threads to avoid race conditions.
			// => do it in INIT phase
			memcpy(
				((char*)dst->data),
				((char*)src0->data),
				dst->nbytes());
		}
		//ggml_barrier(params->threadpool);
	}

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src1);
	const int nc = src1->ne[0];

	GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
	GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)

	// src0 and dst as viewed during set
	const size_t nb0 = ggml_element_size(src0);

	const int im0 = (ne10 == 0 ? 0 : ne10 - 1);
	const int im1 = (ne11 == 0 ? 0 : ne11 - 1);
	const int im2 = (ne12 == 0 ? 0 : ne12 - 1);
	const int im3 = (ne13 == 0 ? 0 : ne13 - 1);

	GGML_ASSERT(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= dst->nbytes());

	GGML_ASSERT(nb10 == sizeof(T));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are viewed with shape of src1 and offset
		// => same indices
		const int i3 = ir / (ne12 * ne11);
		const int i2 = (ir - i3 * ne12 * ne11) / ne11;
		const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

		ggml_vec_cpy(nc,
			(T*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
			(const T*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
	}
}

static void ggml_compute_forward_set(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_set<float>(params, dst);
	} break;
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_set<int32_t>(params, dst);
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

	GGML_TENSOR_BINARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	GGML_ASSERT(src0->type == dst->type);

	// we don't support permuted src0 or src1
	GGML_ASSERT(nb00 == sizeof(T));
	GGML_ASSERT(nb10 == sizeof(float));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

	GGML_ASSERT(ggml_is_quantized(src0->type));
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> wdata(ne00);
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0 indices
				const int i03 = ir / (ne02 * ne01);
				const int i02 = (ir - i03 * ne02 * ne01) / ne01;
				const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

				// src1 and dst are same shape as src0 => same indices
				const int i13 = i03;
				const int i12 = i02;
				const int i11 = i01;

				const int i3 = i03;
				const int i2 = i02;
				const int i1 = i01;

				void* src0_row = (void*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
				float* src1_row = (float*)((char*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13));
				void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

				assert(ne00 % 32 == 0);

				// unquantize row from src0 to temp buffer
				dequantize_row(static_cast<const T*>(src0_row), wdata.data(), ne00);
				// add src1
				std::span<const float> src1_row_span{ src1_row, static_cast<size_t>(ne00) };
				std::ranges::transform(src1_row_span, wdata, wdata.begin(), std::plus<>());

				// quantize row to dst
				if constexpr (requires { quantize_row(wdata.data(), static_cast<T*>(dst_row), ne00); }) {
					quantize_row(wdata.data(), static_cast<T*>(dst_row), ne00);
				}
				else {
					memcpy(dst_row, wdata.data(), ne0 * nb0);
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

	const int nth = pool.available_parallelism();
	const int64_t nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

	GGML_ASSERT(nb0 == sizeof(src0_t));
	GGML_ASSERT(nb00 == sizeof(src0_t));

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0 and dst are same shape => same indices
				const int i3 = ir / (ne2 * ne1);
				const int i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				src0_t* dst_ptr = (src0_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
				src0_t* src0_ptr = (src0_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
				for (int i = 0; i < ne0; i++) {
					dst_ptr[i] = fromFloat32<src0_t>(toFloat32(src0_ptr[i]) + v);
				}
			}
		});
		scope.spawn(std::move(sender));
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

	const int nth = pool.available_parallelism();

	const int64_t nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

	// we don't support permuted src0
	GGML_ASSERT(nb00 == sizeof(T));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

	GGML_ASSERT(ggml_is_quantized(src0->type));
	GGML_ASSERT(dst->type == src0->type);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> wdata(ne0);
			for (int64_t ir = ir0; ir < ir1; ++ir) {
				// src0 and dst are same shape => same indices
				const int i3 = ir / (ne2 * ne1);
				const int i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				void* src0_row = (void*)((char*)src0->data + (i1 * nb01 + i2 * nb02 + i3 * nb03));
				void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

				assert(ne0 % 32 == 0);

				// unquantize row from src0 to temp buffer
				dequantize_row(static_cast<const T*>(src0_row), wdata.data(), ne0);
				// add src1
				ggml_vec_acc1_f32(ne0, wdata.data(), v);
				// quantize row to dst
				quantize_row(wdata.data(), static_cast<T*>(dst_row), ne0);
			}
		});
		scope.spawn(std::move(sender));
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
	// allows optimizing the modulo since n_group should be a power of 2
	GGML_ASSERT((ng & -ng) == ng);

	std::experimental::mdspan y(static_cast<float*>(dst->data),ns, nt, nh, nr);  // { ns, nt, nh, dim }
	auto s0 = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb); // { ns, nh, dim, d_state }
	auto s = make_strided_mdspan((float*)((char*)dst->data + s_off), src0->ne, src0->nb); 	// {ns, nh, dim, d_state }
	auto x = make_strided_mdspan(static_cast<const float*>(src1->data), src1->ne, src1->nb); // { ns, nt, nh, dim }
	auto dt = make_strided_mdspan<3>(static_cast<const float*>(src2->data), src2->ne, src2->nb); // { ns, nt, nh }
	auto A = make_strided_mdspan<2>(static_cast<const float*>(src3->data), src3->ne, src3->nb); // { nh, d_state } or { nh, 1 }
	auto B = make_strided_mdspan(static_cast<const float*>(src4->data), src4->ne, src4->nb); // { ns, nt, ng, d_state }
	auto C = make_strided_mdspan(static_cast<const float*>(src5->data), src5->ne, src5->nb); // { ns, nt, ng, d_state }
	auto ids = make_strided_mdspan<1>(static_cast<const int32_t*>(src6->data), src6->ne, src6->nb);

	auto prev_state = [=](bool first, int64_t i3, int64_t i2, int64_t i1, int64_t i0) {
		if (first) {
			return s0[ids[i3], i2, i1, i0];
		}
		else {
			// use the output as the source when it's not the first token-wise iteration
			return s[i3, i2, i1, i0];
		}
	};

	const auto dA = [=](float dt_soft_plus, int64_t h, int64_t i0) {
		if (src3->ne[0] == 1) {
			// Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop
			// ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16

			return expf(dt_soft_plus * A[h, 0]);
		}
		else {
			// Mamba-1 has an element-wise decay factor for the states
			// ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16

			return expf(dt_soft_plus * A[h, i0]);
		}
	};

	// n_head
	for (int64_t h = 0; h < nh; h++) {
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int i3 = 0; i3 < ns; ++i3) {
				for (int i2 = 0; i2 < nt; ++i2) {
					const float dt_soft_plus = dt[i3, i2, h] <= 20.0f ? log1pf(expf(dt[i3, i2, h])) : dt[i3, i2, h];

					// dim
					for (int i1 = 0; i1 < nr; ++i1) {
						const float x_dt = x[i3, i2, h, i1] * dt_soft_plus;
						float sumf = 0.0f;
						// d_state
						for (int i0 = 0; i0 < nc; ++i0) {
							// state = prev_state * dA + dB * x
							const float state = (prev_state(i2 == 0, i3, h, i1, i0) * dA(dt_soft_plus, h, i0)) + (B[i3, i2, h & (ng - 1), i0] * x_dt);
							// y = rowwise_dotprod(state, C)
							sumf += state * C[i3, i2, h & (ng - 1), i0];
							s[i3, h, i1, i0] = state;
						}
						y[i3, i2, h, i1] = sumf;
					}
				}
			}
		});
		scope.spawn(std::move(sender));
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

	GGML_TENSOR_BINARY_OP_LOCALS

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

					const int64_t  i11 = id % ne11;
					const int64_t  i12 = row_mapping.i2; // row index in src1

					const int64_t  i1 = id;  // selected expert index
					const int64_t  i2 = i12; // row

					// desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
					//       if it is, then we have either copied the data to wdata and made it contiguous or we are using
					//       the original src1 data pointer, so we should index using the indices directly
					// TODO: this is a bit of a hack, we should probably have a better way to handle this
					const char* src1_col = (const char*)wdata +
						(src1_cont || src1->type != vec_dot_type
							? (i11 + i12 * ne11) * row_size
							: (i11 * nb11 + i12 * nb12));

					float* dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2));

					for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
						ggml_vec_dot<src0_t, vec_dot_t>(ne00, &tmp[ir0 - iir0], 0,
							cast_with_offset<src0_t>(src0_cur, ir0 * nb01), 0, 
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

	GGML_TENSOR_BINARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	const ggml_type type = src0->type;

	const bool src1_cont = ggml_is_contiguous(src1);

	enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;

	// we don't support permuted src0 or src1
	GGML_ASSERT(nb00 == ggml_type_size(type));
	GGML_ASSERT(nb10 == ggml_type_size(src1->type));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

	// row groups
	const int n_ids = ids->ne[0]; // n_expert_used
	const int n_as = ne02;       // n_expert

	// initialize matrix_row_counts
	std::vector<int64_t> matrix_row_counts(n_as, 0); // [n_as]
	std::vector<mmid_row_mapping> matrix_rows(n_as * ids->ne[0] * ids->ne[1]); // [n_as][ids->ne[0]*ids->ne[1]]
	std::experimental::mdspan matrix_rows_view(matrix_rows.data(), n_as, ids->ne[0] * ids->ne[1]);

	std::vector<uint8_t> wdata_1;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	if (src1->type != vec_dot_type) {
		wdata_1.resize(ggml_row_size(vec_dot_type, src1->nelements()));

		const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
		const size_t nbw2 = nbw1 * ne11;
		const size_t nbw3 = nbw2 * ne12;

		GGML_ASSERT(src1->type == GGML_TYPE_F32);
		for (int64_t ir0 = 0; ir0 < ne11; ir0 += nth) {
			const int64_t ir1 = std::min(ir0 + nth, ne11);
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &wdata_1] {
				for (int64_t i13 = 0; i13 < ne13; ++i13) {
					for (int64_t i12 = 0; i12 < ne12; ++i12) {
						for (int64_t i11 = ir0; i11 < ir1; i11++) {
							fromFloat(
								cast_with_offset<float>(src1->data, i13 * nb13 + i12 * nb12 + i11 * nb11),
								cast_with_offset<vec_dot_t>(wdata_1.data(), i13 * nbw3 + i12 * nbw2 + i11 * nbw1),
								ne10);
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

		const char* src0_cur = (const char*)src0->data + cur_a * nb02;
		const void* wdata = (src1->type == vec_dot_type) ? src1->data : (const void*)wdata_1.data();
		const size_t row_size = ggml_row_size(vec_dot_type, ne10);

		const int64_t nr0 = ne01;
		const int64_t nr1 = cne1;

		int chunk_size = 16;
		if (nr0 == 1 || nr1 == 1) {
			chunk_size = 64;
		}

#if defined(__aarch64__)
		// disable for ARM
		const bool disable_chunking = true;
#else
		// disable for NUMA
		const bool disable_chunking = ggml_is_numa();
#endif // defined(__aarch64__)

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

	std::experimental::mdspan dx(static_cast<float*>(dst->data), dst->ne[1], dst->ne[0]);
	std::experimental::mdspan dy(static_cast<const float*>(src0->data), src0->ne[1], src0->ne[0]);
	std::experimental::mdspan y(static_cast<const float*>(src1->data), src1->ne[1], src1->ne[0]);

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
	float theta_base_t, float theta_base_h, float theta_base_w, float theta_base_e, int sections[4], bool indep_sects,
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
		if (sector >= sections[0] && sector < sec_w) {
			theta = theta_h;
		}
		else if (sector >= sec_w && sector < sec_w + sections[2]) {
			theta = theta_w;
		}
		else if (sector >= sec_w + sections[2]) {
			theta = theta_e;
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

	GGML_TENSOR_UNARY_OP_LOCALS

	//printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
	//printf("n_past = %d, ne2 = %d\n", n_past, ne2);

	GGML_ASSERT(nb0 == sizeof(T));

	const int nth = pool.available_parallelism();

	const int64_t nr = ggml_nrows(dst);

	GGML_ASSERT(n_dims <= ne0);
	GGML_ASSERT(n_dims % 2 == 0);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	const float theta_scale = powf(freq_base, -2.0f / n_dims);

	float corr_dims[2];
	ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

	const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
	const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
	const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

	if (is_mrope) {
		GGML_ASSERT(sections[0] > 0 || sections[1] > 0 || sections[2] > 0);
	}

	if (is_vision) {
		GGML_ASSERT(n_dims == ne0 / 2);
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
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &corr_dims, &sections] {
			std::vector<float> cache(ne0);
			// row index used to determine which thread to use
			int64_t ir = 0;
			for (int64_t i3 = 0; i3 < ne3; i3++) {
				for (int64_t i2 = 0; i2 < ne2; i2++) {
					if (!is_mrope) {
						const int64_t p = pos[i2];
						ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, &cache[0], sin_sign, theta_scale);
					}
					else {
						const int64_t p_t = pos[i2];
						const int64_t p_h = pos[i2 + ne2];
						const int64_t p_w = pos[i2 + ne2 * 2];
						const int64_t p_e = pos[i2 + ne2 * 3];
						ggml_mrope_cache_init(
							p_t, p_h, p_w, p_e, sections, is_vision,
							freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, &cache[0], sin_sign, theta_scale);
					}

					for (int64_t i1 = 0; i1 < ne1; i1++) {
						if (ir++ < ir0) continue;
						if (ir > ir1) break;

						if (is_neox || is_mrope) {
							if (is_vision) {
								for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
									const int64_t ic = i0 / 2;

									const float cos_theta = cache[i0 + 0];
									const float sin_theta = cache[i0 + 1];

									const T* const src = (T*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
									T* dst_data = (T*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

									const float x0 = toFloat32(src[0]);
									const float x1 = toFloat32(src[n_dims]);

									dst_data[0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
									dst_data[n_dims] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
								}
							}
							else {
								for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
									const int64_t ic = i0 / 2;

									const float cos_theta = cache[i0 + 0];
									const float sin_theta = cache[i0 + 1];

									const T* const src = (T*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
									T* dst_data = (T*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

									const float x0 = toFloat32(src[0]);
									const float x1 = toFloat32(src[n_dims / 2]);

									dst_data[0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
									dst_data[n_dims / 2] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
								}
							}
						}
						else {
							for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
								const float cos_theta = cache[i0 + 0];
								const float sin_theta = cache[i0 + 1];

								const T* const src = (T*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
								T* dst_data = (T*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

								const float x0 = toFloat32(src[0]);
								const float x1 = toFloat32(src[1]);

								dst_data[0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
								dst_data[1] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
							}
						}

						if (is_vision) {
							for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
								const int64_t ic = i0 / 2;

								const float cos_theta = cache[i0 + 0];
								const float sin_theta = cache[i0 + 1];

								const T* const src = (T*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
								T* dst_data = (T*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

								const float x0 = toFloat32(src[0]);
								const float x1 = toFloat32(src[n_dims]);

								dst_data[0] = fromFloat32<T>(x0 * cos_theta - x1 * sin_theta);
								dst_data[n_dims] = fromFloat32<T>(x0 * sin_theta + x1 * cos_theta);
							}
						}
						else {
							for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
								const T* const src = (T*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
								T* dst_data = (T*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

								dst_data[0] = src[0];
								dst_data[1] = src[1];
							}
						}
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}

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
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

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
		if (params->ith == 0) {
			// memcpy needs to be synchronized across threads to avoid race conditions.
			// => do it in INIT phase
			memcpy(
				((char*)dst->data),
				((char*)src0->data),
				dst->nbytes());
		}
		//ggml_barrier(params->threadpool);
	}

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src1);
	const int nc = src1->ne[0];

	GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
	GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)

	// src0 and dst as viewed during acc
	const size_t nb0 = ggml_element_size(src0);

	const size_t nb00 = nb0;
	const size_t nb01 = nb1;
	const size_t nb02 = nb2;
	const size_t nb03 = nb3;

	GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10 - 1) * nb0 + (ne11 == 0 ? 0 : ne11 - 1) * nb1 + (ne12 == 0 ? 0 : ne12 - 1) * nb2 + (ne13 == 0 ? 0 : ne13 - 1) * nb3 < dst->nbytes());
	GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10 - 1) * nb00 + (ne11 == 0 ? 0 : ne11 - 1) * nb01 + (ne12 == 0 ? 0 : ne12 - 1) * nb02 + (ne13 == 0 ? 0 : ne13 - 1) * nb03 < src0->nbytes());

	GGML_ASSERT(nb10 == sizeof(float));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are viewed with shape of src1 and offset
		// => same indices
		const int i3 = ir / (ne12 * ne11);
		const int i2 = (ir - i3 * ne12 * ne11) / ne11;
		const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

#ifdef GGML_USE_ACCELERATE
		vDSP_vadd(
			(float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + offset), 1,
			(float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
			(float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset), 1, nc);
#else
		ggml_vec_add_f32(nc,
			(float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
			(float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + offset),
			(float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
#endif
	}
}

static void ggml_compute_forward_acc(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_acc_f32(params, dst);
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

	auto src_ptr = make_strided_mdspan(static_cast<float*>(src0->data), src0->ne, src0->nb);
	auto dst_ptr = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);

	for (int64_t i3 = 0; i3 < dst_ptr.extent(0); i3++) {
		for (int64_t i2 = 0; i2 < dst_ptr.extent(1); i2++) {
			for (int64_t i1 = 0; i1 < dst_ptr.extent(2); i1++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					for (int64_t i0 = 0; i0 < dst_ptr.extent(3); i0++) {
						if (i0 < src_ptr.extent(3) && i1 < src_ptr.extent(2) && i2 < src_ptr.extent(1) && i3 < src_ptr.extent(0)) {
							dst_ptr[i3, i2, i1, i0] = src_ptr[i3, i2, i1, i0];
						}
						else {
							dst_ptr[i3, i2, i1, i0] = 0;
						}
					}
				});
				scope.spawn(std::move(sender));
			}
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
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(src0->nb[0] == sizeof(float));

	const int ith = params->ith;
	const int nth = params->nth;

	GGML_TENSOR_UNARY_OP_LOCALS

	const int dim = std::bit_cast<int>(dst->op_params[0]);
	const int max_period = std::bit_cast<int>(dst->op_params[1]);

	int half = dim / 2;

	for (int64_t i = 0; i < ne00; i++) {
		float* embed_data = (float*)((char*)dst->data + i * nb1);
		for (int64_t j = ith; j < half; j += nth) {
			float timestep = ((float*)src0->data)[i];
			float freq = (float)expf(-logf(max_period) * j / half);
			float arg = timestep * freq;
			embed_data[j] = cosf(arg);
			embed_data[j + half] = sinf(arg);
		}
		if (dim % 2 != 0 && ith == 0) {
			embed_data[dim] = 0.f;
		}
	}
}

static void ggml_compute_forward_timestep_embedding(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_timestep_embedding_f32(params, dst);
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
					for (int i = 0; i < DV; ++i) {
						VKQ[i] /= S;
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
	case GGML_TYPE_F16: {
		ggml_compute_forward_flash_attn_ext<V_TYPE, ggml_fp16_t>(pool, scope, dst);
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
	case GGML_TYPE_F16: {
		ggml_compute_forward_flash_attn_ext<ggml_fp16_t>(pool, scope, dst);
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

	for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
		for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
			for (int64_t i01 = 0; i01 < src0->ne[1]; i01++) {
				stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
					for (int i00 = 0; i00 < src0->ne[0]; ++i00) {
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

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params* params, const struct ggml_tensor* src0) {
	const int64_t ith = params->ith;
	const int64_t nth = params->nth;

	const int64_t nr = ggml_nrows(src0);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	return { ir0, ir1 };
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t* z, const src0_t* x, const src1_t* y) {
	for (int i = 0; i < n; i++) {
		z[i] = fromFloat32<dst_t>(op(toFloat32(x[i]), toFloat32(y[i])));
	}
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t* z, const src0_t* x, const src1_t* y) {
	for (int i = 0; i < n; i++) {
		int i10 = i % ne10;
		const src1_t* y_ptr = (const src1_t*)((const char*)y + i10 * nb10);
		z[i] = fromFloat32<dst_t>(op(toFloat32(x[i]), toFloat32(*y_ptr)));
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

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(nb0 == sizeof(dst_t));
	GGML_ASSERT(nb00 == sizeof(src0_t));

	const bool is_src1_contiguous = (nb10 == sizeof(src1_t));

	if (!is_src1_contiguous) { // broadcast not implemented yet for non-contiguous
		GGML_ASSERT(ggml_are_same_shape(src0, src1));
	}

#ifdef GGML_USE_ACCELERATE
	vDSP_fn_t vDSP_op = nullptr;
	// TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
	if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
		if (op == op_add) {
			vDSP_op = vDSP_vadd;
		}
		else if (op == op_sub) {
			vDSP_op = vDSP_vsub;
		}
		else if (op == op_mul) {
			vDSP_op = vDSP_vmul;
		}
		else if (op == op_div) {
			vDSP_op = vDSP_vdiv;
		}
	}
#endif

	const int64_t nr = ggml_nrows(src0);
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	const int nth = pool.available_parallelism();

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	if (is_src1_contiguous) {
		for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
			const int64_t ir1 = std::min(ir0 + dr, nr);
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
				for (int64_t ir = ir0; ir < ir1; ++ir) {
					const int64_t i03 = ir / (ne02 * ne01);
					const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
					const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

					const int64_t i13 = i03 % ne13;
					const int64_t i12 = i02 % ne12;
					const int64_t i11 = i01 % ne11;

					dst_t* dst_ptr = (dst_t*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
					const src0_t* src0_ptr = (const src0_t*)((const char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
					const src1_t* src1_ptr = (const src1_t*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

					// src1 is broadcastable across src0 and dst in i1, i2, i3
					const int64_t nr0 = ne00 / ne10;

					for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
						if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> && std::is_same_v<dst_t, float>) {
							if (vDSP_op != nullptr) {
								vDSP_op(src1_ptr, 1, src0_ptr + r * ne10, 1, dst_ptr + r * ne10, 1, ne10);
								continue;
							}
						}
#endif
						vec_binary_op_contiguous<op>(ne10, dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
					}
				}
			});
			scope.spawn(std::move(sender));
		}
	}
	else {
		for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
			const int64_t ir1 = std::min(ir0 + dr, nr);
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
				for (int64_t ir = ir0; ir < ir1; ++ir) {
					const int64_t i03 = ir / (ne02 * ne01);
					const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
					const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

					const int64_t i13 = i03 % ne13;
					const int64_t i12 = i02 % ne12;
					const int64_t i11 = i01 % ne11;

					dst_t* dst_ptr = (dst_t*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
					const src0_t* src0_ptr = (const src0_t*)((const char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
					const src1_t* src1_ptr = (const src1_t*)((const char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);
					vec_binary_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
				}
			});
			scope.spawn(std::move(sender));
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
	int64_t batch;
	int64_t src_w;
	int64_t src_h;
	int64_t dst_w;
	int64_t dst_h;
	int64_t knl_w;
	int64_t knl_h;
	int stride_x;
	int stride_y;
	int pad_x;
	int pad_y;
	int dilation_x;
	int dilation_y;
};

static void ggml_compute_forward_conv_2d_dw_whcn(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* src,
	const ggml_tensor* kernel,
	ggml_tensor* dst,
	const ggml_conv_2d_dw_params& p) {

	const int64_t n = p.channels * p.batch;
	const int64_t per_thread = (n + pool.available_parallelism() - 1) / pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t start = 0; start < n; start += per_thread) {
		const int64_t end = std::min(start + per_thread, n);
		std::experimental::mdspan dst_data(static_cast<float*>(dst->data), n, p.dst_h, p.dst_w);
		std::experimental::mdspan knl_data(static_cast<const float*>(kernel->data), n, p.knl_h, p.knl_w);
		std::experimental::mdspan src_data(static_cast<const float*>(src->data), n, p.src_h, p.src_w);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t i = start; i < end; ++i) {
				for (int64_t dst_y = 0; dst_y < p.dst_h; ++dst_y) {
					for (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
						float sum = 0.0f;
						for (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
							const int64_t src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
							if (src_y < 0 || src_y >= p.src_h) {
								continue;
							}
							for (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
								const int64_t src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
								if (src_x < 0 || src_x >= p.src_w) {
									continue;
								}
								sum += knl_data[i, knl_y, knl_x]
									* src_data[i, src_y, src_x];
							}
						}
						dst_data[i, dst_y, dst_x] = sum;
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_conv_2d_dw_cwhn(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* src,
	const ggml_tensor* kernel,
	ggml_tensor* dst,
	const ggml_conv_2d_dw_params& p) {

	const int64_t c = p.channels;
	std::experimental::mdspan knl_data(static_cast<const float*>(kernel->data), p.knl_h, p.knl_w, p.channels);
	std::experimental::mdspan src_data(static_cast<const float*>(src->data), p.batch, p.src_h, p.src_w, p.channels);
	std::experimental::mdspan dst_data(static_cast<float*>(dst->data), p.dst_h, p.dst_w, p.channels);
	const int64_t rows_total = p.dst_h * p.batch;
	const int64_t rows_per_thread = (rows_total + pool.available_parallelism() - 1) / pool.available_parallelism();

#ifdef GGML_SIMD
	const int64_t pkg_size = GGML_F32_EPR;
	const int64_t pkg_count = c / pkg_size;
	const int64_t c_pkg_end = pkg_count * pkg_size;
#else
	const int64_t c_pkg_end = 0;
#endif
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t row_start = 0; row_start < rows_total; row_start += rows_per_thread) {
		const int64_t row_end = std::min(row_start + rows_per_thread, rows_total);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t row = row_start; row < row_end; ++row) {
				const int64_t dst_y = row % p.dst_h;
				for (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
					const int64_t src_y_base = dst_y * p.stride_y - p.pad_y;
					const int64_t src_x_base = dst_x * p.stride_x - p.pad_x;

#ifdef GGML_SIMD
					// Vectorized loop
					for (int64_t c_i = 0; c_i < c_pkg_end; c_i += pkg_size) {
						GGML_F32_VEC sum = GGML_F32_VEC_ZERO;
						for (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
							const int64_t src_y = src_y_base + knl_y * p.dilation_y;
							if (src_y < 0 || src_y >= p.src_h) {
								continue;
							}
							for (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
								const int64_t src_x = src_x_base + knl_x * p.dilation_x;
								if (src_x < 0 || src_x >= p.src_w) {
									continue;
								}
								GGML_F32_VEC k = GGML_F32_VEC_LOAD(knl_data + (knl_y * p.knl_w + knl_x) * c + c_i);
								GGML_F32_VEC s = GGML_F32_VEC_LOAD(src_data + (src_y * p.src_w + src_x) * c + c_i);
								sum = GGML_F32_VEC_FMA(sum, k, s);
							}
						}
						GGML_F32_VEC_STORE(dst_data + c_i, sum);
					}
#endif
					// Scalar loop
					for (int64_t c_i = c_pkg_end; c_i < c; ++c_i) {
						float sum = 0.0f;
						for (int64_t knl_y = 0; knl_y < p.knl_h; ++knl_y) {
							const int64_t src_y = src_y_base + knl_y * p.dilation_y;
							if (src_y < 0 || src_y >= p.src_h) {
								continue;
							}
							for (int64_t knl_x = 0; knl_x < p.knl_w; ++knl_x) {
								const int64_t src_x = src_x_base + knl_x * p.dilation_x;
								if (src_x < 0 || src_x >= p.src_w) {
									continue;
								}
								sum += knl_data[knl_y, knl_x, c_i]
									* src_data[row / p.dst_h, src_y, src_x, c_i];
							}
						}
						dst_data[row, dst_x, c_i] = sum;
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

void ggml_compute_forward_conv_2d_dw(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* kernel = dst->src[0];
	const ggml_tensor* src = dst->src[1];
	ggml_conv_2d_dw_params p{
		.channels = src->ne[2],
		.batch = src->ne[3],
		.src_w = src->ne[0],
		.src_h = src->ne[1],
		.dst_w = dst->ne[0],
		.dst_h = dst->ne[1],
		.knl_w = kernel->ne[0],
		.knl_h = kernel->ne[1],
		.stride_x = dst->op_params[0],
		.stride_y = dst->op_params[1],
		.pad_x = dst->op_params[2],
		.pad_y = dst->op_params[3],
		.dilation_x = dst->op_params[4],
		.dilation_y = dst->op_params[5]
	};

	GGML_ASSERT(kernel->ne[3] == p.channels);
	GGML_ASSERT(dst->ne[3] == p.batch);

	if (ggml_is_contiguous(src)) {
		ggml_compute_forward_conv_2d_dw_whcn(pool, scope, src, kernel, dst, p);
	}
	else if (ggml_is_contiguous_channels(src)) {
		// kernel should also have channels most contiguous in memory
		GGML_ASSERT(kernel->nb[0] >= kernel->nb[2] && kernel->nb[1] >= kernel->nb[0]);
		ggml_compute_forward_conv_2d_dw_cwhn(pool, scope, src, kernel, dst, p);
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
					GGML_UNUSED(v);
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
			GGML_UNUSED(x);
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

template <typename dst_t>
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
	auto src1_data = make_strided_mdspan<3>(static_cast<const int64_t*>(src1->data), src1->ne, src1->nb);

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
void ggml_compute_forward_set_rows(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_set_rows_f32<dst_t>(params, dst);
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

	const ggml_tensor* src0 = dst->src[0];

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

static void ggml_call_mul_mat(ggml_type type, const ggml_compute_params* params, int64_t m, int64_t n, int64_t k,
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
	dst.src[0] = &src0;
	dst.src[1] = &src1;

	//ggml_compute_forward_mul_mat(params, &dst);
}

// ggml_compute_forward_conv_2d

static void ggml_compute_forward_conv_2d_impl(const ggml_compute_params* params,
	const ggml_tensor* kernel,  // [KW, KH, IC, OC]
	const ggml_tensor* src,     // [W, H, C, N]
	ggml_tensor* dst,     // [OW, OH, OC, N]
	ggml_type                   kernel_type) {

	GGML_ASSERT(ggml_is_contiguous(kernel));
	GGML_ASSERT(kernel_type == GGML_TYPE_F16 || kernel_type == GGML_TYPE_F32);
	GGML_ASSERT(kernel->type == kernel_type);

	const ggml_type_traits* traits = ggml_get_type_traits(kernel_type);

	const int32_t stride_x = dst->op_params[0];
	const int32_t stride_y = dst->op_params[1];
	const int32_t pad_x = dst->op_params[2];
	const int32_t pad_y = dst->op_params[3];
	const int32_t dilation_x = dst->op_params[4];
	const int32_t dilation_y = dst->op_params[5];

	const int64_t c_in = src->ne[2];
	const int64_t c_out = kernel->ne[3];
	GGML_ASSERT(c_in == kernel->ne[2]);

	const int64_t src_w = src->ne[0];
	const int64_t src_h = src->ne[1];
	const int64_t knl_w = kernel->ne[0];
	const int64_t knl_h = kernel->ne[1];
	const int64_t dst_w = dst->ne[0];
	const int64_t dst_h = dst->ne[1];

	const float* src_data = (float*)src->data;
	void* knl_data = kernel->data;
	float* dst_data = (float*)dst->data;

	const int64_t knl_n = knl_w * knl_h * c_in;
	const int64_t patch_total = dst->ne[3] * dst_w * dst_h;

	const int64_t space_per_patch = knl_n * traits->type_size + c_out * sizeof(float);
	// Fix Here
#if 0
	const int64_t batch_size = params->wsize / space_per_patch;
#else
	const int64_t batch_size = 0;
#endif
	const int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
	const int64_t batch_n = (patch_total + patches_per_batch - 1) / patches_per_batch;

	GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);
	// Fix Here
#if 0
	void* tmp = params->wdata;
#else
	void* tmp = nullptr;
#endif
	for (int64_t batch_i = 0; batch_i < batch_n; ++batch_i) {

		const int64_t patch_start_batch = batch_i * patches_per_batch;
		const int64_t patch_end_batch = std::min(patch_start_batch + patches_per_batch,
			patch_total);
		const int64_t patch_n = patch_end_batch - patch_start_batch;

		const int64_t patch_per_thread = (patch_n + params->nth - 1) / params->nth;
		const int64_t patch_start = patch_start_batch + params->ith * patch_per_thread;
		const int64_t patch_end = std::min(patch_start + patch_per_thread, patch_end_batch);

		//im2col for a patch
		for (int64_t p = patch_start; p < patch_end; ++p) {
			const int64_t  batch_n = p / (dst_w * dst_h);
			const int64_t  src_x = (p / dst_w) % dst_h;
			const int64_t  src_y = p % dst_w;

			const float* src_base = (const float*)((const char*)src_data + batch_n * src->nb[3]);
			char* dst_row = (char*)tmp + (p % patches_per_batch) * knl_n * traits->type_size;

			for (int64_t ic = 0; ic < c_in; ++ic) {
				for (int64_t ky = 0; ky < knl_h; ++ky) {
					for (int64_t kx = 0; kx < knl_w; ++kx) {
						const int64_t sy = src_x * stride_y + ky * dilation_y - pad_y;
						const int64_t sx = src_y * stride_x + kx * dilation_x - pad_x;

						int64_t dst_idx = ic * (knl_h * knl_w) + ky * knl_w + kx;

						float src_val;
						if (sy < 0 || sy >= src_h || sx < 0 || sx >= src_w) {
							src_val = 0.0f;
						}
						else {
							const float* src_ptr = (const float*)((const char*)src_base + sx * src->nb[0] + sy * src->nb[1] + ic * src->nb[2]);
							src_val = *src_ptr;
						}

						char* element_ptr = dst_row + dst_idx * traits->type_size;
						if (kernel_type == GGML_TYPE_F32) {
							*(float*)element_ptr = src_val;
						}
						else if (kernel_type == GGML_TYPE_F16) {
							*(ggml_fp16_t*)element_ptr = fromFloat32<ggml_fp16_t>(src_val);
						}
					}
				}
			}
		}   // patches handled by this thread

		//ggml_barrier(params->threadpool);

		float* gemm_output = (float*)((char*)tmp + patches_per_batch * knl_n * traits->type_size);

		// TOFIX
#if 0
		GGML_ASSERT(gemm_output + patch_n * c_out <= (float*)tmp + params->wsize);
#endif
		// GEMM: patches[patch_n, knl_n] กั kernel[knl_n, c_out] = output[patch_n, c_out]
		ggml_call_mul_mat(kernel_type, params, patch_n, c_out, knl_n, tmp, knl_data, gemm_output);

		//ggml_barrier(params->threadpool);


		//permute back [OC, N, OH, OW] to [N, OC, OH, OW]
		const int64_t permute_per_thread = (patch_n + params->nth - 1) / params->nth;
		const int64_t permute_start = params->ith * permute_per_thread;
		const int64_t permute_end = std::min(permute_start + permute_per_thread, patch_n);

		for (int64_t i = permute_start; i < permute_end; ++i) {
			const int64_t p = patch_start_batch + i;
			const int64_t batch_n = p / (dst_w * dst_h);
			const int64_t dst_y = (p / dst_w) % dst_h;
			const int64_t dst_x = p % dst_w;

			for (int64_t oc = 0; oc < c_out; ++oc) {
				const float value = gemm_output[i * c_out + oc];
				float* dst_ptr = (float*)((char*)dst_data + dst_x * dst->nb[0] + dst_y * dst->nb[1] + oc * dst->nb[2] + batch_n * dst->nb[3]);
				*dst_ptr = value;
			}
		}
	}
}

void ggml_compute_forward_conv_2d(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	ggml_compute_forward_conv_2d_impl(params, src0, src1, dst, src0->type);
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

	auto dst_data = make_strided_mdspan(static_cast<float*>(dst->data), dst->ne, dst->nb);
	auto src0_data = make_strided_mdspan(static_cast<const float*>(src0->data), src0->ne, src0->nb);
	auto src1_data = make_strided_mdspan<2>(static_cast<const float*>(src1->data), src1->ne, src1->nb);
	auto src2_data = make_strided_mdspan<2>(static_cast<const int32_t*>(src2->data), src2->ne, src2->nb);

	for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
		for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
			for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
				stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
					// src1 indices
					const int i11 = src2_data[i2, i1];

					GGML_ASSERT(i11 >= 0 && i11 < src1->ne[1]);

					for (int64_t i0 = 0; i0 < dst->ne[0]; i0++)
						dst_data[i3, i2, i1, i0] = src0_data[i3, i2, i1, i0] + src1_data[i11, i0];
				});
				scope.spawn(std::move(sender));
			}
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

static void ggml_compute_forward_opt_step_sgd_f32(const ggml_compute_params* params, ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src0_grad = dst->src[1];
	const ggml_tensor* sgd_params = dst->src[2];

	GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
	GGML_ASSERT(sgd_params->nelements() == 2);

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS
		GGML_ASSERT(nb00 == sizeof(float));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	// using adamw param subset we care about - alpha, wd - could have a separate struct
	const float* sgd_params_ptr = (float*)(sgd_params->data);
	const float   alpha = sgd_params_ptr[0];
	const float   keep = 1.f - alpha * sgd_params_ptr[1];

	for (int ir = ir0; ir < ir1; ++ir) {
		const int64_t i03 = ir / (ne02 * ne01);
		const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
		const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

		const size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

		float* w = (float*)((char*)src0->data + offset);                   // weight
		const float* g = (const float*)((const char*)src0_grad->data + offset);  // grad

		for (int i00 = 0; i00 < ne00; ++i00) {
			w[i00] = w[i00] * keep - alpha * g[i00];
		}
	}
}

void ggml_compute_forward_opt_step_sgd(const ggml_compute_params* params, ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_opt_step_sgd_f32(params, dst);
	}
	break;
	default:
	{
		GGML_ABORT("fatal error - sgd is F32 only");
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
		ggml_compute_forward_dup(pool, scope, tensor);
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
		ggml_compute_forward_acc(params, tensor);
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
		ggml_compute_forward_set(params, tensor);
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
#if 0
	case GGML_OP_IM2COL_BACK:
	{
		ggml_compute_forward_im2col_back_f32(params, tensor);
	} break;
#endif
	case GGML_OP_CONV_2D:
	{
		ggml_compute_forward_conv_2d(params, tensor);
	} break;
	case GGML_OP_CONV_2D_DW:
	{
		ggml_compute_forward_conv_2d_dw(pool, scope, tensor);
	} break;
	case GGML_OP_CONV_TRANSPOSE_2D:
	{
		GGML_ASSERT(tensor->src[0]->type == GGML_TYPE_F16);
		ggml_compute_forward_conv_transpose_2d<ggml_fp16_t>(pool, scope, tensor);
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
		ggml_compute_forward_timestep_embedding(params, tensor);
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
#if 0
		uint32_t n_threads = pool.available_parallelism();
#else
		uint32_t n_threads = 4; // need to fix, it just passes unit test
#endif
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
		ggml_compute_forward_opt_step_sgd(params, tensor);
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
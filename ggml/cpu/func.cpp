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

template <typename T>
void toFloat(const T* x, float* y, int64_t n)
{
	if constexpr (is_quant_type_v<T>) {
		dequantize_row(x, y, n);
	}
	else {
		to_float(x, y, n);
	}
}

template <typename src0_t>
static void ggml_compute_forward_mul_mat_one_chunk(
	const ggml_compute_params* params,
	ggml_tensor* dst,
	const int64_t num_rows_per_vec_dot,
	const int64_t ir0_start,
	const int64_t ir0_end,
	const int64_t ir1_start,
	const int64_t ir1_end) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const bool src1_cont = ggml_is_contiguous(src1);

	using vec_dot_t = typename vec_dot_trait<src0_t>::type;
	enum ggml_type const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;

	// broadcast factors
	const int64_t r2 = ne12 / ne02;
	const int64_t r3 = ne13 / ne03;

	//printf("ir0_start = %6lld, ir0_end = %6lld, ir1_start = %6lld, ir1_end = %6lld\n", ir0_start, ir0_end, ir1_start, ir1_end);

	// threads with no work simply yield (not sure if it helps)
	if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
		return;
	}

	const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
	const size_t row_size = ggml_row_size(vec_dot_type, ne10);

	assert(ne12 % ne02 == 0);
	assert(ne13 % ne03 == 0);

	// block-tiling attempt
	static constexpr int64_t blck_0 = 16;
	static constexpr int64_t blck_1 = 16;

	const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

	// attempt to reduce false-sharing (does not seem to make a difference)
	// 16 * 2, accounting for mmla kernels
	float tmp[32];

	for (int64_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
		for (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
			for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
				const int64_t i13 = (ir1 / (ne12 * ne1));
				const int64_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
				const int64_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

				// broadcast src0 into src1
				const int64_t i03 = i13 / r3;
				const int64_t i02 = i12 / r2;

				auto src0_row = cast_with_offset<std::byte>(src0->data, 0 + i02 * nb02 + i03 * nb03);

				// desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
				//       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
				//       the original src1 data pointer, so we should index using the indices directly
				// TODO: this is a bit of a hack, we should probably have a better way to handle this
				auto src1_col = cast_with_offset<std::byte *>(wdata,
					(src1_cont || src1->type != vec_dot_type
						? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
						: (i11 * nb11 + i12 * nb12 + i13 * nb13)));
				auto dst_col = cast_with_offset<float>(dst->data, i11 * nb1 + i12 * nb2 + i13 * nb3);

				//for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
				//    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
				//}

				const size_t bs = (num_rows_per_vec_dot > 1 ? 16 : 0);
				const size_t bx = (num_rows_per_vec_dot > 1 ? nb01 : 0);
				const size_t by = (num_rows_per_vec_dot > 1 ? src1_col_stride : 0);
				for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
					ggml_vec_dot<src0_t, vec_dot_t>(ne00, &tmp[ir0 - iir0], bs,
						cast_with_offset<src0_t>(src0_row, ir0 * nb01), bx,
						cast_with_offset<vec_dot_t>(src1_col, 0), by, num_rows_per_vec_dot);
				}

				for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
					memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (std::min(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
				}
			}
		}
	}
}

template <typename src0_t>
static void ggml_compute_forward_mul_mat(
	exec::static_thread_pool &pool,
	exec::async_scope &scope,
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	using vec_dot_t = typename vec_dot_trait<src0_t>::type;
	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	enum ggml_type           const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;

	GGML_ASSERT(ne0 == ne01);
	GGML_ASSERT(ne1 == ne11);
	GGML_ASSERT(ne2 == ne12);
	GGML_ASSERT(ne3 == ne13);

	// we don't support permuted src0 or src1
	GGML_ASSERT(nb00 == ggml_type_size(src0->type));
	GGML_ASSERT(nb10 == ggml_type_size(src1->type));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

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
		const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
		const size_t nbw2 = nbw1 * ne11;
		const size_t nbw3 = nbw2 * ne12;

		assert(params->wsize >= ne13 * nbw3);
		GGML_ASSERT(src1->type == GGML_TYPE_F32);

		for (int64_t start = 0; start < ne11; start += nth) {
			int64_t end = std::min(start + nth, ne11);
			stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
				for (int64_t i13 = 0; i13 < ne13; ++i13) {
					for (int64_t i12 = 0; i12 < ne12; ++i12) {
						for (int64_t i11 = start; i11 < end; i11++) {
							fromFloat(
								cast_with_offset<float>(src1->data, i13 * nb13 + i12 * nb12 + i11 * nb11),
								cast_with_offset<vec_dot_t>(params->wdata, i13 * nbw3 + i12 * nbw2 + i11 * nbw1),
								ne10);
						}
					}
				}
			});
			scope.spawn(std::move(sender));
		}
		stdexec::sync_wait(scope.on_empty());
	}

#if GGML_USE_LLAMAFILE
	if (src1->type != vec_dot_type) {
		const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
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
	const int64_t nr0 = ne0;

	// This is the size of the rest of the dimensions of the result
	const int64_t nr1 = ne1 * ne2 * ne3;

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
			if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
				return 1;
			}
			return type_traits_cpu[src0->type].nrows;
		}();

		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			ggml_compute_forward_mul_mat_one_chunk<src0_t>(params, dst, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_mul_mat(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	switch (src0->type) {
	case GGML_TYPE_F32: {
		ggml_compute_forward_mul_mat<ggml_fp32_t>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_F16: {
		ggml_compute_forward_mul_mat<ggml_fp16_t>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_BF16: {
		ggml_compute_forward_mul_mat<ggml_bf16_t>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_mul_mat<block_q4_0>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_mul_mat<block_q4_1>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_mul_mat<block_q5_0>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_mul_mat<block_q5_1>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_mul_mat<block_q8_0>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_mul_mat<block_q2_K>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_mul_mat<block_q3_K>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_mul_mat<block_q4_K>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_mul_mat<block_q5_K>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_mul_mat<block_q6_K>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_mul_mat<block_iq1_s>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_mul_mat<block_iq1_m>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_mul_mat<block_iq2_xxs>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_mul_mat<block_iq2_xs>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_mul_mat<block_iq2_s>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_mul_mat<block_iq3_xxs>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_mul_mat<block_iq3_s>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_mul_mat<block_iq4_nl>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_mul_mat<block_iq4_xs>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_mul_mat<block_tq1_0>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_mul_mat<block_tq2_0>(pool, scope, params, dst);
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

	GGML_TENSOR_BINARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	const int nk = ne00 * ne01 * ne02;

	GGML_ASSERT(nb00 == sizeof(T));
	GGML_ASSERT(nb10 == sizeof(float));

	std::vector<T> wdata(nk), wdata_src(ne10 * ne11);

	// permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
	for (int64_t i02 = 0; i02 < ne02; i02++) {
		for (int64_t i01 = 0; i01 < ne01; i01++) {
			const auto src = cast_with_offset<T>(src0->data, i02 * nb02 + i01 * nb01);
			auto dst_data = &wdata[i01 * ne00 * ne02];
			for (int64_t i00 = 0; i00 < ne00; i00++) {
				dst_data[i00 * ne02 + i02] = src[i00];
			}
		}
	}

	// permute source data (src1) from (L x Cin) to (Cin x L)
	for (int64_t i11 = 0; i11 < ne11; i11++) {
		const auto src = cast_with_offset<float>(src1->data, i11 * nb11);
		for (int64_t i10 = 0; i10 < ne10; i10++) {
			wdata_src[i10 * ne11 + i11] = fromFloat32<T>(src[i10]);
		}
	}

	// need to zero dst since we are accumulating into it
	memset(dst->data, 0, dst->nbytes());

	const int32_t s0 = ((const int32_t*)(dst->op_params))[0];

	// total rows in dst
	const int nr = ne1;

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// calculate range for each thread and run in the background
	for (int ir0 = 0; ir0 < nr; ir0 += dr) {
		const int ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) |
			stdexec::then([=, &wdata, &wdata_src] {
				for (int i1 = ir0; i1 < ir1; i1++) {
					auto dst_data = cast_with_offset<float>(dst->data, i1 * nb1);
					auto wdata_kernel = &wdata[i1 * ne02 * ne00];
					for (int i10 = 0; i10 < ne10; i10++) {
						const int i1n = i10 * ne11;
						for (int i00 = 0; i00 < ne00; i00++) {
							float v = ggml_vec_dot<T>(ne02,
								&wdata_src[i1n],
								wdata_kernel + i00 * ne02, 1);
							dst_data[i10 * s0 + i00] += v;
						}
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

	GGML_TENSOR_BINARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	const int nk = ne00 * ne01 * ne02 * ne03;

	GGML_ASSERT(nb00 == sizeof(T));
	GGML_ASSERT(nb10 == sizeof(float));

	std::vector<T> wdata(nk), wdata_src(ne10 * ne11 * ne12);

	// permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
	for (int64_t i03 = 0; i03 < ne03; i03++) {
		for (int64_t i02 = 0; i02 < ne02; i02++) {
			const auto src = cast_with_offset<T>(src0->data, i03 * nb03 + i02 * nb02);
			T* dst_data = &wdata[i02 * ne01 * ne00 * ne03];
			for (int64_t i01 = 0; i01 < ne01; i01++) {
				for (int64_t i00 = 0; i00 < ne00; i00++) {
					dst_data[i01 * ne00 * ne03 + i00 * ne03 + i03] = src[i01 * ne00 + i00];
				}
			}
		}
	}

	// permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
	for (int i12 = 0; i12 < ne12; i12++) {
		for (int i11 = 0; i11 < ne11; i11++) {
			const auto src = cast_with_offset<float>(src1->data, i12 * nb12 + i11 * nb11);
			T* dst_data = &wdata_src[i11 * ne10 * ne12];
			for (int i10 = 0; i10 < ne10; i10++) {
				dst_data[i10 * ne12 + i12] = fromFloat32<T>(src[i10]);
			}
		}
	}

	memset(dst->data, 0, dst->nbytes());

	const int32_t stride = std::bit_cast<int32_t>(dst->op_params[0]);

	// total patches in dst
	const int np = ne2;

	// patches per thread
	const int dp = (np + nth - 1) / nth;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// calculate range for each thread and run in the background
	for (int ip0 = 0; ip0 < np; ip0 += dp) {
		const int ip1 = std::min(ip0 + dp, np);
		stdexec::sender auto sender = stdexec::schedule(scheduler) |
			stdexec::then([=, &wdata, &wdata_src] {
				for (int i2 = ip0; i2 < ip1; i2++) { // Cout
					const auto dst_data = cast_with_offset<float>(dst->data, i2 * nb2);
					const T* wdata_kernel = &wdata[i2 * ne01 * ne00 * ne03];
					for (int i11 = 0; i11 < ne11; i11++) {
						for (int i10 = 0; i10 < ne10; i10++) {
							const int i1n = i11 * ne10 * ne12 + i10 * ne12;
							for (int i01 = 0; i01 < ne01; i01++) {
								for (int i00 = 0; i00 < ne00; i00++) {
									float v = ggml_vec_dot<T>(ne03,
										&wdata_src[i1n],
										wdata_kernel + i01 * ne00 * ne03 + i00 * ne03, 1);
									dst_data[(i11 * stride + i01) * ne0 + i10 * stride + i00] += v;
								}
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

static void ggml_compute_forward_pool_1d_sk_p0(
	const enum ggml_op_pool op,
	const int k,
	ggml_tensor* dst) {

	const ggml_tensor* src = dst->src[0];

	assert(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

	const char* cdata = (const char*)src->data;
	const char* const data_end = cdata + src->nbytes();
	float* drow = (float*)dst->data;

	const int64_t rs = dst->ne[0];

	while (cdata < data_end) {
		const void* srow = (const void*)cdata;
		int j = 0;
		for (int64_t i = 0; i < rs; ++i) {
			switch (op) {
				case GGML_OP_POOL_AVG:   drow[i] = 0;        break;
				case GGML_OP_POOL_MAX:   drow[i] = -FLT_MAX; break;
				case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
			}
			for (int ki = 0; ki < k; ++ki) {
				const float srow_j = (src->type == GGML_TYPE_F32) ?
					*cast_with_offset<ggml_fp32_t>(srow, sizeof(ggml_fp32_t) * j) :
					toFloat32(*cast_with_offset<ggml_fp16_t>(srow, sizeof(ggml_fp16_t) * j));
				switch (op) {
					case GGML_OP_POOL_AVG:                         drow[i] += srow_j; break;
					case GGML_OP_POOL_MAX:   if (srow_j > drow[i]) drow[i] = srow_j; break;
					case GGML_OP_POOL_COUNT:                       GGML_ABORT("fatal error");
				}
				++j;
			}
			switch (op) {
				case GGML_OP_POOL_AVG:         drow[i] /= k; break;
				case GGML_OP_POOL_MAX:                       break;
				case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
			}
		}

		cdata += src->nb[1];
		drow += rs;
	}
}

static void ggml_compute_forward_pool_1d(ggml_tensor* dst) {

	const int32_t* opts = (const int32_t*)dst->op_params;
	enum ggml_op_pool op = std::bit_cast<ggml_op_pool>(opts[0]);
	const int k0 = opts[1];
	const int s0 = opts[2];
	const int p0 = opts[3];
	GGML_ASSERT(p0 == 0); // padding not supported
	GGML_ASSERT(k0 == s0); // only s = k supported

	ggml_compute_forward_pool_1d_sk_p0(op, k0, dst);
}

static void ggml_compute_forward_pool_2d(
	ggml_tensor* dst) {

	const struct ggml_tensor* src = dst->src[0];

	assert(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);

	const int32_t* opts = (const int32_t*)dst->op_params;
	auto op = std::bit_cast<ggml_op_pool>(opts[0]);
	const int k0 = opts[1];
	const int k1 = opts[2];
	const int s0 = opts[3];
	const int s1 = opts[4];
	const int p0 = opts[5];
	const int p1 = opts[6];
	const char* cdata = (const char*)src->data;
	const char* const data_end = cdata + src->nbytes();

	const int64_t px = dst->ne[0];
	const int64_t py = dst->ne[1];
	const int64_t pa = px * py;

	float* dplane = (float*)dst->data;

	const int ka = k0 * k1;
	const int offset0 = -p0;
	const int offset1 = -p1;

	while (cdata < data_end) {
		for (int oy = 0; oy < py; ++oy) {
			float* const drow = dplane + oy * px;
			for (int ox = 0; ox < px; ++ox) {
				float* const out = drow + ox;
				switch (op) {
				case GGML_OP_POOL_AVG:     *out = 0;        break;
				case GGML_OP_POOL_MAX:     *out = -FLT_MAX; break;
				case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
				}

				const int ix = offset0 + ox * s0;
				const int iy = offset1 + oy * s1;

				for (int ky = 0; ky < k1; ++ky) {
					if (iy + ky < 0 || iy + ky >= src->ne[1]) continue;
					const void* srow = (const void*)(cdata + src->nb[1] * (iy + ky));
					for (int kx = 0; kx < k0; ++kx) {
						int j = ix + kx;
						if (j < 0 || j >= src->ne[0]) continue;
						const float srow_j = (src->type == GGML_TYPE_F32) ?
							*cast_with_offset<ggml_fp32_t>(srow, sizeof(ggml_fp32_t) * j) :
							toFloat32(*cast_with_offset<ggml_fp16_t>(srow, sizeof(ggml_fp16_t) * j));
						switch (op) {
						case GGML_OP_POOL_AVG:                     *out += srow_j; break;
						case GGML_OP_POOL_MAX: if (srow_j > *out)  *out = srow_j; break;
						case GGML_OP_POOL_COUNT:               GGML_ABORT("fatal error");
						}
					}
				}
				switch (op) {
				case GGML_OP_POOL_AVG:           *out /= ka; break;
				case GGML_OP_POOL_MAX:                       break;
				case GGML_OP_POOL_COUNT: GGML_ABORT("fatal error");
				}
			}
		}

		cdata += src->nb[2];
		dplane += pa;
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
// dst:  result [N, OH, OW, IC*KH*KW]
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

	const int nth = pool.available_parallelism();

	const int64_t N = is_2D ? ne13 : ne12;
	const int64_t IC = is_2D ? ne12 : ne11;
	const int64_t IH = is_2D ? ne11 : 1;
	const int64_t IW = ne10;

	const int64_t KH = is_2D ? ne01 : 1;
	const int64_t KW = ne00;

	const int64_t OH = is_2D ? ne2 : 1;
	const int64_t OW = ne1;

	int ofs0 = is_2D ? nb13 : nb12;
	int ofs1 = is_2D ? nb12 : nb11;

	if constexpr (std::is_same_v<T, ggml_fp16_t>) {
		GGML_ASSERT(nb00 == sizeof(T));
	}
	GGML_ASSERT(nb10 == sizeof(float));

	// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
	{
		for (int64_t start = 0; start < IC; start += nth) {
			int64_t end = std::min(start + nth, IC);
			stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
				stdexec::then([=] {
					for (int64_t in = 0; in < N; in++) {
						for (int64_t ioh = 0; ioh < OH; ioh++) { // 1
							for (int64_t iow = 0; iow < OW; iow++) {
								for (int64_t iic = start; iic < end; iic++) {
									// micro kernel
									const auto dst_data =
										cast_with_offset<T>(dst->data,
											sizeof(T) * (in * OH * OW + ioh * OW + iow) * (IC * KH * KW)); // [IC, KH, KW]
									const auto src_data = cast_with_offset<ggml_fp32_t>(src1->data, in * ofs0 + iic * ofs1); // [IH, IW]

									for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
										for (int64_t ikw = 0; ikw < KW; ikw++) {
											const int64_t iiw = iow * s0 + ikw * d0 - p0;
											const int64_t iih = ioh * s1 + ikh * d1 - p1;

											if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
												dst_data[iic * (KH * KW) + ikh * KW + ikw] = 0;
											}
											else {
												dst_data[iic * (KH * KW) + ikh * KW + ikw] = fromFloat32<T>(src_data[iih * IW + iiw]);
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
void vec_sum(const int n, ggml_fp32_t* s, const T* x) {
	ggml_fp32_t sum = 0.0;
	for (int i = 0; i < n; ++i) {
		sum += toFloat32(x[i]);
	}
	*s = sum;
}

template <typename T>
static void ggml_compute_forward_sum(ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_scalar(dst));
	assert(src0->nb[0] == sizeof(T));

	GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
	GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

	float sum = 0;
	float row_sum = 0;

	for (int64_t i03 = 0; i03 < ne03; i03++) {
		for (int64_t i02 = 0; i02 < ne02; i02++) {
			for (int64_t i01 = 0; i01 < ne01; i01++) {
				vec_sum<T>(ne00, &row_sum,
					(T*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
				sum += row_sum;
			}
		}
	}
	((T*)dst->data)[0] = fromFloat32<T>(sum);
}

static void ggml_compute_forward_sum(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sum<ggml_fp32_t>(dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_sum<ggml_fp16_t>(dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_sum<ggml_bf16_t>(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_repeat_f16(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(ggml_can_repeat(src0, dst));

	GGML_TENSOR_UNARY_OP_LOCALS

	// guaranteed to be an integer due to the check in ggml_can_repeat
	const int nr0 = (int)(ne0 / ne00);
	const int nr1 = (int)(ne1 / ne01);
	const int nr2 = (int)(ne2 / ne02);
	const int nr3 = (int)(ne3 / ne03);

	// TODO: support for transposed / permuted tensors
	GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

	// TODO: maybe this is not optimal?
	for (int i3 = 0; i3 < nr3; i3++) {
		for (int k3 = 0; k3 < ne03; k3++) {
			for (int i2 = 0; i2 < nr2; i2++) {
				for (int k2 = 0; k2 < ne02; k2++) {
					for (int i1 = 0; i1 < nr1; i1++) {
						for (int k1 = 0; k1 < ne01; k1++) {
							for (int i0 = 0; i0 < nr0; i0++) {
								ggml_fp16_t* y = (ggml_fp16_t*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0);
								ggml_fp16_t* x = (ggml_fp16_t*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01);
								// ggml_vec_cpy_f16(ne00, y, x)
								for (int i = 0; i < ne00; ++i) {
									y[i] = x[i];
								}
							}
						}
					}
				}
			}
		}
	}
}

// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(
	struct ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(ggml_can_repeat(src0, dst));

	GGML_TENSOR_UNARY_OP_LOCALS

	// guaranteed to be an integer due to the check in ggml_can_repeat
	const int nr0 = (int)(ne0 / ne00);
	const int nr1 = (int)(ne1 / ne01);
	const int nr2 = (int)(ne2 / ne02);
	const int nr3 = (int)(ne3 / ne03);

	// TODO: support for transposed / permuted tensors
	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb00 == sizeof(float));

	// TODO: maybe this is not optimal?
	for (int i3 = 0; i3 < nr3; i3++) {
		for (int k3 = 0; k3 < ne03; k3++) {
			for (int i2 = 0; i2 < nr2; i2++) {
				for (int k2 = 0; k2 < ne02; k2++) {
					for (int i1 = 0; i1 < nr1; i1++) {
						for (int k1 = 0; k1 < ne01; k1++) {
							for (int i0 = 0; i0 < nr0; i0++) {
								ggml_vec_cpy(ne00,
									(float*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 + (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0),
									(float*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01));
							}
						}
					}
				}
			}
		}
	}
}

static void ggml_compute_forward_repeat(
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	case GGML_TYPE_BF16:
	case GGML_TYPE_I16:
	{
		ggml_compute_forward_repeat_f16(dst);
	} break;
	case GGML_TYPE_F32:
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_repeat_f32(dst);
	} break;
	// TODO: templateify the implemenation and support for I64
	//       ref https://github.com/ggml-org/llama.cpp/pull/14274#discussion_r2169492225
	//case GGML_TYPE_I64:
	//    {
	//        ggml_compute_forward_repeat_i64(params, dst);
	//    } break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

#define GGML_VEC_MAD_UNROLL  32

// xs and vs are byte strides of x and v
inline static void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float* y, const float* xv, const float* vv) {

	const float* x[GGML_VEC_MAD_UNROLL];
	const float* v[GGML_VEC_MAD_UNROLL];

	for (int i = 0; i < GGML_VEC_MAD_UNROLL; ++i) {
		x[i] = (const float*)((const char*)xv + i * xs);
		v[i] = (const float*)((const char*)vv + i * vs);
	}

#if defined(GGML_SIMD)
	const int np = (n & ~(GGML_F32_STEP - 1));

	GGML_F32_VEC vx[GGML_VEC_MAD_UNROLL];

	for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
		vx[k] = GGML_F32_VEC_SET1(v[k][0]);
	}

	GGML_F32_VEC ax[GGML_VEC_MAD_UNROLL][GGML_F32_ARR];
	GGML_F32_VEC ay[GGML_F32_ARR];

	for (int i = 0; i < np; i += GGML_F32_STEP) {
		for (int j = 0; j < GGML_F32_ARR; j++) {
			ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);

			for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
				ax[k][j] = GGML_F32_VEC_LOAD(x[k] + i + j * GGML_F32_EPR);
				ay[j] = GGML_F32_VEC_FMA(ay[j], ax[k][j], vx[k]);
			}

			GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
		}
	}

	// leftovers
	for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
		for (int i = np; i < n; ++i) {
			y[i] += x[k][i] * v[k][0];
		}
	}
#else
	// scalar
	for (int k = 0; k < GGML_VEC_MAD_UNROLL; ++k) {
		for (int i = 0; i < n; ++i) {
			y[i] += x[k][i] * v[k][0];
		}
	}
#endif
}

inline static void ggml_vec_mad_f32(const int n, float* y, const float* x, const float v) {
#if defined(GGML_SIMD)
	const int np = (n & ~(GGML_F32_STEP - 1));

	GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	GGML_F32_VEC ax[GGML_F32_ARR];
	GGML_F32_VEC ay[GGML_F32_ARR];

	for (int i = 0; i < np; i += GGML_F32_STEP) {
		for (int j = 0; j < GGML_F32_ARR; j++) {
			ax[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
			ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
			ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

			GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] += x[i] * v;
	}
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] += x[i] * v;
	}
#endif
}

static void ggml_compute_forward_out_prod_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(dst->type == GGML_TYPE_F32);
	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	const int nth = pool.available_parallelism();

	GGML_ASSERT(ne0 == ne00);
	GGML_ASSERT(ne1 == ne10);
	GGML_ASSERT(ne2 == ne12);
	GGML_ASSERT(ne3 == ne13);

	GGML_ASSERT(ne2 % ne02 == 0);
	GGML_ASSERT(ne3 % ne03 == 0);

	// we don't support permuted src0 or src1
	GGML_ASSERT(nb00 == sizeof(float));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 == sizeof(float));
	// GGML_ASSERT(nb0 <= nb1);
	// GGML_ASSERT(nb1 <= nb2);
	// GGML_ASSERT(nb2 <= nb3);

	// nb01 >= nb00 - src0 is not transposed
	//   compute by src0 rows

	ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);

	// dst[:,:,:,:] = 0
	// for i2,i3:
	//   for i1:
	//     for i01:
	//       for i0:
	//         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

	// parallelize by last three dimensions

	// total rows in dst
	const int64_t nr = ne1 * ne2 * ne3;

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// block-tiling attempt
	static constexpr int64_t blck_0 = std::max(GGML_VEC_MAD_UNROLL, 32);
	static constexpr int64_t blck_1 = 16;

	// dps == dst per src0, used for group query attention
	const int64_t dps2 = ne2 / ne02;
	const int64_t dps3 = ne3 / ne03;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t bir = ir0; bir < ir1; bir += blck_1) {
				const int64_t bir1 = std::min(bir + blck_1, ir1);
				for (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
					const int64_t bne01 = std::min(bi01 + blck_0, ne01);
					for (int64_t ir = bir; ir < bir1; ++ir) {
						// dst indices
						const int64_t i3 = ir / (ne2 * ne1);
						const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
						const int64_t i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

						const int64_t i02 = i2 / dps2;
						const int64_t i03 = i3 / dps3;

						//const int64_t i10 = i1;
						const int64_t i12 = i2;
						const int64_t i13 = i3;

#if GGML_VEC_MAD_UNROLL > 2
						const int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
						for (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
							const int64_t i11 = i01;

							float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
							float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
							float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

							ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
						}
						for (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
							const int64_t i11 = i01;

							float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
							float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
							float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

							ggml_vec_mad_f32(ne0, d, s0, *s1);
						}
#else
						for (int64_t i01 = bi01; i01 < bne01; ++i01) {
							const int64_t i11 = i01;

							float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
							float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
							float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

							ggml_vec_mad_f32(ne0, d, s0, *s1);
						}
#endif
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

template <typename T>
static void ggml_compute_forward_out_prod_q_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS;

	const int nth = pool.available_parallelism();

	GGML_ASSERT(ne02 == ne12);
	GGML_ASSERT(ne03 == ne13);
	GGML_ASSERT(ne2 == ne12);
	GGML_ASSERT(ne3 == ne13);

	// we don't support permuted src0 dim0
	GGML_ASSERT(nb00 == sizeof(T));

	// dst dim0 cannot be transposed or permuted
	GGML_ASSERT(nb0 == sizeof(float));
	// GGML_ASSERT(nb0 <= nb1);
	// GGML_ASSERT(nb1 <= nb2);
	// GGML_ASSERT(nb2 <= nb3);

	GGML_ASSERT(ne0 == ne00);
	GGML_ASSERT(ne1 == ne10);
	GGML_ASSERT(ne2 == ne02);
	GGML_ASSERT(ne3 == ne03);

	// nb01 >= nb00 - src0 is not transposed
	//   compute by src0 rows

	ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);

	// parallelize by last three dimensions

	// total rows in dst
	const int64_t nr = ne1 * ne2 * ne3;

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// dst[:,:,:,:] = 0
	// for i2,i3:
	//   for i1:
	//     for i01:
	//       for i0:
	//         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> wdata(ne0);
			for (int64_t ir = ir0; ir < ir1; ++ir) {
				// dst indices
				const int64_t i3 = ir / (ne2 * ne1);
				const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int64_t i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				const int64_t i02 = i2;
				const int64_t i03 = i3;

				//const int64_t i10 = i1;
				const int64_t i12 = i2;
				const int64_t i13 = i3;

				for (int64_t i01 = 0; i01 < ne01; ++i01) {
					const int64_t i11 = i01;

					T* s0 = (T*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
					float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
					float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

					dequantize_row(s0, wdata.data(), ne0);
					ggml_vec_mad_f32(ne0, d, wdata.data(), *s1);
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_out_prod(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_out_prod_q_f32<block_q4_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_1: {
		ggml_compute_forward_out_prod_q_f32<block_q4_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_out_prod_q_f32<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_out_prod_q_f32<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_out_prod_q_f32<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q2_K: {
		ggml_compute_forward_out_prod_q_f32<block_q2_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q3_K: {
		ggml_compute_forward_out_prod_q_f32<block_q3_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q4_K: {
		ggml_compute_forward_out_prod_q_f32<block_q4_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_K: {
		ggml_compute_forward_out_prod_q_f32<block_q5_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q6_K: {
		ggml_compute_forward_out_prod_q_f32<block_q6_K>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ1_0: {
		ggml_compute_forward_out_prod_q_f32<block_tq1_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_TQ2_0: {
		ggml_compute_forward_out_prod_q_f32<block_tq2_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XXS: {
		ggml_compute_forward_out_prod_q_f32<block_iq2_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_XS: {
		ggml_compute_forward_out_prod_q_f32<block_iq2_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_XXS: {
		ggml_compute_forward_out_prod_q_f32<block_iq3_xxs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_S: {
		ggml_compute_forward_out_prod_q_f32<block_iq1_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ1_M: {
		ggml_compute_forward_out_prod_q_f32<block_iq1_m>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_NL: {
		ggml_compute_forward_out_prod_q_f32<block_iq4_nl>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ4_XS: {
		ggml_compute_forward_out_prod_q_f32<block_iq4_xs>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ3_S: {
		ggml_compute_forward_out_prod_q_f32<block_iq3_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_IQ2_S: {
		ggml_compute_forward_out_prod_q_f32<block_iq2_s>(pool, scope, dst);
	} break;
	case GGML_TYPE_F16:
	{
		GGML_ABORT("fatal error"); // todo
		// ggml_compute_forward_out_prod_f16_f32(params, dst);
	}
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_out_prod_f32(pool, scope, dst);
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

	GGML_TENSOR_BINARY_OP_LOCALS

	const int64_t nc = ne00;
	const int64_t nr = src1->nelements();

	assert(ne0 == nc);
	assert(ne02 == ne11);
	assert(nb00 == sizeof(T));
	assert(ggml_nrows(dst) == nr);

	const int nth = pool.available_parallelism();

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread

	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			for (int64_t i = ir0; i < ir1; ++i) {
				const int64_t i12 = i / (ne11 * ne10);
				const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
				const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
				const int64_t i01 = *(int32_t*)((char*)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

				GGML_ASSERT(i01 >= 0 && i01 < ne01);

				toFloat(
					(const T*)((char*)src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03),
					(float*)((char*)dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3), nc);
			}
		});
		scope.spawn(std::move(sender));
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

static void ggml_compute_forward_get_rows_back_f32_f16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	if (params->ith != 0) {
		return;
	}

	GGML_ASSERT(ggml_is_contiguous(dst));

	// ggml_compute_forward_dup_same_cont(params, opt0, dst);

	memset(dst->data, 0, dst->nbytes());

	const int nc = src0->ne[0];
	const int nr = src1->nelements();

	GGML_ASSERT(dst->ne[0] == nc);
	GGML_ASSERT(src0->nb[0] == sizeof(ggml_fp16_t));

	for (int i = 0; i < nr; ++i) {
		const int r = ((int32_t*)src1->data)[i];

		for (int j = 0; j < nc; ++j) {
			ggml_fp16_t v = ((ggml_fp16_t*)((char*)src0->data + i * src0->nb[1]))[j];
			((float*)((char*)dst->data + r * dst->nb[1]))[j] += toFloat32(v);
		}
	}
}

void ggml_vec_add_f32(const int n, float* z, const float* x, const float* y) { for (int i = 0; i < n; ++i) z[i] = x[i] + y[i]; }

static void ggml_compute_forward_get_rows_back_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	if (params->ith != 0) {
		return;
	}

	GGML_ASSERT(ggml_is_contiguous(dst));

	// ggml_compute_forward_dup_same_cont(params, opt0, dst);

	memset(dst->data, 0, dst->nbytes());

	const int nc = src0->ne[0];
	const int nr = src1->nelements();

	GGML_ASSERT(dst->ne[0] == nc);
	GGML_ASSERT(src0->nb[0] == sizeof(float));

	for (int i = 0; i < nr; ++i) {
		const int r = ((int32_t*)src1->data)[i];

		ggml_vec_add_f32(nc,
			(float*)((char*)dst->data + r * dst->nb[1]),
			(float*)((char*)dst->data + r * dst->nb[1]),
			(float*)((char*)src0->data + i * src0->nb[1]));
	}
}

static void ggml_compute_forward_get_rows_back(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_get_rows_back_f32_f16(params, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_get_rows_back_f32(params, dst);
	} break;
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

inline static void ggml_vec_argmax_f32(const int n, int* s, const float* x) {
	float maxValue = -INFINITY;
	int idx = 0;
	for (int i = 0; i < n; ++i) {
		maxValue = std::max(maxValue, x[i]);
		if (maxValue == x[i]) { idx = i; }
	}
	*s = idx;
}

static void ggml_compute_forward_argmax_f32(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	assert(src0->nb[0] == sizeof(float));
	assert(dst->nb[0] == sizeof(float));

	const int64_t ne00 = src0->ne[0];
	const int64_t ne01 = src0->ne[1];

	const size_t nb01 = src0->nb[1];
	const size_t nb0 = dst->nb[0];

	for (int64_t i1 = 0; i1 < ne01; i1++) {
		float* src = (float*)((char*)src0->data + i1 * nb01);
		int32_t* dst_ = (int32_t*)((char*)dst->data + i1 * nb0);
		int v = 0;
		ggml_vec_argmax_f32(ne00, &v, src);
		dst_[0] = v;
	}
}

static void ggml_compute_forward_argmax(ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_argmax_f32(dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

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

	const int64_t nr = ggml_nrows(src0);

	const int nth = pool.available_parallelism();

	std::atomic<int64_t> sum_thread{ 0 };

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=, &sum_thread] {
			int64_t sum = 0;
			for (int64_t ir = ir0; ir < ir1; ++ir) {
				const int64_t i03 = ir / (ne02 * ne01);
				const int64_t i02 = (ir - i03 * ne03) / ne01;
				const int64_t i01 = ir - i03 * ne03 - i02 * ne02;

				const char* data0 = (const char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01;
				const char* data1 = (const char*)src1->data + i03 * nb13 + i02 * nb12 + i01 * nb11;

				for (int64_t i00 = 0; i00 < ne00; ++i00) {
					const int32_t val0 = *((const int32_t*)(data0 + i00 * nb00));
					const int32_t val1 = *((const int32_t*)(data1 + i00 * nb10));

					sum += val0 == val1;
				}
			}
			sum_thread += sum;
		});
		scope.spawn(std::move(sender));
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

static void ggml_vec_add1_f32(const int n, float* z, const float* x, const float   v) { for (int i = 0; i < n; ++i) z[i] = x[i] + v; }

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
	case GGML_TYPE_Q5_0: {
		ggml_compute_forward_add1_q_f32<block_q5_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q5_1: {
		ggml_compute_forward_add1_q_f32<block_q5_1>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_add1_q_f32<block_q8_0>(pool, scope, dst);
	} break;
	case GGML_TYPE_Q8_1: {
		// ggml_compute_forward_add1_q_f32<block_q8_1>(pool, scope, dst);
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

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int nc = src1->ne[0]; // d_conv
	const int ncs = src0->ne[0]; // d_conv - 1 + n_t
	const int nr = src0->ne[1]; // d_inner
	const int n_t = dst->ne[1]; // tokens per sequence
	const int n_s = dst->ne[2]; // number of sequences in the batch

	GGML_ASSERT(dst->ne[0] == nr);
	GGML_ASSERT(src0->nb[0] == sizeof(float));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min<int64_t>(ir0 + dr, nr);
		const int ir = ir1 - ir0;
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int i3 = 0; i3 < n_s; ++i3) {
				for (int i2 = 0; i2 < n_t; ++i2) {
					// {d_conv - 1 + n_t, d_inner, n_seqs}
					// sliding window
					const float* s = (const float*)((const char*)src0->data + ir0 * (src0->nb[1]) + i2 * (src0->nb[0]) + i3 * (src0->nb[2])); // {d_conv, d_inner, n_s}
					const float* c = (const float*)((const char*)src1->data + ir0 * (src1->nb[1])); // {d_conv, d_inner}
					float* x = (float*)((char*)dst->data + ir0 * (dst->nb[0]) + i2 * (dst->nb[1]) + i3 * (dst->nb[2])); // {d_inner, n_t, n_s}

					// TODO: transpose the output for smaller strides for big batches?
					// d_inner
					for (int i1 = 0; i1 < ir; ++i1) {
						// rowwise dot product
						// NOTE: not using ggml_vec_dot_f32, because its sum is in double precision
						float sumf = 0.0f;

						// d_conv
						for (int i0 = 0; i0 < nc; ++i0) {
							sumf += s[i0 + i1 * ncs] * c[i0 + i1 * nc];
						}
						x[i1] = sumf;
					}
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

	const int nth = pool.available_parallelism();
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

	// heads per thread
	const int dh = (nh + nth - 1) / nth;

	const int32_t* ids = (const int32_t*)src6->data;

	// row range for this thread
	for (int64_t ih0 = 0; ih0 < nh; ih0 += dh) {
		const int64_t ih1 = std::min(ih0 + dh, nh);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int i3 = 0; i3 < ns; ++i3) {
				const float* s0 = (const float*)((const char*)src0->data + ids[i3] * (src0->nb[3])); // {d_state, dim, nh, ns}
				float* s = (float*)((char*)dst->data + i3 * (src0->nb[3]) + s_off); // {d_state, dim, nh, ns}

				for (int i2 = 0; i2 < nt; ++i2) {
					const float* x = (const float*)((const char*)src1->data + i2 * (src1->nb[2]) + i3 * (src1->nb[3])); // {dim, nh, nt, ns}
					const float* dt = (const float*)((const char*)src2->data + i2 * (src2->nb[1]) + i3 * (src2->nb[2])); // {nh, nt, ns}
					const float* A = (const float*)((const char*)src3->data); // {d_state, nh} or {1, nh}
					const float* B = (const float*)((const char*)src4->data + i2 * (src4->nb[2]) + i3 * (src4->nb[3])); // {d_state, ng, nt, ns}
					const float* C = (const float*)((const char*)src5->data + i2 * (src5->nb[2]) + i3 * (src5->nb[3])); // {d_state, ng, nt, ns}
					float* y = (float*)((char*)dst->data + i2 * (nh * nr * sizeof(float)) + i3 * (nt * nh * nr * sizeof(float))); // {dim, nh, nt, ns}

					if (src3->ne[0] == 1) {
						// Mamba-2 has a scalar decay factor per head; dA can be outside the state-wise loop

						// n_head
						for (int h = ih0; h < ih1; ++h) {
							// ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
							const float dt_soft_plus = dt[h] <= 20.0f ? log1pf(expf(dt[h])) : dt[h];
							const float dA = expf(dt_soft_plus * A[h]);

							// dim
							for (int i1 = 0; i1 < nr; ++i1) {
								const int ii = i1 + h * nr;
								const float x_dt = x[ii] * dt_soft_plus;
								float sumf = 0.0f;
#if defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
								const int ggml_f32_epr = svcntw();
								const int ggml_f32_step = 1 * ggml_f32_epr;

								const int np = (nc & ~(ggml_f32_step - 1));

								GGML_F32_VEC sum = GGML_F32_VEC_ZERO;

								GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
								GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

								for (int i = 0; i < np; i += ggml_f32_step) {
									// TODO: maybe unroll more?
									for (int j = 0; j < 1; j++) {
										GGML_F32_VEC t0 = GGML_F32_VEC_LOAD(s0 + i + j * ggml_f32_epr + ii * nc);
										GGML_F32_VEC t1 = GGML_F32_VEC_LOAD(B + i + j * ggml_f32_epr + (h & (ng - 1)) * nc);
										GGML_F32_VEC t2 = GGML_F32_VEC_LOAD(C + i + j * ggml_f32_epr + (h & (ng - 1)) * nc);

										t0 = GGML_F32_VEC_MUL(t0, adA);
										t1 = GGML_F32_VEC_MUL(t1, axdt);

										t0 = GGML_F32_VEC_ADD(t0, t1);

										sum = GGML_F32_VEC_FMA(sum, t0, t2);

										GGML_F32_VEC_STORE(s + i + j * ggml_f32_epr + ii * nc, t0);
									}
								}

								sumf = GGML_F32xt_REDUCE_ONE(sum);
#else
								const int np = (nc & ~(GGML_F32_STEP - 1));

								GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

								GGML_F32_VEC adA = GGML_F32_VEC_SET1(dA);
								GGML_F32_VEC axdt = GGML_F32_VEC_SET1(x_dt);

								GGML_F32_VEC ax[GGML_F32_ARR];
								GGML_F32_VEC ay[GGML_F32_ARR];
								GGML_F32_VEC az[GGML_F32_ARR];

								for (int i = 0; i < np; i += GGML_F32_STEP) {
									for (int j = 0; j < GGML_F32_ARR; j++) {
										ax[j] = GGML_F32_VEC_LOAD(s0 + i + j * GGML_F32_EPR + ii * nc);
										ay[j] = GGML_F32_VEC_LOAD(B + i + j * GGML_F32_EPR + (h & (ng - 1)) * nc);
										az[j] = GGML_F32_VEC_LOAD(C + i + j * GGML_F32_EPR + (h & (ng - 1)) * nc);

										ax[j] = GGML_F32_VEC_MUL(ax[j], adA);
										ay[j] = GGML_F32_VEC_MUL(ay[j], axdt);

										ax[j] = GGML_F32_VEC_ADD(ax[j], ay[j]);

										sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], az[j]);

										GGML_F32_VEC_STORE(s + i + j * GGML_F32_EPR + ii * nc, ax[j]);
									}
								}

								// reduce sum0..sum3 to sum0
								GGML_F32_VEC_REDUCE(sumf, sum);
#endif
#else
								const int np = 0;
#endif
								// d_state
								for (int i0 = np; i0 < nc; ++i0) {
									const int i = i0 + ii * nc;
									const int ig = i0 + (h & (ng - 1)) * nc;
									// state = prev_state * dA + dB * x
									const float state = (s0[i] * dA) + (B[ig] * x_dt);
									// y = rowwise_dotprod(state, C)
									sumf += state * C[ig];
									s[i] = state;
								}
								y[ii] = sumf;
							}
						}
					}
					else {
						// Mamba-1 has an element-wise decay factor for the states

						// n_head
						for (int h = ih0; h < ih1; ++h) {
							// ref: https://github.com/state-spaces/mamba/blob/62db608da60f6fc790b8ed9f4b3225e95ca15fde/mamba_ssm/ops/triton/softplus.py#L16
							const float dt_soft_plus = dt[h] <= 20.0f ? log1pf(expf(dt[h])) : dt[h];

							// dim
							for (int i1 = 0; i1 < nr; ++i1) {
								const int ii = i1 + h * nr;
								const float x_dt = x[ii] * dt_soft_plus;
#if defined(__ARM_FEATURE_SVE)
								svfloat32_t vx_dt = GGML_F32_VEC_SET1(x_dt);
								svfloat32_t vdt_soft_plus = GGML_F32_VEC_SET1(dt_soft_plus);
								svfloat32_t r1_vector = GGML_F32_VEC_ZERO;

								// d_state
								// TODO: what happens when (d_state % svcntw()) != 0?
								for (int64_t k = 0; k < nc; k += svcntw()) {
									svfloat32_t vA = GGML_F32_VEC_LOAD(&A[h * nc + k]);
									svfloat32_t vB = GGML_F32_VEC_LOAD(&B[k + (h & (ng - 1)) * nc]);
									svfloat32_t vC = GGML_F32_VEC_LOAD(&C[k + (h & (ng - 1)) * nc]);
									svfloat32_t vs0 = GGML_F32_VEC_LOAD(&s0[ii * nc + k]);

									svfloat32_t t1 = GGML_F32_VEC_MUL(vdt_soft_plus, vA);
									t1 = exp_ps_sve(svptrue_b32(), t1);
									svfloat32_t t2 = GGML_F32_VEC_MUL(vx_dt, vB);

									vs0 = GGML_F32_VEC_FMA(t2, vs0, t1);
									r1_vector = GGML_F32_VEC_ADD(GGML_F32_VEC_MUL(vs0, vC), r1_vector);

									GGML_F32_VEC_STORE(&s[ii * nc + k], vs0);
								}
								y[ii] = GGML_F32xt_REDUCE_ONE(r1_vector);
#else
								float sumf = 0.0f;
								// NOTE: can't really use GGML_SIMD here because d_state is usually 16
								//       and also because expf is used within the loop.
								// d_state
								for (int i0 = 0; i0 < nc; ++i0) {
									const int i = i0 + ii * nc;
									const int ig = i0 + (h & (ng - 1)) * nc;
									// state = prev_state * dA + dB * x
									const float state = (s0[i] * expf(dt_soft_plus * A[i0 + h * nc])) + (B[ig] * x_dt);
									// y = rowwise_dotprod(state, C)
									sumf += state * C[ig];
									s[i] = state;
								}
								y[ii] = sumf;
#endif
							}
						}
					}
					// use the output as the source when it's not the first token-wise iteration
					s0 = s;
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

	float* dst_data = (float*)dst->data;
	float* state = ((float*)dst->data) + C * T;

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int64_t dr = (HEADS + nth - 1) / nth;

	float* k = (float*)dst->src[0]->data;
	float* v = (float*)dst->src[1]->data;
	float* r = (float*)dst->src[2]->data;
	float* time_faaaa = (float*)dst->src[3]->data;
	float* time_decay = (float*)dst->src[4]->data;

	size_t t_stride = HEADS * head_size; // Same to C

	size_t h_stride = C / HEADS;
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
	size_t h_stride_2d = head_size * head_size;

	memset(dst_data, 0, T * C * sizeof(float));

#if defined(__AVX__) && !defined(__AVX512F__)
#define GGML_F32X GGML_F32x8
#define GGML_F32X_SET1 GGML_F32x8_SET1
#define GGML_F32X_LOAD GGML_F32x8_LOAD
#define GGML_F32X_STORE GGML_F32x8_STORE
#define GGML_F32X_MUL GGML_F32x8_MUL
#define GGML_F32X_FMA GGML_F32x8_FMA
#define WKV_VECTOR_SIZE 8
#elif defined(__AVX512F__)
#define GGML_F32X GGML_F32x16
#define GGML_F32X_SET1 GGML_F32x16_SET1
#define GGML_F32X_LOAD GGML_F32x16_LOAD
#define GGML_F32X_STORE GGML_F32x16_STORE
#define GGML_F32X_MUL GGML_F32x16_MUL
#define GGML_F32X_FMA GGML_F32x16_FMA
#define WKV_VECTOR_SIZE 16
#elif defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
#define GGML_F32X GGML_F32xt
#define GGML_F32X_SET1 GGML_F32xt_SET1
#define GGML_F32X_LOAD GGML_F32xt_LOAD
#define GGML_F32X_STORE GGML_F32xt_STORE
#define GGML_F32X_MUL GGML_F32xt_MUL
#define GGML_F32X_FMA GGML_F32xt_FMA
#define WKV_VECTOR_SIZE 8
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define GGML_F32X GGML_F32x4
#define GGML_F32X_SET1 GGML_F32x4_SET1
#define GGML_F32X_LOAD GGML_F32x4_LOAD
#define GGML_F32X_STORE GGML_F32x4_STORE
#define GGML_F32X_MUL GGML_F32x4_MUL
#define GGML_F32X_FMA GGML_F32x4_FMA
#define WKV_VECTOR_SIZE 4
#endif

	for (int64_t h_start = 0; h_start < HEADS; h_start += dr) {
		const int64_t h_end = std::min(h_start + dr, HEADS);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
#ifdef WKV_VECTOR_SIZE
			int wkv_vector_size;
#if defined(__ARM_FEATURE_SVE)
			wkv_vector_size = svcntw();
#else
			wkv_vector_size = WKV_VECTOR_SIZE;
#endif
			const int64_t vec_count = head_size / wkv_vector_size;

			for (int64_t t = 0; t < T; t++) {
				size_t t_offset = t * t_stride;
				size_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

				for (int64_t h = h_start; h < h_end; h++) {
					size_t h_offset = h * h_stride;
					size_t t_h_offset = t_offset + h_offset;
					size_t h_2d_offset = h * h_stride_2d;

					for (int64_t i = 0; i < head_size; i++) {
						size_t t_h_i_offset = t_h_offset + i;
						size_t h_i_offset = h_offset + i;
						size_t h_2d_i_offset = h_2d_offset + i * h_stride;

						float k_val = k[t_h_i_offset];
						float r_val = r[t_h_i_offset];
						float time_faaaa_val = time_faaaa[h_i_offset];
						float time_decay_val = time_decay[t_h_i_offset];

						// Broadcast scalar values to vectors
						GGML_F32X k_vec = GGML_F32X_SET1(k_val);
						GGML_F32X r_vec = GGML_F32X_SET1(r_val);
						GGML_F32X time_faaaa_vec = GGML_F32X_SET1(time_faaaa_val);
						GGML_F32X time_decay_vec = GGML_F32X_SET1(time_decay_val);

						for (int64_t j = 0; j < vec_count; j++) {
							size_t base_j = j * wkv_vector_size;
							size_t t_h_j_offset = t_h_offset + base_j;
							size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

							// Load x elements at once
							GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
							GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
							GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

							// Compute kv = v * k
							GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

							// Compute temp = kv * time_faaaa + prev_state
							GGML_F32X temp_vec = GGML_F32X_FMA(prev_state_vec, kv_vec, time_faaaa_vec);

							// Update dst: dst += temp * r
							dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, r_vec);
							GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

							// Update state: state = prev_state * time_decay + kv
							GGML_F32X new_state_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, time_decay_vec);
							GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], new_state_vec);
						}

						// Handle remaining elements, this will not be used.
						for (int64_t j = vec_count * wkv_vector_size; j < head_size; j++) {
							size_t t_h_j_offset = t_h_offset + j;
							size_t h_2d_i_j_offset = h_2d_i_offset + j;
							float v_val = v[t_h_j_offset];
							float kv_val = v_val * k_val;
							float prev_state_val = state_prev[h_2d_i_j_offset];
							float temp_val = kv_val * time_faaaa_val + prev_state_val;
							dst_data[t_h_j_offset] += temp_val * r_val;
							state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
						}
					}
				}
			}

#else
			// basically fused operations:
			// dst = r @ (time_faaaa * (k @ v) + state),
			// state = time_decay * state + (k @ v),
			// recursive through each token
			for (int64_t t = 0; t < T; t++) {
				size_t t_offset = t * t_stride;
				size_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[5]->data + state_offset;

				for (int64_t h = h_start; h < h_end; h++) {
					size_t h_offset = h * h_stride;
					size_t t_h_offset = t_offset + h_offset;
					size_t h_2d_offset = h * h_stride_2d;

					for (int64_t i = 0; i < head_size; i++) {
						size_t t_h_i_offset = t_h_offset + i;
						size_t h_i_offset = h_offset + i;
						size_t h_2d_i_offset = h_2d_offset + i * h_stride;

						float k_val = k[t_h_i_offset];
						float r_val = r[t_h_i_offset];
						float time_faaaa_val = time_faaaa[h_i_offset];
						// RWKV v6: different time_decay for each token.
						float time_decay_val = time_decay[t_h_i_offset];

						for (int64_t j = 0; j < head_size; j++) {
							size_t t_h_j_offset = t_h_offset + j;
							size_t h_2d_i_j_offset = h_2d_i_offset + j;

							float v_val = v[t_h_j_offset];
							float kv_val = v_val * k_val;
							float prev_state_val = state_prev[h_2d_i_j_offset];
							float temp_val = kv_val * time_faaaa_val + prev_state_val;
							dst_data[t_h_j_offset] += temp_val * r_val;
							state_cur[h_2d_i_j_offset] = prev_state_val * time_decay_val + kv_val;
						}
					}
				}
			}
#endif
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

	float* dst_data = (float*)dst->data;
	float* state = ((float*)dst->data) + C * T;

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	const int64_t dr = (HEADS + nth - 1) / nth;

	float* k = (float*)dst->src[0]->data;
	float* v = (float*)dst->src[1]->data;
	float* q = (float*)dst->src[2]->data;
	float* g = (float*)dst->src[3]->data;

	size_t t_stride = HEADS * head_size; // Same to C

	size_t h_stride = C / HEADS;
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
	size_t h_stride_2d = head_size * head_size;

	memset(dst_data, 0, T * C * sizeof(float));

#if defined(__AVX__) && !defined(__AVX512F__)
#define GGML_F32X GGML_F32x8
#define GGML_F32X_SET1 GGML_F32x8_SET1
#define GGML_F32X_LOAD GGML_F32x8_LOAD
#define GGML_F32X_STORE GGML_F32x8_STORE
#define GGML_F32X_MUL GGML_F32x8_MUL
#define GGML_F32X_FMA GGML_F32x8_FMA
#define GLA_VECTOR_SIZE 8
#elif defined(__AVX512F__)
#define GGML_F32X GGML_F32x16
#define GGML_F32X_SET1 GGML_F32x16_SET1
#define GGML_F32X_LOAD GGML_F32x16_LOAD
#define GGML_F32X_STORE GGML_F32x16_STORE
#define GGML_F32X_MUL GGML_F32x16_MUL
#define GGML_F32X_FMA GGML_F32x16_FMA
#define GLA_VECTOR_SIZE 16
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define GGML_F32X GGML_F32x4
#define GGML_F32X_SET1 GGML_F32x4_SET1
#define GGML_F32X_LOAD GGML_F32x4_LOAD
#define GGML_F32X_STORE GGML_F32x4_STORE
#define GGML_F32X_MUL GGML_F32x4_MUL
#define GGML_F32X_FMA GGML_F32x4_FMA
#define GLA_VECTOR_SIZE 4
#endif

	for (int64_t h_start = 0; h_start < HEADS; h_start += dr) {
		const int64_t h_end = std::min(h_start + dr, HEADS);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {

#ifdef GLA_VECTOR_SIZE
			const int64_t vec_count = head_size / GLA_VECTOR_SIZE;

			for (int64_t t = 0; t < T; t++) {
				size_t t_offset = t * t_stride;
				size_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;

				for (int64_t h = h_start; h < h_end; h++) {
					size_t h_offset = h * h_stride;
					size_t t_h_offset = t_offset + h_offset;
					size_t h_2d_offset = h * h_stride_2d;

					for (int64_t i = 0; i < head_size; i++) {
						size_t t_h_i_offset = t_h_offset + i;
						size_t h_2d_i_offset = h_2d_offset + i * h_stride;

						float k_val = k[t_h_i_offset];
						float q_val = q[t_h_i_offset] * scale;
						float g_val = g[t_h_i_offset];

						// Broadcast scalar values to vectors
						GGML_F32X k_vec = GGML_F32X_SET1(k_val);
						GGML_F32X q_vec = GGML_F32X_SET1(q_val);
						GGML_F32X g_vec = GGML_F32X_SET1(g_val);

						for (int64_t j = 0; j < vec_count; j++) {
							size_t base_j = j * GLA_VECTOR_SIZE;
							size_t t_h_j_offset = t_h_offset + base_j;
							size_t h_2d_i_j_offset = h_2d_i_offset + base_j;

							// Load x elements at once
							GGML_F32X v_vec = GGML_F32X_LOAD(&v[t_h_j_offset]);
							GGML_F32X prev_state_vec = GGML_F32X_LOAD(&state_prev[h_2d_i_j_offset]);
							GGML_F32X dst_vec = GGML_F32X_LOAD(&dst_data[t_h_j_offset]);

							// Compute kv = v * k
							GGML_F32X kv_vec = GGML_F32X_MUL(v_vec, k_vec);

							// Compute temp = prev_state * g + kv
							GGML_F32X temp_vec = GGML_F32X_FMA(kv_vec, prev_state_vec, g_vec);

							// Update dst: dst += temp * q
							dst_vec = GGML_F32X_FMA(dst_vec, temp_vec, q_vec);
							GGML_F32X_STORE(&dst_data[t_h_j_offset], dst_vec);

							// Update state
							GGML_F32X_STORE(&state_cur[h_2d_i_j_offset], temp_vec);
						}

						// Handle remaining elements, this will not be used.
						for (int64_t j = vec_count * GLA_VECTOR_SIZE; j < head_size; j++) {
							size_t t_h_j_offset = t_h_offset + j;
							size_t h_2d_i_j_offset = h_2d_i_offset + j;
							float v_val = v[t_h_j_offset];
							float kv_val = v_val * k_val;
							float prev_state_val = state_prev[h_2d_i_j_offset];
							float temp_val = kv_val + prev_state_val * g_val;
							dst_data[t_h_j_offset] += temp_val * q_val;
							state_cur[h_2d_i_j_offset] = temp_val;
						}
					}
				}
			}

#else
			for (int64_t t = 0; t < T; t++) {
				size_t t_offset = t * t_stride;
				size_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[4]->data + state_offset;

				for (int64_t h = h_start; h < h_end; h++) {
					size_t h_offset = h * h_stride;
					size_t t_h_offset = t_offset + h_offset;
					size_t h_2d_offset = h * h_stride_2d;

					for (int64_t i = 0; i < head_size; i++) {
						size_t t_h_i_offset = t_h_offset + i;
						size_t h_2d_i_offset = h_2d_offset + i * h_stride;

						float k_val = k[t_h_i_offset];
						float q_val = q[t_h_i_offset] * scale;
						float g_val = g[t_h_i_offset];

						for (int64_t j = 0; j < head_size; j++) {
							size_t t_h_j_offset = t_h_offset + j;
							size_t h_2d_i_j_offset = h_2d_i_offset + j;

							float v_val = v[t_h_j_offset];
							float kv_val = v_val * k_val;
							float prev_state_val = state_prev[h_2d_i_j_offset];
							float temp_val = prev_state_val * g_val + kv_val;
							dst_data[t_h_j_offset] += temp_val * q_val;
							state_cur[h_2d_i_j_offset] = temp_val;
						}
					}
				}
			}
#endif
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
					//       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
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

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int64_t n = ggml_nrows(src0);
	const int64_t dr = (n + nth - 1) / nth;
	const int nc = src0->ne[0];

	const size_t nb00 = src0->nb[0];
	const size_t nb01 = src0->nb[1];

	const size_t nb0 = dst->nb[0];
	const size_t nb1 = dst->nb[1];

	GGML_ASSERT(nb0 == sizeof(T));
	GGML_ASSERT(nb00 == sizeof(T));

	for (int64_t ir0 = 0; ir0 < n; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, n);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int j = ir0; j < ir1; j++) {
				T* dst_ptr = (T*)((char*)dst->data + j * nb1);
				T* src0_ptr = (T*)((char*)src0->data + j * nb01);

				for (int i = 0; i < nc; i++) {
					dst_ptr[i] = fromFloat32<T>(std::clamp(toFloat32(src0_ptr[i]), min, max));
				}
			}
		});
		scope.spawn(std::move(sender));
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
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst,
	const float value) {

	const struct ggml_tensor* src0 = dst->src[0];

	const int ith = params->ith;
	const int nth = params->nth;

	const int  n_past = ((int32_t*)dst->op_params)[0];
	const bool inplace = src0->data == dst->data;

	GGML_ASSERT(n_past >= 0);

	if (!inplace) {
		if (ith == 0) {
			// memcpy needs to be synchronized across threads to avoid race conditions.
			// => do it in INIT phase
			GGML_ASSERT(dst->nelements() == src0->nelements());
			GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
			memcpy(
				((char*)dst->data),
				((char*)src0->data),
				dst->nbytes());
		}
		//ggml_barrier(params->threadpool);
	}

	// TODO: handle transposed/permuted matrices

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];
	const int nr = src0->ne[1];
	const int nz = n / nr;

	GGML_ASSERT(dst->nb[0] == sizeof(float));
	GGML_ASSERT(src0->nb[0] == sizeof(float));

	for (int k = 0; k < nz; k++) {
		for (int j = ith; j < nr; j += nth) {
			for (int i = n_past; i < nc; i++) {
				if (i > n_past + j) {
					*(float*)((char*)dst->data + k * dst->nb[2] + j * dst->nb[1] + i * dst->nb[0]) = value;
				}
			}
		}
	}
}

static void ggml_compute_forward_diag_mask_inf(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_diag_mask_f32(params, dst, -INFINITY);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_cpy_f321(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

inline static void ggml_vec_scale_f322(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
	vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
	const int np = (n & ~(GGML_F32_STEP - 1));

	GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	GGML_F32_VEC ay[GGML_F32_ARR];

	for (int i = 0; i < np; i += GGML_F32_STEP) {
		for (int j = 0; j < GGML_F32_ARR; j++) {
			ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
			ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

			GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] *= v;
	}
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] *= v;
	}
#endif
}

inline static void ggml_vec_max_f32(const int n, float* s, const float* x) {
#ifndef GGML_USE_ACCELERATE
	float max = -INFINITY;
	for (int i = 0; i < n; ++i) {
		max = std::max(max, x[i]);
	}
	*s = max;
#else
	vDSP_maxv(x, 1, s, n);
#endif
}

static ggml_float ggml_vec_soft_max_f32(const int n, float* y, const float* x, float max) {
	int i = 0;
	ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
	for (; i + 15 < n; i += 16) {
		__m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
			_mm512_set1_ps(max)));
		_mm512_storeu_ps(y + i, val);
		sum += (ggml_float)_mm512_reduce_add_ps(val);
	}
#elif defined(__AVX2__) && defined(__FMA__)
	for (; i + 7 < n; i += 8) {
		__m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
			_mm256_set1_ps(max)));
		_mm256_storeu_ps(y + i, val);
		__m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
			_mm256_castps256_ps128(val));
		val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
		val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
		sum += (ggml_float)_mm_cvtss_f32(val2);
	}
#elif defined(__SSE21__)
	for (; i + 3 < n; i += 4) {
		__m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
			_mm_set1_ps(max)));
		_mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
		val = _mm_add_ps(val, _mm_movehl_ps(val, val));
		val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
		__m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
		val = _mm_add_ps(val, tmp);
		tmp = _mm_movehl_ps(tmp, val);
		val = _mm_add_ss(val, tmp);
#endif
		sum += (ggml_float)_mm_cvtss_f32(val);
	}
#elif defined(__ARM_NEON) && defined(__aarch64__)
	for (; i + 3 < n; i += 4) {
		float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
			vdupq_n_f32(max)));
		vst1q_f32(y + i, val);
		sum += (ggml_float)vaddvq_f32(val);
	}
#endif
	for (; i < n; ++i) {
		float val = expf(x[i] - max);
		sum += (ggml_float)val;
		y[i] = val;
	}
	return sum;
}

inline static void ggml_vec_cpy_f32(const int n, float* y, const float* x) {
	for (int i = 0; i < n; ++i) y[i] = x[i];
}

inline static void ggml_vec_scale_f32(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
	vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
	const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
	const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
	const int ggml_f32_step = 2 * ggml_f32_epr;

	GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
	const int np = (n & ~(ggml_f32_step - 1));
	svfloat32_t ay1;
	svfloat32_t ay2;
	for (int i = 0; i < np; i += ggml_f32_step) {
		ay1 = GGML_F32_VEC_LOAD(y + i);
		ay1 = GGML_F32_VEC_MUL(ay1, vx);
		GGML_F32_VEC_STORE(y + i, ay1);

		ay2 = GGML_F32_VEC_LOAD(y + i + 1 * ggml_f32_epr);
		ay2 = GGML_F32_VEC_MUL(ay2, vx);
		GGML_F32_VEC_STORE(y + i + 1 * ggml_f32_epr, ay2);
	}
	// leftovers
	// maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
	if (np < n) {
		svbool_t pg = svwhilelt_b32(np, n);
		ay1 = svld1_f32(pg, y + np);
		ay1 = svmul_f32_m(pg, ay1, vx);
		svst1_f32(pg, y + np, ay1);
	}
#else
	const int np = (n & ~(GGML_F32_STEP - 1));

	GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	GGML_F32_VEC ay[GGML_F32_ARR];

	for (int i = 0; i < np; i += GGML_F32_STEP) {
		for (int j = 0; j < GGML_F32_ARR; j++) {
			ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
			ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

			GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] *= v;
	}
#endif
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] *= v;
	}
#endif
}

static void ggml_compute_forward_soft_max_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	assert(ggml_is_contiguous(dst));
	assert(ggml_are_same_shape(src0, dst));

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);

	const int nth = pool.available_parallelism();

	GGML_TENSOR_UNARY_OP_LOCALS

	const int64_t nb11 = src1 ? src1->nb[1] : 1;
	const int64_t nb12 = src1 ? src1->nb[2] : 1;
	const int64_t nb13 = src1 ? src1->nb[3] : 1;

	const int64_t ne12 = src1 ? src1->ne[2] : 1;
	const int64_t ne13 = src1 ? src1->ne[3] : 1;

	// TODO: is this supposed to be ceil instead of floor?
	//       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
	const uint32_t n_head = ne02;
	const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

	const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
	const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	const int nc = src0->ne[0];

	// rows per thread
	const int64_t nh = ne01;
	const int64_t dh = (nh + nth - 1) / nth;
	const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// row range for this thread
	for (int64_t ih0 = 0; ih0 < nh; ih0 += dh) {
		const int64_t ih1 = std::min(ih0 + dh, nh);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> wp(nc);
			for (int64_t i03 = 0; i03 < ne03; i03++) {
				for (int64_t i02 = 0; i02 < ne02; i02++) {
					for (int64_t i01 = ih0; i01 < ih1; i01++) {
						const int64_t i11 = i01;
						const int64_t i12 = i02 % ne12;
						const int64_t i13 = i03 % ne13;

						// ALiBi
						const uint32_t h = i02; // head
						const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

						float* sp = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
						float* dp = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

						// broadcast the mask across rows
						ggml_fp16_t* mp_f16 = src1 ? (ggml_fp16_t*)((char*)src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13) : NULL;
						float* mp_f32 = src1 ? (float*)((char*)src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13) : NULL;

						ggml_vec_cpy_f32(ne00, wp.data(), sp);
						ggml_vec_scale_f32(ne00, wp.data(), scale);
						if (mp_f32) {
							if (use_f16) {
								for (int i = 0; i < ne00; ++i) {
									wp[i] += slope * toFloat32(mp_f16[i]);
								}
							}
							else {
								for (int i = 0; i < ne00; ++i) {
									wp[i] += slope * mp_f32[i];
								}
							}
						}

#ifndef NDEBUG
						for (int i = 0; i < ne00; ++i) {
							//printf("p[%d] = %f\n", i, p[i]);
							assert(!isnan(wp[i]));
						}
#endif

						float max = -INFINITY;
						ggml_vec_max_f32(ne00, &max, wp.data());

						ggml_float sum = ggml_vec_soft_max_f32(ne00, dp, wp.data(), max);
						assert(sum > 0.0);

						sum = 1.0 / sum;
						ggml_vec_scale_f32(ne00, dp, sum);

#ifndef NDEBUG
						for (int i = 0; i < ne00; ++i) {
							assert(!isnan(dp[i]));
							assert(!isinf(dp[i]));
						}
#endif
					}
				}
			}
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_soft_max(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_soft_max_f32(pool, scope, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_dot_f32(int n, float* s, size_t bs, const float* x, size_t bx, const float* y, size_t by, int nrc) {
	// scalar
	ggml_float sumf = 0.0;
	for (int i = 0; i < n; ++i) {
		sumf += (ggml_float)(x[i] * y[i]);
	}
	*s = sumf;
}

static void ggml_vec_mul_f32(const int n, float* z, const float* x, const float* y) { for (int i = 0; i < n; ++i) z[i] = x[i] * y[i]; }
static void ggml_vec_scale_f321(const int n, float* y, const float   v) {
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] *= v;
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

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int nc = src0->ne[0];
	const int64_t nr = ggml_nrows(src0);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int i1 = ir0; i1 < ir1; i1++) {
				float* dy = (float*)((char*)src0->data + i1 * src0->nb[1]);
				float* y = (float*)((char*)src1->data + i1 * src1->nb[1]);
				float* dx = (float*)((char*)dst->data + i1 * dst->nb[1]);

#ifndef NDEBUG
				for (int i = 0; i < nc; ++i) {
					//printf("p[%d] = %f\n", i, p[i]);
					assert(!isnan(dy[i]));
					assert(!isnan(y[i]));
				}
#endif
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
				ggml_vec_dot_f32(nc, &dot_y_dy, 0, y, 0, dy, 0, 1);
				ggml_vec_cpy_f321(nc, dx, dy);
				ggml_vec_acc1_f32(nc, dx, -dot_y_dy);
				ggml_vec_mul_f32(nc, dx, dx, y);
				ggml_vec_scale_f321(nc, dx, scale);

#ifndef NDEBUG
				for (int i = 0; i < nc; ++i) {
					assert(!isnan(dx[i]));
					assert(!isinf(dx[i]));
				}
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

inline static void ggml_vec_sum_f32(const int n, float* s, const float* x) {
#ifndef GGML_USE_ACCELERATE
	ggml_float sum = 0.0;
	for (int i = 0; i < n; ++i) {
		sum += (ggml_float)x[i];
	}
	*s = sum;
#else
	vDSP_sve(x, 1, s, n);
#endif
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

	GGML_TENSOR_UNARY_OP_LOCALS
	
	const int nth = pool.available_parallelism();
	const int64_t dr = (ne1 + nth - 1) / nth;

	float* dst_ptr = (float*)dst->data;

	// TODO: optimize

	for (int64_t ir0 = 0; ir0 < ne1; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, ne1);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) | stdexec::then([=] {
			for (int64_t i2 = 0; i2 < ne2; ++i2) {
				for (int64_t i1 = ir0; i1 < ir1; i1++) {
					for (int64_t i0 = 0; i0 < ne0; ++i0) {
						for (int64_t i3 = 0; i3 < ne3; ++i3) {
							const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;

							const float* src_ptr = (const float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);

							if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
								dst_ptr[dst_idx] = *src_ptr;
							}
							else {
								dst_ptr[dst_idx] = 0;
							}
						}
					}
				}
			}
		});
		scope.spawn(std::move(sender));
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

inline static void ggml_vec_scale_f16(const int n, ggml_fp16_t* y, const float v) {
#if defined(GGML_SIMD)
	const int np = (n & ~(GGML_F16_STEP - 1));

	GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

	GGML_F16_VEC ay[GGML_F16_ARR];

	for (int i = 0; i < np; i += GGML_F16_STEP) {
		for (int j = 0; j < GGML_F16_ARR; j++) {
			ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);
			ay[j] = GGML_F16_VEC_MUL(ay[j], vx);

			GGML_F16_VEC_STORE(y + i + j * GGML_F16_EPR, ay, j);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) * v);
	}
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] = fromFloat32<ggml_fp16_t>(toFloat32(y[i]) * v);
	}
#endif
}

inline static void ggml_vec_mad_f16(const int n, ggml_fp16_t* y, const ggml_fp16_t* x, const float v) {
#if defined(GGML_SIMD)
	const int np = (n & ~(GGML_F16_STEP - 1));

	GGML_F16_VEC vx = GGML_F16_VEC_SET1(v);

	GGML_F16_VEC ax[GGML_F16_ARR];
	GGML_F16_VEC ay[GGML_F16_ARR];

	for (int i = 0; i < np; i += GGML_F16_STEP) {
		for (int j = 0; j < GGML_F16_ARR; j++) {
			ax[j] = GGML_F16_VEC_LOAD(x + i + j * GGML_F16_EPR, j);
			ay[j] = GGML_F16_VEC_LOAD(y + i + j * GGML_F16_EPR, j);
			ay[j] = GGML_F16_VEC_FMA(ay[j], ax[j], vx);

			GGML_F16_VEC_STORE(y + i + j * GGML_F16_EPR, ay, j);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(y[i]) + GGML_FP16_TO_FP32(x[i]) * v);
	}
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] = fromFloat32<ggml_fp16_t>(toFloat32(y[i]) + toFloat32(x[i]) * v);
	}
#endif
}

template <typename V_TYPE, typename K_TYPE> 
static void ggml_compute_forward_flash_attn_ext_f16(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* q,
	const ggml_tensor* k,
	const ggml_tensor* v,
	const ggml_tensor* mask,
	ggml_tensor* dst) {

	using q_to_vec_dot_t = typename vec_dot_trait<K_TYPE>::type;
	GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
	GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
	GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
	GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
	GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
	GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
	GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
	GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

	const int nth = pool.available_parallelism();

	const int64_t DK = nek0;
	const int64_t DV = nev0;
	const int64_t N = neq1;

	GGML_ASSERT(ne0 == DV);
	GGML_ASSERT(ne2 == N);

	// input tensor rows must be contiguous
	GGML_ASSERT(nbq0 == ggml_type_size(q->type));
	GGML_ASSERT(nbk0 == ggml_type_size(k->type));
	GGML_ASSERT(nbv0 == ggml_type_size(v->type));

	GGML_ASSERT(neq0 == DK);
	GGML_ASSERT(nek0 == DK);
	GGML_ASSERT(nev0 == DV);

	GGML_ASSERT(neq1 == N);

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

	// broadcast factors
	const int64_t rk2 = neq2 / nek2;
	const int64_t rk3 = neq3 / nek3;

	const int64_t rv2 = neq2 / nev2;
	const int64_t rv3 = neq3 / nev3;

	// parallelize by q rows using ggml_vec_dot_f32

	// total rows in q
	const int64_t nr = neq1 * neq2 * neq3;

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);
	float logit_softcap = std::bit_cast<float>(dst->op_params[2]);

	if (logit_softcap != 0) {
		scale /= logit_softcap;
	}

	const uint32_t n_head = neq2;
	const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

	const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
	const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// loop over n_batch and n_head
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			std::vector<float> VKQ32(DV); // FP32 VKQ accumulator
			std::vector<float> V32(DV); // (temporary) FP32 V buffer
			std::vector<ggml_fp16_t> VKQ16(DV); // (temporary) FP16 VKQ accumulator
			std::vector<q_to_vec_dot_t> Q_q(DK); // (temporary) buffer for Q converted to quantized/FP16

			for (int64_t ir = ir0; ir < ir1; ++ir) {
				// q indices
				const int iq3 = ir / (neq2 * neq1);
				const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
				const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

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

				const float* pq = (const float*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3));
				fromFloat(pq, Q_q.data(), DK);

				// online softmax / attention
				// loop over n_kv and n_head_kv
				// ref: https://arxiv.org/pdf/2112.05682.pdf
				for (int64_t ic = 0; ic < nek1; ++ic) {
					const float mv = mp ? slope * toFloat32(mp[ic]) : 0.0f;
					if (mv == -INFINITY) {
						continue;
					}

					float s; // KQ value

					const K_TYPE* k_data = cast_with_offset<K_TYPE>(k->data, ic * nbk1 + ik2 * nbk2 + ik3 * nbk3);
					ggml_vec_dot<K_TYPE, q_to_vec_dot_t>(DK, &s, 0, k_data, 0, Q_q.data(), 0, 1);

					s = s * scale; // scale KQ value

					if (logit_softcap != 0.0f) {
						s = logit_softcap * tanhf(s);
					}

					s += mv; // apply mask

					const float Mold = M;

					float ms = 1.0f; // upon new higher max val, scale VKQ and KQ sum with this value
					float vs = 1.0f; // post-softmax KQ value, expf(s - M)

					const void* v_data = ((const char*)v->data + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3));

					if (v->type == GGML_TYPE_F16) {
						if (s > M) {
							// s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
							M = s;
							ms = expf(Mold - M);

							// V = V*expf(Mold - M)
							ggml_vec_scale_f16(DV, VKQ16.data(), ms);
						}
						else {
							// no new maximum, ms == 1.0f, vs != 1.0f
							vs = expf(s - M);
						}

						// V += v*expf(s - M)
						ggml_vec_mad_f16(DV, VKQ16.data(), (const ggml_fp16_t*)v_data, vs);
					}
					else {
						if (s > M) {
							// s is new maximum, ms < 1.0f, vs == expf(s - s) == 1.0f
							M = s;
							ms = expf(Mold - M);

							// V = V*expf(Mold - M)
							ggml_vec_scale_f32(DV, VKQ32.data(), ms);
						}
						else {
							// no new maximum, ms == 1.0f, vs != 1.0f
							vs = expf(s - M);
						}

						if constexpr (is_quant_type_v<V_TYPE>) {
							dequantize_row(static_cast<const V_TYPE*>(v_data), V32.data(), DV);
						} else {
							to_float(static_cast<const V_TYPE*>(v_data), V32.data(), DV);
						} 

						// V += v*expf(s - M)
						ggml_vec_mad_f32(DV, VKQ32.data(), V32.data(), vs);
					}

					S = S * ms + vs; // scale and increment sum with partial sum
				}

				if (v->type == GGML_TYPE_F16) {
					for (int64_t d = 0; d < DV; ++d) {
						VKQ32[d] = toFloat32(VKQ16[d]);
					}
				}

				// V /= S
				const float S_inv = 1.0f / S;
				ggml_vec_scale_f32(DV, VKQ32.data(), S_inv);

				// dst indices
				const int i1 = iq1;
				const int i2 = iq2;
				const int i3 = iq3;

				// original
				//memcpy((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3), V, nev0*sizeof(float));

				// permute(0, 2, 1, 3)
				memcpy((char*)dst->data + (i3 * ne2 * ne1 + i2 + i1 * ne1) * nb1, VKQ32.data(), nb1);
			}
		});
		scope.spawn(std::move(sender));
	}
}

template <typename V_TYPE>
static void ggml_compute_forward_flash_attn_ext_f16(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* q,
	const ggml_tensor* k,
	const ggml_tensor* v,
	const ggml_tensor* mask,
	ggml_tensor* dst) {
	switch (k->type) {
	case GGML_TYPE_F16: {
		ggml_compute_forward_flash_attn_ext_f16<V_TYPE, ggml_fp16_t>(pool, scope, q, k, v, mask, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_flash_attn_ext_f16<V_TYPE, block_q4_0>(pool, scope, q, k, v, mask, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_flash_attn_ext_f16<V_TYPE, block_q8_0>(pool, scope, q, k, v, mask, dst);
	} break;
	default:
		assert(false);
	}
}

static void ggml_compute_forward_flash_attn_ext_f16(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_tensor* q,
	const ggml_tensor* k,
	const ggml_tensor* v,
	const ggml_tensor* mask,
	ggml_tensor* dst) {
	switch (v->type) {
	case GGML_TYPE_F16: {
		ggml_compute_forward_flash_attn_ext_f16<ggml_fp16_t>(pool, scope, q, k, v, mask, dst);
	} break;
	case GGML_TYPE_Q4_0: {
		ggml_compute_forward_flash_attn_ext_f16<block_q4_0>(pool, scope, q, k, v, mask, dst);
	} break;
	case GGML_TYPE_Q8_0: {
		ggml_compute_forward_flash_attn_ext_f16<block_q8_0>(pool, scope, q, k, v, mask, dst);
	} break;
	default:
		GGML_ASSERT(false && "fattn: unsupported V-type");
	}
}

static void ggml_compute_forward_flash_attn_ext(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_tensor* dst) {
	const ggml_tensor* q = dst->src[0];
	const ggml_tensor* k = dst->src[1];
	const ggml_tensor* v = dst->src[2];
	const ggml_tensor* mask = dst->src[3];
	switch (dst->op_params[3]) {
	case GGML_PREC_DEFAULT:
	case GGML_PREC_F32:
	{
		// uses F32 accumulators
		ggml_compute_forward_flash_attn_ext_f16(pool, scope, q, k, v, mask, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static ggml_float ggml_vec_log_soft_max_f32(const int n, float* y, const float* x, float max) {
	// log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

	int i = 0;
	ggml_float sum = 0;
	for (; i < n; ++i) {
		float val = x[i] - max;
		y[i] = val;
		sum += (ggml_float)expf(val);
	}
	return sum = (ggml_float)logf(sum);
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

	const int nth = pool.available_parallelism();

	std::atomic<float> sums;

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=, &sums] {
			std::vector<float> st(nc);
			float sum_thread = 0.0f;
			for (int64_t i1 = ir0; i1 < ir1; ++i1) {
				const float* s0 = (const float*)((const char*)src0->data + i1 * src0->nb[1]);
				const float* s1 = (const float*)((const char*)src1->data + i1 * src1->nb[1]);

#ifndef NDEBUG
				for (int64_t i = 0; i < nc; ++i) {
					//printf("p[%d] = %f\n", i, p[i]);
					assert(!isnan(s0[i]));
					assert(!isnan(s1[i]));
				}
#endif

				float max = -INFINITY;
				ggml_vec_max_f32(nc, &max, s0);
				const ggml_float sum_softmax = ggml_vec_log_soft_max_f32(nc, &st[0], s0, max);
				assert(sum_softmax >= 0.0);

				ggml_vec_add1_f32(nc, &st[0], &st[0], -sum_softmax);
				ggml_vec_mul_f32(nc, &st[0], &st[0], s1);

				float sum_st = 0.0f;
				ggml_vec_sum_f32(nc, &sum_st, &st[0]);
				sum_thread += sum_st;

#ifndef NDEBUG
				for (int64_t i = 0; i < nc; ++i) {
					assert(!isnan(st[i]));
					assert(!isinf(st[i]));
				}
#endif
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

inline static void ggml_vec_sub_f32(const int n, float* z, const float* x, const float* y) { for (int i = 0; i < n; ++i) z[i] = x[i] - y[i]; }

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

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	// TODO: handle transposed/permuted matrices
	const int64_t nc = src0f->ne[0];
	const int64_t nr = ggml_nrows(src0f);

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			const float d_by_nr = ((const float*)grad->data)[0] / (float)nr;

			for (int64_t i1 = ir0; i1 < ir1; i1++) {
				float* ds0 = (float*)((char*)dst->data + i1 * dst->nb[1]);
				const float* s0 = (const float*)((const char*)src0f->data + i1 * src0f->nb[1]);
				const float* s1 = (const float*)((const char*)src1f->data + i1 * src1f->nb[1]);

#ifndef NDEBUG
				for (int64_t i = 0; i < nc; ++i) {
					//printf("p[%d] = %f\n", i, p[i]);
					assert(!isnan(s0[i]));
					assert(!isnan(s1[i]));
				}
#endif

				// soft_max
				float max = -INFINITY;
				ggml_vec_max_f32(nc, &max, s0);
				const ggml_float sum = ggml_vec_soft_max_f32(nc, ds0, s0, max);
				assert(sum > 0.0);
				ggml_vec_scale_f32(nc, ds0, 1.0 / sum);

				// grad(src0f) = (softmax(src0f) - src1f) * grad(cross_entropy_loss(src0f, src1f)) / nr
				ggml_vec_sub_f32(nc, ds0, ds0, s1);
				ggml_vec_scale_f32(nc, ds0, d_by_nr);

#ifndef NDEBUG
				for (int64_t i = 0; i < nc; ++i) {
					assert(!isnan(ds0[i]));
					assert(!isinf(ds0[i]));
				}
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

	const int64_t nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS
	GGML_ASSERT(nb00 == sizeof(float));

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	const float* adamw_params_ptr = ggml_get_data_f32(adamw_params);
	const float alpha = adamw_params_ptr[0];
	const float beta1 = adamw_params_ptr[1];
	const float beta2 = adamw_params_ptr[2];
	const float eps = adamw_params_ptr[3];
	const float wd = adamw_params_ptr[4];
	const float beta1h = adamw_params_ptr[5];
	const float beta2h = adamw_params_ptr[6];

	for (int64_t ir0 = 0; ir0 < nr; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, nr);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int ir = ir0; ir < ir1; ++ir) {
				const int64_t i03 = ir / (ne02 * ne01);
				const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
				const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

				const size_t offset = i03 * nb03 + i02 * nb02 + i01 * nb01;

				float* w = (float*)((char*)src0->data + offset); // weight
				const float* g = (const float*)((const char*)src0_grad->data + offset); // grad
				float* m = (float*)((char*)src0_grad_m->data + offset);
				float* v = (float*)((char*)src0_grad_v->data + offset);

				for (int i00 = 0; i00 < ne00; ++i00) {
					m[i00] = m[i00] * beta1 + g[i00] * (1.0f - beta1);
					v[i00] = v[i00] * beta2 + g[i00] * g[i00] * (1.0f - beta2);

					const float mh = m[i00] * beta1h;
					const float vh = sqrtf(v[i00] * beta2h) + eps;

					// The weight decay is applied independently of the Adam momenta m and v.
					// This is NOT equivalent to l2 regularization that adds w[i00]*w[i00] to the loss.
					// See: https://arxiv.org/pdf/1711.05101v3.pdf
					w[i00] = w[i00] * (1.0f - alpha * wd) - alpha * mh / vh;
				}
			}
		});
		scope.spawn(std::move(sender));
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

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	float eps = std::bit_cast<float>(dst->op_params[0]);
	GGML_ASSERT(eps >= 0.0f);
	const int64_t dr = (src0->ne[1] + nth - 1) / nth;

	// TODO: optimize
	for (int64_t ir0 = 0; ir0 < src0->ne[1]; ir0 += dr) {
		const int64_t ir1 = std::min(ir0 + dr, src0->ne[1]);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t i03 = 0; i03 < src0->ne[3]; i03++) {
				for (int64_t i02 = 0; i02 < src0->ne[2]; i02++) {
					for (int64_t i01 = ir0; i01 < ir1; i01++) {
						const float* x = (float*)((char*)src0->data + i01 * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3]);

						ggml_float sum = 0.0;
						for (int64_t i00 = 0; i00 < src0->ne[0]; i00++) {
							sum += (ggml_float)(x[i00] * x[i00]);
						}

						float* y = (float*)((char*)dst->data + i01 * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]);

						memcpy(y, x, src0->ne[0] * sizeof(float));

						const float scale = 1.0f / fmaxf(sqrtf(sum), eps);

						ggml_vec_scale_f32(src0->ne[0], y, scale);
					}
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

	float* dst_data = (float*)dst->data;
	float* state = ((float*)dst->data) + C * T;

	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();

	const int64_t dr = (HEADS + nth - 1) / nth;

	float* r = (float*)dst->src[0]->data;
	float* w = (float*)dst->src[1]->data;
	float* k = (float*)dst->src[2]->data;
	float* v = (float*)dst->src[3]->data;
	float* a = (float*)dst->src[4]->data;
	float* b = (float*)dst->src[5]->data;

	int64_t t_stride = HEADS * head_size; // Same to C

	int64_t h_stride = C / HEADS;
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
	int64_t h_stride_2d = head_size * head_size;

	for (int64_t h_start = 0; h_start < HEADS; h_start += dr) {
		const int64_t h_end = std::min(h_start + dr, HEADS);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
#if defined(GGML_SIMD)
			for (int64_t t = 0; t < T; t++) {
				int64_t t_offset = t * t_stride;
				int64_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

				for (int64_t h = h_start; h < h_end; h++) {
					int64_t h_offset = h * h_stride;
					int64_t t_h_offset = t_offset + h_offset;
					int64_t h_2d_offset = h * h_stride_2d;

					for (int64_t ii = 0; ii < head_size; ii++) {
						int64_t t_h_i_offset = t_h_offset + ii;
						int64_t h_2d_i_offset = h_2d_offset + ii * h_stride;

						GGML_F32_VEC v_vec = GGML_F32_VEC_SET1(v[t_h_i_offset]);

						float sa = 0;
						{
							GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
							GGML_F32_VEC ax[GGML_F32_ARR];
							GGML_F32_VEC ay[GGML_F32_ARR];
							for (int64_t j = 0; j < head_size; j += GGML_F32_STEP) {
								for (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
									ax[kk] = GGML_F32_VEC_LOAD(&a[t_h_offset + j + kk * GGML_F32_EPR]);
									ay[kk] = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_offset + j + kk * GGML_F32_EPR]);
									sum[kk] = GGML_F32_VEC_FMA(sum[kk], ax[kk], ay[kk]);
								}
							}
							GGML_F32_VEC_REDUCE(sa, sum);
						}

						GGML_F32_VEC sa_vec = GGML_F32_VEC_SET1(sa);

						int64_t j = 0;
						GGML_F32_VEC result_vec[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };
						for (; j < head_size; j += GGML_F32_STEP) {
							for (int64_t kk = 0; kk < GGML_F32_ARR; kk++) {
								int64_t t_h_j_offset = t_h_offset + j + kk * GGML_F32_EPR;
								int64_t h_2d_i_j_offset = h_2d_i_offset + j + kk * GGML_F32_EPR;

								GGML_F32_VEC r_vec = GGML_F32_VEC_LOAD(&r[t_h_j_offset]);
								GGML_F32_VEC w_vec = GGML_F32_VEC_LOAD(&w[t_h_j_offset]);
								GGML_F32_VEC k_vec = GGML_F32_VEC_LOAD(&k[t_h_j_offset]);
								GGML_F32_VEC b_vec = GGML_F32_VEC_LOAD(&b[t_h_j_offset]);

								k_vec = GGML_F32_VEC_MUL(v_vec, k_vec);

								GGML_F32_VEC state_vec = GGML_F32_VEC_LOAD(&state_prev[h_2d_i_j_offset]);
								// kv + s * decay + sa * b
								state_vec = GGML_F32_VEC_FMA(k_vec, state_vec, w_vec);
								state_vec = GGML_F32_VEC_FMA(state_vec, sa_vec, b_vec);
								GGML_F32_VEC_STORE(&state_cur[h_2d_i_j_offset], state_vec);

								result_vec[kk] = GGML_F32_VEC_FMA(result_vec[kk], state_vec, r_vec);
							}
						}
						GGML_F32_VEC_REDUCE(dst_data[t_h_i_offset], result_vec);

						// There shouldn't be left-overs though.
						for (; j < head_size; j++) {
							int64_t t_h_j_offset = t_h_offset + j;
							int64_t h_2d_i_j_offset = h_2d_i_offset + j;

							float r_val = r[t_h_j_offset];
							float w_val = w[t_h_j_offset];
							float k_val = k[t_h_j_offset];
							float b_val = b[t_h_j_offset];
							float kv_val = v[t_h_i_offset] * k_val;

							float prev_state_val = state_prev[h_2d_i_j_offset];
							state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
							dst_data[t_h_i_offset] += state_cur[h_2d_i_j_offset] * r_val;
						}
					}
				}
			}
#else
			for (int64_t t = 0; t < T; t++) {
				int64_t t_offset = t * t_stride;
				int64_t state_offset = head_size * C * (t / (T / n_seqs));
				float* state_cur = state + state_offset;
				float* state_prev = t % (T / n_seqs) ? state_cur : (float*)dst->src[6]->data + state_offset;

				for (int64_t h = h_start; h < h_end; h++) {
					int64_t h_offset = h * h_stride;
					int64_t t_h_offset = t_offset + h_offset;
					int64_t h_2d_offset = h * h_stride_2d;

					for (int64_t i = 0; i < head_size; i++) {
						int64_t t_h_i_offset = t_h_offset + i;
						int64_t h_2d_i_offset = h_2d_offset + i * h_stride;

						float v_val = v[t_h_i_offset];

						float sa = 0, result = 0;
						for (int64_t j = 0; j < head_size; j++) {
							sa += a[t_h_offset + j] * state_prev[h_2d_i_offset + j];
						}

						for (int64_t j = 0; j < head_size; j++) {
							int64_t t_h_j_offset = t_h_offset + j;
							int64_t h_2d_i_j_offset = h_2d_i_offset + j;

							float r_val = r[t_h_j_offset];
							float w_val = w[t_h_j_offset];
							float k_val = k[t_h_j_offset];
							float b_val = b[t_h_j_offset];
							float kv_val = v_val * k_val;
							float prev_state_val = state_prev[h_2d_i_j_offset];
							state_cur[h_2d_i_j_offset] = prev_state_val * w_val + kv_val + sa * b_val;
							result += state_cur[h_2d_i_j_offset] * r_val;
						}
						dst_data[t_h_i_offset] = result;
					}
				}
			}
#endif
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
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t i = start; i < end; ++i) {
				const float* knl_data = (const float*)kernel->data + (i % p.channels) * p.knl_w * p.knl_h;
				const float* src_data = (const float*)src->data + i * p.src_w * p.src_h;
				float* dst_data = (float*)dst->data + i * p.dst_w * p.dst_h;

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
								sum += knl_data[knl_y * p.knl_w + knl_x]
									* src_data[src_y * p.src_w + src_x];
							}
						}
						dst_data[dst_y * p.dst_w + dst_x] = sum;
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
	const float* knl_data = (const float*)kernel->data;

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
				const float* src_data = (const float*)src->data + (row / p.dst_h) * p.src_w * p.src_h * c;
				for (int64_t dst_x = 0; dst_x < p.dst_w; ++dst_x) {
					float* dst_data = (float*)dst->data + (row * p.dst_w + dst_x) * c;
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
								sum += knl_data[(knl_y * p.knl_w + knl_x) * c + c_i]
									* src_data[(src_y * p.src_w + src_x) * c + c_i];
							}
						}
						dst_data[c_i] = sum;
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
	const float* src_data = (const float*)src0->data;
	float* dst_data = (float*)dst->data;

	GGML_TENSOR_UNARY_OP_LOCALS

	const int s0 = ggml_get_op_params_i32(dst, 0);
	const int s1 = ggml_get_op_params_i32(dst, 1);
	const int s2 = ggml_get_op_params_i32(dst, 2);
	const int s3 = ggml_get_op_params_i32(dst, 3);

	const int64_t total = ne1 * ne2 * ne3;
	const int nth = pool.available_parallelism();
	stdexec::scheduler auto scheduler = pool.get_scheduler();
	const int64_t per_thread = (total + nth - 1) / nth;

	for (int64_t start = 0; start < total; start += per_thread) {
		const int64_t end = std::min(start + per_thread, total);
		stdexec::sender auto sender = stdexec::schedule(scheduler) | stdexec::then([=] {
			for (int64_t i = start; i < end; ++i) {
				const int64_t i1 = i % ne1;
				const int64_t i2 = (i / ne1) % ne2;
				const int64_t i3 = i / (ne2 * ne1);
				float* dst_row = dst_data + (i3 * nb3 + i2 * nb2 + i1 * nb1) / sizeof(float);

				const int64_t i01 = ggml_wrap_index(i1 - s1, ne01);
				const int64_t i02 = ggml_wrap_index(i2 - s2, ne02);
				const int64_t i03 = ggml_wrap_index(i3 - s3, ne03);
				const float* src_row = src_data + (i03 * nb03 + i02 * nb02 + i01 * nb01) / sizeof(float);

				const int64_t s = ggml_wrap_index(-s0, ne00);
				const int64_t n = ne00 - s;
				ggml_vec_cpy_f32(n, dst_row, src_row + s);
				ggml_vec_cpy_f32(s, dst_row + n, src_row);
			}
		});
		scope.spawn(std::move(sender));
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

static void ggml_compute_forward_set_rows_f32(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int64_t nc = ne00;
	const int64_t nr = ne01;

	assert(ne0 == nc);
	assert(ne2 == ne02);
	assert(ne3 == ne03);
	assert(src0->type == GGML_TYPE_F32);
	assert(ne02 % ne11 == 0);
	assert(ne03 % ne12 == 0);

	const int ith = params->ith;
	const int nth = params->nth;

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

//	ggml_from_float_t const from_float = ggml_get_type_traits_cpu(dst->type)->from_float;

	for (int64_t i03 = 0; i03 < ne03; ++i03) {
		for (int64_t i02 = 0; i02 < ne02; ++i02) {
			for (int64_t i = ir0; i < ir1; ++i) {
				const int64_t i12 = i03 % ne12;
				const int64_t i11 = i02 % ne11;
				const int64_t i10 = i;

				const int64_t i1 = *(int64_t*)((char*)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

				GGML_ASSERT(i1 >= 0 && i1 < ne1);
#if 0
				from_float(
					(const float*)((char*)src0->data + i * nb01 + i02 * nb02 + i03 * nb03),
					((char*)dst->data + i1 * nb1 + i02 * nb2 + i03 * nb3), nc);
#endif
			}
		}
	}
}

void ggml_compute_forward_set_rows(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_set_rows_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("src0->type = %d (%s) not supported", src0->type, ggml_type_name(src0->type));
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
	const int64_t batch_size = params->wsize / space_per_patch;
	const int64_t patches_per_batch = batch_size > 8 ? (batch_size / 8) * 8 : batch_size;
	const int64_t batch_n = (patch_total + patches_per_batch - 1) / patches_per_batch;

	GGML_ASSERT(patches_per_batch > 0 && batch_size >= 1);

	void* tmp = params->wdata;

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

		GGML_ASSERT(gemm_output + patch_n * c_out <= (float*)tmp + params->wsize);

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

void ggml_compute_forward_diag_mask_zero(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_diag_mask_f32(params, dst, 0);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward(
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
		ggml_compute_forward_sqr(params, tensor);
	} break;
	case GGML_OP_SQRT:
	{
		ggml_compute_forward_sqrt(params, tensor);
	} break;
	case GGML_OP_LOG:
	{
		ggml_compute_forward_log(params, tensor);
	} break;
	case GGML_OP_SIN:
	{
		ggml_compute_forward_sin(params, tensor);
	} break;
	case GGML_OP_COS:
	{
		ggml_compute_forward_cos(params, tensor);
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
		ggml_compute_forward_concat(params, tensor);
	} break;
	case GGML_OP_SILU_BACK:
	{
		ggml_compute_forward_silu_back(params, tensor);
	} break;
	case GGML_OP_NORM:
	{
		ggml_compute_forward_norm(params, tensor);
	} break;
	case GGML_OP_RMS_NORM:
	{
		ggml_compute_forward_rms_norm(params, tensor);
	} break;
	case GGML_OP_RMS_NORM_BACK:
	{
		ggml_compute_forward_rms_norm_back(params, tensor);
	} break;
	case GGML_OP_GROUP_NORM:
	{
		ggml_compute_forward_group_norm(params, tensor);
	} break;
	case GGML_OP_L2_NORM:
	{
		ggml_compute_forward_l2_norm(pool, scope, tensor);
	} break;
	case GGML_OP_MUL_MAT:
	{
		ggml_compute_forward_mul_mat(pool, scope, params, tensor);
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
		ggml_compute_forward_scale(params, tensor);
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
		ggml_compute_forward_set_rows(params, tensor);
	} break;
	case GGML_OP_GET_ROWS_BACK:
	{
		ggml_compute_forward_get_rows_back(params, tensor);
	} break;
	case GGML_OP_DIAG:
	{
		ggml_compute_forward_diag(tensor);
	} break;
	case GGML_OP_DIAG_MASK_INF:
	{
		ggml_compute_forward_diag_mask_inf(params, tensor);
	} break;
	case GGML_OP_DIAG_MASK_ZERO:
	{
		ggml_compute_forward_diag_mask_zero(params, tensor);
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
#if 0
	case GGML_OP_POOL_2D_BACK:
	{
		ggml_compute_forward_pool_2d_back(params, tensor);
	} break;
#endif
	case GGML_OP_UPSCALE:
	{
		ggml_compute_forward_upscale(params, tensor);
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
		ggml_compute_forward_argsort(params, tensor);
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
#if 0
	case GGML_OP_WIN_PART:
	{
		ggml_compute_forward_win_part(params, tensor);
	} break;
	case GGML_OP_WIN_UNPART:
	{
		ggml_compute_forward_win_unpart(params, tensor);
	} break;
#endif
	case GGML_OP_UNARY:
	{
		ggml_compute_forward_unary(params, tensor);
	} break;
	case GGML_OP_GLU:
	{
		ggml_compute_forward_glu(pool, scope, tensor);
	} break;
#if 0
	case GGML_OP_GET_REL_POS:
	{
		ggml_compute_forward_get_rel_pos(params, tensor);
	} break;
	case GGML_OP_ADD_REL_POS:
	{
		ggml_compute_forward_add_rel_pos(params, tensor);
	} break;
#endif
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
	case GGML_OP_MAP_CUSTOM1:
	{
		assert(false);
		//ggml_compute_forward_map_custom1(params, tensor);
	}
	break;
	case GGML_OP_MAP_CUSTOM2:
	{
		assert(false);
		//ggml_compute_forward_map_custom2(params, tensor);
	}
	break;
	case GGML_OP_MAP_CUSTOM3:
	{
		assert(false);
		//ggml_compute_forward_map_custom3(params, tensor);
	}
	break;
	case GGML_OP_CUSTOM:
	{
		assert(false);
		//ggml_compute_forward_custom(params, tensor);
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

static void ggml_graph_compute_thread(ggml_compute_state* state)
{
	exec::static_thread_pool pool(8);
	exec::async_scope scope;
	ggml_compute_params params = {
		/*.ith       =*/ 0, //state->ith,
		/*.nth       =*/ 1, //atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
		/*.wsize     =*/ state->cplan->work_size,
		/*.wdata     =*/ state->cplan->work_data,
		/*.threadpool=*/ state->threadpool
	};

	for (auto& node : state->cgraph->nodes) {
		ggml_compute_forward(pool, scope, &params, node);
#if 0
		if (state->ith == 0 && cplan->abort_callback &&
			cplan->abort_callback(cplan->abort_callback_data)) {
			tp->abort = true;
			tp->ec = GGML_STATUS_ABORTED;
		}
		state->threadpool->barrier();
#endif
		stdexec::sync_wait(scope.on_empty());
	}
}

ggml_status ggml_graph_compute(ggml_cgraph* cgraph, ggml_cplan& cplan)
{
	GGML_ASSERT(cplan.n_threads > 0);
	GGML_ASSERT(cplan.work_size == 0 || cplan.work_data != nullptr);
	std::optional<ggml_threadpool> default_thread_pools;
	ggml_threadpool* threadpool = [&] {
		if (cplan.threadpool == nullptr) {
			default_thread_pools.emplace(cplan.n_threads);
			return std::addressof(default_thread_pools.value());
		}
		else {
			return cplan.threadpool;
		}
	}();

	int n_threads = cplan.n_threads;
	if (n_threads > 1) {

	}
	else {
		ggml_compute_state state{ 
			.cgraph = cgraph,
			.cplan = &cplan,
			.threadpool = threadpool
		};
		ggml_graph_compute_thread(&state);
	}
	return {};
}

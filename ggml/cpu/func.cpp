module;
#include <assert.h>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <bit>
#include <barrier>
#include <cstdlib>
#include <new>
#include <numbers>
#include <iostream>
#include <exec/static_thread_pool.hpp>
#include <exec/async_scope.hpp>

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

static const size_t CACHE_LINE_SIZE_F32 = std::hardware_destructive_interference_size / sizeof(float);

module ggml;
import :ds;
import :tensor;
import :utility;
import :cpu.ds;
import :cpu.from_float;
import :cpu.to_float;
import :cpu.traits;
import :cpu.vec_dot;
import :cpu.op.norm;
import :cpu.op.scale;
import :cpu.op.unary;

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

static void ggml_compute_forward_mul_mat_one_chunk(
	const ggml_compute_params* params,
	ggml_tensor* dst,
	const enum ggml_type type,
	const int64_t num_rows_per_vec_dot,
	const int64_t ir0_start,
	const int64_t ir0_end,
	const int64_t ir1_start,
	const int64_t ir1_end) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const bool src1_cont = ggml_is_contiguous(src1);

	ggml_vec_dot_t const vec_dot = type_traits_cpu[type].vec_dot;
	enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

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
					vec_dot(ne00, tmp + ir0 - iir0, bs, src0_row + ir0 * nb01, bx, src1_col, by, num_rows_per_vec_dot);
				}

				for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
					memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (std::min(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
				}
			}
		}
	}
}

static void ggml_compute_forward_mul_mat(
	exec::static_thread_pool &pool,
	exec::async_scope &scope,
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int nth = pool.available_parallelism();

	enum ggml_type           const vec_dot_type = type_traits_cpu[src0->type].vec_dot_type;
	ggml_from_float_t        const from_float = type_traits_cpu[vec_dot_type].from_float;

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
							from_float(
								cast_with_offset<float>(src1->data, i13 * nb13 + i12 * nb12 + i11 * nb11),
								cast_with_offset<void>(params->wdata, i13 * nbw3 + i12 * nbw2 + i11 * nbw1),
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
			ggml_compute_forward_mul_mat_one_chunk(params, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);
		});
		scope.spawn(std::move(sender));
	}
}

static void ggml_compute_forward_arange_f32(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_compute_params* params,
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
	ggml_compute_params* params,
	ggml_tensor* dst) {
	switch (dst->type) {
		case GGML_TYPE_F32:
		{
			ggml_compute_forward_arange_f32(pool, scope, params, dst);
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
	const ggml_compute_params* params,
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

	memset(params->wdata, 0, params->wsize);

	// permute kernel data (src0) from (K x Cout x Cin) to (Cin x K x Cout)
	{
		const auto wdata = cast_with_offset<T>(params->wdata, 0);

		for (int64_t i02 = 0; i02 < ne02; i02++) {
			for (int64_t i01 = 0; i01 < ne01; i01++) {
				const auto src = cast_with_offset<T>(src0->data, i02 * nb02 + i01 * nb01);
				auto dst_data = &wdata[i01 * ne00 * ne02];
				for (int64_t i00 = 0; i00 < ne00; i00++) {
					dst_data[i00 * ne02 + i02] = src[i00];
				}
			}
		}
	}

	// permute source data (src1) from (L x Cin) to (Cin x L)
	{
		const auto dst_data = cast_with_offset<T>(params->wdata, sizeof(T) * nk);

		for (int64_t i11 = 0; i11 < ne11; i11++) {
			const auto src = cast_with_offset<float>(src1->data, i11 * nb11);
			for (int64_t i10 = 0; i10 < ne10; i10++) {
				dst_data[i10 * ne11 + i11] = fromFloat32<T>(src[i10]);
			}
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
			stdexec::then([=] {
				const T* wdata = cast_with_offset<T>(params->wdata, 0);
				const T* wdata_src = &wdata[nk];

				for (int i1 = ir0; i1 < ir1; i1++) {
					auto dst_data = cast_with_offset<float>(dst->data, i1 * nb1);
					auto wdata_kernel = &wdata[i1* ne02* ne00];
					for (int i10 = 0; i10 < ne10; i10++) {
						const int i1n = i10 * ne11;
						for (int i00 = 0; i00 < ne00; i00++) {
							float v = ggml_vec_dot<T>(ne02,
								wdata_src + i1n,
								wdata_kernel + i00 * ne02, 1);
							dst_data[i10 * s0 + i00] += v;
						}
					}
				}
			});
		scope.spawn(std::move(sender));
	}
}

template <typename T>
static void ggml_compute_forward_conv_transpose_2d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_compute_params* params,
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

	memset(params->wdata, 0, params->wsize);

	// permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
	{
		const auto wdata = cast_with_offset<T>(params->wdata, 0);

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
	}

	// permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
	{
		const auto wdata = cast_with_offset<T>(params->wdata, sizeof(T) * nk);
		for (int i12 = 0; i12 < ne12; i12++) {
			for (int i11 = 0; i11 < ne11; i11++) {
				const auto src = cast_with_offset<float>(src1->data, i12 * nb12 + i11 * nb11);
				T* dst_data = &wdata[i11 * ne10 * ne12];
				for (int i10 = 0; i10 < ne10; i10++) {
					dst_data[i10 * ne12 + i12] = fromFloat32<T>(src[i10]);
				}
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
			stdexec::then([=] {
				const T* wdata = cast_with_offset<T>(params->wdata, 0);
				const T* wdata_src = &wdata[nk];

				for (int i2 = ip0; i2 < ip1; i2++) { // Cout
					const auto dst_data = cast_with_offset<float>(dst->data, i2 * nb2);
					const T* wdata_kernel = &wdata[i2 * ne01 * ne00 * ne03];
					for (int i11 = 0; i11 < ne11; i11++) {
						for (int i10 = 0; i10 < ne10; i10++) {
							const int i1n = i11 * ne10 * ne12 + i10 * ne12;
							for (int i01 = 0; i01 < ne01; i01++) {
								for (int i00 = 0; i00 < ne00; i00++) {
									float v = ggml_vec_dot<T>(ne03,
										wdata_src + i1n,
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
}

static void ggml_compute_forward_conv_transpose_1d(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_conv_transpose_1d<ggml_fp16_t>(pool, scope, params, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_conv_transpose_1d<ggml_fp32_t>(pool, scope, params, dst);
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

	// parallelize by elements
	const int ne = dst->nelements();
	const int dr = (ne + nth - 1) / nth;
	for (int ie0 = 0; ie0 < ne; ie0 += dr) {
		const int ie1 = std::min(ie0 + dr, ne);
		stdexec::sender auto sender = stdexec::schedule(pool.get_scheduler()) |
			stdexec::then([=] {
				memcpy(
					cast_with_offset<char>(dst->data, ie0 * nb0),
					cast_with_offset<char>(src0->data, ie0 * nb0),
					(ie1 - ie0) * nb0);
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
				ne00 == ne0 &&
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

static void ggml_compute_forward_dup_f16(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());

	GGML_TENSOR_UNARY_OP_LOCALS

	const int ith = params->ith; // thread index
	const int nth = params->nth; // number of threads

	// parallelize by rows
	const int nr = ne01;
	// number of rows per thread
	const int dr = (nr + nth - 1) / nth;
	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (src0->type == dst->type &&
		ne00 == ne0 &&
		nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
		// copy by rows
		const size_t rs = ne00 * nb00;
		for (int64_t i03 = 0; i03 < ne03; i03++) {
			for (int64_t i02 = 0; i02 < ne02; i02++) {
				for (int64_t i01 = ir0; i01 < ir1; i01++) {
					memcpy(
						((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
						((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
						rs);
				}
			}
		}
		return;
	}

	// TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

	if (ggml_is_contiguous(dst)) {
		if (nb00 == sizeof(ggml_fp16_t)) {
			if (dst->type == GGML_TYPE_F16) {
				size_t id = 0;
				const size_t rs = ne00 * nb00;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const auto src0_ptr = cast_with_offset<char>(src0->data, i01 * nb01 + i02 * nb02 + i03 * nb03);
							memcpy(cast_with_offset<char>(dst->data, id), src0_ptr, rs);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_F32) {
				size_t id = 0;
				float* dst_ptr = (float*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
							for (int i00 = 0; i00 < ne00; i00++) {
								dst_ptr[id] = toFloat32(src0_ptr[i00]);
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (ggml_exist_cpu_from_float(dst->type)) {
				ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
				float* src0_f32 = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

				size_t id = 0;
				size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
				char* dst_ptr = (char*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

							for (int i00 = 0; i00 < ne00; i00++) {
								src0_f32[i00] = toFloat32(src0_ptr[i00]);
							}

							quantize_row_q(src0_f32, dst_ptr + id, ne00);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			}
			else {
				GGML_ABORT("fatal error"); // TODO: implement
			}
		}
		else {
			//printf("%s: this is not optimal - fix me\n", __func__);

			if (dst->type == GGML_TYPE_F32) {
				size_t id = 0;
				float* dst_ptr = (float*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = toFloat32(*src0_ptr);
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_F16) {
				size_t id = 0;
				ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = *src0_ptr;
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else {
				GGML_ABORT("fatal error"); // TODO: implement
			}
		}
		return;
	}

	// dst counters
	int64_t i10 = 0;
	int64_t i11 = 0;
	int64_t i12 = 0;
	int64_t i13 = 0;

	if (dst->type == GGML_TYPE_F16) {
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

						memcpy(dst_ptr, src0_ptr, sizeof(ggml_fp16_t));

						if (++i10 == ne00) {
							i10 = 0;
							if (++i11 == ne01) {
								i11 = 0;
								if (++i12 == ne02) {
									i12 = 0;
									if (++i13 == ne03) {
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
	}
	else if (dst->type == GGML_TYPE_F32) {
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

						*(float*)dst_ptr = toFloat32(*(const ggml_fp16_t*)src0_ptr);

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
	}
	else {
		GGML_ABORT("fatal error"); // TODO: implement
	}
}

static void ggml_compute_forward_dup_bf16(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());

	GGML_TENSOR_UNARY_OP_LOCALS

	const int ith = params->ith; // thread index
	const int nth = params->nth; // number of threads

	// parallelize by rows
	const int nr = ne01;
	// number of rows per thread
	const int dr = (nr + nth - 1) / nth;
	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (src0->type == dst->type &&
		ne00 == ne0 &&
		nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
		// copy by rows
		const size_t rs = ne00 * nb00;
		for (int64_t i03 = 0; i03 < ne03; i03++) {
			for (int64_t i02 = 0; i02 < ne02; i02++) {
				for (int64_t i01 = ir0; i01 < ir1; i01++) {
					memcpy(
						((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
						((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
						rs);
				}
			}
		}
		return;
	}

	// TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

	if (ggml_is_contiguous(dst)) {
		if (nb00 == sizeof(ggml_bf16_t)) {
			if (dst->type == GGML_TYPE_BF16) {
				size_t id = 0;
				const size_t rs = ne00 * nb00;
				char* dst_ptr = (char*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
							memcpy(dst_ptr + id, src0_ptr, rs);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_F16) {
				size_t id = 0;
				ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
							for (int i00 = 0; i00 < ne00; i00++) {
								dst_ptr[id] = fromFloat32<ggml_fp16_t>(toFloat32(src0_ptr[i00]));
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_F32) {
				size_t id = 0;
				float* dst_ptr = (float*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
							for (int i00 = 0; i00 < ne00; i00++) {
								dst_ptr[id] = toFloat32(src0_ptr[i00]);
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (ggml_exist_cpu_from_float(dst->type)) {
				ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
				float* src0_f32 = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

				size_t id = 0;
				size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
				char* dst_ptr = (char*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

							for (int i00 = 0; i00 < ne00; i00++) {
								src0_f32[i00] = toFloat32(src0_ptr[i00]);
							}
							quantize_row_q(src0_f32, dst_ptr + id, ne00);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			}
			else {
				GGML_ABORT("fatal error"); // TODO: implement
			}
		}
		else {
			//printf("%s: this is not optimal - fix me\n", __func__);

			if (dst->type == GGML_TYPE_F32) {
				size_t id = 0;
				float* dst_ptr = (float*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = toFloat32(*src0_ptr);
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_BF16) {
				size_t id = 0;
				ggml_bf16_t* dst_ptr = (ggml_bf16_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = *src0_ptr;
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_F16) {
				size_t id = 0;
				ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = fromFloat32<ggml_fp16_t>(toFloat32(*src0_ptr));
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else {
				GGML_ABORT("fatal error"); // TODO: implement
			}
		}
		return;
	}

	// dst counters
	int64_t i10 = 0;
	int64_t i11 = 0;
	int64_t i12 = 0;
	int64_t i13 = 0;

	if (dst->type == GGML_TYPE_BF16) {
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

						memcpy(dst_ptr, src0_ptr, sizeof(ggml_bf16_t));

						if (++i10 == ne00) {
							i10 = 0;
							if (++i11 == ne01) {
								i11 = 0;
								if (++i12 == ne02) {
									i12 = 0;
									if (++i13 == ne03) {
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
	}
	else if (dst->type == GGML_TYPE_F16) {
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

						*(ggml_fp16_t*)dst_ptr = fromFloat32<ggml_fp16_t>(toFloat32(*(const ggml_bf16_t*)src0_ptr));

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
	}
	else if (dst->type == GGML_TYPE_F32) {
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

						*(float*)dst_ptr = toFloat32(*(const ggml_bf16_t*)src0_ptr);

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
	}
	else {
		GGML_ABORT("fatal error"); // TODO: implement
	}
}

static void ggml_compute_forward_dup_f32(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	GGML_ASSERT(dst->nelements() == src0->nelements());

	GGML_TENSOR_UNARY_OP_LOCALS

	const int ith = params->ith; // thread index
	const int nth = params->nth; // number of threads

	// parallelize by rows
	const int nr = ne01;
	// number of rows per thread
	const int dr = (nr + nth - 1) / nth;
	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (src0->type == dst->type &&
		ne00 == ne0 &&
		nb00 == ggml_type_size(src0->type) && nb0 == ggml_type_size(dst->type)) {
		// copy by rows
		const size_t rs = ne00 * nb00;
		for (int64_t i03 = 0; i03 < ne03; i03++) {
			for (int64_t i02 = 0; i02 < ne02; i02++) {
				for (int64_t i01 = ir0; i01 < ir1; i01++) {
					memcpy(
						((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
						((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03),
						rs);
				}
			}
		}
		return;
	}

	if (ggml_is_contiguous(dst)) {
		// TODO: simplify
		if (nb00 == sizeof(float)) {
			if (dst->type == GGML_TYPE_F32) {
				size_t id = 0;
				const size_t rs = ne00 * nb00;
				char* dst_ptr = (char*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
							memcpy(dst_ptr + id, src0_ptr, rs);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			}
			else if (ggml_exist_cpu_from_float(dst->type)) {
				ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(dst->type)->from_float;
				size_t id = 0;
				size_t rs = nb0 * (ne00 / ggml_blck_size(dst->type));
				char* dst_ptr = (char*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += rs * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							const float* src0_ptr = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
							quantize_row_q(src0_ptr, dst_ptr + id, ne00);
							id += rs;
						}
						id += rs * (ne01 - ir1);
					}
				}
			}
			else {
				GGML_ABORT("fatal error"); // TODO: implement
			}
		}
		else {
			//printf("%s: this is not optimal - fix me\n", __func__);

			if (dst->type == GGML_TYPE_F32) {
				size_t id = 0;
				float* dst_ptr = (float*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = *src0_ptr;
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_F16) {
				size_t id = 0;
				ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = fromFloat32<ggml_fp16_t>(*src0_ptr);
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else if (dst->type == GGML_TYPE_BF16) {
				size_t id = 0;
				ggml_bf16_t* dst_ptr = (ggml_bf16_t*)dst->data;

				for (int i03 = 0; i03 < ne03; i03++) {
					for (int i02 = 0; i02 < ne02; i02++) {
						id += ne00 * ir0;
						for (int i01 = ir0; i01 < ir1; i01++) {
							for (int i00 = 0; i00 < ne00; i00++) {
								const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

								dst_ptr[id] = fromFloat32<ggml_bf16_t>(*src0_ptr);
								id++;
							}
						}
						id += ne00 * (ne01 - ir1);
					}
				}
			}
			else {
				GGML_ABORT("fatal error"); // TODO: implement
			}
		}

		return;
	}

	// dst counters

	int64_t i10 = 0;
	int64_t i11 = 0;
	int64_t i12 = 0;
	int64_t i13 = 0;

	if (dst->type == GGML_TYPE_F32) {
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

						memcpy(dst_ptr, src0_ptr, sizeof(float));

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
	}
	else if (dst->type == GGML_TYPE_F16) {
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

						*(ggml_fp16_t*)dst_ptr = fromFloat32<ggml_fp16_t>(*(const float*)src0_ptr);

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
	}
	else if (dst->type == GGML_TYPE_BF16) {
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

						*(ggml_bf16_t*)dst_ptr = fromFloat32<ggml_bf16_t>(*(const float*)src0_ptr);

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
	}
	else {
		GGML_ABORT("fatal error"); // TODO: implement
	}
}

static void ggml_compute_forward_dup_q(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

		const enum ggml_type type = src0->type;
	ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;

	size_t qk = ggml_blck_size(type);
	const int64_t nr = src1->nelements() / qk;

	// destination must be contiguous in the first dimension
	GGML_ASSERT(nb10 == ggml_type_size(dst->type));
	// must either have first dimension large enough to hold a row, or fully contiguous
	GGML_ASSERT((ne10 % qk) == 0 || ggml_is_contiguous(dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	for (int64_t ir = ir0; ir < ir1; ++ir) {

		uint32_t i = ir * qk;

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

		dequantize_row_q(
			(const void*)((char*)src0->data + x_offset),
			(float*)((char*)dst->data + dst_offset), qk);
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
		ggml_compute_forward_dup_f16(params, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_dup_bf16(params, dst);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_dup_f32(params, dst);
	} break;
	default:
	{
		if (ggml_is_quantized(src0->type) && dst->type == GGML_TYPE_F32) {
			ggml_compute_forward_dup_q(params, dst);
			break;
		}
		GGML_ABORT("fatal error");
	}
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

static void ggml_compute_forward_sum_f32(ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	assert(ggml_is_scalar(dst));
	assert(src0->nb[0] == sizeof(float));

	GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
	GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

	float sum = 0;
	float row_sum = 0;

	for (int64_t i03 = 0; i03 < ne03; i03++) {
		for (int64_t i02 = 0; i02 < ne02; i02++) {
			for (int64_t i01 = 0; i01 < ne01; i01++) {
				vec_sum<ggml_fp32_t>(ne00, &row_sum,
					(float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
				sum += row_sum;
			}
		}
	}
	((float*)dst->data)[0] = sum;
}

static void ggml_compute_forward_sum_f16(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];
	assert(ggml_is_scalar(dst));

	assert(src0->nb[0] == sizeof(ggml_fp16_t));

	GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
	GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

	float sum = 0;
	float row_sum = 0;

	for (int64_t i03 = 0; i03 < ne03; i03++) {
		for (int64_t i02 = 0; i02 < ne02; i02++) {
			for (int64_t i01 = 0; i01 < ne01; i01++) {
#if 0
				ggml_vec_sum_f16_ggf(ne00,
					&row_sum,
					(ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
#endif
				sum += row_sum;
			}
		}
	}
#if 0
	((ggml_fp16_t*)dst->data)[0] = fromFloat32<ggml_fp16_t>(sum);
#endif
}

static void ggml_compute_forward_sum_bf16(ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	assert(ggml_is_scalar(dst));

	assert(src0->nb[0] == sizeof(ggml_bf16_t));

	GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
	GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

	float sum = 0;
	float row_sum = 0;

	for (int64_t i03 = 0; i03 < ne03; i03++) {
		for (int64_t i02 = 0; i02 < ne02; i02++) {
			for (int64_t i01 = 0; i01 < ne01; i01++) {
#if 0
				ggml_vec_sum_bf16_ggf(ne00,
					&row_sum,
					(ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
#endif
				sum += row_sum;
			}
		}
	}
#if 0
	((ggml_bf16_t*)dst->data)[0] = GGML_FP32_TO_BF16(sum);
#endif
}

static void ggml_compute_forward_sum(ggml_tensor* dst) {
	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sum_f32(dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_sum_f16(dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_sum_bf16(dst);
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

	GGML_ASSERT(ggml_can_repeat(*src0, *dst));

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

	GGML_ASSERT(ggml_can_repeat(*src0, *dst));

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
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(dst->type == GGML_TYPE_F32);
	GGML_ASSERT(src0->type == GGML_TYPE_F32);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	const int ith = params->ith;
	const int nth = params->nth;

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

	if (ith == 0) {
		ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);
	}
	//ggml_barrier(params->threadpool);

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

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	// block-tiling attempt
	const int64_t blck_0 = std::max(GGML_VEC_MAD_UNROLL, 32);
	const int64_t blck_1 = 16;

	// dps == dst per src0, used for group query attention
	const int64_t dps2 = ne2 / ne02;
	const int64_t dps3 = ne3 / ne03;

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
}

static void ggml_compute_forward_out_prod_q_f32(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];
	const ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS;

	const int ith = params->ith;
	const int nth = params->nth;

	const enum ggml_type type = src0->type;
	ggml_to_float_t const dequantize_row_q = nullptr;// ggml_get_type_traits(type)->to_float;

	GGML_ASSERT(ne02 == ne12);
	GGML_ASSERT(ne03 == ne13);
	GGML_ASSERT(ne2 == ne12);
	GGML_ASSERT(ne3 == ne13);

	// we don't support permuted src0 dim0
	GGML_ASSERT(nb00 == ggml_type_size(type));

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

	if (ith == 0) {
		ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);
	}
	//ggml_barrier(params->threadpool);

	// parallelize by last three dimensions

	// total rows in dst
	const int64_t nr = ne1 * ne2 * ne3;

	// rows per thread
	const int64_t dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	// dst[:,:,:,:] = 0
	// for i2,i3:
	//   for i1:
	//     for i01:
	//       for i0:
	//         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

	float* wdata = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

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

			float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
			float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
			float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

			dequantize_row_q(s0, wdata, ne0);
			ggml_vec_mad_f32(ne0, d, wdata, *s1);
		}
	}
}

static void ggml_compute_forward_out_prod(
	const ggml_compute_params* params,
	ggml_tensor* dst) {
	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
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
	{
		ggml_compute_forward_out_prod_q_f32(params, dst);
	} break;
	case GGML_TYPE_F16:
	{
		GGML_ABORT("fatal error"); // todo
		// ggml_compute_forward_out_prod_f16_f32(params, dst);
	}
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_out_prod_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_get_rows_q(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int64_t nc = ne00;
	const int64_t nr = src1->nelements();

	const enum ggml_type type = src0->type;
	ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;

	assert(ne0 == nc);
	assert(ne02 == ne11);
	assert(nb00 == ggml_type_size(type));
	assert(ggml_nrows(dst) == nr);

	const int ith = params->ith;
	const int nth = params->nth;

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	for (int64_t i = ir0; i < ir1; ++i) {
		const int64_t i12 = i / (ne11 * ne10);
		const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
		const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
		const int64_t i01 = *(int32_t*)((char*)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

		GGML_ASSERT(i01 >= 0 && i01 < ne01);

		dequantize_row_q(
			(const void*)((char*)src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03),
			(float*)((char*)dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3), nc);
	}
}

static void ggml_compute_forward_get_rows_f16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int64_t nc = ne00;
	const int64_t nr = src1->nelements();

	assert(ne0 == nc);
	assert(ne02 == ne11);
	assert(nb00 == sizeof(ggml_fp16_t));
	assert(ggml_nrows(dst) == nr);

	const int ith = params->ith;
	const int nth = params->nth;

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	for (int64_t i = ir0; i < ir1; ++i) {
		const int64_t i12 = i / (ne11 * ne10);
		const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
		const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
		const int64_t i01 = *(int32_t*)((char*)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

		GGML_ASSERT(i01 >= 0 && i01 < ne01);

		to_float<ggml_fp16_t>(
			(const ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03),
			(float*)((char*)dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3), nc);
	}
}

static void ggml_compute_forward_get_rows_bf16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int64_t nc = ne00;
	const int64_t nr = src1->nelements();

	assert(ne0 == nc);
	assert(ne02 == ne11);
	assert(nb00 == sizeof(ggml_bf16_t));
	assert(ggml_nrows(dst) == nr);

	const int ith = params->ith;
	const int nth = params->nth;

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	for (int64_t i = ir0; i < ir1; ++i) {
		const int64_t i12 = i / (ne11 * ne10);
		const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
		const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
		const int64_t i01 = *(int32_t*)((char*)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

		GGML_ASSERT(i01 >= 0 && i01 < ne01);
		assert(false); //TODO
#if 0
		to_float<ggml_bf16_t>(
			(const ggml_bf16_t*)((char*)src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03),
			(float*)((char*)dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3), nc);
#endif
	}
}

static void ggml_compute_forward_get_rows_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int64_t nc = ne00;
	const int64_t nr = src1->nelements();

	assert(ne0 == nc);
	assert(ne02 == ne11);
	assert(nb00 == sizeof(float));
	assert(ggml_nrows(dst) == nr);

	const int ith = params->ith;
	const int nth = params->nth;

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);

	for (int64_t i = ir0; i < ir1; ++i) {
		const int64_t i12 = i / (ne11 * ne10);
		const int64_t i11 = (i - i12 * ne11 * ne10) / ne10;
		const int64_t i10 = (i - i12 * ne11 * ne10 - i11 * ne10);
		const int64_t i01 = *(int32_t*)((char*)src1->data + i10 * nb10 + i11 * nb11 + i12 * nb12);

		GGML_ASSERT(i01 >= 0 && i01 < ne01);

		ggml_vec_cpy(nc,
			(float*)((char*)dst->data + i10 * nb1 + i11 * nb2 + i12 * nb3),
			(float*)((char*)src0->data + i01 * nb01 + i11 * nb02 + i12 * nb03));
	}
}

static void ggml_compute_forward_get_rows(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
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
	{
		ggml_compute_forward_get_rows_q(params, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_get_rows_f16(params, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_get_rows_bf16(params, dst);
	} break;
	case GGML_TYPE_F32:
	case GGML_TYPE_I32:
	{
		ggml_compute_forward_get_rows_f32(params, dst);
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

static void ggml_compute_forward_add_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_can_repeat(*src1, *src0) && ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb00 == sizeof(float));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (nb10 == sizeof(float)) {
		for (int ir = ir0; ir < ir1; ++ir) {
			// src1 is broadcastable across src0 and dst in i1, i2, i3
			const int64_t i03 = ir / (ne02 * ne01);
			const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
			const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

			const int64_t i13 = i03 % ne13;
			const int64_t i12 = i02 % ne12;
			const int64_t i11 = i01 % ne11;
			const int64_t nr0 = ne00 / ne10;

			float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
			float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
			float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

			for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
				vDSP_vadd(src0_ptr + r * ne10, 1, src1_ptr, 1, dst_ptr + r * ne10, 1, ne10);
#else
				ggml_vec_add_f32(ne10, dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
#endif
			}
		}
	}
	else {
		// src1 is not contiguous
		for (int ir = ir0; ir < ir1; ++ir) {
			// src1 is broadcastable across src0 and dst in i1, i2, i3
			const int64_t i03 = ir / (ne02 * ne01);
			const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
			const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

			const int64_t i13 = i03 % ne13;
			const int64_t i12 = i02 % ne12;
			const int64_t i11 = i01 % ne11;

			float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
			float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

			for (int64_t i0 = 0; i0 < ne0; ++i0) {
				const int64_t i10 = i0 % ne10;
				float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i10 * nb10);

				dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
			}
		}
	}
}

static void ggml_compute_forward_add_f16_f16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(src0->type == GGML_TYPE_F16);
	GGML_ASSERT(src1->type == GGML_TYPE_F16);
	GGML_ASSERT(dst->type == GGML_TYPE_F16);

	GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (nb10 == sizeof(ggml_fp16_t)) {
		for (int ir = ir0; ir < ir1; ++ir) {
			// src0, src1 and dst are same shape => same indices
			const int i3 = ir / (ne2 * ne1);
			const int i2 = (ir - i3 * ne2 * ne1) / ne1;
			const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

			ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
			ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
			ggml_fp16_t* src1_ptr = (ggml_fp16_t*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

			for (int i = 0; i < ne0; i++) {
				dst_ptr[i] = fromFloat32<ggml_fp16_t>(toFloat32(src0_ptr[i]) + toFloat32(src1_ptr[i]));
			}
		}
	}
	else {
		// src1 is not contiguous
		GGML_ABORT("fatal error");
	}
}

static void ggml_compute_forward_add_f16_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(src0->type == GGML_TYPE_F16);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	if (dst->type == GGML_TYPE_F32) {
		GGML_ASSERT(nb0 == sizeof(float));
	}
	else {
		GGML_ASSERT(dst->type == GGML_TYPE_F16);
		GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
	}

	GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (nb10 == sizeof(float)) {
		if (dst->type == GGML_TYPE_F16) {
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0, src1 and dst are same shape => same indices
				const int i3 = ir / (ne2 * ne1);
				const int i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
				ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
				float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

				for (int i = 0; i < ne0; i++) {
					dst_ptr[i] = fromFloat32<ggml_fp16_t>(toFloat32(src0_ptr[i]) + src1_ptr[i]);
				}
			}
		}
		else {
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0, src1 and dst are same shape => same indices
				const int i3 = ir / (ne2 * ne1);
				const int i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
				ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
				float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

				for (int i = 0; i < ne0; i++) {
					dst_ptr[i] = toFloat32(src0_ptr[i]) + src1_ptr[i];
				}
			}
		}
	}
	else {
		// src1 is not contiguous
		GGML_ABORT("fatal error");
	}
}

static void ggml_compute_forward_add_bf16_bf16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(src0->type == GGML_TYPE_BF16);
	GGML_ASSERT(src1->type == GGML_TYPE_BF16);
	GGML_ASSERT(dst->type == GGML_TYPE_BF16);

	GGML_ASSERT(nb0 == sizeof(ggml_bf16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (nb10 == sizeof(ggml_bf16_t)) {
		for (int ir = ir0; ir < ir1; ++ir) {
			// src0, src1 and dst are same shape => same indices
			const int i3 = ir / (ne2 * ne1);
			const int i2 = (ir - i3 * ne2 * ne1) / ne1;
			const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

			ggml_bf16_t* dst_ptr = (ggml_bf16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
			ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
			ggml_bf16_t* src1_ptr = (ggml_bf16_t*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

			for (int i = 0; i < ne0; i++) {
				dst_ptr[i] = fromFloat32<ggml_bf16_t>(toFloat32(src0_ptr[i]) + toFloat32(src1_ptr[i]));
			}
		}
	}
	else {
		// src1 is not contiguous
		GGML_ABORT("fatal error");
	}
}

static void ggml_compute_forward_add_bf16_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

		GGML_ASSERT(src0->type == GGML_TYPE_BF16);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	if (dst->type == GGML_TYPE_F32) {
		GGML_ASSERT(nb0 == sizeof(float));
	}
	else {
		GGML_ASSERT(dst->type == GGML_TYPE_BF16);
		GGML_ASSERT(nb0 == sizeof(ggml_bf16_t));
	}

	GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	if (nb10 == sizeof(float)) {
		if (dst->type == GGML_TYPE_BF16) {
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0, src1 and dst are same shape => same indices
				const int i3 = ir / (ne2 * ne1);
				const int i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				ggml_bf16_t* dst_ptr = (ggml_bf16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
				ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
				float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

				for (int i = 0; i < ne0; i++) {
					dst_ptr[i] = fromFloat32<ggml_bf16_t>(toFloat32(src0_ptr[i]) + src1_ptr[i]);
				}
			}
		}
		else {
			for (int ir = ir0; ir < ir1; ++ir) {
				// src0, src1 and dst are same shape => same indices
				const int i3 = ir / (ne2 * ne1);
				const int i2 = (ir - i3 * ne2 * ne1) / ne1;
				const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

				float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
				ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
				float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

				for (int i = 0; i < ne0; i++) {
					dst_ptr[i] = toFloat32(src0_ptr[i]) + src1_ptr[i];
				}
			}
		}
	}
	else {
		// src1 is not contiguous
		GGML_ABORT("fatal error");
	}
}

static void ggml_compute_forward_add_q_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

		const int ith = params->ith;
	const int nth = params->nth;

	const enum ggml_type type = src0->type;
	const enum ggml_type dtype = dst->type;
	ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;
	ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(dtype)->from_float;

	// we don't support permuted src0 or src1
	GGML_ASSERT(nb00 == ggml_type_size(type));
	GGML_ASSERT(nb10 == sizeof(float));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

	GGML_ASSERT(ggml_is_quantized(src0->type));
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	float* wdata = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

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
		dequantize_row_q(src0_row, wdata, ne00);
		// add src1
		std::span<float> wdata_span{ wdata, static_cast<size_t>(ne00) };
		std::span<const float> src1_row_span{ src1_row, static_cast<size_t>(ne00) };
		std::ranges::transform(src1_row_span, wdata_span, wdata_span.begin(), std::plus<>());

		// quantize row to dst
		if (quantize_row_q != NULL) {
			quantize_row_q(wdata, dst_row, ne00);
		}
		else {
			memcpy(dst_row, wdata, ne0 * nb0);
		}
	}
}

static void ggml_compute_forward_add(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add_f32(params, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
	case GGML_TYPE_F16:
	{
		if (src1->type == GGML_TYPE_F16) {
			ggml_compute_forward_add_f16_f16(params, dst);
		}
		else if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add_f16_f32(params, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
	case GGML_TYPE_BF16:
	{
		if (src1->type == GGML_TYPE_BF16) {
			ggml_compute_forward_add_bf16_bf16(params, dst);
		}
		else if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add_bf16_f32(params, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
	case GGML_TYPE_Q4_0:
	case GGML_TYPE_Q4_1:
	case GGML_TYPE_Q5_0:
	case GGML_TYPE_Q5_1:
	case GGML_TYPE_Q8_0:
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
	{
		ggml_compute_forward_add_q_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

template <typename op>
void process_vec(const int n, float* z, const float* x, const float* y) {
	for (int i = 0; i < n; ++i) z[i] = op()(x[i], y[i]);
}

template <typename op>
static void ggml_compute_forward_muldiv_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_can_repeat(*src1, *src0) && ggml_are_same_shape(src0, dst));

	const int ith = params->ith;
	const int nth = params->nth;

	const int64_t nr = ggml_nrows(src0);

	GGML_TENSOR_BINARY_OP_LOCALS

	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb00 == sizeof(float));

	if (nb10 == sizeof(float)) {
		for (int64_t ir = ith; ir < nr; ir += nth) {
			// src0 and dst are same shape => same indices
			const int64_t i03 = ir / (ne02 * ne01);
			const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
			const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

			const int64_t i13 = i03 % ne13;
			const int64_t i12 = i02 % ne12;
			const int64_t i11 = i01 % ne11;
			const int64_t nr0 = ne00 / ne10;

			float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
			float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
			float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

			for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
				UNUSED(ggml_vec_mul_f32);

				vDSP_vmul(src0_ptr + r * ne10, 1, src1_ptr, 1, dst_ptr + r * ne10, 1, ne10);
#else
				process_vec<op>(ne10, dst_ptr + r * ne10, src0_ptr + r * ne10, src1_ptr);
#endif
			}
		}
	}
	else {
		// src1 is not contiguous
		for (int64_t ir = ith; ir < nr; ir += nth) {
			// src0 and dst are same shape => same indices
			// src1 is broadcastable across src0 and dst in i1, i2, i3
			const int64_t i03 = ir / (ne02 * ne01);
			const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
			const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

			const int64_t i13 = i03 % ne13;
			const int64_t i12 = i02 % ne12;
			const int64_t i11 = i01 % ne11;

			float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
			float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

			for (int64_t i0 = 0; i0 < ne00; ++i0) {
				const int64_t i10 = i0 % ne10;
				float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i10 * nb10);

				dst_ptr[i0] = op()(src0_ptr[i0], (*src1_ptr));
			}
		}
	}
}

static void ggml_compute_forward_muldiv(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(src1->type == GGML_TYPE_F32 && "only f32 src1 supported for now");

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		if (dst->op == GGML_OP_MUL) {
			ggml_compute_forward_muldiv_f32<std::multiplies<>>(params, dst);
		}
		else {
			ggml_compute_forward_muldiv_f32<std::divides<>>(params, dst);
		}
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_vec_add1_f32(const int n, float* z, const float* x, const float   v) { for (int i = 0; i < n; ++i) z[i] = x[i] + v; }

static void ggml_compute_forward_add1_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

		GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb00 == sizeof(float));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are same shape => same indices
		const int i3 = ir / (ne2 * ne1);
		const int i2 = (ir - i3 * ne2 * ne1) / ne1;
		const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

#ifdef GGML_USE_ACCELERATE
		UNUSED(ggml_vec_add1_f32);

		vDSP_vadd(
			(float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1,
			(float*)((char*)src1->data), 0,
			(float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1,
			ne0);
#else
		ggml_vec_add1_f32(ne0,
			(float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
			(float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01),
			*(float*)src1->data);
#endif
	}
}

static void ggml_compute_forward_add1_f16_f16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = toFloat32(*(ggml_fp16_t*)src1->data);

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

		GGML_ASSERT(src0->type == GGML_TYPE_F16);
	GGML_ASSERT(src1->type == GGML_TYPE_F16);
	GGML_ASSERT(dst->type == GGML_TYPE_F16);

	GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are same shape => same indices
		const int i3 = ir / (ne2 * ne1);
		const int i2 = (ir - i3 * ne2 * ne1) / ne1;
		const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

		ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
		ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
		for (int i = 0; i < ne0; i++) {
			dst_ptr[i] = fromFloat32<ggml_fp16_t>(toFloat32(src0_ptr[i]) + v);
		}
	}
}

static void ggml_compute_forward_add1_f16_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = *(float*)src1->data;

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

		GGML_ASSERT(src0->type == GGML_TYPE_F16);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_F16);

	GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are same shape => same indices
		const int i3 = ir / (ne2 * ne1);
		const int i2 = (ir - i3 * ne2 * ne1) / ne1;
		const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

		ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
		ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
		for (int i = 0; i < ne0; i++) {
			dst_ptr[i] = fromFloat32<ggml_fp16_t>(toFloat32(src0_ptr[i]) + v);
		}
	}
}

static void ggml_compute_forward_add1_bf16_bf16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = toFloat32(*(ggml_bf16_t*)src1->data);

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

	GGML_ASSERT(src0->type == GGML_TYPE_BF16);
	GGML_ASSERT(src1->type == GGML_TYPE_BF16);
	GGML_ASSERT(dst->type == GGML_TYPE_BF16);

	GGML_ASSERT(nb0 == sizeof(ggml_bf16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are same shape => same indices
		const int i3 = ir / (ne2 * ne1);
		const int i2 = (ir - i3 * ne2 * ne1) / ne1;
		const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

		ggml_bf16_t* dst_ptr = (ggml_bf16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
		ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
		for (int i = 0; i < ne0; i++) {
			dst_ptr[i] = fromFloat32<ggml_bf16_t>(toFloat32(src0_ptr[i]) + v);
		}
	}
}

static void ggml_compute_forward_add1_bf16_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = *(float*)src1->data;

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

	GGML_ASSERT(src0->type == GGML_TYPE_BF16);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);
	GGML_ASSERT(dst->type == GGML_TYPE_BF16);

	GGML_ASSERT(nb0 == sizeof(ggml_bf16_t));
	GGML_ASSERT(nb00 == sizeof(ggml_bf16_t));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are same shape => same indices
		const int i3 = ir / (ne2 * ne1);
		const int i2 = (ir - i3 * ne2 * ne1) / ne1;
		const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

		ggml_bf16_t* dst_ptr = (ggml_bf16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
		ggml_bf16_t* src0_ptr = (ggml_bf16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
		for (int i = 0; i < ne0; i++) {
			dst_ptr[i] = fromFloat32<ggml_bf16_t>(toFloat32(src0_ptr[i]) + v);
		}
	}
}

static void ggml_vec_acc1_f32(const int n, float* y, const float   v) { for (int i = 0; i < n; ++i) y[i] += v; }

static void ggml_compute_forward_add1_q_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_is_scalar(src1));

	// scalar to add
	const float v = *(float*)src1->data;

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(src0);

	GGML_TENSOR_UNARY_OP_LOCALS

	const enum ggml_type type = src0->type;
	ggml_to_float_t const dequantize_row_q = ggml_get_type_traits(type)->to_float;
	ggml_from_float_t const quantize_row_q = ggml_get_type_traits_cpu(type)->from_float;

	// we don't support permuted src0
	GGML_ASSERT(nb00 == ggml_type_size(type));

	// dst cannot be transposed or permuted
	GGML_ASSERT(nb0 <= nb1);
	GGML_ASSERT(nb1 <= nb2);
	GGML_ASSERT(nb2 <= nb3);

	GGML_ASSERT(ggml_is_quantized(src0->type));
	GGML_ASSERT(dst->type == src0->type);
	GGML_ASSERT(src1->type == GGML_TYPE_F32);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	float* wdata = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

	for (int ir = ir0; ir < ir1; ++ir) {
		// src0 and dst are same shape => same indices
		const int i3 = ir / (ne2 * ne1);
		const int i2 = (ir - i3 * ne2 * ne1) / ne1;
		const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

		void* src0_row = (void*)((char*)src0->data + (i1 * nb01 + i2 * nb02 + i3 * nb03));
		void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

		assert(ne0 % 32 == 0);

		// unquantize row from src0 to temp buffer
		dequantize_row_q(src0_row, wdata, ne0);
		// add src1
		ggml_vec_acc1_f32(ne0, wdata, v);
		// quantize row to dst
		quantize_row_q(wdata, dst_row, ne0);
	}
}

static void ggml_compute_forward_add1(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_add1_f32(params, dst);
	} break;
	case GGML_TYPE_F16:
	{
		if (src1->type == GGML_TYPE_F16) {
			ggml_compute_forward_add1_f16_f16(params, dst);
		}
		else if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add1_f16_f32(params, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
	case GGML_TYPE_BF16:
	{
		if (src1->type == GGML_TYPE_BF16) {
			ggml_compute_forward_add1_bf16_bf16(params, dst);
		}
		else if (src1->type == GGML_TYPE_F32) {
			ggml_compute_forward_add1_bf16_f32(params, dst);
		}
		else {
			GGML_ABORT("fatal error");
		}
	} break;
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
	{
		ggml_compute_forward_add1_q_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_ssm_conv_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {
	const struct ggml_tensor* src0 = dst->src[0]; // conv_x
	const struct ggml_tensor* src1 = dst->src[1]; // conv1d.weight

	const int ith = params->ith;
	const int nth = params->nth;

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
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);
	const int ir = ir1 - ir0;

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
}

static void ggml_compute_forward_ssm_conv(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {
	switch (dst->src[0]->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_ssm_conv_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_ssm_scan_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {
	const struct ggml_tensor* src0 = dst->src[0]; // s
	const struct ggml_tensor* src1 = dst->src[1]; // x
	const struct ggml_tensor* src2 = dst->src[2]; // dt
	const struct ggml_tensor* src3 = dst->src[3]; // A
	const struct ggml_tensor* src4 = dst->src[4]; // B
	const struct ggml_tensor* src5 = dst->src[5]; // C

	const int ith = params->ith;
	const int nth = params->nth;

	const int64_t nc = src0->ne[0]; // d_state
	const int64_t nr = src0->ne[1]; // d_inner
	const int64_t n_t = src1->ne[1]; // number of tokens per sequence
	const int64_t n_s = src0->ne[2]; // number of sequences in the batch

	GGML_ASSERT(src1->nelements() + src0->nelements() == dst->nelements());
	GGML_ASSERT(src0->nb[0] == sizeof(float));
	GGML_ASSERT(src1->nb[0] == sizeof(float));
	GGML_ASSERT(src2->nb[0] == sizeof(float));
	GGML_ASSERT(src3->nb[0] == sizeof(float));
	GGML_ASSERT(src4->nb[0] == sizeof(float));
	GGML_ASSERT(src5->nb[0] == sizeof(float));
	// required for the dot product between s and C
	GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
	// required for per-sequence offsets for states
	GGML_ASSERT(src0->nb[2] == src0->ne[0] * src0->ne[1] * sizeof(float));
	// required to get correct offset for state destination (i.e. src1->nb[3])
	GGML_ASSERT(src1->nb[3] == src1->ne[0] * src1->ne[1] * src1->ne[2] * sizeof(float));

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int64_t ir0 = dr * ith;
	const int64_t ir1 = std::min(ir0 + dr, nr);
	const int64_t ir = ir1 - ir0;

	for (int i3 = 0; i3 < n_s; ++i3) {
		for (int i2 = 0; i2 < n_t; ++i2) {
			const float* s0 = (const float*)((const char*)src0->data + ir0 * (src0->nb[1]) + i3 * (src0->nb[2])); // {d_state, d_inner, n_s}
			const float* x = (const float*)((const char*)src1->data + ir0 * (src1->nb[0]) + i2 * (src1->nb[1]) + i3 * (src1->nb[2])); // {d_inner, n_t, n_s}
			const float* dt = (const float*)((const char*)src2->data + ir0 * (src2->nb[0]) + i2 * (src2->nb[1]) + i3 * (src2->nb[2])); // {d_inner, n_t, n_s}
			const float* A = (const float*)((const char*)src3->data + ir0 * (src3->nb[1])); // {d_state, d_inner}
			const float* B = (const float*)((const char*)src4->data + i2 * (src4->nb[1]) + i3 * (src4->nb[2])); // {d_state, n_t, n_s}
			const float* C = (const float*)((const char*)src5->data + i2 * (src5->nb[1]) + i3 * (src5->nb[2])); // {d_state, n_t, n_s}
			float* y = (float*)((char*)dst->data + ir0 * (src1->nb[0]) + i2 * (src1->nb[1]) + i3 * (src1->nb[2])); // {d_inner, n_t, n_s}
			float* s = (float*)((char*)dst->data + ir0 * (src0->nb[1]) + i3 * (src0->nb[2]) + src1->nb[3]);  // {d_state, d_inner, n_s}

			// use the output as the source for the next token-wise iterations
			if (i2 > 0) { s0 = s; }

			// d_inner
			for (int i1 = 0; i1 < ir; ++i1) {
				// ref: https://github.com/state-spaces/mamba/blob/34076d664838588a3c97727b263478ab9f621a07/mamba_ssm/ops/triton/selective_state_update.py#L78
				float dt_soft_plus = dt[i1] <= 20.0f ? log1pf(expf(dt[i1])) : dt[i1];
				float x_dt = x[i1] * dt_soft_plus;
				float sumf = 0.0f;
				// d_state
				for (int i0 = 0; i0 < nc; ++i0) {
					int i = i0 + i1 * nc;
					// state = prev_state * dA + dB * x
					float state = (s0[i] * expf(dt_soft_plus * A[i])) + (B[i0] * x_dt);
					// y = rowwise_dotprod(state, C)
					sumf += state * C[i0];
					s[i] = state;
				}
				y[i1] = sumf;
			}
		}
	}
}

static void ggml_compute_forward_ssm_scan(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {
	switch (dst->src[0]->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_ssm_scan_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_rwkv_wkv6_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {
	const int64_t T = dst->src[1]->ne[2];
	const int64_t C = dst->ne[0];
	const int64_t HEADS = dst->src[1]->ne[1];
	const int64_t n_seqs = dst->src[5]->ne[1];
	const int64_t head_size = C / HEADS;

	float* dst_data = (float*)dst->data;
	float* state = ((float*)dst->data) + C * T;

	const int ith = params->ith;
	const int nth = params->nth;

	if (ith >= HEADS) {
		return;
	}

	const int h_start = (HEADS * ith) / nth;
	const int h_end = ((HEADS * (ith + 1)) / nth < HEADS) ?
		(HEADS * (ith + 1)) / nth : HEADS;

	float* k = (float*)dst->src[0]->data;
	float* v = (float*)dst->src[1]->data;
	float* r = (float*)dst->src[2]->data;
	float* time_faaaa = (float*)dst->src[3]->data;
	float* time_decay = (float*)dst->src[4]->data;

	size_t t_stride = HEADS * head_size; // Same to C

	size_t h_stride = C / HEADS;
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
	size_t h_stride_2d = head_size * head_size;

	if (ith == 0) {
		memset(dst_data, 0, T * C * sizeof(float));
	}
	//ggml_barrier(params->threadpool);


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
#elif defined(__ARM_NEON) && defined(__aarch64__)
#define GGML_F32X GGML_F32x4
#define GGML_F32X_SET1 GGML_F32x4_SET1
#define GGML_F32X_LOAD GGML_F32x4_LOAD
#define GGML_F32X_STORE GGML_F32x4_STORE
#define GGML_F32X_MUL GGML_F32x4_MUL
#define GGML_F32X_FMA GGML_F32x4_FMA
#define WKV_VECTOR_SIZE 4
#endif

#ifdef WKV_VECTOR_SIZE
	const int64_t vec_count = head_size / WKV_VECTOR_SIZE;

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
					size_t base_j = j * WKV_VECTOR_SIZE;
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
				for (int64_t j = vec_count * WKV_VECTOR_SIZE; j < head_size; j++) {
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
}

static void ggml_compute_forward_rwkv_wkv6(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rwkv_wkv6_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_gla_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {
	const int64_t T = dst->src[1]->ne[2];
	const int64_t C = dst->ne[0];
	const int64_t HEADS = dst->src[1]->ne[1];
	const int64_t n_seqs = dst->src[4]->ne[1];
	const int64_t head_size = C / HEADS;
	const float scale = std::bit_cast<float>(dst->op_params[0]);

	float* dst_data = (float*)dst->data;
	float* state = ((float*)dst->data) + C * T;

	const int ith = params->ith;
	const int nth = params->nth;

	if (ith >= HEADS) {
		return;
	}

	const int h_start = (HEADS * ith) / nth;
	const int h_end = ((HEADS * (ith + 1)) / nth < HEADS) ?
		(HEADS * (ith + 1)) / nth : HEADS;

	float* k = (float*)dst->src[0]->data;
	float* v = (float*)dst->src[1]->data;
	float* q = (float*)dst->src[2]->data;
	float* g = (float*)dst->src[3]->data;

	size_t t_stride = HEADS * head_size; // Same to C

	size_t h_stride = C / HEADS;
	GGML_ASSERT(C % HEADS == 0); // C must be divisible by HEADS
	size_t h_stride_2d = head_size * head_size;

	if (ith == 0) {
		memset(dst_data, 0, T * C * sizeof(float));
	}
	//ggml_barrier(params->threadpool);


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
}

static void ggml_compute_forward_gla(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_gla_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_mul_mat_id(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];
	const struct ggml_tensor* ids = dst->src[2];

	GGML_TENSOR_BINARY_OP_LOCALS

	const int ith = params->ith;
	const int nth = params->nth;

	const enum ggml_type type = src0->type;

	const bool src1_cont = ggml_is_contiguous(src1);

	ggml_vec_dot_t    const vec_dot = type_traits_cpu[type].vec_dot;
	enum ggml_type    const vec_dot_type = type_traits_cpu[type].vec_dot_type;
	ggml_from_float_t const from_float = type_traits_cpu[vec_dot_type].from_float;

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

	struct mmid_row_mapping {
		int32_t i1;
		int32_t i2;
	};
	
	char* wdata_src1_end = (src1->type == vec_dot_type) ?
		(char*)params->wdata :
		(char*)params->wdata + GGML_PAD(ggml_row_size(vec_dot_type, src1->nelements()), sizeof(int64_t));

	int64_t* matrix_row_counts = (int64_t*)(wdata_src1_end); // [n_as]
	struct mmid_row_mapping* matrix_rows = (struct mmid_row_mapping*)(matrix_row_counts + n_as); // [n_as][ne11]

	if (src1->type != vec_dot_type) {
		char* wdata = (char *)params->wdata;

		const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
		const size_t nbw2 = nbw1 * ne11;
		const size_t nbw3 = nbw2 * ne12;

		assert(params->wsize >= ne13 * nbw3);
		GGML_ASSERT(src1->type == GGML_TYPE_F32);

		for (int64_t i13 = 0; i13 < ne13; ++i13) {
			for (int64_t i12 = 0; i12 < ne12; ++i12) {
				for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
					from_float((float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11),
						(void*)(wdata + i13 * nbw3 + i12 * nbw2 + i11 * nbw1),
						ne10);
				}
			}
		}
	}

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id)*ne12 + (i1)]

	if (ith == 0) {
		// initialize matrix_row_counts
		memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

		// group rows by src0 matrix
		for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
			for (int id = 0; id < n_ids; ++id) {
				const int32_t i02 = *(const int32_t*)((const char*)ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

				assert(i02 >= 0 && i02 < n_as);

				mmid_row_mapping new_map{ id, static_cast<int32_t>(iid1) };
				MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = new_map;
				matrix_row_counts[i02] += 1;
			}
		}
	}

	//ggml_barrier(params->threadpool);

	// compute each matrix multiplication in sequence
	for (int cur_a = 0; cur_a < n_as; ++cur_a) {
		const int64_t cne1 = matrix_row_counts[cur_a];

		if (cne1 == 0) {
			continue;
		}

		const char* src0_cur = (const char*)src0->data + cur_a * nb02;

		const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
		const size_t row_size = ggml_row_size(vec_dot_type, ne10);

		const int64_t nr0 = ne01; // src0 rows
		const int64_t nr1 = cne1; // src1 rows

		// distribute the thread work across the inner or outer loop based on which one is larger

		const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
		const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

		const int64_t ith0 = ith % nth0;
		const int64_t ith1 = ith / nth0;

		const int64_t dr0 = (nr0 + nth0 - 1) / nth0;
		const int64_t dr1 = (nr1 + nth1 - 1) / nth1;

		const int64_t ir010 = dr0 * ith0;
		const int64_t ir011 = std::min(ir010 + dr0, nr0);

		const int64_t ir110 = dr1 * ith1;
		const int64_t ir111 = std::min(ir110 + dr1, nr1);

		// threads with no work simply yield (not sure if it helps)
		//if (ir010 >= ir011 || ir110 >= ir111) {
		//    sched_yield();
		//    continue;
		//}

		// block-tiling attempt
		const int64_t blck_0 = 16;
		const int64_t blck_1 = 16;

		// attempt to reduce false-sharing (does not seem to make a difference)
		float tmp[16];

		for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
			for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
				for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ++ir1) {
					const int64_t _i12 = ir1; // logical row index for this expert

					struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
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

					//for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
					//    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
					//}

					for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
						vec_dot(ne00, &tmp[ir0 - iir0], 0, src0_cur + ir0 * nb01, 0, src1_col, 0, 1);
					}

					memcpy(&dst_col[iir0], tmp, (std::min(iir0 + blck_0, ir011) - iir0) * sizeof(float));
				}
			}
		}
	}

#undef MMID_MATRIX_ROW
}

static void ggml_compute_forward_clamp_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	float min = std::bit_cast<float>(dst->op_params[0]);
	float max = std::bit_cast<float>(dst->op_params[1]);
	GGML_ASSERT(min <= max);

	const int ith = params->ith;
	const int nth = params->nth;

	const int n = ggml_nrows(src0);
	const int nc = src0->ne[0];

	const size_t nb00 = src0->nb[0];
	const size_t nb01 = src0->nb[1];

	const size_t nb0 = dst->nb[0];
	const size_t nb1 = dst->nb[1];

	GGML_ASSERT(nb0 == sizeof(float));
	GGML_ASSERT(nb00 == sizeof(float));

	for (int j = ith; j < n; j += nth) {
		float* dst_ptr = (float*)((char*)dst->data + j * nb1);
		float* src0_ptr = (float*)((char*)src0->data + j * nb01);

		for (int i = 0; i < nc; i++) {
			dst_ptr[i] = std::clamp(src0_ptr[i], min, max);
		}
	}
}

static void ggml_compute_forward_clamp(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_clamp_f32(params, dst);
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

static void ggml_compute_forward_soft_max_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	assert(ggml_is_contiguous(dst));
	assert(ggml_are_same_shape(src0, dst));

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);

	// TODO: handle transposed/permuted matrices

	const int ith = params->ith;
	const int nth = params->nth;

	GGML_TENSOR_UNARY_OP_LOCALS

	//const int64_t ne11 = src1 ? src1->ne[1] : 1;

	// TODO: is this supposed to be ceil instead of floor?
	//       https://huggingface.co/mosaicml/mpt-7b/blob/main/attention.py#L370
	const uint32_t n_head = ne02;
	const uint32_t n_head_log2 = 1u << (uint32_t)floor(log2(n_head));

	const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
	const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

	const int nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	float* wp = (float*)params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith;

	const bool use_f16 = (src1 && src1->type == GGML_TYPE_F16);

	for (int i1 = ir0; i1 < ir1; i1++) {
		// ALiBi
		const uint32_t h = (i1 / ne01) % ne02; // head
		const float slope = (max_bias > 0.0f) ? h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1) : 1.0f;

		float* sp = (float*)((char*)src0->data + i1 * src0->nb[1]);
		float* dp = (float*)((char*)dst->data + i1 * dst->nb[1]);

		// broadcast the mask across rows
		ggml_fp16_t* mp_f16 = src1 ? (ggml_fp16_t*)((char*)src1->data) + (i1 % ne01) * ne00 : NULL;
		float* mp_f32 = src1 ? (float*)((char*)src1->data) + (i1 % ne01) * ne00 : NULL;

		ggml_vec_cpy_f321(nc, wp, sp);
		ggml_vec_scale_f322(nc, wp, scale);
		if (mp_f32) {
			if (use_f16) {
				for (int i = 0; i < nc; ++i) {
					wp[i] += slope * toFloat32(mp_f16[i]);
				}
			}
			else {
				for (int i = 0; i < nc; ++i) {
					wp[i] += slope * mp_f32[i];
				}
			}
		}

#ifndef NDEBUG
		for (int i = 0; i < nc; ++i) {
			//printf("p[%d] = %f\n", i, p[i]);
			assert(!isnan(wp[i]));
		}
#endif

		float max = -INFINITY;
		ggml_vec_max_f32(nc, &max, wp);

		ggml_float sum = ggml_vec_soft_max_f32(nc, dp, wp, max);
		assert(sum > 0.0);

		sum = 1.0 / sum;
		ggml_vec_scale_f322(nc, dp, sum);

#ifndef NDEBUG
		for (int i = 0; i < nc; ++i) {
			assert(!isnan(dp[i]));
			assert(!isinf(dp[i]));
		}
#endif
	}
}

static void ggml_compute_forward_soft_max(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_soft_max_f32(params, dst);
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
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];

	GGML_ASSERT(ggml_is_contiguous(src0));
	GGML_ASSERT(ggml_is_contiguous(src1));
	GGML_ASSERT(ggml_is_contiguous(dst));
	GGML_ASSERT(ggml_are_same_shape(src0, dst));
	GGML_ASSERT(ggml_are_same_shape(src1, dst));

	float scale = std::bit_cast<float>(dst->op_params[0]);
	float max_bias = std::bit_cast<float>(dst->op_params[1]);

	GGML_ASSERT(max_bias == 0.0f);

	// TODO: handle transposed/permuted matrices

	const int ith = params->ith;
	const int nth = params->nth;

	const int nc = src0->ne[0];
	const int nr = ggml_nrows(src0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

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
}

static void ggml_compute_forward_soft_max_ext_back(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_soft_max_ext_back_f32(params, dst);
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

static void ggml_compute_forward_rope_f16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst,
	const bool forward) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];
	const struct ggml_tensor* src2 = dst->src[2];

	float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
	int sections[4];

	//const int n_past     = ((int32_t *) dst->op_params)[0];
	const int n_dims = ((int32_t*)dst->op_params)[1];
	const int mode = ((int32_t*)dst->op_params)[2];
	//const int n_ctx      = ((int32_t *) dst->op_params)[3];
	const int n_ctx_orig = ((int32_t*)dst->op_params)[4];
	memcpy(&freq_base, (int32_t*)dst->op_params + 5, sizeof(float));
	memcpy(&freq_scale, (int32_t*)dst->op_params + 6, sizeof(float));
	memcpy(&ext_factor, (int32_t*)dst->op_params + 7, sizeof(float));
	memcpy(&attn_factor, (int32_t*)dst->op_params + 8, sizeof(float));
	memcpy(&beta_fast, (int32_t*)dst->op_params + 9, sizeof(float));
	memcpy(&beta_slow, (int32_t*)dst->op_params + 10, sizeof(float));
	memcpy(&sections, (int32_t*)dst->op_params + 11, sizeof(int) * 4);


	GGML_TENSOR_UNARY_OP_LOCALS

	//printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
	//printf("n_past = %d, ne2 = %d\n", n_past, ne2);

	GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(dst);

	GGML_ASSERT(n_dims <= ne0);
	GGML_ASSERT(n_dims % 2 == 0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	// row index used to determine which thread to use
	int ir = 0;

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
	const float sin_sign = forward ? 1.0f : -1.0f;

	const int32_t* pos = (const int32_t*)src1->data;

	for (int64_t i3 = 0; i3 < ne3; i3++) {
		for (int64_t i2 = 0; i2 < ne2; i2++) {

			float* cache = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;
			if (!is_mrope) {
				const int64_t p = pos[i2];
				ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
			}
			else {
				const int64_t p_t = pos[i2];
				const int64_t p_h = pos[i2 + ne2];
				const int64_t p_w = pos[i2 + ne2 * 2];
				const int64_t p_e = pos[i2 + ne2 * 3];
				ggml_mrope_cache_init(
					p_t, p_h, p_w, p_e, sections, is_vision,
					freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
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

							const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
							ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

							const float x0 = toFloat32(src[0]);
							const float x1 = toFloat32(src[n_dims]);

							dst_data[0] = fromFloat32<ggml_fp16_t>(x0 * cos_theta - x1 * sin_theta);
							dst_data[n_dims] = fromFloat32<ggml_fp16_t>(x0 * sin_theta + x1 * cos_theta);
						}
					}
					else {
						for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
							const int64_t ic = i0 / 2;

							const float cos_theta = cache[i0 + 0];
							const float sin_theta = cache[i0 + 1];

							const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
							ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

							const float x0 = toFloat32(src[0]);
							const float x1 = toFloat32(src[n_dims / 2]);

							dst_data[0] = fromFloat32<ggml_fp16_t>(x0 * cos_theta - x1 * sin_theta);
							dst_data[n_dims / 2] = fromFloat32<ggml_fp16_t>(x0 * sin_theta + x1 * cos_theta);
						}
					}
				}
				else {
					for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
						const float cos_theta = cache[i0 + 0];
						const float sin_theta = cache[i0 + 1];

						const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
						ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

						const float x0 = toFloat32(src[0]);
						const float x1 = toFloat32(src[1]);

						dst_data[0] = fromFloat32<ggml_fp16_t>(x0 * cos_theta - x1 * sin_theta);
						dst_data[1] = fromFloat32<ggml_fp16_t>(x0 * sin_theta + x1 * cos_theta);
					}
				}

				if (is_vision) {
					for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
						const int64_t ic = i0 / 2;

						const float cos_theta = cache[i0 + 0];
						const float sin_theta = cache[i0 + 1];

						const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
						ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

						const float x0 = toFloat32(src[0]);
						const float x1 = toFloat32(src[n_dims]);

						dst_data[0] = fromFloat32<ggml_fp16_t>(x0 * cos_theta - x1 * sin_theta);
						dst_data[n_dims] = fromFloat32<ggml_fp16_t>(x0 * sin_theta + x1 * cos_theta);
					}
				}
				else {
					for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
						const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
						ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

						dst_data[0] = src[0];
						dst_data[1] = src[1];
					}
				}
			}
		}
	}
}

static void ggml_compute_forward_rope_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst,
	const bool forward) {

	const struct ggml_tensor* src0 = dst->src[0];
	const struct ggml_tensor* src1 = dst->src[1];
	const struct ggml_tensor* src2 = dst->src[2];

	float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
	int sections[4];

	//const int n_past     = ((int32_t *) dst->op_params)[0];
	const int n_dims = ((int32_t*)dst->op_params)[1];
	const int mode = ((int32_t*)dst->op_params)[2];
	//const int n_ctx      = ((int32_t *) dst->op_params)[3];
	const int n_ctx_orig = ((int32_t*)dst->op_params)[4];

	memcpy(&freq_base, (int32_t*)dst->op_params + 5, sizeof(float));
	memcpy(&freq_scale, (int32_t*)dst->op_params + 6, sizeof(float));
	memcpy(&ext_factor, (int32_t*)dst->op_params + 7, sizeof(float));
	memcpy(&attn_factor, (int32_t*)dst->op_params + 8, sizeof(float));
	memcpy(&beta_fast, (int32_t*)dst->op_params + 9, sizeof(float));
	memcpy(&beta_slow, (int32_t*)dst->op_params + 10, sizeof(float));
	memcpy(&sections, (int32_t*)dst->op_params + 11, sizeof(int) * 4);

	GGML_TENSOR_UNARY_OP_LOCALS

	//printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
	//printf("n_past = %d, ne2 = %d\n", n_past, ne2);

	GGML_ASSERT(nb00 == sizeof(float));

	const int ith = params->ith;
	const int nth = params->nth;

	const int nr = ggml_nrows(dst);

	GGML_ASSERT(n_dims <= ne0);
	GGML_ASSERT(n_dims % 2 == 0);

	// rows per thread
	const int dr = (nr + nth - 1) / nth;

	// row range for this thread
	const int ir0 = dr * ith;
	const int ir1 = std::min(ir0 + dr, nr);

	// row index used to determine which thread to use
	int ir = 0;

	const float theta_scale = powf(freq_base, -2.0f / n_dims);

	float corr_dims[2];
	ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

	const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
	const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;  // ggml_rope_multi, multimodal rotary position embedding
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
	const float sin_sign = forward ? 1.0f : -1.0f;

	const int32_t* pos = (const int32_t*)src1->data;

	for (int64_t i3 = 0; i3 < ne3; i3++) { // batch
		for (int64_t i2 = 0; i2 < ne2; i2++) { // seq-len

			float* cache = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;
			if (!is_mrope) {
				const int64_t p = pos[i2];
				ggml_rope_cache_init(p, freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
			}
			else {
				const int64_t p_t = pos[i2];
				const int64_t p_h = pos[i2 + ne2];
				const int64_t p_w = pos[i2 + ne2 * 2];
				const int64_t p_e = pos[i2 + ne2 * 3];
				ggml_mrope_cache_init(
					p_t, p_h, p_w, p_e, sections, is_vision,
					freq_scale, freq_factors, corr_dims, ne0, ext_factor, attn_factor, cache, sin_sign, theta_scale);
			}

			for (int64_t i1 = 0; i1 < ne1; i1++) { // attn-heads
				if (ir++ < ir0) continue;
				if (ir > ir1) break;

				if (is_neox || is_mrope) {
					if (is_vision) {
						for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
							const int64_t ic = i0 / 2;

							const float cos_theta = cache[i0 + 0];
							const float sin_theta = cache[i0 + 1];

							const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
							float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

							const float x0 = src[0];
							const float x1 = src[n_dims];

							dst_data[0] = x0 * cos_theta - x1 * sin_theta;
							dst_data[n_dims] = x0 * sin_theta + x1 * cos_theta;
						}
					}
					else {
						for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
							const int64_t ic = i0 / 2;

							const float cos_theta = cache[i0 + 0];
							const float sin_theta = cache[i0 + 1];

							const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
							float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

							const float x0 = src[0];
							const float x1 = src[n_dims / 2];

							dst_data[0] = x0 * cos_theta - x1 * sin_theta;
							dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
						}
					}
				}
				else {
					for (int64_t i0 = 0; i0 < n_dims; i0 += 2) {
						const float cos_theta = cache[i0 + 0];
						const float sin_theta = cache[i0 + 1];

						const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
						float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

						const float x0 = src[0];
						const float x1 = src[1];

						dst_data[0] = x0 * cos_theta - x1 * sin_theta;
						dst_data[1] = x0 * sin_theta + x1 * cos_theta;
					}
				}

				if (is_vision) {
					for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
						const int64_t ic = i0 / 2;

						const float cos_theta = cache[i0 + 0];
						const float sin_theta = cache[i0 + 1];

						const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + ic * nb00);
						float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + ic * nb0);

						const float x0 = src[0];
						const float x1 = src[n_dims];

						dst_data[0] = x0 * cos_theta - x1 * sin_theta;
						dst_data[n_dims] = x0 * sin_theta + x1 * cos_theta;
					}
				}
				else {
					// fill the remain channels with data from src tensor
					for (int64_t i0 = n_dims; i0 < ne0; i0 += 2) {
						const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
						float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

						dst_data[0] = src[0];
						dst_data[1] = src[1];
					}
				}
			}
		}
	}
}

static void ggml_compute_forward_rope(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_rope_f16(params, dst, true);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rope_f32(params, dst, true);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

static void ggml_compute_forward_rope_back(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_rope_f16(params, dst, false);
	} break;
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_rope_f32(params, dst, false);
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
		ggml_compute_forward_dup(pool, scope, params, tensor);
	} break;
	case GGML_OP_ADD:
	{
		ggml_compute_forward_add(params, tensor);
	} break;
	case GGML_OP_ADD1:
	{
		ggml_compute_forward_add1(params, tensor);
	} break;
#if 0
	case GGML_OP_ACC:
	{
		ggml_compute_forward_acc(params, tensor);
	} break;
	case GGML_OP_SUB:
	{
		ggml_compute_forward_sub(params, tensor);
	} break;
#endif
	case GGML_OP_MUL:
	case GGML_OP_DIV:
	{
		ggml_compute_forward_muldiv(params, tensor);
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
#if 0
	case GGML_OP_SUM_ROWS:
	{
		ggml_compute_forward_sum_rows(params, tensor);
	} break;
	case GGML_OP_MEAN:
	{
		ggml_compute_forward_mean(params, tensor);
	} break;
#endif
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
#if 0
	case GGML_OP_REPEAT_BACK:
	{
		ggml_compute_forward_repeat_back(params, tensor);
	} break;
	case GGML_OP_CONCAT:
	{
		ggml_compute_forward_concat(params, tensor);
	} break;
#endif
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
#if 0
	case GGML_OP_GROUP_NORM:
	{
		ggml_compute_forward_group_norm(params, tensor);
	} break;
#endif
	case GGML_OP_MUL_MAT:
	{
		ggml_compute_forward_mul_mat(pool, scope, params, tensor);
	} break;
	case GGML_OP_MUL_MAT_ID:
	{
		ggml_compute_forward_mul_mat_id(params, tensor);
	} break;
	case GGML_OP_OUT_PROD:
	{
		ggml_compute_forward_out_prod(params, tensor);
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
		ggml_compute_forward_get_rows(params, tensor);
	} break;
	case GGML_OP_GET_ROWS_BACK:
	{
		ggml_compute_forward_get_rows_back(params, tensor);
	} break;
#if 0
	case GGML_OP_DIAG:
	{
		ggml_compute_forward_diag(params, tensor);
	} break;
#endif
	case GGML_OP_DIAG_MASK_INF:
	{
		ggml_compute_forward_diag_mask_inf(params, tensor);
	} break;
#if 0
	case GGML_OP_DIAG_MASK_ZERO:
	{
		ggml_compute_forward_diag_mask_zero(params, tensor);
	} break;
#endif
	case GGML_OP_SOFT_MAX:
	{
		ggml_compute_forward_soft_max(params, tensor);
	} break;
	case GGML_OP_SOFT_MAX_BACK:
	{
		ggml_compute_forward_soft_max_ext_back(params, tensor);
	} break;
	case GGML_OP_ROPE:
	{
		ggml_compute_forward_rope(params, tensor);
	} break;
	case GGML_OP_ROPE_BACK:
	{
		ggml_compute_forward_rope_back(params, tensor);
	} break;
	case GGML_OP_CLAMP:
	{
		ggml_compute_forward_clamp(params, tensor);
	} break;
	case GGML_OP_CONV_TRANSPOSE_1D:
	{
		ggml_compute_forward_conv_transpose_1d(pool, scope, params, tensor);
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
	case GGML_OP_CONV_TRANSPOSE_2D:
	{
		GGML_ASSERT(tensor->src[0]->type == GGML_TYPE_F16);
		ggml_compute_forward_conv_transpose_2d<ggml_fp16_t>(pool, scope, params, tensor);
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
	case GGML_OP_UPSCALE:
	{
		ggml_compute_forward_upscale(params, tensor);
	} break;
	case GGML_OP_PAD:
	{
		ggml_compute_forward_pad(params, tensor);
	} break;
#endif
	case GGML_OP_PAD_REFLECT_1D:
	{
		ggml_compute_forward_pad_reflect_1d(pool, scope, tensor);
	} break;
	case GGML_OP_ARANGE:
	{
		ggml_compute_forward_arange(pool, scope, params, tensor);
	} break;
#if 0
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
		ggml_compute_forward_leaky_relu(params, tensor);
	} break;
	case GGML_OP_FLASH_ATTN_EXT:
	{
		ggml_compute_forward_flash_attn_ext(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3], tensor);
	} break;
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
		ggml_compute_forward_ssm_conv(params, tensor);
	} break;
	case GGML_OP_SSM_SCAN:
	{
		ggml_compute_forward_ssm_scan(params, tensor);
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
		ggml_compute_forward_rwkv_wkv6(params, tensor);
	} break;
	case GGML_OP_GATED_LINEAR_ATTN:
	{
		ggml_compute_forward_gla(params, tensor);
	} break;
#if 0
	case GGML_OP_MAP_UNARY:
	{
		ggml_unary_op_f32_t fun;
		memcpy(&fun, tensor->op_params, sizeof(fun));
		ggml_compute_forward_map_unary(params, tensor, fun);
	}
	break;
	case GGML_OP_MAP_BINARY:
	{
		ggml_binary_op_f32_t fun;
		memcpy(&fun, tensor->op_params, sizeof(fun));
		ggml_compute_forward_map_binary(params, tensor, fun);
	}
	break;
	case GGML_OP_MAP_CUSTOM1_F32:
	{
		ggml_custom1_op_f32_t fun;
		memcpy(&fun, tensor->op_params, sizeof(fun));
		ggml_compute_forward_map_custom1_f32(params, tensor, fun);
	}
	break;
	case GGML_OP_MAP_CUSTOM2_F32:
	{
		ggml_custom2_op_f32_t fun;
		memcpy(&fun, tensor->op_params, sizeof(fun));
		ggml_compute_forward_map_custom2_f32(params, tensor, fun);
	}
	break;
	case GGML_OP_MAP_CUSTOM3_F32:
	{
		ggml_custom3_op_f32_t fun;
		memcpy(&fun, tensor->op_params, sizeof(fun));
		ggml_compute_forward_map_custom3_f32(params, tensor, fun);
	}
	break;
	case GGML_OP_MAP_CUSTOM1:
	{
		ggml_compute_forward_map_custom1(params, tensor);
	}
	break;
	case GGML_OP_MAP_CUSTOM2:
	{
		ggml_compute_forward_map_custom2(params, tensor);
	}
	break;
	case GGML_OP_MAP_CUSTOM3:
	{
		ggml_compute_forward_map_custom3(params, tensor);
	}
	break;
	case GGML_OP_CROSS_ENTROPY_LOSS:
	{
		ggml_compute_forward_cross_entropy_loss(params, tensor);
	}
	break;
	case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
	{
		ggml_compute_forward_cross_entropy_loss_back(params, tensor);
	}
	break;
	case GGML_OP_OPT_STEP_ADAMW:
	{
		ggml_compute_forward_opt_step_adamw(params, tensor);
	}
	break;
#endif
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

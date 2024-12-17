module;
#include <assert.h>
#include <float.h>
#include <math.h>
#include <algorithm>
#include <bit>
#include <barrier>
#include <cstdlib>
#include <new>
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

static void ggml_vec_cpy_f32(const int n, float* y, const float* x) { for (int i = 0; i < n; ++i) y[i] = x[i]; }

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

							ggml_vec_cpy_f32(ne00, left, cast_with_offset<float>(src0->data, i3 * nb03 + i2 * nb02 + i1 * nb01));

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

static void ggml_compute_forward_sum_f32(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

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

static void ggml_compute_forward_sum_f16(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

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

static void ggml_compute_forward_sum_bf16(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	if (params->ith != 0) {
		return;
	}

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

static void ggml_compute_forward_sum(
	const ggml_compute_params* params,
	ggml_tensor* dst) {

	const ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_sum_f32(params, dst);
	} break;
	case GGML_TYPE_F16:
	{
		ggml_compute_forward_sum_f16(params, dst);
	} break;
	case GGML_TYPE_BF16:
	{
		ggml_compute_forward_sum_bf16(params, dst);
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
								ggml_vec_cpy_f32(ne00,
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

		ggml_vec_cpy_f32(nc,
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
	{
		ggml_compute_forward_dup(pool, scope, params, tensor);
	} break;
#if 0
	case GGML_OP_ADD:
	{
		ggml_compute_forward_add(params, tensor);
	} break;
	case GGML_OP_ADD1:
	{
		ggml_compute_forward_add1(params, tensor);
	} break;
	case GGML_OP_ACC:
	{
		ggml_compute_forward_acc(params, tensor);
	} break;
	case GGML_OP_SUB:
	{
		ggml_compute_forward_sub(params, tensor);
	} break;
	case GGML_OP_MUL:
	{
		ggml_compute_forward_mul(params, tensor);
	} break;
	case GGML_OP_DIV:
	{
		ggml_compute_forward_div(params, tensor);
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
#endif
	case GGML_OP_SUM:
	{
		ggml_compute_forward_sum(params, tensor);
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
	case GGML_OP_ARGMAX:
	{
		ggml_compute_forward_argmax(params, tensor);
	} break;
	case GGML_OP_COUNT_EQUAL:
	{
		ggml_compute_forward_count_equal(params, tensor);
	} break;
#endif
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
#endif
	case GGML_OP_MUL_MAT:
	{
		ggml_compute_forward_mul_mat(pool, scope, params, tensor);
	} break;
#if 0
	case GGML_OP_MUL_MAT_ID:
	{
		ggml_compute_forward_mul_mat_id(params, tensor);
	} break;
#endif
	case GGML_OP_OUT_PROD:
	{
		ggml_compute_forward_out_prod(params, tensor);
	} break;
#if 0
	case GGML_OP_SCALE:
	{
		ggml_compute_forward_scale(params, tensor);
	} break;
	case GGML_OP_SET:
	{
		ggml_compute_forward_set(params, tensor);
	} break;
	case GGML_OP_CPY:
	{
		ggml_compute_forward_cpy(params, tensor);
	} break;
#endif
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
		ggml_compute_forward_soft_max(params, tensor);
	} break;
	case GGML_OP_SOFT_MAX_BACK:
	{
		ggml_compute_forward_soft_max_back(params, tensor);
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
#endif
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
	case GGML_OP_SSM_CONV:
	{
		ggml_compute_forward_ssm_conv(params, tensor);
	} break;
	case GGML_OP_SSM_SCAN:
	{
		ggml_compute_forward_ssm_scan(params, tensor);
	} break;
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
	case GGML_OP_RWKV_WKV6:
	{
		ggml_compute_forward_rwkv_wkv6(params, tensor);
	} break;
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

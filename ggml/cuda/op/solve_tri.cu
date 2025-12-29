#include <span>
#include "common.cuh"
#include "cuda_func.h"
#include "launch.cuh"
#include "mdspan_helper.h"
#include "reduce.cuh"

static constexpr int64_t MAX_N_FAST = 64;
static constexpr int64_t MAX_K_FAST = 32;

void solve_tri_f32_cublas(const solve_tri_context &ctx, ggml_cuda_pool& pool, cublasHandle_t cublas_handle, cudaStream_t stream) {
    const float   alpha = 1.0f;
    const int64_t total_batches = ctx.A_ne[2] * ctx.A_ne[3];
    if (total_batches == 0) {
        return;
    }

    const int64_t n = ctx.A_ne[0];
    const int64_t k = ctx.B_ne[0];

    // Bulk copy B -> X (contiguous tensors)
    if (ctx.X != ctx.B) {
        const int64_t total_elements_BX = n * k * total_batches;
        CUDA_CHECK(cudaMemcpyAsync(ctx.X, ctx.B, total_elements_BX * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    ggml_cuda_pool_alloc<const float*> A_ptrs_alloc(pool, total_batches);
    ggml_cuda_pool_alloc<float*>       X_ptrs_alloc(pool, total_batches);

    auto A_data = make_strided_mdspan(ctx.A, ctx.A_ne, ctx.A_nb);
    auto X_data = make_strided_mdspan(ctx.X, ctx.X_ne, ctx.X_nb);
    std::span<const float*> A_ptrs { A_ptrs_alloc.get(), (size_t)total_batches };
    std::span<float*> X_ptrs { X_ptrs_alloc.get(), (size_t)total_batches };

    launch_functor(stream, std::make_tuple(ctx.A_ne[3], ctx.A_ne[2]),
        [=] __device__(int64_t i3, int64_t i2)  {
            const int64_t idx = i3 * ctx.A_ne[2] + i2;
            A_ptrs[idx] = &A_data(i3, i2, 0, 0);
            X_ptrs[idx] = &X_data(i3, i2, 0, 0);
        }
    );

    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));

    // Yes, this is necessary, without this we get RMSE errors
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasStrsmBatched(cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, k, n, &alpha, A_ptrs.data(), n, X_ptrs.data(), k, total_batches));

    // revert to standard mode from common.cuh
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
}

// ======================
// Fast Kernel (n <= 64, k <= 32) - Warp-based parallel reduction
// ======================
// When ncols_template == 0 the bounds for the loops in this function are not
// known and can't be unrolled. As we want to keep pragma unroll for all other
// cases we supress the clang transformation warning here.
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wpass-failed"
#endif  // __clang__
template <int n_template, int k_template>
static __global__ void solve_tri_f32_fast(solve_tri_context ctx,
    const uint3  ne02,
    const int    n_arg,
    const int    k_arg) {
    const int n = n_template == 0 ? n_arg : n_template;
    const int k = k_template == 0 ? k_arg : k_template;

    const int batch_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int col_idx = threadIdx.y;

    if (col_idx >= k) {
        return;
    }

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<WARP_SIZE>(block);

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02 = i02_i03.y;
    const int64_t i03 = i02_i03.x;

    auto A_data = make_strided_mdspan(ctx.A, ctx.A_ne, ctx.A_nb);
    auto A_batch = std::submdspan(A_data, i03, i02, std::full_extent, std::full_extent);
    auto B_data = make_strided_mdspan(ctx.B, ctx.B_ne, ctx.B_nb);
    auto B_batch = std::submdspan(B_data, i03, i02, std::full_extent, std::full_extent);
    auto X_data = make_strided_mdspan(ctx.X, ctx.X_ne, ctx.X_nb);
    auto X_batch = std::submdspan(X_data, i03, i02, std::full_extent, std::full_extent);

    __shared__ float sA[MAX_N_FAST * MAX_N_FAST];

    const int offset = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
    for (int i = 0; i < n * n; i += k * WARP_SIZE) {
        const int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch(i0 / n, i0 % n);
        }
    }

    block.sync();

    float x_low = (lane < n) ? B_batch(lane, col_idx) : 0.0f;
    float x_high = (WARP_SIZE + lane < n) ? B_batch((WARP_SIZE + lane), col_idx) : 0.0f;

    const int half = WARP_SIZE;
    const int nrows_low = (n < half) ? n : half;

#pragma unroll
    for (int row = 0; row < nrows_low; ++row) {
        float sum = 0.0f;
        if (lane < row) {
            sum += sA[row * n + lane] * x_low;
        }
        sum = cooperative_groups::reduce(tile, sum, cooperative_groups::plus<float>());

        if (lane == row) {
            x_low = (x_low - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int row = half; row < n; ++row) {
        float     sum = sA[row * n + lane] * x_low;
        const int j = half + lane;
        if (j < row) {
            sum += sA[row * n + j] * x_high;
        }
        sum = cooperative_groups::reduce(tile, sum, cooperative_groups::plus<float>());

        if (lane == row - half) {
            x_high = (x_high - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int row = rr * WARP_SIZE + lane;
        if (row < n) {
            const float val = (row < half) ? x_low : x_high;
            X_batch(row, col_idx) = val;
        }
    }
}
#ifdef __clang__
#    pragma clang diagnostic pop
#endif  // __clang__

void solve_tri_f32_fast_cuda(const solve_tri_context &ctx, cudaStream_t  stream) {
    const uint3 ne02_fd = init_fastdiv_values((uint32_t)ctx.A_ne[2]);
    const int64_t n = ctx.A_ne[0];
    const int64_t k = ctx.B_ne[0];
    dim3        threads(WARP_SIZE, k);
    dim3        grid(ctx.A_ne[2] * ctx.A_ne[3]);
    if (n == 64) {
        switch (k) {
        case 32:
            solve_tri_f32_fast<64, 32>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 16:
            solve_tri_f32_fast<64, 16>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 14:
            solve_tri_f32_fast<64, 14>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 12:
            solve_tri_f32_fast<64, 12>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 10:
            solve_tri_f32_fast<64, 10>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 8:
            solve_tri_f32_fast<64, 8>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 6:
            solve_tri_f32_fast<64, 6>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 4:
            solve_tri_f32_fast<64, 4>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 2:
            solve_tri_f32_fast<64, 2>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        case 1:
            solve_tri_f32_fast<64, 1>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, 0, 0);
            break;
        default:
            solve_tri_f32_fast<0, 0>
                << <grid, threads, 0, stream >> > (ctx, ne02_fd, n, k);
        }
    }
    else {  // run general case
        solve_tri_f32_fast<0, 0>
            << <grid, threads, 0, stream >> > (ctx, ne02_fd, n, k);
    }
}

void solve_tri_f32_cuda(const solve_tri_context& ctx, ggml_cuda_pool& pool, cublasHandle_t cublas_handle, cudaStream_t stream) {
    const int64_t n = ctx.A_ne[0];
    const int64_t k = ctx.B_ne[0];

    if (n <= MAX_N_FAST && k <= MAX_K_FAST) {
        solve_tri_f32_fast_cuda(ctx, stream);
    }
    else {
        solve_tri_f32_cublas(ctx, pool, cublas_handle, stream);
    }
}
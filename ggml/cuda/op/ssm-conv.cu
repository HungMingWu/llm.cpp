#include "cuda_func.h"
#include "mdspan_helper.h"
#include "unary.cuh"
#define GGML_ABORT(...)

template <bool apply_silu, size_t split_d_inner, size_t d_conv, typename src0_t, typename src1_t, typename out_t>
static __global__ void ssm_conv_f32(
    const int64_t n_t, src0_t src0_data, src1_t src1_data, const float* bias, out_t out_data) {
    const int tid = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = src1_data(bidy * split_d_inner + tid, j);
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = src0_data(bidx, bidy * split_d_inner + tid, j);
            }
        }
        else {
            x[(i - 1) % d_conv] = src0_data(bidx, bidy * split_d_inner + tid, i + d_conv - 1);// x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        out_data(bidx, i, bidy * split_d_inner + tid) = apply_silu ? silu(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t, typename src0_t, typename src1_t, typename out_t>
static __global__ void ssm_conv_long_token_f32(const int64_t n_t, src0_t src0_data, src1_t src1_data, const float* bias, out_t out_data) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const auto x_block = std::submdspan(src0_data,
    bidx, std::pair { bidy * split_d_inner, src0_data.extent(1) }, 
    std::pair { bidz * split_n_t, src0_data.extent(2) });
    const auto w_block = std::submdspan(src1_data, 
        std::pair { bidy * split_d_inner, src1_data.extent(0) }, std::full_extent);
    auto y_block = std::submdspan(out_data,
    bidx, std::pair { bidz * split_n_t, bidz * split_n_t + local_n_t }, 
    std::pair { bidy * split_d_inner, out_data.extent(2) });

    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block(row, col);
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block(tid, j);
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block(i, tid) = apply_silu ? silu(sumf) : sumf;
    }
}

template <bool apply_silu>
void ssm_conv_f32_cuda(const ssm_conv_context& ctx, cudaStream_t stream) {
    const int threads = 128;
    assert(ctx.nr % threads == 0);

    auto src0_data = make_strided_mdspan<3>(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<2>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    const float * bias = ctx.bias_d;
    auto out_data = make_strided_mdspan<3>(ctx.out_d, ctx.out_ne, ctx.out_nb);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (ctx.n_t <= 32) {
            const dim3 blocks(ctx.n_s, (ctx.nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC> << <blocks, threads, 0, stream >> > (ctx.n_t, src0_data, src1_data, bias, out_data);
        }
        else {
            const int64_t split_n_t = 32;
            dim3          blocks(ctx.n_s, (ctx.nr + threads - 1) / threads, (ctx.n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                ctx.n_t, src0_data, src1_data, bias, out_data);
        }
    };

    switch (ctx.nc) {
    case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
    case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
    case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
    case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
    default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

void ssm_conv_f32_cuda(const ssm_conv_context& ctx, cudaStream_t stream) {
    if (ctx.fuse_silu) {
        ssm_conv_f32_cuda<true>(ctx, stream);
    } else {
        ssm_conv_f32_cuda<false>(ctx, stream);
    }
}
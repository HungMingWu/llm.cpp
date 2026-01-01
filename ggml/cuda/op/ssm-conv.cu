#include "cuda_func.h"
#include "mdspan_helper.h"
#define GGML_ABORT(...)

template <size_t split_d_inner, size_t d_conv, typename src0_t, typename src1_t, typename dst_t>
static __global__ void ssm_conv_f32(
    const int64_t n_t, src0_t src0_data, src1_t src1_data, dst_t dst_data) {
    const int tid = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = src1_data(bidy * split_d_inner + tid, j);
    }

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
        dst_data(bidx, i, bidy * split_d_inner + tid) = sumf;
    }
}

template <size_t split_d_inner, size_t d_conv, int64_t split_n_t, typename src0_t, typename src1_t, typename dst_t>
static __global__ void ssm_conv_long_token_f32(
    const int64_t n_t, src0_t src0_data, src1_t src1_data, dst_t dst_data) {
    const int tid = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = src1_data(bidy * split_d_inner + tid, j);
    }

#pragma unroll
    for (int64_t i = 0; i < split_n_t; i++) {
        if (bidz * split_n_t + i < n_t) {
            float sumf = 0.0f;

            if (i == 0) {
                for (size_t j = 0; j < d_conv; j++) {
                    x[j] = src0_data(bidx, bidy * split_d_inner + tid, bidz * split_n_t + j);
                }
            }
            else {
                x[(i - 1) % d_conv] = src0_data(bidx, bidy * split_d_inner + tid, bidz * split_n_t + i + d_conv - 1);
            }

#pragma unroll
            for (size_t j = 0; j < d_conv; j++) {
                sumf += x[(i + j) % d_conv] * w[j];
            }
            dst_data(bidx, bidz * split_n_t + i, bidy * split_d_inner + tid) = sumf;
        }
    }
}

void ssm_conv_f32_cuda(const ssm_conv_context& ctx, cudaStream_t stream) {
    const int threads = 128;
    assert(ctx.nr % threads == 0);

    auto src0_data = make_strided_mdspan<3>(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan<2>(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto dst_data = make_strided_mdspan<3>(ctx.dst_d, ctx.dst_ne, ctx.dst_nb);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (ctx.n_t <= 32) {
            const dim3 blocks(ctx.n_s, (ctx.nr + threads - 1) / threads, 1);
            ssm_conv_f32<threads, kNC> << <blocks, threads, 0, stream >> > (ctx.n_t, src0_data, src1_data, dst_data);
        }
        else {
            const int64_t split_n_t = 32;
            dim3          blocks(ctx.n_s, (ctx.nr + threads - 1) / threads, (ctx.n_t + split_n_t - 1) / split_n_t);
            ssm_conv_long_token_f32<threads, kNC, split_n_t> << <blocks, threads, 0, stream >> > (
                ctx.n_t, src0_data, src1_data, dst_data);
        }
    };

    switch (ctx.nc) {
    case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
    case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
    case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
    default: GGML_ABORT("Only support kernel sizes 3, 4, 9 right now.");
    }
}
#include "cuda_func.h"
#include "common.cuh"
#include "helper.h"
#include "reduce.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

template <size_t splitD, size_t N>
__global__ void __launch_bounds__(splitD, 1)
ssm_scan_f32(
    const int32_t* __restrict__ src6,
    const int64_t d_inner, const int64_t L_param, 
    auto src0_data, auto src1_data, auto src2_data, auto src3_data, auto src4_data, auto src5_data, auto y_data, auto s_data)
{
    auto block = cooperative_groups::this_thread_block();
    const size_t L = L_param;

    float regA[N];
    float regs0[N];

    __shared__ float smemB[N];
    __shared__ float smemC[N];

#ifdef USE_CUB
    using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    union CubTempStorage {
        typename BlockLoad::TempStorage load_temp;
        typename BlockStore::TempStorage store_temp;
    };
    __shared__ CubTempStorage cub_temp_storage;

    BlockLoad(cub_temp_storage.load_temp).Load(&src3_data(blockIdx.y* splitD, 0), regA);
    BlockLoad(cub_temp_storage.load_temp).Load(&src0_data(src6[blockIdx.x], blockIdx.y* splitD, 0, 0), regs0);
#else
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        regA[n] = src3_data(blockIdx.y * splitD + threadIdx.x, n);
        regs0[n] = src0_data(src6[blockIdx.x], blockIdx.y * splitD, threadIdx.x, n);
    }
#endif

#pragma unroll
    for (size_t i = 0; i < L; i++)
    {
        if (threadIdx.x < N)
        {
            smemB[threadIdx.x] = src4_data(blockIdx.x, i, 0, threadIdx.x);
            smemC[threadIdx.x] = src5_data(blockIdx.x, i, 0, threadIdx.x);
        }
        block.sync();
        const int64_t h = blockIdx.y * splitD + threadIdx.x;
        float dt_soft_plus = src2_data(blockIdx.x, i, h);
        if (dt_soft_plus <= 20.0f)
        {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        float x_dt = src1_data(blockIdx.x, i, h, 0) * dt_soft_plus; 

        float sumf = 0.0f;
#pragma unroll
        for (size_t n = 0; n < N; n++)
        {
            float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_data(blockIdx.x, i, h, 0) = sumf;
    }

    const int64_t h = blockIdx.y * splitD;
#ifdef USE_CUB
    BlockStore(cub_temp_storage.store_temp).Store(&s_data(blockIdx.x, h, 0, 0), regs0);
#else
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        s_data(blockIdx.x, blockIdx.y* splitD, threadIdx.x, n) = regs0[n];;
    }
#endif
}

// assumes as many threads as d_state
template <int splitH, int d_state>
__global__ void __launch_bounds__(d_state, 1)
ssm_scan_f32_group(
    const int32_t* __restrict__ src6,
    const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok,
    auto src0_data, auto src1_data, auto src2_data, auto src3_data, auto src4_data, auto src5_data,
    auto y_data, auto s_data) {

    const int head_idx = (blockIdx.x * splitH) / d_head;
    const int head_off = ((blockIdx.x * splitH) % d_head);
    const int seq_idx = blockIdx.y;

    const int group_off = head_idx / (n_head / n_group);

    float state[splitH];
    // for the parallel accumulation
    __shared__ float stateC[splitH * d_state];
    std::experimental::mdspan stateC_data(stateC, splitH, d_state);

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);

#pragma unroll
    for (int j = 0; j < splitH; j++) {
        state[j] = src0_data(src6[seq_idx], head_idx, head_off + j, threadIdx.x);// s0_block[j * d_state + ];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // TODO: only calculate dA and dt_soft_plus once per head instead of every splitH head elements
        // TODO: only calculate B and C once per head group
        // NOTE: dt_soft_plus, dA and x_dt have the same value across threads here.
        float dt_soft_plus = src2_data(seq_idx, i, head_idx);
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        const float dA = expf(dt_soft_plus * src3_data(head_idx, 0));
        const float B = src4_data(seq_idx, i, group_off, threadIdx.x);
        const float C = src5_data(seq_idx, i, group_off, threadIdx.x);

        // across d_head
#pragma unroll
        for (int j = 0; j < splitH; j++) {
            const float x_dt = src1_data(seq_idx, i, head_idx, head_off + j) * dt_soft_plus;

            state[j] = (state[j] * dA) + (B * x_dt);

            stateC_data(j, threadIdx.x) = state[j] * C;
        }

        block.sync();

        // parallel accumulation for stateC
        // TODO: simplify
        {
            static_assert((d_state & -d_state) == d_state, "the state size has to be a power of 2");
            static_assert((splitH & -splitH) == splitH, "splitH has to be a power of 2");

            // reduce until w matches the warp size
            // TODO: does this work even when the physical warp size is 64?
#pragma unroll
            for (int w = d_state; w > WARP_SIZE; w >>= 1) {
                // (assuming there are d_state threads)
#pragma unroll
                for (int j = 0; j < ((w >> 1) * splitH + d_state - 1) / d_state; j++) {
                    // TODO: check for bank conflicts
                    const int k = (threadIdx.x % (w >> 1)) + (d_state * (threadIdx.x / (w >> 1))) + j * d_state * (d_state / (w >> 1));
                    //stateC[k] += stateC[k + (w >> 1)];
                    stateC_data(threadIdx.x / (w >> 1) + j * (d_state / (w >> 1)), threadIdx.x % (w >> 1)) +=
                        stateC_data(threadIdx.x / (w >> 1) + j * (d_state / (w >> 1)), threadIdx.x % (w >> 1) + (w >> 1));

                }
                block.sync();
            }

            static_assert(splitH >= d_state / WARP_SIZE);
#pragma unroll
            for (int j = 0; j < splitH / (d_state / WARP_SIZE); j++) {
                const float y = cooperative_groups::reduce(tile,
                    stateC_data(threadIdx.x / WARP_SIZE + j * (d_state / WARP_SIZE), threadIdx.x % WARP_SIZE),
                    cooperative_groups::plus<float>());;

                // store the above accumulations
                if (threadIdx.x % WARP_SIZE == 0) {
                    const int k = threadIdx.x / WARP_SIZE + j * (d_state / WARP_SIZE);
                    y_data(seq_idx, i, head_idx, head_off + k) = y;
                }
            }
        }
    }

    // write back the state
#pragma unroll
    for (int j = 0; j < splitH; j++) {
        s_data(seq_idx, head_idx, head_off + j, threadIdx.x) = state[j];
    }
}

void ssm_scan_f32_cuda(const ssm_scan_context& ctx, cudaStream_t stream) {
    const int threads = 128;
    std::experimental::mdspan y_data(static_cast<float*>(ctx.dst_d), ctx.n_seq, ctx.n_tok, ctx.n_head, ctx.head_dim);
    auto s_data = make_strided_mdspan((float*)((char *)ctx.dst_d + ctx.s_off), ctx.src0_ne, ctx.src0_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.src0_ne, ctx.src0_nb);
    auto src1_data = make_strided_mdspan(ctx.src1_d, ctx.src1_ne, ctx.src1_nb);
    auto src2_data = make_strided_mdspan<3>(ctx.src2_d, ctx.src2_ne, ctx.src2_nb);
    auto src3_data = make_strided_mdspan<2>(ctx.src3_d, ctx.src3_ne, ctx.src3_nb);
    auto src4_data = make_strided_mdspan(ctx.src4_d, ctx.src4_ne, ctx.src4_nb);
    auto src5_data = make_strided_mdspan(ctx.src5_d, ctx.src5_ne, ctx.src5_nb);
    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    if (ctx.src3_nb[1] == sizeof(float)) {
        // Mamba-2
        if (ctx.d_state == 128) {
            GGML_ASSERT(ctx.d_state % threads == 0);
            // NOTE: can be any power of two between 4 and 64
            const int splitH = 16;
            GGML_ASSERT(ctx.head_dim % splitH == 0);
            const dim3 blocks((ctx.n_head * ctx.head_dim + (splitH - 1)) / splitH, ctx.n_seq, 1);
            ssm_scan_f32_group<16, 128> << <blocks, threads, 0, stream >> > (
                ctx.src6_d, ctx.n_head, ctx.head_dim,
                ctx.n_group, ctx.n_tok, src0_data, src1_data, src2_data, src3_data, src4_data, src5_data, y_data, s_data);
        }
        else if (ctx.d_state == 256) { // Falcon-H1
            const int threads = 256;
            // NOTE: can be any power of two between 8 and 64
            const int splitH = 16;
            GGML_ASSERT(ctx.head_dim % splitH == 0);
            const dim3 blocks((ctx.n_head * ctx.head_dim + (splitH - 1)) / splitH, ctx.n_seq, 1);
            ssm_scan_f32_group<16, 256> << <blocks, threads, 0, stream >> > (
                ctx.src6_d, ctx.n_head, ctx.head_dim,
                ctx.n_group, ctx.n_tok, src0_data, src1_data, src2_data, src3_data, src4_data, src5_data, y_data, s_data);
        }
        else {
            GGML_ABORT("doesn't support d_state!=(128 or 256).");
        }
    }
    else {
        // Mamba-1
        GGML_ASSERT(ctx.n_head % threads == 0);
        GGML_ASSERT(ctx.head_dim == 1);
        GGML_ASSERT(ctx.n_group == 1);
        const dim3 blocks(ctx.n_seq, (ctx.n_head + threads - 1) / threads, 1);
        const int  smem_size = (threads * (ctx.d_state + 1) * 2) * sizeof(float);
        if (ctx.d_state == 16) {
            ssm_scan_f32<threads, 16> << <blocks, threads, smem_size, stream >> > (
                ctx.src6_d, ctx.n_head, ctx.n_tok,
                src0_data, src1_data, src2_data, src3_data, src4_data, src5_data, y_data, s_data);
        }
        else {
            GGML_ABORT("doesn't support d_state!=16.");
        }
    }
}
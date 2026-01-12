#include <utility>
#include "cuda_func.h"
#include "common.cuh"
#include "convert.cuh"
#include "mdspan_helper.h"
#include "reduce.cuh"

static constexpr size_t CUDA_SOFT_MAX_BLOCK_SIZE = 1024;
#define GGML_PAD1(x, n) (((x) + (n) - 1) & ~((n) - 1))

template <bool use_shared, typename src_t, typename mask_t, typename dst_t>
static __global__ void soft_max_f32(
    src_t x, mask_t mask, std::span<const float> sinks, dst_t dst, const soft_max_params p) {
    const int ncols = p.ncols;

    const int64_t i03 = blockIdx.z;
    const int64_t i02 = blockIdx.y;
    const int64_t i01 = blockIdx.x;

    auto block = cooperative_groups::this_thread_block();
    const int tid = block.thread_rank();
    auto tile = cooperative_groups::tiled_partition<32>(block);

    const int64_t i11 = i01;
    const int64_t i12 = i02 % p.src1_ne[2];
    const int64_t i13 = i03 % p.src1_ne[3];

    const int block_size = blockDim.x;

    const int tile_id = tid / tile.size();
    const int lane_id = tile.thread_rank();

    const float slope = get_alibi_slope(p.max_bias, i02, p.n_head_log2, p.m0, p.m1);

    extern __shared__ float data_soft_max_f32[];
    float* buf_iw = data_soft_max_f32; // shared memory buffer for inter-warp communication
    // shared memory buffer to cache values between iterations:
    float* vals = use_shared ? buf_iw + WARP_SIZE : &dst(i03, i02, i01, 0);

    float max_val = !sinks.empty() ? sinks[i02] : -INFINITY;

#pragma unroll
    for (int col = tid; col < ncols; col += block_size) {
        const float val = [&]() {
            float val = x(i03, i02, i01, col) * p.scale;
            if (!mask.empty()) {
                val += slope * ggml_cuda_cast<float>(mask(i13, i12, i11, col));
            }
            return val;
        }();

        vals[col] = val;
        max_val = max(max_val, val);
    }

    // find the max value in the block
    max_val = reduceWithBlock<cooperative_groups::greater>(block, tile, -INFINITY, max_val, buf_iw);

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col = tid; col < ncols; col += block_size) {
        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    // find the sum of exps in the block
    tmp = reduceWithBlock<cooperative_groups::plus>(block, tile, 0.0f, tmp, buf_iw);

    if (!sinks.empty()) {
        tmp += expf(sinks[i02] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col = tid; col < ncols; col += block_size) {
        dst(i03, i02, i01, col) = vals[col] * inv_sum;
    }
}

// TODO: This is a common pattern used across kernels that could be moved to common.cuh + templated
static __device__ float two_stage_warp_reduce_max(float val) {
    val = warp_reduce_max(val);
    if (blockDim.x > WARP_SIZE) {
        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
        __shared__ float local_vals[32];
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            local_vals[warp_id] = val;
        }
        __syncthreads();
        val = -INFINITY;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            val = local_vals[lane_id];
        }
        return warp_reduce_max(val);
    }
    else {
        return val;
    }
}

static __device__ float two_stage_warp_reduce_sum(float val) {
    val = warp_reduce_sum(val);
    if (blockDim.x > WARP_SIZE) {
        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
        __shared__ float local_vals[32];
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            local_vals[warp_id] = val;
        }
        __syncthreads();
        val = 0.0f;
        if (lane_id < (static_cast<int>(blockDim.x) / WARP_SIZE)) {
            val = local_vals[lane_id];
        }
        return warp_reduce_sum(val);
    }
    else {
        return val;
    }
}

// TODO: Template to allow keeping ncols in registers if they fit
static __device__ void soft_max_f32_parallelize_cols_single_row(const float* __restrict__ x,
    float* __restrict__ dst,
    float* __restrict__ tmp_maxs,
    float* __restrict__ tmp_sums,
    const soft_max_params p) {
    namespace cg = cooperative_groups;

    const cg::grid_group g = cg::this_grid();

    const int tid = threadIdx.x;
    const int col_start = blockIdx.x * blockDim.x + tid;
    const int n_elem_per_thread = 4;

    float     local_vals[n_elem_per_thread] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float     local_max = -INFINITY;
    const int step_size = gridDim.x * blockDim.x;

    // Compute thread-local max
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            local_max = fmaxf(local_max, local_vals[i]);
        }
        col += step_size * n_elem_per_thread;
    }

    // Compute CTA-level max
    local_max = two_stage_warp_reduce_max(local_max);

    // Store CTA-level max to GMEM
    if (tid == 0) {
        tmp_maxs[blockIdx.x] = local_max;
    }
    g.sync();

    // Compute compute global max from CTA-level maxs
    assert(gridDim.x < blockDim.x);  // currently we only support this case
    if (tid < gridDim.x) {
        local_max = tmp_maxs[tid];
    }
    else {
        local_max = -INFINITY;
    }
    local_max = two_stage_warp_reduce_max(local_max);

    // Compute softmax dividends, accumulate divisor
    float tmp_expf = 0.0f;
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? x[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.ncols) {
                const float tmp = expf(local_vals[i] - local_max);
                tmp_expf += tmp;
                dst[idx] = tmp;
            }
        }
        col += step_size * n_elem_per_thread;
    }

    // Reduce divisor within CTA
    tmp_expf = two_stage_warp_reduce_sum(tmp_expf);

    // Store CTA-level sum to GMEM
    if (tid == 0) {
        tmp_sums[blockIdx.x] = tmp_expf;
    }
    g.sync();

    // Compute global sum from CTA-level sums
    if (tid < gridDim.x) {
        tmp_expf = tmp_sums[tid];
    }
    else {
        tmp_expf = 0.0f;
    }
    tmp_expf = two_stage_warp_reduce_sum(tmp_expf);

    // Divide dividend by global sum + store data
    for (int col = col_start; col < p.ncols;) {
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            local_vals[i] = idx < p.ncols ? dst[idx] : -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_elem_per_thread; i++) {
            const int idx = col + i * step_size;
            if (idx < p.ncols) {
                dst[idx] = local_vals[i] / tmp_expf;
            }
        }
        col += step_size * n_elem_per_thread;
    }
}

__launch_bounds__(8 * WARP_SIZE, 1) static __global__ void soft_max_f32_parallelize_cols(const float* __restrict__ x,
    float* __restrict__ dst,
    float* __restrict__ tmp_maxs,
    float* __restrict__ tmp_sums,
    const soft_max_params p)
    // We loop over all instead of parallelizing across gridDim.y as cooperative groups
    // currently only support synchronizing the complete grid if not launched as a cluster group
    // (which requires CC > 9.0)
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#grid-synchronization
    // https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html#class-cluster-group
{
    for (int rowx = 0; rowx < p.src0_ne[1] * p.src0_ne[2] * p.src0_ne[3]; rowx++) {
        soft_max_f32_parallelize_cols_single_row(x + int64_t(rowx) * p.ncols, dst + int64_t(rowx) * p.ncols, tmp_maxs,
            tmp_sums, p);
    }
}

template <typename T>
static void soft_max_f32_cuda(const softmax_context& ctx, cudaStream_t stream) {
    int nth = WARP_SIZE;
    const int64_t ncols_x = ctx.params.ncols;

    while (nth < ncols_x && nth < CUDA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth, 1, 1);
    const dim3 block_nums(ctx.params.src0_ne[1], ctx.params.src0_ne[2], ctx.params.src0_ne[3]);
    const size_t nbytes_shared = (GGML_PAD1(ncols_x, WARP_SIZE) + WARP_SIZE) * sizeof(float);
    static_assert(CUDA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");


    const int id = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    auto dst_data = make_strided_mdspan(ctx.dst_d, ctx.params.dst_ne, ctx.params.dst_nb);
    auto src0_data = make_strided_mdspan(ctx.src0_d, ctx.params.src0_ne, ctx.params.src0_nb);
    auto mask = [&]() {
        using dst_t = decltype(make_strided_mdspan(static_cast<const T*>(ctx.src1_d), ctx.params.src1_ne, ctx.params.src1_nb));
        if (!ctx.src1_d) return dst_t{};
        else {
            return make_strided_mdspan(static_cast<const T*>(ctx.src1_d), ctx.params.src1_ne, ctx.params.src1_nb);
        }
    }();

    auto sinks = [&]() -> std::span<const float> {
        if (!ctx.src2_d) return {};
        else {
            assert(ctx.params.src2_ne[0] == ctx.params.src0_ne[2]);
            return { static_cast<const float*>(ctx.src2_d), static_cast<size_t>(ctx.params.src2_ne[0]) };
        }
    }();
    if (nbytes_shared <= smpbo) {
        soft_max_f32<true> << <block_nums, block_dims, nbytes_shared, stream >> > (src0_data,
            mask, sinks, dst_data, ctx.params);
    }
    else {
        // Parallelize across SMs for top-p/dist-sampling
        // The heuristic for parallelizing rows across SMs vs parallelizing single row & looping over all rows was done on the basis of a B6000 GPU and
        // Can be adapted further for lower-SM-count GPUs, though keeping data in registers should be implemented first as that is the optimal solution.
        if (ggml_cuda_info().devices[id].supports_cooperative_launch &&
            ncols_x / (ctx.params.src0_ne[1] * ctx.params.src0_ne[2] * ctx.params.src0_ne[3]) > 8192 && mask.empty() && sinks.empty() &&
            ctx.params.scale == 1.0f && ctx.params.max_bias == 0.0f) {
            ggml_cuda_pool_alloc<float> tmp_maxs_alloc(ctx.pool, ggml_cuda_info().devices[id].nsm * sizeof(float));
            ggml_cuda_pool_alloc<float> tmp_sums_alloc(ctx.pool, ggml_cuda_info().devices[id].nsm * sizeof(float));

            void* kernel_args[] = { (void*)&ctx.src0_d, (void*)&ctx.dst_d, (void*)&tmp_maxs_alloc.ptr,
                                     (void*)&tmp_sums_alloc.ptr, (void*) &ctx.params };
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)soft_max_f32_parallelize_cols,
                        dim3(ggml_cuda_info().devices[id].nsm, 1, 1),
                        dim3(WARP_SIZE * 8, 1, 1), kernel_args, 0, stream));
        }
        else {
            const size_t nbytes_shared_low = WARP_SIZE * sizeof(float);

            soft_max_f32<false> << <block_nums, block_dims, nbytes_shared_low, stream >> > (src0_data,
                mask, sinks, dst_data, ctx.params);
        }
    }
}

void soft_max_f32_cuda(const softmax_context &ctx, cudaStream_t stream)
{
    if (ctx.use_f16) {
        soft_max_f32_cuda<half>(ctx, stream);
    }
    else {
        soft_max_f32_cuda<float>(ctx, stream);
    }
}

static __global__ void soft_max_back_f32(
    auto grad, auto dstf, auto dst, const int ncols, const float scale) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    float dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dgf_dot += dstf(row, col) * grad(row, col);
    }

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<32>(block);
    dgf_dot = cooperative_groups::reduce(tile, dgf_dot, cooperative_groups::plus<float>{});

    for (int col = tid; col < ncols; col += WARP_SIZE) {
        dst(row, col) = scale * (grad(row, col) - dgf_dot) * dstf(row, col);
    }
}

void soft_max_back_f32_cuda(const softmax_back_context &ctx, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(ctx.nrows, 1, 1);

    std::mdspan grad(ctx.src0_d, ctx.nrows, ctx.ncols);
    std::mdspan dstf(ctx.src1_d, ctx.nrows, ctx.ncols);
    std::mdspan dst(ctx.dst_d, ctx.nrows, ctx.ncols);
    soft_max_back_f32 << <block_nums, block_dims, 0, stream >> > (grad, dstf, dst, ctx.ncols, ctx.scale);
}

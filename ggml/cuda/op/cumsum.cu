#include <algorithm>
#include "common.cuh"
#include "convert.cuh"
#include "cuda_func.h"
#include "mdspan_helper.h"
#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
#include <cooperative_groups/reduce.h>

#define GGML_ABORT(...)

#ifdef GGML_CUDA_USE_CUB
#   include <cub/cub.cuh>
#endif // GGML_CUDA_USE_CUB

template<typename T, int BLOCK_SIZE>
static __global__ void cumsum_cub_kernel(
    mdspan_stiride_t<const T, 4> src_data, mdspan_stiride_t<T, 4> dst_data) {
#ifdef GGML_CUDA_USE_CUB
    using BlockScanT = cub::BlockScan<T, BLOCK_SIZE>;
    auto block = cooperative_groups::this_thread_block();
    __shared__ typename BlockScanT::TempStorage temp_storage;
    __shared__ T block_carry;

    const int tid = threadIdx.x;
    constexpr int UNROLL_FACTOR = 4;
    constexpr int TILE_SIZE = BLOCK_SIZE * UNROLL_FACTOR;

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y;
    const int64_t i3 = blockIdx.z;

    if (i1 >= src_data.extent(2) || i2 >= src_data.extent(1) || i3 >= src_data.extent(0)) {
        return;
    }

    if (tid == 0) {
        block_carry = 0;
    }
    block.sync();

    for (int64_t start = 0; start < src_data.extent(3); start += TILE_SIZE) {
        T items[UNROLL_FACTOR];
        T thread_sum = T(0);

#pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
            int64_t idx = start + tid * UNROLL_FACTOR + i;
            T val = (idx < src_data.extent(3)) ? src_data(i3, i2, i1, idx) : T(0);
            thread_sum += val;
            items[i] = thread_sum;
        }

        // Block-wide scan on thread sums
        T thread_prefix;
        T block_total;
        BlockScanT(temp_storage).InclusiveSum(thread_sum, thread_prefix, block_total);
        block.sync();

        // Add offset to each item and store
        T thread_offset = thread_prefix - thread_sum + block_carry;
#pragma unroll
        for (int i = 0; i < UNROLL_FACTOR; i++) {
            int64_t idx = start + tid * UNROLL_FACTOR + i;
            if (idx < src_data.extent(3)) {
                dst_data(i3, i2, i1, idx) = items[i] + thread_offset;
            }
        }

        block.sync();

        // Update carry for next tile
        if (tid == 0) {
            block_carry += block_total;
        }
    }
#else
    NO_DEVICE_CODE;
#endif // GGML_CUDA_USE_CUB
}

// Fallback kernel implementation
template <typename T>
static __global__ void cumsum_kernel(
    mdspan_stiride_t<const T, 4> src_data, mdspan_stiride_t<T, 4> dst_data) {

    const int tid = threadIdx.x;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float smem[];
    float* s_vals = smem;
    float* s_warp_sums = smem + blockDim.x;
    float* s_carry = smem + blockDim.x + warps_per_block;
    float* s_chunk_total = s_carry + 1;

    // Initialize carry
    if (tid == 0) {
        *s_carry = 0.0f;
    }
    __syncthreads();

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;
    if (i3 >= src_data.extent(0) || i2 >= src_data.extent(1) || i1 >= src_data.extent(2)) {
        return;
    }

    // register blocking: process 4 elements per thread to hide latency
    // and reduce synchronization overhead
    constexpr int num_unroll = 4;
    T             temp[num_unroll];

    auto block = cooperative_groups::this_thread_block();
    auto tile = cooperative_groups::tiled_partition<warp_size>(block);

    for (int64_t i = 0; i < src_data.extent(3); i += num_unroll * blockDim.x) {
        int64_t idx = i + tid * num_unroll;

        // thread local sequential scan
        temp[0] = (idx < src_data.extent(3) ? src_data(i3, i2, i1, idx) : T(0));
#pragma unroll
        for (int64_t j = 1; j < num_unroll; j++) {
            temp[j] = temp[j - 1];
            if (idx + j < src_data.extent(3)) {
                temp[j] += src_data(i3, i2, i1, idx + j);
            }
            else {
                temp[j] += 0;
            }
        }

        // last emenent is sum of all values assigned to thread
        float val = (idx < src_data.extent(3)) ? ggml_cuda_cast<float, T>(temp[num_unroll - 1]) : 0.0f;

        // Warp inclusive scan
        val = cooperative_groups::inclusive_scan(tile, val, cooperative_groups::plus<float>());
        s_vals[tid] = val;

        if (lane == warp_size - 1) {
            s_warp_sums[warp] = val;
        }
        block.sync();

        // Exclusive scan of warp sums (warp 0 only)
        if (warp == 0) {
            float w = (tid < warps_per_block) ? s_warp_sums[tid] : 0.0f;
            float inc = cooperative_groups::inclusive_scan(tile, w, cooperative_groups::plus<float>());
            if (tid < warps_per_block) {
                s_warp_sums[tid] = inc - w;   // exclusive sum
            }
            if (tid == warps_per_block - 1) {
                *s_chunk_total = inc;          // total sum of this chunk
            }
        }
        block.sync();

        // write back results
        float carry = *s_carry;
        // calculate sum offset for this thread
        float final_val_offset = s_vals[tid] + s_warp_sums[warp] + carry - temp[num_unroll - 1];

#pragma unroll
        for (int32_t j = 0; j < num_unroll; j++) {
            if (idx + j < src_data.extent(3)) {
                dst_data(i3, i2, i1, idx + j) = temp[j] + ggml_cuda_cast<T, float>(final_val_offset);
            }
        }

        block.sync();

        // Update carry for next chunk
        if (tid == 0) {
            *s_carry += *s_chunk_total;
        }
    }
}

#ifdef GGML_CUDA_USE_CUB
template <typename T>
static void cumsum_cub(ggml_cuda_pool& pool,
    const T* src,
    T* dst,
    int64_t          ne,
    cudaStream_t     stream) {
    size_t tmp_size = 0;

    // Query how much temp storage CUDA UnBound (CUB) needs
    cub::DeviceScan::InclusiveSum(nullptr,   // d_temp_storage (null = just query size)
        tmp_size,  // reference to size (will be set by CUB)
        src,       // input pointer
        dst,       // output pointer
        ne,        // number of elements
        stream     // CUDA stream to use
    );

    ggml_cuda_pool_alloc<uint8_t> tmp_alloc(pool, tmp_size);

    // Perform the inclusive scan
    cub::DeviceScan::InclusiveSum((void*)tmp_alloc.get(), tmp_size, src, dst, ne, stream);
}
#endif // GGML_CUDA_USE_CUB

template <typename T>
static void cumsum_cuda(const cumsum_context& ctx, cudaStream_t stream) {
    static constexpr size_t CUDA_CUMSUM_BLOCK_SIZE = 256;
    const size_t type_size = sizeof(T);
    bool use_cub = false;
#ifdef GGML_CUDA_USE_CUB
    // Check if we can use CUB (data must be contiguous along innermost dimension)
    const bool is_contiguous = (ctx.src0_nb[0] == type_size) && (ctx.dst_nb[0] == type_size);

    if (is_contiguous) {
        use_cub = true;
        const int64_t nrows = ctx.src0_ne[1] * ctx.src0_ne[2] * ctx.src0_ne[3];
        // TODO: Compare with DeviceSegmentedScan::InclusiveSegmentedSum for nrows > 1 once InclusiveSegmentedSum is released
        // Heuristics were determined as part of https://github.com/ggml-org/llama.cpp/pull/17004
        if (((nrows == 1) && (ctx.src0_ne[0] > 1024)) || (ctx.src0_ne[0] / nrows > 4096)) {
            for (int i = 0; i < nrows; i++) {
                cumsum_cub(ctx.pool, static_cast<const T*>(ctx.src0_d) + i * ctx.src0_ne[0], 
                    static_cast<T*>(ctx.dst_d) + i * ctx.src0_ne[0], ctx.src0_ne[0], stream);
            }
            return;
        }
    }
#endif // GGML_CUDA_USE_CUB
    dim3 grid_dims(ctx.src0_ne[1], ctx.src0_ne[2], ctx.src0_ne[3]);
    auto src_data = make_strided_mdspan(static_cast<const T*>(ctx.src0_d), ctx.src0_ne, ctx.src0_nb);
    auto dst_data = make_strided_mdspan(static_cast<T*>(ctx.dst_d), ctx.dst_ne, ctx.dst_nb);

    if (use_cub && ctx.src0_ne[0] >= 1024) {
        cumsum_cub_kernel<T, CUDA_CUMSUM_BLOCK_SIZE> << <grid_dims, CUDA_CUMSUM_BLOCK_SIZE, 0, stream >> > (
            src_data, dst_data);
    }
    else {
        const auto& info = ggml_cuda_info().devices[ggml_cuda_get_device()];
        const int warp_size = info.warp_size;
        const int num_warps = (ctx.src0_ne[0] + warp_size - 1) / warp_size;
        int block_size = num_warps * warp_size;
        block_size = std::min<int>(block_size, CUDA_CUMSUM_BLOCK_SIZE);
        dim3 block_dims(block_size, 1, 1);
        const int warps_per_block = block_size / warp_size;
        const size_t shmem_size = (block_size + warps_per_block + 2) * sizeof(float);
        cumsum_kernel << <grid_dims, block_dims, shmem_size, stream >> > (src_data, dst_data);
    }
}

void cumsum_cuda(const cumsum_context& ctx, cudaStream_t  stream) {
    switch (ctx.src0_type) {
    case internal::GGML_TYPE_F32:
    {
        cumsum_cuda<float>(ctx, stream);
    } break;
    // We do not support those on CPU for now anyway, so comment them out because they cause errors on some CI platforms
    /*case internal::GGML_TYPE_F16:
        {
            cumsum_cuda<half>(ctx, stream);
        } break;
    case internal::GGML_TYPE_BF16:
        {
            cumsum_cuda<nv_bfloat16>(ctx, stream);
        } break;*/
    default:
        GGML_ABORT("fatal error");
    }
}

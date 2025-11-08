#pragma once
#include <cooperative_groups/reduce.h>

template <template<typename> class Op, typename block_t, typename tile_t, typename T>
__device__ __forceinline__ auto reduceWithBlock(block_t block, tile_t tile, T initial_val, T val, T* buffer)
{
    const int tid = block.thread_rank();
    const int tile_id = tid / tile.size();
    const int lane_id = tile.thread_rank();
    val = cooperative_groups::reduce(tile, val, Op<T>());
    if (block.size() > tile.size()) {
        assert(block.size() <= 1024 && block.size() % 32 == 0);
        if (tile_id == 0) {
            buffer[lane_id] = initial_val;
        }
        block.sync();
        if (lane_id == 0) {
            buffer[tile_id] = val;
        }
        block.sync();
        val = cooperative_groups::reduce(tile, buffer[lane_id], Op<T>());;
    }
    return val;
}

template <>
struct cooperative_groups::plus<float2>
{
    __device__ float2 operator()(float2 a, float2 b) {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

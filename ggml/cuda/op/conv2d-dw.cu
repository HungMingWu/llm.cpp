#include "cuda_func.h"
#define GGML_ABORT(...)

struct conv_params {
    int in_w, in_h;
    int out_w, out_h;
    int kernel_w, kernel_h;
    int stride_x, stride_y;
    int padding_x, padding_y;
    int dilation_x, dilation_y;
    int channels, batches;
};

struct kernel_bounds {
    int y_min, y_max;
    int x_min, x_max;
};

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int out_x, int out_y, const conv_params& params) {
    kernel_bounds bounds;
    bounds.y_min = max(0, (params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.y_max =
        min(params.kernel_h,
            (params.in_h + params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.x_min = max(0, (params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    bounds.x_max =
        min(params.kernel_w,
            (params.in_w + params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int input_index(int n, int c, int y, int x, const conv_params& params) {
        return n * (params.channels * params.in_w * params.in_h) + c * params.in_w * params.in_h + y * params.in_w + x;
    }

    __device__ static int kernel_index(int c, int ky, int kx, const conv_params& params) {
        return c * params.kernel_h * params.kernel_w + ky * params.kernel_w + kx;
    }

    __device__ static int output_index(int n, int c, int y, int x, const conv_params& params) {
        return n * (params.channels * params.out_w * params.out_h) + c * params.out_w * params.out_h +
            y * params.out_w + x;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params& params, int& n, int& c, int& out_y,
        int& out_x) {
        out_x = global_idx % params.out_w;
        out_y = (global_idx / params.out_w) % params.out_h;
        c = (global_idx / (params.out_w * params.out_h)) % params.channels;
        n = global_idx / (params.out_w * params.out_h * params.channels);
    }
};

struct cwhn_layout {
    __device__ static int input_index(int n, int c, int y, int x, const conv_params& params) {
        return n * (params.channels * params.in_w * params.in_h) + (y * params.in_w + x) * params.channels + c;
    }

    __device__ static int kernel_index(int c, int ky, int kx, const conv_params& params) {
        return (ky * params.kernel_w + kx) * params.channels + c;
    }

    __device__ static int output_index(int n, int c, int y, int x, const conv_params& params) {
        return n * (params.channels * params.out_w * params.out_h) + y * (params.out_w * params.channels) +
            x * params.channels + c;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params& params, int& n, int& c, int& out_y,
        int& out_x) {
        c = global_idx % params.channels;
        out_x = (global_idx / params.channels) % params.out_w;
        out_y = (global_idx / (params.channels * params.out_w)) % params.out_h;
        n = global_idx / (params.channels * params.out_w * params.out_h);
    }
};

template <typename T, typename Layout>
__global__ void conv2d_dw_kernel(const T* __restrict__ input, const T* __restrict__ kernel, T* __restrict__ output,
    const int in_w, const int in_h, const int out_w, const int out_h,
    const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
    const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
    const int channels, const int batches) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batches * channels * out_h * out_w;

    if (global_idx >= total_elements) {
        return;
    }

    conv_params params = { in_w,     in_h,      out_w,     out_h,      kernel_w,   kernel_h, stride_x,
                           stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches };

    int batch_idx, channel_idx, out_y_idx, out_x_idx;
    Layout::unpack_indices(global_idx, params, batch_idx, channel_idx, out_y_idx, out_x_idx);

    T accumulator = 0;
    kernel_bounds bounds = calculate_kernel_bounds(out_x_idx, out_y_idx, params);

    for (int kern_y = bounds.y_min; kern_y < bounds.y_max; ++kern_y) {
        int in_y_idx = calculate_input_coord(out_y_idx, kern_y, params.stride_y, params.dilation_y, params.padding_y);

        for (int kern_x = bounds.x_min; kern_x < bounds.x_max; ++kern_x) {
            int in_x_idx = calculate_input_coord(out_x_idx, kern_x, params.stride_x, params.dilation_x, params.padding_x);

            const T input_val = input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)];
            const T kernel_val = kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];

            accumulator += input_val * kernel_val;
        }
    }

    output[Layout::output_index(batch_idx, channel_idx, out_y_idx, out_x_idx, params)] = accumulator;
}

void conv2d_dw_cuda(conv2d_dw_context* ctx, cudaStream_t stream)
{
    static constexpr size_t CUDA_CONV2D_DW_BLOCK_SIZE = 256;
    const int total = ctx->batches * ctx->channels * ctx->out_h * ctx->out_w;
    const int blocks = (total + CUDA_CONV2D_DW_BLOCK_SIZE - 1) / CUDA_CONV2D_DW_BLOCK_SIZE;

    if (ctx->input_is_contiguous) {
        conv2d_dw_kernel<float, whcn_layout> << <blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, stream >> > (
            ctx->x_d, ctx->w_d, ctx->y_d,
            ctx->in_w, ctx->in_h, ctx->out_w, ctx->out_h,
            ctx->kernel_w, ctx->kernel_h, ctx->stride_x, ctx->stride_y, ctx->padding_x, ctx->padding_y,
            ctx->dilation_x, ctx->dilation_y, ctx->channels, ctx->batches);
    }
    else if (ctx->input_is_contiguous_channels) {
        conv2d_dw_kernel<float, cwhn_layout> << <blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, stream >> > (
            ctx->x_d, ctx->w_d, ctx->y_d,
            ctx->in_w, ctx->in_h, ctx->out_w, ctx->out_h,
            ctx->kernel_w, ctx->kernel_h, ctx->stride_x, ctx->stride_y, ctx->padding_x, ctx->padding_y,
            ctx->dilation_x, ctx->dilation_y, ctx->channels, ctx->batches);
    }
    else {
        GGML_ABORT("Unsupported memory layout for conv_2d_dw");
    }
}
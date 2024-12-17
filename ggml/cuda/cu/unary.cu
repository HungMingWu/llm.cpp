#include "common.cuh"
#include <assert.h>
#include "cuda_func.h"
#define GGML_ASSERT(...) assert(__VA_ARGS__)

static constexpr size_t CUDA_RELU_BLOCK_SIZE = 256;

static __host__ __device__ float neg(float x) {
    return -x;
}

static __host__ __device__ float step(float x) {
    return x > 0.0f;
}

static __host__ __device__ float gelu(float x) {
    static const float GELU_COEF_A = 0.044715f;
    static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

static __host__ __device__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

static __host__ __device__ float gelu_quick(float x) {
    static const float GELU_QUICK_COEF = -1.702f;
    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __host__ __device__ float relu(float x) {
    return fmaxf(x, 0);
}

static __host__ __device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __host__ __device__ float hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __host__ __device__ float hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __host__ __device__ float sgn(float x) {
    return (x > 0.f) ? 1.f : ((x < 0.f) ? -1.f : 0.f);
}

static __host__ __device__ float elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __host__ __device__ float sqr(float x) {
    return x * x;
}

template <float (*Func)(float)>
static __global__ void transform(const float* x, float* dst, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = Func(x[i]);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T* x, T* dst, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op((float)x[i]);
}

template <float (*op)(float), typename T>
static void unary_cuda(const T* x, T* dst, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_UNARY_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_UNARY_BLOCK_SIZE - 1) / CUDA_UNARY_BLOCK_SIZE;
    unary_op_kernel<op> << <num_blocks, CUDA_UNARY_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

template <float (*op)(float)>
void op_unary(const unary_context* ctx) {
    GGML_ASSERT(ctx->src0_type == GGML_TYPE_F32 || ctx->src0_type == GGML_TYPE_F16);
    GGML_ASSERT(ctx->dst_type == GGML_TYPE_F32 || ctx->dst_type == GGML_TYPE_F16);
    GGML_ASSERT(ctx->src0_type == ctx->dst_type);

    if (ctx->src0_type == GGML_TYPE_F16) {
        unary_cuda<op>((const half*)ctx->src0_d, (half*)ctx->dst_d, ctx->nelements, ctx->stream);
    }
    else {
        unary_cuda<op>((const float*)ctx->src0_d, (float*)ctx->dst_d, ctx->nelements, ctx->stream);
    }
}

static __global__ void silu_back_f32(
    const float* grad, const float* xf, float* dst, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xfi = xf[i];
    const float s = 1.0f / (1.0f + expf(-xfi));
    dst[i] = grad[i] * s * (1.0f + xfi * (1.0f - s));
}

void neg_cuda(const unary_context* ctx)
{
    op_unary<neg>(ctx);
}

void step_cuda(const unary_context* ctx)
{
    op_unary<step>(ctx);
}

void gelu_cuda(const unary_context* ctx)
{
    op_unary<gelu>(ctx);
}

void silu_cuda(const unary_context* ctx)
{
    op_unary<silu>(ctx);
}

void gelu_quick_cuda(const unary_context* ctx)
{
    op_unary<gelu_quick>(ctx);
}

void tanh_cuda(const unary_context* ctx)
{
    op_unary<tanhf>(ctx);
}

void relu_cuda(const unary_context* ctx) {
    op_unary<relu>(ctx);
}

void sigmoid_cuda(const unary_context* ctx)
{
    op_unary<sigmoid>(ctx);
}

void hardsigmoid_cuda(const unary_context* ctx)
{
    op_unary<hardsigmoid>(ctx);
}

void hardswish_cuda(const unary_context* ctx)
{
    op_unary<hardswish>(ctx);
}

void exp_cuda(const unary_context* ctx)
{
    op_unary<expf>(ctx);
}

void abs_cuda(const unary_context* ctx)
{
    op_unary<fabsf>(ctx);
}

void sgn_cuda(const unary_context* ctx)
{
    op_unary<sgn>(ctx);
}

void elu_cuda(const unary_context* ctx)
{
    op_unary<elu>(ctx);
}

void silu_back_f32_cuda(const float* grad, const float* x, float* dst, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_SILU_BACK_BLOCK_SIZE = 256;
    static constexpr size_t CUDA_SILU_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_f32 << <num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream >> > (grad, x, dst, k);
}

void sqr_cuda(const unary_context* ctx) {
    op_unary<sqr>(ctx);
}

void sqrt_cuda(const unary_context* ctx) {
    op_unary<sqrtf>(ctx);
}

void sin_cuda(const unary_context* ctx) {
    op_unary<sinf>(ctx);
}

void cos_cuda(const unary_context* ctx) {
    op_unary<cosf>(ctx);
}

void log_cuda(const unary_context* ctx)
{
    op_unary<logf>(ctx);
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T* x, T* dst, const int k, const float negative_slope) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T* x, T* dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel << <num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream >> > (x, dst, k, negative_slope);
}

void leaky_relu_cuda(bool is_half, const void* x, void* dst,
    const int k, const float negative_slope, cudaStream_t stream)
{
    if (is_half) {
        leaky_relu_cuda((const half*)x, (half*)dst, k, negative_slope, stream);
    }
    else {
        leaky_relu_cuda((const float*)x, (float*)dst, k, negative_slope, stream);
    }
}
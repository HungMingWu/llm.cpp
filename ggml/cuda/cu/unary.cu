static constexpr size_t CUDA_GELU_BLOCK_SIZE = 256;
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256

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

template <float (*Func)(float)>
static __global__ void transform(const float* x, float* dst, const int k) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = Func(x[i]);
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

void neg_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_NEG_BLOCK_SIZE = 256;
	const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    transform<neg> << <num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void step_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_STEP_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_STEP_BLOCK_SIZE - 1) / CUDA_STEP_BLOCK_SIZE;
    transform<step> << <num_blocks, CUDA_STEP_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void gelu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    transform<gelu> << <num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void silu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_SILU_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    transform<silu> << <num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void gelu_quick_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    const int num_blocks = (k + CUDA_GELU_BLOCK_SIZE - 1) / CUDA_GELU_BLOCK_SIZE;
    transform<gelu_quick> << <num_blocks, CUDA_GELU_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void tanh_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_TANH_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_TANH_BLOCK_SIZE - 1) / CUDA_TANH_BLOCK_SIZE;
    transform<tanhf> << <num_blocks, CUDA_TANH_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void relu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_RELU_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    transform<relu> << <num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void sigmoid_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_SIGMOID_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SIGMOID_BLOCK_SIZE - 1) / CUDA_SIGMOID_BLOCK_SIZE;
    transform<sigmoid> << <num_blocks, CUDA_SIGMOID_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void hardsigmoid_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_HARDSIGMOID_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_HARDSIGMOID_BLOCK_SIZE - 1) / CUDA_HARDSIGMOID_BLOCK_SIZE;
    transform<hardsigmoid> << <num_blocks, CUDA_HARDSIGMOID_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void hardswish_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_HARDSWISH_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_HARDSWISH_BLOCK_SIZE - 1) / CUDA_HARDSWISH_BLOCK_SIZE;
    transform<hardswish> << <num_blocks, CUDA_HARDSWISH_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void exp_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_EXP_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_EXP_BLOCK_SIZE - 1) / CUDA_EXP_BLOCK_SIZE;
    transform<expf> << <num_blocks, CUDA_EXP_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void abs_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_ABS_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_ABS_BLOCK_SIZE - 1) / CUDA_ABS_BLOCK_SIZE;
    transform<fabsf> << <num_blocks, CUDA_ABS_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void sgn_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_SGN_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SGN_BLOCK_SIZE - 1) / CUDA_SGN_BLOCK_SIZE;
    transform<sgn> << <num_blocks, CUDA_SGN_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void elu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream)
{
    static constexpr size_t CUDA_ELU_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_ELU_BLOCK_SIZE - 1) / CUDA_ELU_BLOCK_SIZE;
    transform<elu> << <num_blocks, CUDA_ELU_BLOCK_SIZE, 0, stream >> > (x, dst, k);
}

void silu_back_f32_cuda(const float* grad, const float* x, float* dst, const int k, cudaStream_t stream) {
    static constexpr size_t CUDA_SILU_BACK_BLOCK_SIZE = 256;
    static constexpr size_t CUDA_SILU_BLOCK_SIZE = 256;
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_f32 << <num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream >> > (grad, x, dst, k);
}
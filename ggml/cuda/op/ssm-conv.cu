#define GGML_ASSERT(...)
#define GGML_ABORT(...)
#define GGML_UNUSED(x) (void)(x)

template <size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float* __restrict__ src0, const float* __restrict__ src1,
    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
    float* __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
    const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float* x_block = (const float*)((const char*)src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float* w_block = (const float*)((const char*)src1 + bidy * split_d_inner * src1_nb1);
    float* y_block = (float*)((char*)dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        }
        else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        y_block[i * stride_y + tid] = sumf;
    }
}

template <size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float* __restrict__ src0, const float* __restrict__ src1,
    const int src0_nb0, const int src0_nb1, const int src0_nb2,
    const int src1_nb1, float* __restrict__ dst, const int dst_nb0,
    const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float* x_block = (const float*)((const char*)src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
        bidz * split_n_t * src0_nb0);
    const float* w_block = (const float*)((const char*)src1 + bidy * split_d_inner * src1_nb1);
    float* y_block =
        (float*)((char*)dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

#pragma unroll
    for (int64_t i = 0; i < split_n_t; i++) {
        if (bidz * split_n_t + i < n_t) {
            float sumf = 0.0f;

            if (i == 0) {
                for (size_t j = 0; j < d_conv; j++) {
                    x[j] = x_block[tid * stride_x + j];
                }
            }
            else {
                x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
            }

#pragma unroll
            for (size_t j = 0; j < d_conv; j++) {
                sumf += x[(i + j) % d_conv] * w[j];
            }
            y_block[i * stride_y + tid] = sumf;
        }
    }
}

void ssm_conv_f32_cuda(const float* src0, const float* src1, const int src0_nb0, const int src0_nb1,
    const int src0_nb2, const int src1_nb1, float* dst, const int dst_nb0, const int dst_nb1,
    const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
    const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    if (n_t <= 32) {
        const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
        if (nc == 4) {
            ssm_conv_f32<threads, 4> << <blocks, threads, 0, stream >> > (src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
        else if (nc == 3) {
            ssm_conv_f32<threads, 3> << <blocks, threads, 0, stream >> > (src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
        else {
            GGML_ABORT("Only support kernel size = 3 or size = 4 right now.");
        }
    }
    else {
        if (nc == 4) {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            ssm_conv_long_token_f32<threads, 4, split_n_t> << <blocks, threads, 0, stream >> > (
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
        else if (nc == 3) {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            ssm_conv_long_token_f32<threads, 3, split_n_t> << <blocks, threads, 0, stream >> > (
                src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
        else {
            GGML_ABORT("Only support kernel size = 3 or size = 4 right now.");
        }
    }
}
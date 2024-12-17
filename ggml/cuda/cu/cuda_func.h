#pragma once
#include "../common.h"
#include <stdint.h>

enum ggml_prec : int;
enum ggml_op_pool : int;
enum ggml_type : int;

struct mmq_args {
    const char* x; const char* y; float* dst;
    int64_t ne00; int64_t ne01; int64_t stride01;
    int64_t ne10; int64_t ne11; int64_t stride11;
    int64_t ne0;
    bool use_stream_k;
};

void arange_f32_cuda(float* dst, size_t dst_size, const float start, const float step, cudaStream_t stream);

void conv_transpose_1d_f32_f32_cuda(
    const int s0,
    const int src0_ne0, const int src0_ne1, const int src0_ne2,
    const int src1_ne0, const int src1_ne1,
    const int dst_ne0, const int dst_ne1,
    const float* src0, const float* src1, float* dst, size_t dst_size,
    cudaStream_t stream);

void mul_mat_vec_cuda(
    const half* x, const float* y, float* dst,
    const int64_t ncols, const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y,
    const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst,
    enum ggml_prec prec, cudaStream_t stream);

// From quantize.cu
void quantize_row_q8_1_cuda(
    const float* x, void* vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float* x, void* vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream);

void mul_mat_vec_q4_0_q8_1_cuda(
    const void* vx, const void* vy, float* dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst, cudaStream_t stream);

void pool2d_nchw_kernel_f32_f32_cuda(
    const int ih, const int iw, const int oh, const int ow,
    const int kh, const int kw, const int sh, const int sw,
    const int ph, const int pw, const int parallel_elements,
    const float* src, float* dst, enum ggml_op_pool op,
    cudaStream_t stream);

void im2col_cuda_f16(const float* x, half* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream);

void im2col_cuda_f32(const float* x, float* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t batch, int64_t batch_offset, int64_t offset_delta,
    int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream);

// unary
void neg_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void step_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void gelu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void silu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void gelu_quick_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void tanh_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void relu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void sigmoid_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void hardsigmoid_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void hardswish_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void exp_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void abs_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void sgn_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void elu_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void silu_back_f32_cuda(const float* grad, const float* x, float* dst, const int k, cudaStream_t stream);

// get_row
struct get_row_context {
    ggml_type type;
    const void* src0_d;
    const int32_t* src1_d;
    float* dst_d;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    const int64_t ne10, ne11, ne12, ne13;
    // strides in elements
    const size_t s1, s2, s3, s10, s11, s12;
};

struct get_row_back_context {
    const float* src0_d;
    const int32_t* src1_d;
    float* dst_d;
    const int64_t ne00;
    const int64_t ne10;
    const int64_t ne1;
};

void get_rows_cuda(const get_row_context *ctx, cudaStream_t stream);
void get_rows_back_cuda(const get_row_back_context* ctx, cudaStream_t stream);

// argmax
struct argmax_context {
    const float* src0_d;
    int32_t* dst_d;
    const int64_t ne00;
    const int64_t nrows;
};
void argmax_cuda(const argmax_context* ctx, cudaStream_t stream);

// count_equal
struct count_equal_context {
    const int* src0_d;
    const int* src1_d;
    int64_t* dst_d;
    const size_t dst_size;
    const int64_t ne;
};
void count_equal_cuda(const count_equal_context* ctx, cudaStream_t stream);

// bin_bcast
struct bin_bcast_context {
    const void* src0_d;
    const void* src1_d;
    void* dst_d;
    const ggml_type src0_type, src1_type, dst_type;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    const int64_t ne10, ne11, ne12, ne13;
    const size_t nb10, nb11, nb12, nb13;
    const int64_t ne0, ne1, ne2, ne3;
    const size_t nb0, nb1, nb2, nb3;
    const bool src0_is_contiguous, src1_is_contiguous, dst_is_contiguous;
};

void repeat_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void add_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void mul_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void div_cuda(const bin_bcast_context* ctx, cudaStream_t stream);

// cpy
struct dup_context {
    const void* src_d;
    void* dst_d;
    const ggml_type src_type, dst_type;
    const int64_t ne;
    const size_t src_length;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    // rename later
    const int64_t ne10, ne11, ne12, ne13;
    const size_t nb10, nb11, nb12, nb13;
    const bool src_is_contiguous, dst_is_contiguous;
};
void dup_cuda(const dup_context* ctx, cudaStream_t stream);

// scale
void scale_f32_cuda(const float* x, float* dst, const float scale, const int k, cudaStream_t stream);

// norm
void norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream);

void rms_norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream);

void rms_norm_back_f32_cuda(const float* grad, const float* xf, float* dst, const int ncols, const int nrows, const float eps, cudaStream_t stream);

// wkv6
void rwkv_wkv6_cuda(const int B, const int T, const int C,
    const int H, const float* k,
    const float* v, const float* r,
    const float* tf, const float* td, const float* s, float* dst, cudaStream_t stream);

// gla

void rwkv_wkv6_cuda(const int B, const int T, const int C, const int H, const float scale,
    const float* k, const float* v, const float* r,
    const float* td, const float* s, float* dst, cudaStream_t stream);
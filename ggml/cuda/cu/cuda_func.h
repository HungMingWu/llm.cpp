#pragma once
#include "../common.h"
#include "../cuda_pool.h"
#include "block.h"
#include <stdint.h>

enum ggml_prec : int;
enum ggml_op_pool : int;
enum ggml_type : int;

#define MMQ_DP4A_MAX_BATCH_SIZE 64 // Max. batch size to use for dp4a MMQ kernels when FP16 tensor cores are available.

struct block_q8_1_mmq {
    // The y float data is converted to a data layout that can simply be copied to shared memory as a contiguous block.
    // The y float data is first grouped as blocks of 128 values.
    // These blocks are then treated as individual data values and transposed.
    //
    // To avoid shared memory bank conflicts each block is padded with 16 bytes.
    // This padding is also used to store block scales/partial sums.
    // The scales multiplied with the quantized data are equal to the unquantized values.
    // The partial sums are obtained by summing up a subgroup of the contained values (prior to quantization)
    //     and are only needed for performance reasons.
    //
    // The exact data stored depends on the x data type.
    union {
        float d4[4];    // 1 32 bit scale per 32 values, stored as d0,d1,d2,d3
        half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3
        half  d2s6[8];  // 1 16 bit scale per 64 values + 1 16 bit partial sum per 16 values for the first 96 values,
        //     stored as d0,d1,s1,s2,s3,s4,s5
    };
    int8_t qs[4 * QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4 * QK8_1 + 4 * sizeof(half2), "Unexpected block_q8_1_mmq size");
static_assert(sizeof(block_q8_1_mmq) == 4 * sizeof(block_q8_1), "Unexpected block_q8_1_mmq size");

static int get_mmq_x_max_host(const int cc) {
    return new_mma_available(cc) ? 128 :
        ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA && cc < GGML_CUDA_CC_OFFSET_AMD ?
#ifdef GGML_CUDA_FORCE_MMQ
        128 : 64;
#else
        MMQ_DP4A_MAX_BATCH_SIZE : 64;
#endif // GGML_CUDA_FORCE_MMQ
}

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

// mmq
struct mat_vec_context {
    ggml_type src0_type;
    const void* vx;
    const void* vy;
    float* dst;
    const int64_t ncols_x, nrows_x;
    const int64_t ncols_y, nrows_y;
    const int64_t nrows_dst;
};

void mul_mat_vec_q_cuda(const mat_vec_context* ctx, cudaStream_t stream);

struct mat_q_context {
    int id;
    ggml_cuda_pool* pool;
    const char* src0_dd_i;
    ggml_type src0_type;
    const char* src1_ddq_i;
    const int64_t src1_ncols;
    float* dst_dd_i;
    const int64_t nrows_dst;
    const int64_t ne00;
    const int64_t ne11;
    const int64_t row_diff;
    const int64_t stride00;
    const int64_t src1_padded_row_size;
};

void mul_mat_q_cuda(const mat_q_context* ctx, cudaStream_t stream);

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
void sqr_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void sqrt_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void sin_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);
void cos_f32_cuda(const float* x, float* dst, const int k, cudaStream_t stream);

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

// compute_batched_ptrs
void k_compute_batched_ptrs_cuda(
    const half* src0_as_f16, const half* src1_as_f16, char* dst,
    const void** ptrs_src, void** ptrs_dst,
    int64_t ne12, int64_t ne13,
    int64_t ne23,
    size_t  nb02, size_t  nb03,
    size_t  nb12, size_t  nb13,
    size_t  nbd2, size_t  nbd3,
    int64_t r2, int64_t r3, cudaStream_t stream);

// misc

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

struct copy_src1_to_contiguous_context {
    const int64_t num_src1_rows;
    const int64_t ids_ne1;
    const size_t ids_nb0, ids_nb1;
    const int64_t n_ids;
    char* src1_original;
    char* src1_contiguous;
    const char* ids_dev;
    const int64_t i02;
    const int64_t ne10, ne11;
    const size_t nb11, nb12;
    int* dev_cur_src1_row;
    mmid_row_mapping* dev_row_mapping;
};

void k_copy_src1_to_contiguous_cuda(const copy_src1_to_contiguous_context* ctx, cudaStream_t stream);

struct k_copy_dst_from_contiguous_context {
    const int64_t ne0;
    const int64_t num_src1_rows;
    char* dst_original;
    const char* dst_contiguous;
    const mmid_row_mapping* dev_row_mapping;
    const size_t nb1, nb2;
};
void k_copy_dst_from_contiguous_cuda(const k_copy_dst_from_contiguous_context* ctx, cudaStream_t stream);

// clamp
void clamp_f32_cuda(const float* x, float* dst, const float min, const float max, const int k, cudaStream_t stream);

// diagmask
void diag_mask_inf_f32_cuda(const float* x, float* dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream);

// softmax
struct softmax_context {
    const float* src0_d;
    const void* src1_d;
    float* dst_d;
    const int64_t ne00;
    const int64_t nrows_x, nrows_y;
    const float scale, max_bias;
    bool use_f16;
};

void soft_max_f32_cuda(const softmax_context* ctx, cudaStream_t stream);
void soft_max_back_f32_cuda(
    const float* grad, const float* dstf, float* dst,
    const int ncols, const int nrows, const float scale, cudaStream_t stream);

// rope

struct mrope_sections {
    int v[4];
};

struct rope_corr_dims {
    float v[2];
};

struct rope_context {
    const bool forward;
    const bool is_neox;
    const bool is_mrope;
    const bool is_vision;
    const ggml_type src0_type;
    const void* src0_d;
    void* dst_d;
    const int64_t ne00, ne01, ne02;
    const size_t s01, s02;
    const int n_dims;
    const int n_ctx_orig;
    const int64_t nr;
    const int32_t* pos;
    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float* freq_factors;
    mrope_sections sections;
};

void rope_cuda(const rope_context* ctx, cudaStream_t stream);

// concat
struct concat_context {
    const int32_t dim;
    const bool src0_is_contiguous, src1_is_contiguous;
	const float* src0_d, * src1_d;
    float* dst_d;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    const int64_t ne10, ne11, ne12, ne13;
    const size_t nb10, nb11, nb12, nb13;
    const int64_t ne0, ne1, ne2, ne3;
	const size_t nb0, nb1, nb2, nb3;
    const size_t src0_size, src1_size;
};
void concat_cuda(const concat_context* ctx, cudaStream_t stream);
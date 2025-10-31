#pragma once
#include "../common.h"
#include "../cuda_pool.h"
#include "block.h"
#include <stdint.h>

enum ggml_prec : int;
enum ggml_op_pool : int;
enum ggml_type : int;
enum ggml_sort_order : int;
enum ggml_scale_mode : int;

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
    return (amd_mfma_available(cc) || turing_mma_available(cc)) ? 128 :
        GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA ?
#ifdef GGML_CUDA_FORCE_MMQ
        128 : 64;
#else
        MMQ_DP4A_MAX_BATCH_SIZE : 64;
#endif // GGML_CUDA_FORCE_MMQ
}

void arange_f32_cuda(float* dst, size_t dst_size, const float start, const float step, cudaStream_t stream);

// conv-transpose-1d.h
struct conv_transpose_1d_context {
    ggml_type src0_type;
    int64_t src0_ne[4];
    int64_t src1_ne[4];
    int64_t dst_ne[4];
    const void* src0_d;
    const float* src1_d;
    float* dst_d;;
    const int stride, padding, dilation;
};

void conv_transpose_1d_f32_cuda(const conv_transpose_1d_context& ctx, cudaStream_t stream);

// From quantize.cu
void quantize_row_q8_1_cuda(
    const float* x, const int32_t* ids, void* vy,
    ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

void quantize_mmq_q8_1_cuda(
    const float* x, const int32_t* ids, void* vy,
    ggml_type type_src0, int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
    int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, cudaStream_t stream);

// From mmvq.cu

struct mat_vec_q_switch_context {
    ggml_type type_x;
    const void* vx;
    const void* vy;
    const int32_t* ids;
    float* dst;
    const int64_t ncols_x;
    const int64_t nrows_x;
    const int64_t ncols_dst;
    const int64_t stride_row_x;
    const int64_t stride_col_y;
    const int64_t stride_col_dst;
    const int64_t nchannels_x;
    const int64_t nchannels_y;
    const int64_t nchannels_dst;
    const int64_t stride_channel_x;
    const int64_t stride_channel_y;
    const int64_t stride_channel_dst;
    const int64_t nsamples_x;
    const int64_t nsamples_dst;
    const int64_t stride_sample_x;
    const int64_t stride_sample_y;
    const int64_t stride_sample_dst;
};

void mul_mat_vec_q_switch_type(const mat_vec_q_switch_context* ctx, cudaStream_t stream);

struct mmq_args {
    const char* x; ggml_type type_x; const int* y; const int32_t* ids_dst; const int32_t* expert_bounds; float* dst;
    int64_t ncols_x; int64_t nrows_x; int64_t ncols_dst; int64_t stride_row_x; int64_t ncols_y; int64_t nrows_dst;
    int64_t nchannels_x; int64_t nchannels_y; int64_t stride_channel_x; int64_t stride_channel_y; int64_t stride_channel_dst;
    int64_t nsamples_x; int64_t nsamples_y; int64_t stride_sample_x; int64_t stride_sample_y; int64_t stride_sample_dst;
    bool use_stream_k; int64_t ncols_max;
};

void ggml_cuda_mul_mat_q_switch_type(ggml_cuda_pool& pool, const mmq_args& args, cudaStream_t stream);

// mmid.cu
void ggml_cuda_launch_mm_ids_helper(
    const int32_t* ids, int32_t* ids_src1, int32_t* ids_dst, int32_t* expert_bounds,
    const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, cudaStream_t stream);

void pool2d_nchw_kernel_f32_f32_cuda(
    const int ih, const int iw, const int oh, const int ow,
    const int kh, const int kw, const int sh, const int sw,
    const int ph, const int pw, const int parallel_elements,
    const float* src, float* dst, enum ggml_op_pool op,
    cudaStream_t stream);

void im2col_cuda(ggml_type dst_type, const float* x, void* dst,
    int64_t IW, int64_t IH, int64_t OW, int64_t OH, int64_t KW, int64_t KH, int64_t IC,
    int64_t N,
    int s0, int s1, int p0, int p1, int d0, int d1, cudaStream_t stream);

void im2col_3d_cuda(ggml_type dst_type, const float* src1_d, void* dst_d,
    int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
    int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
    size_t stride_q, size_t stride_z, size_t stride_y, size_t stride_x,
    int s0, int s1, int s2, int p0, int p1, int p2, int d0, int d1, int d2, cudaStream_t stream);

// unary
struct unary_context {
    cudaStream_t stream;
    const ggml_type src0_type, dst_type;
    const void* src0_d;
    void* dst_d;
    const int64_t nelements;
};

void abs_cuda(const unary_context* ctx);
void sgn_cuda(const unary_context* ctx);
void elu_cuda(const unary_context* ctx);
void xielu_cuda(ggml_type src0_type, const void* src0_d, void* dst_d, int64_t src0_elements,
    const float alpha_n, const float alpha_p, const float beta, const float eps, cudaStream_t stream);
void neg_cuda(const unary_context* ctx);
void gelu_cuda(const unary_context* ctx);
void gelu_erf_cuda(const unary_context* ctx);
void silu_cuda(const unary_context* ctx);
void gelu_quick_cuda(const unary_context* ctx);
void tanh_cuda(const unary_context* ctx);
void relu_cuda(const unary_context* ctx);
void sigmoid_cuda(const unary_context* ctx);
void hardsigmoid_cuda(const unary_context* ctx);
void hardswish_cuda(const unary_context* ctx);
void exp_cuda(const unary_context* ctx);
void step_cuda(const unary_context* ctx);
void silu_back_f32_cuda(const float* grad, const float* x, float* dst, const int k, cudaStream_t stream);
void sqr_cuda(const unary_context* ctx);
void sqrt_cuda(const unary_context* ctx);
void sin_cuda(const unary_context* ctx);
void cos_cuda(const unary_context* ctx);
void log_cuda(const unary_context* ctx);

struct gated_context {
    cudaStream_t stream;
    ggml_type src0_type;
    const int32_t swapped;
    void* src0_d;
    void* src1_d;
    void* dst_d;
    const size_t src0_o, src1_o;
    const int64_t nc, dst_nelements;
    const bool src1_exist;
};

void reglu_cuda(const gated_context* ctx);
void geglu_cuda(const gated_context* ctx);
void swiglu_cuda(const gated_context* ctx);
void swiglu_oai_cuda(const float* x, const float* g, float* dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream);
void geglu_erf_cuda(const gated_context* ctx);
void geglu_quick_cuda(const gated_context* ctx);
void leaky_relu_cuda(bool, const void*, void*, const int, const float, cudaStream_t);

// get_row
struct get_row_context {
    const void* src0_d;
    ggml_type src0_type;
    const int32_t* src1_d;
    void* dst_d;
    ggml_type dst_type;
    int64_t src0_ne[4];
    size_t src0_nb[4];
    int64_t src1_ne[4];
    size_t src1_nb[4];
    int64_t dst_ne[4];
    size_t dst_nb[4];
};

struct get_row_back_context {
    const float* src0_d;
    const int32_t* src1_d;
    float* dst_d;
    const int64_t ne00;
    const int64_t ne10;
    const int64_t ne1;
};

void get_rows_cuda(const get_row_context &ctx, cudaStream_t stream);
void get_rows_back_cuda(const get_row_back_context &ctx, cudaStream_t stream);

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

// binbcast.cu
struct bin_bcast_context {
    void* dst_d;
    const ggml_type src0_type, src1_type, dst_type;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    const int64_t ne10, ne11, ne12, ne13;
    const size_t nb10, nb11, nb12, nb13;
    const int64_t ne0, ne1, ne2, ne3;
    const size_t nb0, nb1, nb2, nb3;
    const bool src0_is_contiguous, src1_is_contiguous, dst_is_contiguous;
    void* src_data[12];
};

void repeat_cuda(const bin_bcast_context* ctx, cudaStream_t stream);

struct repeat_back_context {
    const ggml_type dst_type;
    const void* src0_d;
    void* dst_d;
    const size_t src0_ts;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    const int64_t ne0, ne1, ne2, ne3;
};

void repeat_back_cuda(const repeat_back_context* ctx, cudaStream_t stream);
void add_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void sub_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void mul_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void div_cuda(const bin_bcast_context* ctx, cudaStream_t stream);
void fused_add_cuda(const bin_bcast_context* ctx, int n_fuse, cudaStream_t stream);

// cpy
struct dup_context {
    const void* src_d;
    void* dst_d;
    const ggml_type src_type, dst_type;
    const int64_t ne;
    const size_t src_length, dst_length;
    const int64_t ne00, ne01, ne02, ne03;
    const size_t nb00, nb01, nb02, nb03;
    // rename later
    const int64_t ne10, ne11, ne12, ne13;
    const size_t nb10, nb11, nb12, nb13;
    const bool src_is_contiguous, dst_is_contiguous;
};
void dup_cuda(const dup_context* ctx, cudaStream_t stream);

// scale
void scale_f32_cuda(const float* x, float* dst, const float scale,
    const float bias, const size_t nelements, cudaStream_t stream);

// norm
void norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream);

void rms_norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream);

void rms_norm_back_f32_cuda(const float* grad, const float* xf, float* dst, const int ncols, const int nrows, const float eps, cudaStream_t stream);

void group_norm_f32_cuda(
    const float* x, float* dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream);

void l2_norm_f32_cuda(
    const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel,
    const int64_t stride_sample, const float eps, cudaStream_t stream);

// norm.cu
void rms_norm_mul_f32_cuda(const float* x,
    const float* mul,
    const float* add,
    float* dst,
    const int      ncols,
    const int      nrows,
    const int      nchannels,
    const int      nsamples,
    const int64_t  stride_row,
    const int64_t  stride_channel,
    const int64_t  stride_sample,
    const int64_t  mul_stride_row,
    const int64_t  mul_stride_channel,
    const int64_t  mul_stride_sample,
    const uint32_t mul_ncols,
    const uint32_t mul_nrows,
    const uint32_t mul_nchannels,
    const uint32_t mul_nsamples,
    const int64_t  add_stride_row,
    const int64_t  add_stride_channel,
    const int64_t  add_stride_sample,
    const uint32_t add_ncols,
    const uint32_t add_nrows,
    const uint32_t add_nchannels,
    const uint32_t add_nsamples,
    const float    eps,
    cudaStream_t   stream);

// gla
struct gla_context {
    const int64_t B;
    const int64_t T;
    const int64_t C;
    const int64_t H;
    const float scale;
    const float* k;
    const float* v;
    const float* r;
    const float* td;
    const float* s;
    float* dst;
};
void gated_linear_attn_cuda(const gla_context *ctx, cudaStream_t stream);

// wkv
void rwkv_wkv_cuda(const int B,
    const int T, const int C, const int H,
    const float* k, const float* v, const float* r,
    const float* tf, const float* td, const float* s, float* dst, cudaStream_t stream);
void rwkv_wkv7_cuda(const int B,
    const int T, const int C, const int H,
    const float* r, const float* w, const float* k,
    const float* v, const float* a, const float* b, const float* s, float* dst, cudaStream_t stream);

// compute_batched_ptrs
void k_compute_batched_ptrs_cuda(
    const void* src0_as_f16, const void* src1_as_f16, char* dst,
    const void** ptrs_src, void** ptrs_dst,
    int64_t ne12, int64_t ne13,
    int64_t ne23,
    size_t  nb02, size_t  nb03,
    size_t  nb12, size_t  nb13,
    size_t  nbd2, size_t  nbd3,
    int64_t r2, int64_t r3, cudaStream_t stream);

// clamp
struct clamp_context {
    cudaStream_t stream;
    const ggml_type src0_type, dst_type;
    const void* src0_d;
    void* dst_d;
    const int64_t nelements;
    const float min, max;
};
void clamp_cuda(const clamp_context* ctx);

// diagmask
void diag_mask_inf_f32_cuda(const float* x, float* dst, const int ncols_x, const int nrows_x, const int rows_per_channel, const int n_past, cudaStream_t stream);

// softmax
struct soft_max_params {
    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

struct softmax_context {
    const float* src0_d;
    const void* src1_d;
	const float* src2_d; // optional
    float* dst_d;
    const int64_t ne00;
    const int64_t nrows_x, nrows_y;
    const float scale, max_bias;
    bool use_f16;
    soft_max_params params;
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

// argsort
void argsort_f32_i32_cuda(
    const float* x, int* dst,
    const int ncols, const int nrows,
    ggml_sort_order order, cudaStream_t stream);

// sum
void sum_f32_cuda(ggml_cuda_pool& pool, const float* x, float* dst, const int64_t ne, cudaStream_t stream);

// sum_rows
void sum_rows_f32_cuda(const float* x, float* dst, const int ncols, const int nrows, cudaStream_t stream);

// upscale
struct upscale_context {
    const float* src0_d;
    float* dst_d;
    int64_t src0_ne[4];
    int64_t dst_ne[4];
    size_t src0_nb[4];
    size_t dst_nb[4];
    float sf0, sf1, sf2, sf3;
};

void upscale_f32_cuda(const upscale_context& ctx, cudaStream_t stream);

void upscale_f32_bilinear_cuda(const upscale_context& ctx, const float pixel_offset, cudaStream_t stream);

// acc
void acc_f32_cuda(const float* x, const float* y, float* dst, const int64_t n_elements,
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    const int64_t s1, const int64_t s2, const int64_t s3, const int64_t offset, cudaStream_t stream);

// pad.cu
struct pad_context {
    const float* src0_d;
    float* dst_d;
    int64_t src0_ne[4];
    int64_t dst_ne[4];
    size_t src0_nb[4];
    size_t dst_nb[4];
    const int lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3;
};

void pad_f32_cuda(const pad_context& ctx, cudaStream_t stream);

// pad_reflect_1d.cu
void pad_reflect_1d_cuda(
    const void* src0, void* dst,
    const int64_t ne0,
    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t nb0, const int64_t nb1, const int64_t nb2, const int64_t nb3,
    const int p0, const int p1, cudaStream_t stream);

// tsembd
void timestep_embedding_f32_cuda(const float* x, float* dst, const int ne00, const int nb1,
    const int dim, const int max_period, cudaStream_t stream);

// fattn related

struct flash_attn_ext_context {
    const int device;
    cudaStream_t main_stream;
    ggml_cuda_pool* pool;
    const float scale;
    const float max_bias;
    const float logit_softcap;
    const ggml_prec precision;

    struct {
        const ggml_type type;
        const void* data;
        const int64_t ne0, ne1, ne2, ne3;
        const size_t nb0, nb1, nb2, nb3;
        size_t element_size;
    } Q;

    struct {
        const ggml_type type;
        const size_t block_size, type_size;
        const void* data;
        const int64_t elements;
        const int64_t ne0, ne1, ne2, ne3;
        const size_t nb0, nb1, nb2, nb3;
        const size_t bs, ts;
        bool contiguously_allocated;
        size_t element_size;
    } K;

    struct {
        const bool exist;
        const ggml_type type;
        const size_t block_size, type_size;
        const void* data;
        const int64_t elements;
        const int64_t ne0, ne1, ne2, ne3;
        const size_t nb0, nb1, nb2, nb3;
        const size_t bs, ts;
        bool contiguously_allocated;
        size_t element_size;
    } V;

    struct {
        const bool exist;
        const ggml_type type;
        const void* data;
        const int64_t ne0, ne1, ne2, ne3;
        const size_t nb0, nb1, nb2, nb3;
    } mask;

    struct {
        const void* data;
    } sinks;

    struct {
        const ggml_type type;
        const void* data;
        const int64_t elements;
        const int64_t nrows;
        const int64_t ne0, ne1, ne2, ne3;
    } KQV;
};

void ggml_cuda_flash_attn_ext_vec(const flash_attn_ext_context& ctx);
void ggml_cuda_flash_attn_ext_tile(const flash_attn_ext_context& ctx);
void ggml_cuda_flash_attn_ext_wmma_f16(const flash_attn_ext_context& ctx);
void ggml_cuda_flash_attn_ext_mma_f16(const flash_attn_ext_context& ctx);

// cross-entropy-loss
struct cross_entropy_context {
    const int id;
    const size_t smpbo;
    ggml_cuda_pool& pool;
    const int64_t nrows;
    const int64_t ne00;
    const float* src0_d;
    const float* src1_d;
    float* dst_d;
};
void cross_entropy_loss_cuda(const cross_entropy_context* ctx, cudaStream_t stream);

struct cross_entropy_back_context {
    const int id;
    const size_t smpbo;
    ggml_cuda_pool& pool;
    const int64_t nrows;
    const int64_t ne00;
    const float* grad_d;
    const float* src0f_d;
    const float* src1f_d;
    float* dst_d;
};
void cross_entropy_loss_back_cuda(const cross_entropy_back_context* ctx, cudaStream_t stream);

// opt-step-adamw
void opt_step_adamw_f32_cuda(
    float* x, const float* g, float* g_m,
    float* g_v, const float* pars, const int64_t k, cudaStream_t stream);

// ssm-conv
void ssm_conv_f32_cuda(const float* src0, const float* src1, const int src0_nb0, const int src0_nb1,
    const int src0_nb2, const int src1_nb1, float* dst, const int dst_nb0, const int dst_nb1,
    const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
    const int64_t n_s, cudaStream_t stream);

// ssm-scan
void ssm_scan_f32_cuda(const float* src0, const float* src1, const float* src2, const float* src3,
    const float* src4, const float* src5, const int32_t* src6, float* dst,
    const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1,
    const int src2_nb2, const int src3_nb1, const int src4_nb2, const int src4_nb3, const int src5_nb2,
    const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
    const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
    cudaStream_t stream);

// conv2d-dw.cu
struct conv2d_dw_context {
    bool input_is_contiguous;
    bool input_is_contiguous_channels;
    const int64_t in_w, in_h;
	const int64_t kernel_w, kernel_h;
    const int64_t out_w, out_h, channels, batches;
    const float* x_d;
    float* y_d;
    const float* w_d;
    const int stride_w, stride_h;
    const int padding_w, padding_h;
    const int dilation_w, dilation_h;
};
void conv2d_dw_cuda(const conv2d_dw_context &ctx, cudaStream_t stream);

// conv2d-transpose.cu
struct conv2d_transpose_context {
    ggml_type kernel_type;
    const int64_t WIn, HIn;
    const int64_t WOut, HOut;
    const int64_t CIn, COut;
    const int64_t Kw, Kh;
    const int64_t N;
    const float* input_data;
    float* output_data;
    const void* kernel_data;
    const int32_t stride_w, stride_h;
    const int32_t padding_w, padding_h;
	const int32_t dilation_w, dilation_h;
};
void conv_2d_transpose_cuda(conv2d_transpose_context &ctx, cudaStream_t stream);

//mean.cu
void mean_fallback(const float* src0_d, float* dst_d, const int64_t ncols, const int64_t nrows, cudaStream_t stream);
void mean_cuda(ggml_cuda_pool& pool, const float* src0_d, float* dst_d, const int64_t ncols, const int64_t nrows, cudaStream_t stream);

//set-rows.cu
struct set_rows_context {
    ggml_type src1_type;
    ggml_type dst_type;
    const void* src0_d;
    const void* src1_d;
    void* dst_d;
    int64_t src0_ne[4];
    size_t src0_nb[4];
    int64_t src1_ne[4];
    size_t src1_nb[4];
    int64_t dst_ne[4];
    size_t dst_nb[4];
};

void set_rows_cuda(const set_rows_context &context, cudaStream_t stream);

// softcap.cu
void softcap_f32_cuda(const float* x, float* dst, const float scale, const float softcap, const int k, cudaStream_t stream);

// roll.cu
void roll_f32_cuda(const float* __restrict__ src,
    float* __restrict__ dst,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t ne02,
    const int64_t ne03,
    const int     s0,
    const int     s1,
    const int     s2,
    const int     s3,
    cudaStream_t stream);

// add_id.cu
struct add_id_context {
    int64_t src0_ne[4];
    int64_t src1_ne[4];
    int64_t src2_ne[4];
    int64_t dst_ne[4];
    size_t src0_nb[4];
    size_t src1_nb[4];
    size_t src2_nb[4];
    size_t dst_nb[4];
    const float* src0_d;
    const float* src1_d;
    const int32_t* src2_d;
    float* dst_d;
};

void add_id_cuda(const add_id_context &ctx, cudaStream_t stream);

// opt-step-sgd.cu
void opt_step_sgd_f32_cuda(
    float* x, const float* g, const float* pars, const int64_t k, cudaStream_t stream);

// mmf.cu

struct mmf_ids_data {
    const int32_t* ids_src_compact = nullptr;
    const int32_t* ids_dst_compact = nullptr;
    const int32_t* expert_bounds_dev = nullptr;
    int n_experts = 0;
    int sis1 = 0;
};

bool ggml_cuda_should_use_mmf(enum ggml_type type, 
    size_t type_size, int cc, int warp_size, const int64_t* scr0_ne, int64_t src1_ncols, bool mul_mat_id);
struct mul_mat_f_context {
    ggml_type src0_type;
	const void* src0_d;
    const float* src1_d;
    const int32_t* ids_d;
	float* dst_d;
    int64_t ne00, ne01, ne02, ne03;
    int64_t ne3;

    const int64_t ncols_dst;
    const int64_t stride_col_y;
    const int64_t stride_col_dst;
    const size_t ids_s0, ids_s1;
    const int64_t nchannels_y;
    const int64_t nchannels_dst;
    const int64_t stride_channel_y;
    const int64_t stride_channel_dst;

    const int64_t s01, s02, s03;
    const int64_t s11, s13;
    const int64_t s1, s3;

    const mmf_ids_data* ids_info_ptr;
};

void mul_mat_f_cuda(const mul_mat_f_context* ctx, cudaStream_t stream);

// mmvf.cu
bool ggml_cuda_should_use_mmvf(enum ggml_type type, int cc, const int64_t* src0_ne, int64_t ne11);
struct mul_mat_vec_f_context {
    ggml_type src0_type;
    const void* src0_d;
    const float* src1_d;
    const int32_t* ids_d;
    float* dst_d;
    int64_t ne00, ne01, ne02, ne03;
    int64_t ne3;

    const int64_t ncols_dst;
    const int64_t nchannels_y;
    const int64_t nchannels_dst;
    const int64_t stride_channel_dst;
    const int64_t stride_channel_y;

    const int64_t s01, s02, s03;
    const int64_t s11, s13;
    const int64_t s1, s3;
    const enum ggml_prec prec;
};

void mul_mat_vec_f_cuda(const mul_mat_vec_f_context* ctx, cudaStream_t stream);

// conv2d.cu
struct conv2d_context {
    ggml_type kernel_type;
    const int64_t N, CIn, IH, IW;
    const int64_t COut, OH, OW;
    const int64_t KH, KW;
    const float* input_d;
    const void* kernel_d;
    float* output_d;
    const int stride_w, stride_h;
    const int pad_w, pad_h;
    const int dilation_w, dilation_h;
};

void conv2d_cuda(const conv2d_context& ctx, cudaStream_t stream);

// topk-moe.cu
void launch_topk_moe_cuda(bool with_norm, const float* logits_d, float* weights_d, int32_t* ids_d,
    const int n_rows, const int n_experts, const int n_expert_used, cudaStream_t stream);
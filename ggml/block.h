#pragma once
#include <stdint.h>

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_0 {
    static constexpr int block_size = 32;
    uint16_t d;            // delta
    uint8_t qs[block_size / 2]; // nibbles / quants
};
static_assert(sizeof(block_q4_0) == sizeof(uint16_t) + block_q4_0::block_size / 2, "wrong q4_0 block size/padding");

struct block_q4_1 {
    static constexpr int block_size = 32;
    uint32_t dm;           // delta, min
    uint8_t qs[block_size / 2]; // nibbles / quants
};
static_assert(sizeof(block_q4_1) == 2 * sizeof(uint16_t) + block_q4_1::block_size / 2, "wrong q4_1 block size/padding");

struct block_q5_0 {
    static constexpr int block_size = 32;
    uint16_t d;            // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[block_size / 2]; // nibbles / quants
};
static_assert(sizeof(block_q5_0) == sizeof(uint16_t) + sizeof(uint32_t) + block_q5_0::block_size / 2, "wrong q5_0 block size/padding");

struct block_q5_1 {
    static constexpr int block_size = 32;
    uint32_t dm;           // delta, min
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[block_size / 2]; // nibbles / quants
};
static_assert(sizeof(block_q5_1) == 2 * sizeof(uint16_t) + sizeof(uint32_t) + block_q5_1::block_size / 2, "wrong q5_1 block size/padding");

struct block_q8_0 {
    static constexpr int block_size = 32;
    uint16_t d;        // delta
    int8_t  qs[block_size]; // quants
};
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + block_q8_0::block_size, "wrong q8_0 block size/padding");

#define QK8_1 32
struct block_q8_1 {
    uint32_t ds;      // delta, d * sum(qs[i])
    int8_t qs[QK8_1]; // quants
};
static_assert(sizeof(block_q8_1) == 2 * sizeof(uint16_t) + QK8_1, "wrong q8_1 block size/padding");

//
// Ternary quantization
//

// 1.6875 bpw
struct block_tq1_0 {
    uint8_t qs[(QK_K - 4 * QK_K / 64) / 5]; // 5 elements per byte (3^5 = 243 < 256)
    uint8_t qh[QK_K / 64]; // 4 elements per byte
    uint16_t d;
};
static_assert(sizeof(block_tq1_0) == sizeof(uint16_t) + QK_K / 64 + (QK_K - 4 * QK_K / 64) / 5, "wrong tq1_0 block size/padding");

// 2.0625 bpw
struct block_tq2_0 {
    uint8_t qs[QK_K / 4]; // 2 bits per element
    uint16_t d;
};
static_assert(sizeof(block_tq2_0) == sizeof(uint16_t) + QK_K / 4, "wrong tq2_0 block size/padding");

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
struct block_q2_K {
    uint8_t scales[QK_K / 16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K / 4];      // quants
    uint32_t dm;               // super-block scale for quantized scales, super-block scale for quantized mins
};
static_assert(sizeof(block_q2_K) == 2 * sizeof(uint16_t) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
struct block_q3_K {
    uint8_t hmask[QK_K / 8]; // quants - high bit
    uint8_t qs[QK_K / 4];    // quants - low 2 bits
    uint8_t scales[12];      // scales, quantized with 6 bits
    uint16_t d;              // super-block scale
};
static_assert(sizeof(block_q3_K) == sizeof(uint16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
struct block_q4_K {
    uint32_t dm;                    // super-block scale for quantized scales, super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];           // 4--bit quants
} ;
static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size/padding");

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
struct block_q5_K {
    uint32_t dm;                    // super-block scale for quantized scales, super-block scale for quantized mins
    uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K / 8];           // quants, high bit
    uint8_t qs[QK_K / 2];           // quants, low 4 bits
};
static_assert(sizeof(block_q5_K) == 2 * sizeof(uint16_t) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8, "wrong q5_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // quants, lower 4 bits
    uint8_t qh[QK_K / 4];      // quants, upper 2 bits
    int8_t  scales[QK_K / 16]; // scales, quantized with 8 bits
    uint16_t d;                // super-block scale
};
static_assert(sizeof(block_q6_K) == sizeof(uint16_t) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
struct block_q8_K {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
};
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K / 16 * sizeof(int16_t), "wrong q8_K block size/padding");

// (Almost) "true" 2-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 2.0625 bpw because of the 16-bit scale for each block of 256.
struct block_iq2_xxs {
    uint16_t d;
    uint16_t qs[QK_K / 8];
};
static_assert(sizeof(block_iq2_xxs) == sizeof(uint16_t) + QK_K / 8 * sizeof(uint16_t), "wrong iq2_xxs block size/padding");

// 2.3125 bpw quants
struct block_iq2_xs {
    uint16_t d;
    uint16_t qs[QK_K / 8];
    uint8_t  scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_xs) == sizeof(uint16_t) + QK_K / 8 * sizeof(uint16_t) + QK_K / 32, "wrong iq2_xs block size/padding");

// 2.5625 bpw quants
struct block_iq2_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_s) == sizeof(uint16_t) + QK_K / 4 + QK_K / 16, "wrong iq2_s block size/padding");

// (Almost) "true" 3-bit quantization.
// Due to the need to use blocks as per ggml design, it ends up using
// 3.0625 bpw because of the 16-bit scale for each block of 256.
struct block_iq3_xxs {
    uint16_t d;
    uint8_t qs[3 * QK_K / 8];
};
static_assert(sizeof(block_iq3_xxs) == sizeof(uint16_t) + 3 * (QK_K / 8), "wrong iq3_xxs block size/padding");

// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
struct block_iq3_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[IQ3S_N_SCALE];
};
static_assert(sizeof(block_iq3_s) == sizeof(uint16_t) + 13 * (QK_K / 32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");

// 1.5625 bpw
struct block_iq1_s {
    uint16_t d;
    uint8_t  qs[QK_K / 8];
    uint16_t qh[QK_K / 32];
};
static_assert(sizeof(block_iq1_s) == sizeof(uint16_t) + QK_K / 8 + QK_K / 16, "wrong iq1_s block size/padding");

// 1.75 bpw
struct block_iq1_m {
    uint8_t  qs[QK_K / 8];      // grid index, low 8 bits
    uint8_t  qh[QK_K / 16];     // grid index, high 3 bits + grid shift bit (for two groups of 8)
    uint8_t  scales[QK_K / 32]; // 3-bit block scales (4-bit if QK_K == 64)
};
static_assert(sizeof(block_iq1_m) == QK_K / 8 + QK_K / 16 + QK_K / 32, "wrong iq1_m block size/padding");

// Used by IQ1_M quants
using iq1m_scale_t = uint16_t;

// Non-linear quants
struct block_iq4_nl {
    static constexpr int block_size = 32;
    uint16_t d;
    uint8_t qs[block_size / 2];
};
static_assert(sizeof(block_iq4_nl) == sizeof(uint16_t) + block_iq4_nl::block_size / 2, "wrong iq4_nl block size/padding");

struct block_iq4_xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K / 64];
    uint8_t  qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == sizeof(uint16_t) + sizeof(uint16_t) + QK_K / 64 + QK_K / 2, "wrong iq4_xs block size/padding");

module;
#include <string.h>
#include <algorithm>
#include <bit>

#define GGML_ASSERT(...)
#define GGML_ABORT(...)

inline static void ggml_vec_scale_f321(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
	vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
	const int np = (n & ~(GGML_F32_STEP - 1));

	GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	GGML_F32_VEC ay[GGML_F32_ARR];

	for (int i = 0; i < np; i += GGML_F32_STEP) {
		for (int j = 0; j < GGML_F32_ARR; j++) {
			ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
			ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

			GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
		}
	}

	// leftovers
	for (int i = np; i < n; ++i) {
		y[i] *= v;
	}
#else
	// scalar
	for (int i = 0; i < n; ++i) {
		y[i] *= v;
	}
#endif
}

module ggml:cpu.op.scale;
import :ds;
import :tensor;
import :cpu.ds;

inline static void ggml_vec_scale_f32(const int n, float* y, const float   v) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmul(y, 1, &v, y, 1, n);
#elif defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
    const int sve_register_length = ggml_cpu_get_sve_cnt() * 8;
    const int ggml_f32_epr = sve_register_length / 32;//8;//svcntw(); // SVE128:4, SVE256:8, SVE512:16
    const int ggml_f32_step = 2 * ggml_f32_epr;

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);
    const int np = (n & ~(ggml_f32_step - 1));
    svfloat32_t ay1;
    svfloat32_t ay2;
    for (int i = 0; i < np; i += ggml_f32_step) {
        ay1 = GGML_F32_VEC_LOAD(y + i);
        ay1 = GGML_F32_VEC_MUL(ay1, vx);
        GGML_F32_VEC_STORE(y + i, ay1);

        ay2 = GGML_F32_VEC_LOAD(y + i + 1 * ggml_f32_epr);
        ay2 = GGML_F32_VEC_MUL(ay2, vx);
        GGML_F32_VEC_STORE(y + i + 1 * ggml_f32_epr, ay2);
    }
    // leftovers
    // maximum number of leftover elements will be less that ggml_f32_epr. Apply predicated svmad on available elements only
    if (np < n) {
        svbool_t pg = svwhilelt_b32(np, n);
        ay1 = svld1_f32(pg, y + np);
        ay1 = svmul_f32_m(pg, ay1, vx);
        svst1_f32(pg, y + np, ay1);
    }
#else
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(y + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] *= v;
    }
#endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] *= v;
    }
#endif
}

inline static void ggml_vec_mad1_f32(const int n, float* y, const float* x, const float s, const float b) {
#if defined(GGML_USE_ACCELERATE)
    vDSP_vsmsa(x, 1, &s, &b, y, 1, n);
#elif defined(GGML_SIMD)
#if defined(__ARM_FEATURE_SVE)
    // scalar ; TODO: Write SVE code
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * s + b;
    }
#else
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC vs = GGML_F32_VEC_SET1(s);
    GGML_F32_VEC vb = GGML_F32_VEC_SET1(b);

    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ay[j] = GGML_F32_VEC_LOAD(x + i + j * GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_FMA(ay[j], vs, vb);

            GGML_F32_VEC_STORE(y + i + j * GGML_F32_EPR, ay[j]);
        }
    }

    // leftovers
    for (int i = np; i < n; ++i) {
        y[i] = x[i] * s + b;
    }
#endif
#else
    // scalar
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] * s + b;
    }
#endif
}

static void ggml_compute_forward_scale_f32(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float s; // scale factor
    float b; // bias

    memcpy(&s, (float*)dst->op_params + 0, sizeof(float));
    memcpy(&b, (float*)dst->op_params + 1, sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1) / nth;

    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = std::min(ir0 + dr, nr);

    const size_t nb01 = src0->nb[1];

    const size_t nb1 = dst->nb[1];

    if (b == 0.0f) {
        for (int i1 = ir0; i1 < ir1; i1++) {
            if (dst->data != src0->data) {
                // src0 is same shape as dst => same indices
                // TODO: add x parameter to ggml_vec_scale_f32 and remove this memcpy
                memcpy((char*)dst->data + i1 * nb1, (char*)src0->data + i1 * nb01, nc * sizeof(float));
            }
            ggml_vec_scale_f32(nc, (float*)((char*)dst->data + i1 * nb1), s);
        }
    }
    else {
        for (int i1 = ir0; i1 < ir1; i1++) {
            ggml_vec_mad1_f32(nc,
                (float*)((char*)dst->data + i1 * nb1),
                (float*)((char*)src0->data + i1 * nb1),
                s, b);
        }
    }
}

void ggml_compute_forward_scale(
	const struct ggml_compute_params* params,
	struct ggml_tensor* dst) {

	const struct ggml_tensor* src0 = dst->src[0];

	switch (src0->type) {
	case GGML_TYPE_F32:
	{
		ggml_compute_forward_scale_f32(params, dst);
	} break;
	default:
	{
		GGML_ABORT("fatal error");
	}
	}
}

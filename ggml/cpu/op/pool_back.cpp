module;
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <string.h>

#define GGML_ABORT(...)

module ggml;
import :types;
import :cpu.op;

void ggml_compute_forward_pool_2d_back(
    ggml_tensor* dst) {

    const ggml_tensor* src = dst->src[0];
    const ggml_tensor* dstf = dst->src[1]; // forward tensor of dst

    assert(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);

    const int32_t* opts = (const int32_t*)dst->op_params;
    ggml_op_pool op = static_cast<ggml_op_pool>(opts[0]);
    const int k0 = opts[1];
    const int k1 = opts[2];
    const int s0 = opts[3];
    const int s1 = opts[4];
    const int p0 = opts[5];
    const int p1 = opts[6];

    char* cdata = (char*)dst->data;
    const char* cdataf = (const char*)dstf->data;
    const char* const data_end = cdata + dst->nbytes();

    memset(cdata, 0, dst->nbytes());

    const int64_t px = src->ne[0];
    const int64_t py = src->ne[1];
    const int64_t pa = px * py;

    const float* splane = (const float*)src->data;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            const float* const srow = splane + oy * px;
            for (int ox = 0; ox < px; ++ox) {
                const float grad0 = srow[ox];

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                if (op == GGML_OP_POOL_MAX) {
                    float maxval = -FLT_MAX;
                    int kxmax = -1;
                    int kymax = -1;

                    for (int ky = 0; ky < k1; ++ky) {
                        if (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            continue;
                        }
                        const void* drowf = (const void*)(cdataf + dst->nb[1] * (iy + ky));
                        for (int kx = 0; kx < k0; ++kx) {
                            int j = ix + kx;
                            if (j < 0 || j >= dst->ne[0]) {
                                continue;
                            }

                            const float val = dst->type == GGML_TYPE_F32 ?
                                ((const float*)drowf)[j] : toFloat32(((const ggml_fp16_t*)drowf)[j]);
                            if (val <= maxval) {
                                continue;
                            }

                            maxval = val;
                            kxmax = kx;
                            kymax = ky;
                        }
                    }

                    if (kxmax == -1 || kymax == -1) {
                        continue;
                    }

                    void* drow = (void*)(cdata + dst->nb[1] * (iy + kymax));
                    const int j = ix + kxmax;
                    if (dst->type == GGML_TYPE_F32) {
                        ((float*)drow)[j] += grad0;
                    }
                    else {
                        ((ggml_fp16_t*)drow)[j] = fromFloat32<ggml_fp16_t>(grad0 + toFloat32(((const ggml_fp16_t*)drow)[j]));
                    }
                }
                else if (op == GGML_OP_POOL_AVG) {
                    const float grad = grad0 / ka;

                    for (int ky = 0; ky < k1; ++ky) {
                        if (iy + ky < 0 || iy + ky >= dst->ne[1]) {
                            continue;
                        }
                        void* drow = (void*)(cdata + dst->nb[1] * (iy + ky));
                        for (int kx = 0; kx < k0; ++kx) {
                            int j = ix + kx;
                            if (j < 0 || j >= dst->ne[0]) {
                                continue;
                            }

                            if (dst->type == GGML_TYPE_F32) {
                                ((float*)drow)[j] += grad;
                            }
                            else {
                                ((ggml_fp16_t*)drow)[j] += fromFloat32<ggml_fp16_t>(grad);
                            }
                        }
                    }
                }
                else {
                    assert(false);
                }
            }
        }

        cdata += dst->nb[2];
        cdataf += dst->nb[2];
        splane += pa;
    }
}
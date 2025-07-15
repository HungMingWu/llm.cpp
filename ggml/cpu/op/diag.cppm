module;
#include <assert.h>
#include <stdint.h>
#define GGML_ABORT(...)

module ggml:cpu.op.diag;
import :cpu.ds;

static void ggml_compute_forward_diag_f32(
    ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];

    // TODO: handle transposed/permuted matrices

    assert(src0->ne[0] == dst->ne[0]);
    assert(src0->ne[0] == dst->ne[1]);
    assert(src0->ne[1] == 1);
    assert(src0->ne[2] == dst->ne[2]);
    assert(src0->ne[3] == dst->ne[3]);

    assert(src0->nb[0] == sizeof(float));
    assert(dst->nb[0] == sizeof(float));

    for (int i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int i2 = 0; i2 < dst->ne[2]; i2++) {
            for (int i1 = 0; i1 < dst->ne[1]; i1++) {
                float* d = (float*)((char*)dst->data + i3 * dst->nb[3] + i2 * dst->nb[2] + i1 * dst->nb[1]);
                float* s = (float*)((char*)src0->data + i3 * src0->nb[3] + i2 * src0->nb[2]);
                for (int i0 = 0; i0 < i1; i0++) {
                    d[i0] = 0;
                }
                d[i1] = s[i1];
                for (int i0 = i1 + 1; i0 < dst->ne[0]; i0++) {
                    d[i0] = 0;
                }
            }
        }
    }
}

void ggml_compute_forward_diag(
    ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F32:
    {
        ggml_compute_forward_diag_f32(dst);
    } break;
    default:
    {
        GGML_ABORT("fatal error");
    }
    }
}
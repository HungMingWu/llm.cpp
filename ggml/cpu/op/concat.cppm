module;
#include <assert.h>
#include <algorithm>
#include <bit>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml:cpu.op.concat;
import :ds;
import :tensor;
import :cpu.ds;

template <typename T>
static void ggml_compute_forward_concat(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];

    GGML_ASSERT(ggml_type_size(src0->type) == sizeof(T));

    const int ith = params->ith;
    const int nth = params->nth;

    const int32_t dim = std::bit_cast<int32_t>(dst->op_params[0]);

    GGML_ASSERT(dim >= 0 && dim < 4);

    int64_t o[4] = { 0, 0, 0, 0 };
    o[dim] = src0->ne[dim];

    const T* x;

    // TODO: smarter multi-theading
    for (int i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int i2 = ith; i2 < dst->ne[2]; i2 += nth) {
            for (int i1 = 0; i1 < dst->ne[1]; i1++) {
                for (int i0 = 0; i0 < dst->ne[0]; i0++) {
                    if (i0 < src0->ne[0] && i1 < src0->ne[1] && i2 < src0->ne[2] && i3 < src0->ne[3]) {
                        x = (const T*)((const char*)src0->data + 
                            (i0)*src0->nb[0] + (i1)*src0->nb[1] + (i2)*src0->nb[2] + (i3)*src0->nb[3]);
                    }
                    else {
                        x = (const T*)((const char*)src1->data + 
                            (i0 - o[0]) * src1->nb[0] + (i1 - o[1]) * 
                            src1->nb[1] + (i2 - o[2]) * src1->nb[2] + (i3 - o[3]) * src1->nb[3]);
                    }

                    T* y = (T*)((char*)dst->data + 
                        i0 * dst->nb[0] + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);

                    *y = *x;
                }
            }
        }
    }
}

void ggml_compute_forward_concat(
    const struct ggml_compute_params* params,
    struct ggml_tensor* dst) {

    const struct ggml_tensor* src0 = dst->src[0];

    switch (src0->type) {
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I16:
    {
        ggml_compute_forward_concat<ggml_fp16_t>(params, dst);
    } break;
    case GGML_TYPE_I8:
    {
        ggml_compute_forward_concat<int8_t>(params, dst);
    } break;
    case GGML_TYPE_F32:
    case GGML_TYPE_I32:
    {
        ggml_compute_forward_concat<ggml_fp32_t>(params, dst);
    } break;
    default:
    {
        ggml_compute_forward_concat<char>(params, dst);
    }
    }
}
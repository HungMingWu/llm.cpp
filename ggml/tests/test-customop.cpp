#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <atomic>
#include <span>
#include <vector>

import ggml;
import test;

std::atomic<int> g_custom1_count = 0;
std::atomic<int> g_custom2_count = 0;
std::atomic<int> g_custom3_count = 0;

void custom1(ggml_tensor* dst, int ith, int nth) {
    assert(dst->src.size() == 1);
    const ggml_tensor* a = dst->src[0];
    assert(ggml_are_same_shape(dst, a));

    g_custom1_count++;

    const float* a_data = ggml_get_data_f32(a);
    float* dst_data = ggml_get_data_f32(dst);

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));

    // parallelize by elements
    const int ne = (int)dst->nelements();
    const int dr = (ne + nth - 1) / nth;
    const int ie0 = dr * ith;
    const int ie1 = std::min(ie0 + dr, ne);

    for (int i = ie0; i < ie1; ++i) {
        dst_data[i] = a_data[i] * 2;
    }
}

void custom2(ggml_tensor* dst, int ith, int nth) {
    assert(dst->src.size() == 2);
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    assert(ggml_are_same_shape(dst, a));
    assert(ggml_are_same_shape(dst, b));

    g_custom2_count++;

    const float* a_data = ggml_get_data_f32(a);
    const float* b_data = ggml_get_data_f32(b);
    float* dst_data = ggml_get_data_f32(dst);

    // parallelize by rows
    const int nr = (int)ggml_nrows(dst);
    // number of rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = std::min(ir0 + dr, nr);

    // number of columns
    const int nc = (int)dst->ne[0];

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));
    assert(ggml_is_contiguous(b));

    for (int ir = ir0; ir < ir1; ++ir) {
        for (int ic = 0; ic < nc; ++ic) {
            const int i = ir * nc + ic;
            dst_data[i] = a_data[i] + b_data[i];
        }
    }
}

void custom3(ggml_tensor* dst, int ith, int nth) {
    assert(dst->src.size() == 3);
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    const ggml_tensor* c = dst->src[2];
    assert(ggml_are_same_shape(dst, a));
    assert(ggml_are_same_shape(dst, b));
    assert(ggml_are_same_shape(dst, c));

    g_custom3_count++;

    const float* a_data = ggml_get_data_f32(a);
    const float* b_data = ggml_get_data_f32(b);
    const float* c_data = ggml_get_data_f32(c);
    float* dst_data = ggml_get_data_f32(dst);

    // dont parallelize
    assert(ith == 0);

    // number of elements
    const int ne = (int)dst->nelements();

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));
    assert(ggml_is_contiguous(b));
    assert(ggml_is_contiguous(c));

    for (int i = 0; i < ne; ++i) {
        dst_data[i] = a_data[i] + b_data[i] + c_data[i];
    }
}

void custom(ggml_tensor* dst, int ith, int nth) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const ggml_tensor* src2 = dst->src[2];
    const ggml_tensor* src3 = dst->src[3];
    const ggml_tensor* src4 = dst->src[4];

    int32_t* dst_data = (int32_t*)ggml_get_data(dst);
    const float* src0_data = ggml_get_data_f32(src0);
    const float* src1_data = ggml_get_data_f32(src1);
    const float* src2_data = ggml_get_data_f32(src2);
    const float* src3_data = ggml_get_data_f32(src3);
    const float* src4_data = ggml_get_data_f32(src4);

    // check that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(src0));
    assert(ggml_is_contiguous(src1));
    assert(ggml_is_contiguous(src2));
    assert(ggml_is_contiguous(src3));
    assert(ggml_is_contiguous(src4));

    // check that the shapes are the same
    assert(ggml_are_same_shape(dst, src0));
    assert(ggml_are_same_shape(dst, src1));
    assert(ggml_are_same_shape(dst, src2));
    assert(ggml_are_same_shape(dst, src3));
    assert(ggml_are_same_shape(dst, src4));


    for (int i = ith; i < dst->nelements(); i += nth) {
        dst_data[i] = src0_data[i] + src1_data[i] * src2_data[i] - src3_data[i] * src4_data[i];
    }
}

int main(int argc, const char** argv) {

    float buf1_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf1_f32[i] = (float)(i + 1);
    }
    float buf2_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf2_f32[i] = (float)(i + 1) * 2;
    }
    float buf3_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf3_f32[i] = (float)(i + 1) * 3;
    }

    // map_custom1
    // 2 tasks, no userdata, parallelized by elements
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 10, 2 });

        ggml_tensor* m1 = ggml_map_custom(&ctx, { t }, false, custom1, 2);

        ggml_cgraph graph;
        graph.build_forward_expand(m1);

        const std::vector<float> output = run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            ggml_backend_tensor_set(tensor, buf1_f32, 0, tensor->nbytes());
        });

        for (int i = 0; i < m1->nelements(); ++i) {
            assert(output[i] == buf1_f32[i] * 2);
        }
        assert(g_custom1_count == 2);
    }

    // map_custom2
    // max tasks (4), userdata, parallelized by rows
    {
        ggml_context ctx;
        ggml_tensor* t1 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t2 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* m2 = ggml_map_custom(&ctx, { t1, t2 }, false, custom2);

        ggml_cgraph graph;
        graph.build_forward_expand(m2);

        const std::vector<float> output = run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t1) {
                ggml_backend_tensor_set(tensor, buf1_f32, 0, tensor->nbytes());
            }
            else {
                ggml_backend_tensor_set(tensor, buf2_f32, 0, tensor->nbytes());
            }
        });

        for (int i = 0; i < m2->nelements(); ++i) {
            assert(output[i] == buf1_f32[i] + buf2_f32[i]);
        }

        assert(g_custom2_count == 4);
    }

    // map_custom3
    // 1 task, userdata, not parallelized
    {
        ggml_context ctx;
        ggml_tensor* t1 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t2 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t3 = ctx.create(GGML_TYPE_F32, { 10, 2 });

        ggml_tensor* m3 = ggml_map_custom(&ctx, { t1, t2, t3 }, false, custom3, 1);

        ggml_cgraph graph;
        graph.build_forward_expand(m3);

        const std::vector<float> output = run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t1) {
                ggml_backend_tensor_set(tensor, buf1_f32, 0, tensor->nbytes());
            }
            else if (tensor == t2) {
                ggml_backend_tensor_set(tensor, buf2_f32, 0, tensor->nbytes());
            }
            else {
                ggml_backend_tensor_set(tensor, buf3_f32, 0, tensor->nbytes());
            }
        });

        for (int i = 0; i < m3->nelements(); ++i) {
            assert(output[i] == buf1_f32[i] + buf2_f32[i] + buf3_f32[i]);
        }

        assert(g_custom3_count == 1);
    }

    // custom
    {
        ggml_context ctx;
        ggml_tensor* t1 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t2 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t3 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t4 = ctx.create(GGML_TYPE_F32, { 10, 2 });
        ggml_tensor* t5 = ctx.create(GGML_TYPE_F32, { 10, 2 });

        ggml_tensor* m4 = ggml_custom(&ctx, GGML_TYPE_I32, { 10, 2, 1, 1 }, { t1, t2, t3, t4, t5 }, custom);

        ggml_cgraph graph;
        graph.build_forward_expand(m4);

        const std::vector<int32_t> output = run_graph_in_cpu<int32_t>(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t1 || tensor == t4) {
                ggml_backend_tensor_set(tensor, buf1_f32, 0, tensor->nbytes());
            }
            else if (tensor == t2 || tensor == t5) {
                ggml_backend_tensor_set(tensor, buf2_f32, 0, tensor->nbytes());
            }
            else {
                ggml_backend_tensor_set(tensor, buf3_f32, 0, tensor->nbytes());
            }
        });

        for (int i = 0; i < m4->nelements(); ++i) {
            assert(output[i] == buf1_f32[i] + buf2_f32[i] * buf3_f32[i] - buf1_f32[i] * buf2_f32[i]);
        }
    }

    return 0;
}

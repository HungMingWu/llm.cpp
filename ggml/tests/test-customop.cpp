#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <atomic>
#include <functional>
#include <span>
#include <vector>
#include "mdspan.hpp"

import ggml;
import test;

task_vector custom1(ggml_tensor* dst) {
    assert(dst->src.size() == 1);
    const ggml_tensor* a = dst->src[0];
    assert(ggml_are_same_shape(dst, a));

    const float* a_data = ggml_get_data_f32(a);
    float* dst_data = ggml_get_data_f32(dst);

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));

    task_vector tasks;
    // parallelize by elements
    const int64_t ne = dst->nelements();
    const int64_t boundary = 1 * 1024 * 1024;
    for (int64_t i0 = 0; i0 < ne; i0 += boundary) {
        const int64_t i1 = std::min(ne, boundary);
        tasks.emplace_back([=] {
            for (int64_t i = i0; i < i1; i++) {
                dst_data[i] = a_data[i] * 2;
            }
        });
    }
    return tasks;
}

task_vector custom2(ggml_tensor* dst) {
    assert(dst->src.size() == 2);
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    assert(ggml_are_same_shape(dst, a));
    assert(ggml_are_same_shape(dst, b));

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));
    assert(ggml_is_contiguous(b));
    std::mdspan dst_data(ggml_get_data_f32(dst), dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]);
    std::mdspan a_data(ggml_get_data_f32(a), a->ne[3], a->ne[2], a->ne[1], a->ne[0]);
    std::mdspan b_data(ggml_get_data_f32(b), b->ne[3], b->ne[2], b->ne[1], b->ne[0]);

    task_vector tasks;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            tasks.emplace_back([=] {
                for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
                        dst_data[i3, i2, i1, i0] = a_data[i3, i2, i1, i0] + b_data[i3, i2, i1, i0];
                    }
                }
            });
        }
    }
    return tasks;
}

task_vector custom3(ggml_tensor* dst) {
    assert(dst->src.size() == 3);
    const ggml_tensor* a = dst->src[0];
    const ggml_tensor* b = dst->src[1];
    const ggml_tensor* c = dst->src[2];
    assert(ggml_are_same_shape(dst, a));
    assert(ggml_are_same_shape(dst, b));
    assert(ggml_are_same_shape(dst, c));

    const float* a_data = ggml_get_data_f32(a);
    const float* b_data = ggml_get_data_f32(b);
    const float* c_data = ggml_get_data_f32(c);
    float* dst_data = ggml_get_data_f32(dst);

    // number of elements
    const int ne = (int)dst->nelements();

    // this assumes that the tensors are contiguous
    assert(ggml_is_contiguous(dst));
    assert(ggml_is_contiguous(a));
    assert(ggml_is_contiguous(b));
    assert(ggml_is_contiguous(c));

    task_vector tasks;
    tasks.emplace_back([=] {
        for (int i = 0; i < ne; ++i) {
            dst_data[i] = a_data[i] + b_data[i] + c_data[i];
        }
    });
    return tasks;
}

task_vector custom(ggml_tensor* dst) {
    const ggml_tensor* src0 = dst->src[0];
    const ggml_tensor* src1 = dst->src[1];
    const ggml_tensor* src2 = dst->src[2];
    const ggml_tensor* src3 = dst->src[3];
    const ggml_tensor* src4 = dst->src[4];

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

    std::mdspan dst_data((int32_t*)ggml_get_data(dst), dst->ne[3], dst->ne[2], dst->ne[1], dst->ne[0]);
    std::mdspan src0_data(ggml_get_data_f32(src0), src0->ne[3], src0->ne[2], src0->ne[1], src0->ne[0]);
    std::mdspan src1_data(ggml_get_data_f32(src1), src1->ne[3], src1->ne[2], src1->ne[1], src1->ne[0]);
    std::mdspan src2_data(ggml_get_data_f32(src2), src2->ne[3], src2->ne[2], src2->ne[1], src2->ne[0]);
    std::mdspan src3_data(ggml_get_data_f32(src3), src3->ne[3], src3->ne[2], src3->ne[1], src3->ne[0]);
    std::mdspan src4_data(ggml_get_data_f32(src4), src4->ne[3], src4->ne[2], src4->ne[1], src4->ne[0]);

    task_vector tasks;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            tasks.emplace_back([=] {
                for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
                        dst_data[i3, i2, i1, i0] = src0_data[i3, i2, i1, i0] +
                            src1_data[i3, i2, i1, i0] * src2_data[i3, i2, i1, i0] -
                            src3_data[i3, i2, i1, i0] * src4_data[i3, i2, i1, i0];
                    }
                }
            });
        }
    }
    return tasks;
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
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, 10, 2);

        ggml_tensor* m1 = ggml_map_custom(&ctx, { t }, false, custom1);

        ggml_cgraph graph;
        graph.build_forward_expand(m1);

        const std::vector<float> output = run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            ggml_backend_tensor_set(tensor, buf1_f32, 0, tensor->nbytes());
        });

        for (int i = 0; i < m1->nelements(); ++i) {
            assert(output[i] == buf1_f32[i] * 2);
        }
    }

    // map_custom2
    // max tasks (4), userdata, parallelized by rows
    {
        ggml_context ctx;
        ggml_tensor* t1 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t2 = ctx.create(GGML_TYPE_F32, 10, 2);
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
    }

    // map_custom3
    // 1 task, userdata, not parallelized
    {
        ggml_context ctx;
        ggml_tensor* t1 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t2 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t3 = ctx.create(GGML_TYPE_F32, 10, 2);

        ggml_tensor* m3 = ggml_map_custom(&ctx, { t1, t2, t3 }, false, custom3);

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
    }

    // custom
    {
        ggml_context ctx;
        ggml_tensor* t1 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t2 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t3 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t4 = ctx.create(GGML_TYPE_F32, 10, 2);
        ggml_tensor* t5 = ctx.create(GGML_TYPE_F32, 10, 2);

        ggml_tensor* m4 = ggml_custom(&ctx, GGML_TYPE_F32, { 10, 2, 1, 1 }, { t1, t2, t3, t4, t5 }, custom);

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

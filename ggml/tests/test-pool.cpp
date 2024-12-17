#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#define GGML_ASSERT(...) assert(__VA_ARGS__)

import ggml;
import test;

int main(int argc, const char** argv) {

    float buf_f32[1024];
    ggml_fp16_t buf_f16[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f32[i] = (float)(i + 1);
        buf_f16[i] = fromFloat32<ggml_fp16_t>(buf_f32[i]);
    }

    // avg pool 1d - Float 32
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_1d(&ctx, t, GGML_OP_POOL_AVG, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f32, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);

        GGML_ASSERT(output[0] == 2);
        GGML_ASSERT(output[1] == 5);
        GGML_ASSERT(output[2] == 8);
        GGML_ASSERT(output[3] == 12);
        GGML_ASSERT(output[4] == 15);
        GGML_ASSERT(output[5] == 18);

    }

    // avg pool 1d - Float 16
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F16, { 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_1d(&ctx, t, GGML_OP_POOL_AVG, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f16, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);

        GGML_ASSERT(output[0] == 2);
        GGML_ASSERT(output[1] == 5);
        GGML_ASSERT(output[2] == 8);
        GGML_ASSERT(output[3] == 12);
        GGML_ASSERT(output[4] == 15);
        GGML_ASSERT(output[5] == 18);

    }

    // max pool 1d - Float 32
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_1d(&ctx, t, GGML_OP_POOL_MAX, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f32, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 3);
        GGML_ASSERT(output[1] == 6);
        GGML_ASSERT(output[2] == 9);
        GGML_ASSERT(output[3] == 13);
        GGML_ASSERT(output[4] == 16);
        GGML_ASSERT(output[5] == 19);

    }

    // max pool 1d - Float 16
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F16, { 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_1d(&ctx, t, GGML_OP_POOL_MAX, 3, 3, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f16, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 3);
        GGML_ASSERT(output[1] == 6);
        GGML_ASSERT(output[2] == 9);
        GGML_ASSERT(output[3] == 13);
        GGML_ASSERT(output[4] == 16);
        GGML_ASSERT(output[5] == 19);

    }

    // avg pool 2d - Float 32
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 10, 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_2d(&ctx, t, GGML_OP_POOL_AVG, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f32, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 17);
        GGML_ASSERT(output[1] == 20);
        GGML_ASSERT(output[2] == 23);
        GGML_ASSERT(output[3] == 57);
        GGML_ASSERT(output[4] == 60);
        GGML_ASSERT(output[5] == 63);
        GGML_ASSERT(output[6] == 117);
        GGML_ASSERT(output[7] == 120);
        GGML_ASSERT(output[8] == 123);
        GGML_ASSERT(output[9] == 157);
        GGML_ASSERT(output[10] == 160);
        GGML_ASSERT(output[11] == 163);

    }

    // avg pool 2d - Float 16
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F16, { 10, 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_2d(&ctx, t, GGML_OP_POOL_AVG, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f16, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 17);
        GGML_ASSERT(output[1] == 20);
        GGML_ASSERT(output[2] == 23);
        GGML_ASSERT(output[3] == 57);
        GGML_ASSERT(output[4] == 60);
        GGML_ASSERT(output[5] == 63);
        GGML_ASSERT(output[6] == 117);
        GGML_ASSERT(output[7] == 120);
        GGML_ASSERT(output[8] == 123);
        GGML_ASSERT(output[9] == 157);
        GGML_ASSERT(output[10] == 160);
        GGML_ASSERT(output[11] == 163);

    }

    // max pool 2d - Float 32
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 10, 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_2d(&ctx, t, GGML_OP_POOL_MAX, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f32, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 33);
        GGML_ASSERT(output[1] == 36);
        GGML_ASSERT(output[2] == 39);
        GGML_ASSERT(output[3] == 73);
        GGML_ASSERT(output[4] == 76);
        GGML_ASSERT(output[5] == 79);
        GGML_ASSERT(output[6] == 133);
        GGML_ASSERT(output[7] == 136);
        GGML_ASSERT(output[8] == 139);
        GGML_ASSERT(output[9] == 173);
        GGML_ASSERT(output[10] == 176);
        GGML_ASSERT(output[11] == 179);

    }

    // max pool 2d - Float 16
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F16, { 10, 10, 2 });

        ggml_tensor* t_pooled = ggml_pool_2d(&ctx, t, GGML_OP_POOL_MAX, 3, 4, 3, 4, 0, 0);
        GGML_ASSERT(t_pooled->ne[0] == 3);
        GGML_ASSERT(t_pooled->ne[1] == 2);
        GGML_ASSERT(t_pooled->ne[2] == 2);
        GGML_ASSERT(t_pooled->ne[3] == 1);

        ggml_cgraph graph;
        graph.build_forward_expand(t_pooled);

        run_graph_in_cpu(&ctx, graph, [&](ggml_tensor* tensor) {
            if (tensor == t) {
                ggml_backend_tensor_set(t, buf_f16, 0, t->nbytes());
            }
        });

        const float* output = ggml_get_data_f32(t_pooled);
        GGML_ASSERT(output[0] == 33);
        GGML_ASSERT(output[1] == 36);
        GGML_ASSERT(output[2] == 39);
        GGML_ASSERT(output[3] == 73);
        GGML_ASSERT(output[4] == 76);
        GGML_ASSERT(output[5] == 79);
        GGML_ASSERT(output[6] == 133);
        GGML_ASSERT(output[7] == 136);
        GGML_ASSERT(output[8] == 139);
        GGML_ASSERT(output[9] == 173);
        GGML_ASSERT(output[10] == 176);
        GGML_ASSERT(output[11] == 179);

    }

    return 0;
}

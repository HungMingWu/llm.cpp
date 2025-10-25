#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <bit>
#include <memory>
#include <unordered_map>
#include <vector>

import ggml;
import test;

void printf_tensor(ggml_tensor* t) {

    if (t->type == GGML_TYPE_F32) {
        const float* t_d = ggml_get_data_f32(t);
        for (int i = 0; i < t->ne[2]; ++i) {
            for (int j = 0; j < t->ne[1]; ++j) {
                for (int k = 0; k < t->ne[0]; ++k) {
                    printf("%.1f ", t_d[i * t->ne[1] * t->ne[0] + j * t->ne[0] + k]);
                }
                printf("\n");
            }
            printf("---\n");
        }
    }
    else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* t_d = (ggml_fp16_t *)ggml_get_data(t);
        for (int i = 0; i < t->ne[2]; ++i) {
            for (int j = 0; j < t->ne[1]; ++j) {
                for (int k = 0; k < t->ne[0]; ++k) {
                    printf("%.1f ", toFloat32(t_d[i * t->ne[1] * t->ne[0] + j * t->ne[0] + k]));
                }
                printf("\n");
            }
            printf("---\n");
        }
    }
    else {
        printf("unknown type\n");
    }
}

void check_tensor(float* t, float* expected_t_d, int ne0, int ne1, int ne2) {
    for (int i2 = 0; i2 < ne2; ++i2) {
        for (int i1 = 0; i1 < ne1; ++i1) {
            for (int i0 = 0; i0 < ne0; ++i0) {
                float expected = *(expected_t_d + i2 * ne1 * ne0 + i1 * ne0 + i0);
                float actual = t[i2 * ne1 * ne0 + i1 * ne0 + i0];
                if (expected != actual) {
                    printf("expected %.1f, got %.1f\n", expected, actual);
                }
                assert(expected == actual);
            }
        }
    }
}

void test_conv_transpose_1d(void) {

    float buf_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f32[i] = (float)i;
    }

    ggml_fp16_t buf_f16[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f16[i] = fromFloat32<ggml_fp16_t>((float)i);
    }

    float expected_out_1[3][4] = {
        {18.0, 45.0, 59.0, 37.0},
        {24.0, 61.0, 83.0, 51.0},
        {30.0, 77.0, 107.0, 65.0},
    };
    float expected_out_2[3][6] = {
        {18.0, 21.0, 24.0, 29.0, 30.0, 37.0},
        {24.0, 27.0, 34.0, 39.0, 44.0, 51.0},
        {30.0, 33.0, 44.0, 49.0, 58.0, 65.0},
    };
    float expected_out_3[3][8] = {
        {18.0, 21.0, 0.0, 24.0, 29.0, 0.0, 30.0, 37.0},
        {24.0, 27.0, 0.0, 34.0, 39.0, 0.0, 44.0, 51.0},
        {30.0, 33.0, 0.0, 44.0, 49.0, 0.0, 58.0, 65.0},
    };

    // conv transpose 1d with stride 1, 2 & 3
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 3, 2 }); // l x cin
        ggml_tensor* k = ctx.create(GGML_TYPE_F16, { 2, 3, 2 }); // k x cout x cin
        ggml_tensor* out_1 = ggml_conv_transpose_1d(&ctx, k, t, 1 /* s0 */, 0 /* p0 */, 1 /* d0 */);
        ggml_tensor* out_2 = ggml_conv_transpose_1d(&ctx, k, t, 2 /* s0 */, 0 /* p0 */, 1 /* d0 */);
        ggml_tensor* out_3 = ggml_conv_transpose_1d(&ctx, k, t, 3 /* s0 */, 0 /* p0 */, 1 /* d0 */);
        ggml_cgraph gf_1, gf_2, gf_3;
        std::unordered_map<ggml_tensor*, bool> visited;
        auto fillTensor = [&](ggml_tensor* tensor) {
            if (visited[tensor]) return;
            if (tensor == t) {
                ggml_backend_tensor_set(tensor, buf_f32, 0, tensor->nbytes());
            }
            else if (tensor == k) {
                ggml_backend_tensor_set(tensor, buf_f16, 0, tensor->nbytes());
            }
            visited[tensor] = true;
        };
        gf_1.build_forward_expand(out_1);
        gf_2.build_forward_expand(out_2);
        gf_3.build_forward_expand(out_3);
        auto result_1 = run_graph_in_cpu(&ctx, gf_1, fillTensor);
        auto result_2 = run_graph_in_cpu(&ctx, gf_2, fillTensor);
        auto result_3 = run_graph_in_cpu(&ctx, gf_3, fillTensor);

        check_tensor(result_1.data(), (float*)expected_out_1, 4, 3, 1);
        check_tensor(result_2.data(), (float*)expected_out_2, 6, 3, 1);
        check_tensor(result_3.data(), (float*)expected_out_3, 8, 3, 1);
    }
}

void test_conv_transpose_2d(void) {
    float buf_f32[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f32[i] = (float)i;
    }

    ggml_fp16_t buf_f16[1024];
    for (int i = 0; i < 1024; ++i) {
        buf_f16[i] = fromFloat32<ggml_fp16_t>((float)i);
    }

    float expected_out_1[3][3][4] = {
        {
            {72.0, 162.0, 188.0, 106.0},
            {192.0, 430.0, 490.0, 274.0},
            {132.0, 292.0, 326.0, 180.0},
        },
        {
            {96.0, 218.0, 260.0, 146.0},
            {264.0, 590.0, 682.0, 378.0},
            {180.0, 396.0, 446.0, 244.0},
        },
        {
            {120.0, 274.0, 332.0, 186.0},
            {336.0, 750.0, 874.0, 482.0},
            {228.0, 500.0, 566.0, 308.0},
        },
    };

    float expected_out_2[3][4][6] = {
        {
            {72.0, 78.0, 84.0, 92.0, 96.0, 106.0},
            {84.0, 90.0, 100.0, 108.0, 116.0, 126.0},
            {108.0, 120.0, 120.0, 134.0, 132.0, 148.0},
            {132.0, 144.0, 148.0, 162.0, 164.0, 180.0},
        },
        {
            {96.0, 102.0, 116.0, 124.0, 136.0, 146.0},
            {108.0, 114.0, 132.0, 140.0, 156.0, 166.0},
            {156.0, 168.0, 176.0, 190.0, 196.0, 212.0},
            {180.0, 192.0, 204.0, 218.0, 228.0, 244.0},
        },
        {
            {120.0, 126.0, 148.0, 156.0, 176.0, 186.0},
            {132.0, 138.0, 164.0, 172.0, 196.0, 206.0},
            {204.0, 216.0, 232.0, 246.0, 260.0, 276.0},
            {228.0, 240.0, 260.0, 274.0, 292.0, 308.0},
        },
    };

    float expected_out_3[3][5][8] = {
        {
            {72.0, 78.0, 0.0, 84.0, 92.0, 0.0, 96.0, 106.0},
            {84.0, 90.0, 0.0, 100.0, 108.0, 0.0, 116.0, 126.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {108.0, 120.0, 0.0, 120.0, 134.0, 0.0, 132.0, 148.0},
            {132.0, 144.0, 0.0, 148.0, 162.0, 0.0, 164.0, 180.0},
        },
        {
            {96.0, 102.0, 0.0, 116.0, 124.0, 0.0, 136.0, 146.0},
            {108.0, 114.0, 0.0, 132.0, 140.0, 0.0, 156.0, 166.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {156.0, 168.0, 0.0, 176.0, 190.0, 0.0, 196.0, 212.0},
            {180.0, 192.0, 0.0, 204.0, 218.0, 0.0, 228.0, 244.0},
        },
        {
            {120.0, 126.0, 0.0, 148.0, 156.0, 0.0, 176.0, 186.0},
            {132.0, 138.0, 0.0, 164.0, 172.0, 0.0, 196.0, 206.0},
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {204.0, 216.0, 0.0, 232.0, 246.0, 0.0, 260.0, 276.0},
            {228.0, 240.0, 0.0, 260.0, 274.0, 0.0, 292.0, 308.0},
        },
    };

    // conv transpose 2d with stride 1, 2 & 3
    {
        ggml_context ctx;
        ggml_tensor* t = ctx.create(GGML_TYPE_F32, { 3, 2, 2, 1 }); // w x h x cin
        ggml_tensor* k = ctx.create(GGML_TYPE_F16, { 2, 2, 3, 2 }); // w x h cin x cout
        ggml_tensor* out_1 = ggml_conv_transpose_2d(&ctx, k, t, 1);
        ggml_tensor* out_2 = ggml_conv_transpose_2d(&ctx, k, t, 2);
        ggml_tensor* out_3 = ggml_conv_transpose_2d(&ctx, k, t, 3);
        ggml_cgraph gf_1;
        ggml_cgraph gf_2;
        ggml_cgraph gf_3;

        gf_1.build_forward_expand(out_1);
        gf_2.build_forward_expand(out_2);
        gf_3.build_forward_expand(out_3);
        std::unordered_map<ggml_tensor*, bool> visited;
        auto fillTensor = [&](ggml_tensor* tensor) {
            if (visited[tensor]) return;
            if (tensor == t) {
                ggml_backend_tensor_set(tensor, buf_f32, 0, tensor->nbytes());
            }
            else if (tensor == k) {
                ggml_backend_tensor_set(tensor, buf_f16, 0, tensor->nbytes());
            }
            visited[tensor] = true;
        };
        auto result_1 = run_graph_in_cpu(&ctx, gf_1, fillTensor);
        auto result_2 = run_graph_in_cpu(&ctx, gf_2, fillTensor);
        auto result_3 = run_graph_in_cpu(&ctx, gf_3, fillTensor);

        // printf("in\n");
        // printf_tensor(t);
        // printf("\n\nkernel\n");
        // printf_tensor(k);
        // printf("\n\nout\n");
        // printf_tensor(out);
        // printf("\n\nout_2\n");
        // printf_tensor(out_2);
        // printf("\n\nout_3\n");
        // printf_tensor(out_3);

        check_tensor(result_1.data(), (float*)expected_out_1, 4, 3, 3);
        check_tensor(result_2.data(), (float*)expected_out_2, 6, 4, 3);
        check_tensor(result_3.data(), (float*)expected_out_3, 8, 5, 3);
    }

}

int main(int argc, const char* argv[]) {
    test_conv_transpose_1d();
    test_conv_transpose_2d();
    return 0;
}

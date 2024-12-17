#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

import ggml;

struct test_model {
    ggml_tensor* a;
    ggml_tensor* b;
    std::unique_ptr<ggml_backend> backend;
    std::unique_ptr<ggml_backend_buffer> buffer;
    ggml_context ctx;
};

void load_model(test_model& model, float* a, float* b, int M, int N, int K, bool use_gpu = false) {
    size_t buffer_size = 0;
    {
        buffer_size += (M * N) * ggml_type_size(GGML_TYPE_F32); // tensor a
        buffer_size += (N * K) * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
    printf("%s: backend buffer size = %d bytes\n", __func__, (int)buffer_size);

    int num_tensors = 2;

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (use_gpu) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        model.backend = ggml_backend_cpu_init();
    }

    model.buffer = model.backend->get_default_buffer_type()->alloc_buffer(buffer_size);

    // create tensors
    model.a = model.ctx.create(GGML_TYPE_F32, { K, M });
    printf("Matrix A: [%i, %i]\n", K, M);
    model.b = model.ctx.create(GGML_TYPE_F32, { K, N });
    printf("Matrix B: [%i, %i]\n", K, N);

    // create a allocator
    ggml_tallocr alloc(model.buffer.get());

    // alloc memory
    alloc.alloc(model.a);

    // load data to buffer
    ggml_backend_tensor_set(model.a, a, 0, model.a->nbytes()); // cuda requires copy the data directly to device

    // alloc memory
    alloc.alloc(model.b);

    ggml_backend_tensor_set(model.b, b, 0, model.b->nbytes());  // cuda requires copy the data directly to device
}

ggml_cgraph build_graph(test_model& model) {
    ggml_cgraph gf;

    auto ctx = &model.ctx;

    // zT = x @ yT
    ggml_tensor* result = ggml_mul_mat(ctx, model.a, ggml_cont(ctx, model.b));

    // z = (zT)T
    gf.build_forward_expand(ggml_cont(ctx, ggml_transpose(ctx, result)));

    return gf;
}

ggml_tensor* compute(ggml_cgraph& gf, const test_model& model, ggml_gallocr_t allocr) {
    // allocate tensors
    allocr->alloc_graph(&gf);

#if 0
    int n_threads = 1;

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#endif

    model.backend->graph_compute(&gf);

    //ggml_graph_print(gf);

    // in this case, the output tensor is the last one in the graph
    return gf.getNodes().back();
}


static void ggml_vec_dot_f16(const int n, float* s, float* x, float* y) {
    float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    *s = sumf;
}

static void gemm_f16_out_f32(int m, int n, int k,
    float* A,
    float* B,
    float* C,
    const int ith, const int nth) {
    // does not seem to make a difference
    int m0, m1, n0, n1;
    // patches per thread
    if (m > n) {
        n0 = 0;
        n1 = n;

        // total patches in dst
        const int np = m;

        // patches per thread
        const int dp = (np + nth - 1) / nth;

        // patch range for this thread
        m0 = dp * ith;
        m1 = std::min(m0 + dp, np);
    }
    else {
        m0 = 0;
        m1 = m;

        // total patches in dst
        const int np = n;

        // patches per thread
        const int dp = (np + nth - 1) / nth;

        // patch range for this thread
        n0 = dp * ith;
        n1 = std::min(n0 + dp, np);
    }

    // block-tiling attempt
    int64_t blck_n = 16;
    int64_t blck_m = 16;

    for (int j = n0; j < n1; j += blck_n) {
        for (int i = m0; i < m1; i += blck_m) {
            // printf("i j k => %d %d %d\n", i, j, K);
            for (int ii = i; ii < i + blck_m && ii < m1; ii++) {
                for (int jj = j; jj < j + blck_n && jj < n1; jj++) {
                    ggml_vec_dot_f16(k,
                        C + ii * n + jj,
                        A + ii * k,
                        B + jj * k);
                }
            }
        }
    }
}


void perform_gemm_test(float* a, float* b, float* expected, int M, int N, int K) {
    printf("\nPerforming gemm_f16_out_f32 test:\n");

    std::vector<float> gemm_out(M * N);
    gemm_f16_out_f32(M, N, K, a, b, gemm_out.data(), 0, 1);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1ff,", gemm_out[i * N + j]);
        }
        printf("\n");
    }

    bool passed = true;

    for (int i = 0; i < M * N; i++) {
        if (gemm_out[i] != expected[i]) {
            passed = false;
            break;
        }
    }

    printf("gemm_mult (%i): %s\n", (M * N), passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
}

int main(void)
{
    //ggml_time_init();
    const int M = 4, N = 16, K = 36;  // a conv2d expected matrix multiplication

    // matrix A (4 X 36)
    float matrixA[M * K] = {
       2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
        10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
        7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
    };

    // matrix B (16 X 36)
    float matrixB[N * K] = {
        9.0f, 7.0f, 1.0f, 3.0f, 5.0f, 9.0f, 7.0f, 6.0f, 1.0f, 10.0f, 1.0f, 1.0f, 7.0f, 2.0f, 4.0f, 9.0f, 10.0f, 4.0f, 5.0f, 5.0f, 7.0f, 1.0f, 7.0f, 7.0f, 2.0f, 9.0f, 5.0f, 10.0f, 7.0f, 4.0f, 8.0f, 9.0f, 9.0f, 3.0f, 10.0f, 2.0f,
        4.0f, 6.0f, 10.0f, 9.0f, 5.0f, 1.0f, 8.0f, 7.0f, 4.0f, 7.0f, 2.0f, 6.0f, 5.0f, 3.0f, 1.0f, 10.0f, 8.0f, 4.0f, 8.0f, 3.0f, 7.0f, 1.0f, 2.0f, 7.0f, 6.0f, 8.0f, 6.0f, 5.0f, 2.0f, 3.0f, 1.0f, 1.0f, 2.0f, 5.0f, 7.0f, 1.0f,
        8.0f, 2.0f, 8.0f, 8.0f, 8.0f, 8.0f, 4.0f, 4.0f, 6.0f, 10.0f, 10.0f, 9.0f, 2.0f, 9.0f, 3.0f, 7.0f, 7.0f, 1.0f, 4.0f, 9.0f, 1.0f, 2.0f, 3.0f, 6.0f, 1.0f, 10.0f, 5.0f, 8.0f, 9.0f, 4.0f, 6.0f, 2.0f, 3.0f, 1.0f, 2.0f, 7.0f,
        5.0f, 1.0f, 7.0f, 2.0f, 9.0f, 10.0f, 9.0f, 5.0f, 2.0f, 5.0f, 4.0f, 10.0f, 9.0f, 9.0f, 1.0f, 9.0f, 8.0f, 8.0f, 9.0f, 4.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 8.0f, 4.0f, 5.0f, 10.0f, 7.0f, 6.0f, 2.0f, 1.0f, 10.0f, 10.0f, 7.0f,
        9.0f, 4.0f, 5.0f, 9.0f, 5.0f, 10.0f, 10.0f, 3.0f, 6.0f, 6.0f, 4.0f, 4.0f, 4.0f, 8.0f, 5.0f, 4.0f, 9.0f, 1.0f, 9.0f, 9.0f, 1.0f, 7.0f, 9.0f, 2.0f, 10.0f, 9.0f, 10.0f, 8.0f, 3.0f, 3.0f, 9.0f, 3.0f, 9.0f, 10.0f, 1.0f, 8.0f,
        9.0f, 2.0f, 6.0f, 9.0f, 7.0f, 2.0f, 3.0f, 5.0f, 3.0f, 6.0f, 9.0f, 7.0f, 3.0f, 7.0f, 6.0f, 4.0f, 10.0f, 3.0f, 5.0f, 7.0f, 2.0f, 9.0f, 3.0f, 2.0f, 2.0f, 10.0f, 8.0f, 7.0f, 3.0f, 10.0f, 6.0f, 3.0f, 1.0f, 1.0f, 4.0f, 10.0f,
        2.0f, 9.0f, 2.0f, 10.0f, 6.0f, 4.0f, 3.0f, 6.0f, 3.0f, 6.0f, 9.0f, 7.0f, 8.0f, 8.0f, 3.0f, 3.0f, 10.0f, 5.0f, 2.0f, 10.0f, 7.0f, 10.0f, 9.0f, 3.0f, 6.0f, 6.0f, 5.0f, 10.0f, 2.0f, 3.0f, 6.0f, 1.0f, 9.0f, 4.0f, 10.0f, 4.0f,
        10.0f, 7.0f, 8.0f, 10.0f, 10.0f, 8.0f, 7.0f, 10.0f, 4.0f, 6.0f, 8.0f, 7.0f, 7.0f, 6.0f, 9.0f, 3.0f, 6.0f, 5.0f, 5.0f, 2.0f, 7.0f, 2.0f, 7.0f, 4.0f, 4.0f, 6.0f, 6.0f, 4.0f, 3.0f, 9.0f, 3.0f, 6.0f, 4.0f, 7.0f, 2.0f, 9.0f,
        7.0f, 3.0f, 2.0f, 5.0f, 7.0f, 3.0f, 10.0f, 2.0f, 6.0f, 1.0f, 4.0f, 7.0f, 5.0f, 10.0f, 3.0f, 10.0f, 4.0f, 5.0f, 5.0f, 1.0f, 6.0f, 10.0f, 7.0f, 4.0f, 5.0f, 3.0f, 9.0f, 9.0f, 8.0f, 6.0f, 9.0f, 2.0f, 3.0f, 6.0f, 8.0f, 5.0f,
        5.0f, 5.0f, 5.0f, 5.0f, 3.0f, 10.0f, 4.0f, 1.0f, 8.0f, 8.0f, 9.0f, 8.0f, 4.0f, 1.0f, 4.0f, 9.0f, 3.0f, 6.0f, 3.0f, 1.0f, 4.0f, 8.0f, 3.0f, 10.0f, 8.0f, 6.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 4.0f, 3.0f, 6.0f, 4.0f,
        6.0f, 2.0f, 3.0f, 3.0f, 3.0f, 7.0f, 5.0f, 1.0f, 8.0f, 1.0f, 4.0f, 5.0f, 1.0f, 1.0f, 6.0f, 4.0f, 2.0f, 1.0f, 7.0f, 8.0f, 6.0f, 1.0f, 1.0f, 5.0f, 6.0f, 5.0f, 10.0f, 6.0f, 7.0f, 5.0f, 9.0f, 3.0f, 2.0f, 7.0f, 9.0f, 4.0f,
        2.0f, 5.0f, 9.0f, 5.0f, 10.0f, 3.0f, 1.0f, 8.0f, 1.0f, 7.0f, 1.0f, 8.0f, 1.0f, 6.0f, 7.0f, 8.0f, 4.0f, 9.0f, 5.0f, 10.0f, 3.0f, 7.0f, 6.0f, 8.0f, 8.0f, 5.0f, 6.0f, 8.0f, 10.0f, 9.0f, 4.0f, 1.0f, 3.0f, 3.0f, 4.0f, 7.0f,
        8.0f, 2.0f, 6.0f, 6.0f, 5.0f, 1.0f, 3.0f, 7.0f, 1.0f, 7.0f, 2.0f, 2.0f, 2.0f, 8.0f, 4.0f, 1.0f, 1.0f, 5.0f, 9.0f, 4.0f, 1.0f, 2.0f, 3.0f, 10.0f, 1.0f, 4.0f, 9.0f, 9.0f, 6.0f, 8.0f, 8.0f, 1.0f, 9.0f, 10.0f, 4.0f, 1.0f,
        8.0f, 5.0f, 8.0f, 9.0f, 4.0f, 8.0f, 2.0f, 1.0f, 1.0f, 9.0f, 4.0f, 5.0f, 6.0f, 1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 3.0f, 1.0f, 4.0f, 6.0f, 7.0f, 7.0f, 7.0f, 8.0f, 7.0f, 8.0f, 8.0f, 2.0f, 10.0f, 2.0f, 7.0f, 3.0f, 8.0f, 3.0f,
        8.0f, 7.0f, 6.0f, 2.0f, 4.0f, 10.0f, 10.0f, 6.0f, 10.0f, 3.0f, 7.0f, 6.0f, 4.0f, 3.0f, 5.0f, 5.0f, 5.0f, 3.0f, 8.0f, 10.0f, 3.0f, 4.0f, 8.0f, 4.0f, 2.0f, 6.0f, 8.0f, 9.0f, 6.0f, 9.0f, 4.0f, 3.0f, 5.0f, 2.0f, 2.0f, 6.0f,
        10.0f, 6.0f, 2.0f, 1.0f, 7.0f, 5.0f, 6.0f, 4.0f, 1.0f, 9.0f, 10.0f, 2.0f, 4.0f, 5.0f, 8.0f, 5.0f, 7.0f, 4.0f, 7.0f, 6.0f, 3.0f, 9.0f, 2.0f, 1.0f, 4.0f, 2.0f, 6.0f, 6.0f, 3.0f, 3.0f, 2.0f, 8.0f, 5.0f, 9.0f, 3.0f, 4.0f,
    };

    // matrix C (4 x 16)
    float expected_result[M * N] = {
        1224.0f, 1023.0f, 1158.0f,1259.0f,1359.0f,1194.0f,1535.0f,1247.0f,1185.0f,1029.0f,889.0f,1182.0f,955.0f,1179.0f,1147.0f,1048.0f,
        1216.0f, 1087.0f, 1239.0f,1361.0f,1392.0f,1260.0f,1247.0f,1563.0f,1167.0f,1052.0f,942.0f,1214.0f,1045.0f,1134.0f,1264.0f,1126.0f,
        1125.0f, 966.0f, 1079.0f,1333.0f,1287.0f,1101.0f,1185.0f,1167.0f,1368.0f,990.0f,967.0f,1121.0f,971.0f,1086.0f,1130.0f,980.0f,
        999.0f, 902.0f, 1020.0f,1056.0f,1076.0f,929.0f,1029.0f,1052.0f,990.0f,1108.0f,823.0f,989.0f,759.0f,1041.0f,1003.0f,870.0f
    };

    bool passed = true;

    perform_gemm_test(matrixA, matrixB, expected_result, M, N, K);

    test_model model;
    load_model(model, matrixA, matrixB, M, N, K, true);

    //create the worst case graph for memory usage estimation
    ggml_cgraph gf = build_graph(model);

    ggml_gallocr allocr = [&] {
        ggml_gallocr allocr(model.backend->get_default_buffer_type());
        // compute the required memory
        allocr.reserve(&gf);
        size_t mem_size = allocr.get_buffer_size(0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size / 1024.0f / 1024.0f);
        return allocr;
    }();

    ggml_tensor* result = compute(gf, model, &allocr);

    std::vector<float> out_data(result->nelements());

    ggml_backend_tensor_get(result, out_data.data(), 0, result->nbytes());

    printf("\nPerforming ggml_mul_mat test:\n");

    passed = true;
    for (int i = 0; i < M * N; i++) {
        if (out_data[i] != expected_result[i]) {
            passed = false;
            break;
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.1f ", out_data[i * N + j]);
        }
        printf("\n");
    }

    printf("ggml_mul_mat (%d): %s\n", (int)result->nelements(), passed && (result->nelements() == M * N) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return 0;
}

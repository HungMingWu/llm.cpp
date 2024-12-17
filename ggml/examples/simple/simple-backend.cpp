#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <print>
#include <string>
#include <vector>

import ggml;

static void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

// This is a simple model with two tensors a and b
struct simple_model {
    ggml_tensor* a;
    ggml_tensor* b;

    // the backend to perform the computation (CPU, CUDA, METAL)
    std::unique_ptr<ggml_backend> backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    std::unique_ptr<ggml_backend_buffer> buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    ggml_context ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model& model, float* a, float* b, int rows_A, int cols_A, int rows_B, int cols_B) {
    ggml_log_set(ggml_log_callback_default);
    // initialize the backend
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    model.backend = ggml_backend_metal_init();
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 2;

    // create tensors
    model.a = model.ctx.create(GGML_TYPE_F32, { cols_A, rows_A });
    model.b = model.ctx.create(GGML_TYPE_F32, { cols_B, rows_B });

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(&model.ctx, model.backend.get());

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, a, 0, model.a->nbytes());
    ggml_backend_tensor_set(model.b, b, 0, model.b->nbytes());
}

// build the compute graph to perform a matrix multiplication
ggml_cgraph build_graph(simple_model& model) {
    ggml_cgraph gf{};

    // result = a*b^T
    ggml_tensor* result = ggml_mul_mat(&model.ctx, model.a, model.b);

    // build operations nodes
    gf.build_forward_expand(result);

    return gf;
}

// compute with backend
ggml_tensor* compute(ggml_cgraph& gf, const simple_model& model, ggml_gallocr_t allocr) {
    // allocate tensors
    allocr->alloc_graph(&gf);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

#if 0
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }
#endif
    model.backend->compute(&gf);

    // in this case, the output tensor is the last one in the graph
    return gf.nodes.back();
}

int main(void) {
    //ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    const int rows_B = 3, cols_B = 2;
    /* Transpose([
        10, 9, 5,
        5, 9, 4
    ]) 2 rows, 3 cols */
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    simple_model model;
    load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

    // create the worst case graph for memory usage estimation
    ggml_cgraph gf = build_graph(model);

    // calculate the temporaly memory required to compute
    ggml_gallocr allocr = [&] {
        ggml_gallocr allocr(model.backend->get_default_buffer_type());

        allocr.reserve(&gf);
        size_t mem_size = allocr.get_buffer_size(0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);
        return allocr;
    }();

    // perform computation
    struct ggml_tensor* result = compute(gf, model, &allocr);

    // create a array to print result
    std::vector<float> out_data(result->nelements());

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, result->nbytes());

    // expected result:
    // [ 60.00 110.00 54.00 29.00
    //  55.00 90.00 126.00 28.00
    //  50.00 54.00 42.00 64.00 ]

    printf("mul mat (%d x %d) (transposed result):\n[", (int)result->ne[0], (int)result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[i * result->ne[1] + j]);
        }
    }
    printf(" ]\n");

    return 0;
}

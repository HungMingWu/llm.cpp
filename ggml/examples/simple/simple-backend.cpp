#include <array>
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
    ggml_tensor* a = nullptr;
    ggml_tensor* b = nullptr;

    // the backend to perform the computation (CPU, CUDA, METAL)
    std::unique_ptr<ggml_backend> backend;
    std::unique_ptr<ggml_backend> cpu_backend;
    std::array<ggml_backend*, 2> backend_view;
    std::array< ggml_backend_buffer_type*, 2> buffer_type_view;
    std::unique_ptr<ggml_backend_sched> sched;

    // the backend buffer to storage the tensors data of a and b
    std::unique_ptr<ggml_backend_buffer> buffer;
};

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

// initialize the tensors of the model in this case two matrices 2x2
void init_model(simple_model& model) {
    ggml_log_set(ggml_log_callback_default);

    ggml_backend_load_all();
    model.backend = ggml_backend_init_best();
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    model.backend_view[0] = model.backend.get();
    model.backend_view[1] = model.cpu_backend.get();
	model.buffer_type_view[0] = model.backend_view[0]->get_default_buffer_type();
    model.buffer_type_view[1] = model.backend_view[1]->get_default_buffer_type();
    model.sched = std::make_unique<ggml_backend_sched>(model.backend_view, model.buffer_type_view, false, true);
}

// build the compute graph to perform a matrix multiplication
ggml_cgraph build_graph(ggml_context &ctx, simple_model& model) {
    ggml_cgraph gf;

    // create tensors
    model.a = ctx.create(GGML_TYPE_F32, { cols_A, rows_A });
    model.b = ctx.create(GGML_TYPE_F32, { cols_B, rows_B });

    // result = a*b^T
    ggml_tensor* result = ggml_mul_mat(&ctx, model.a, model.b);

    // build operations nodes
    gf.build_forward_expand(result);

    return gf;
}

// compute with backend
ggml_tensor* compute(simple_model& model, ggml_cgraph &gf) {
    model.sched->reset();
    model.sched->alloc_graph(gf);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, matrix_A, 0, model.a->nbytes());
    ggml_backend_tensor_set(model.b, matrix_B, 0, model.b->nbytes());

    // compute the graph
    model.sched->graph_compute(gf);

    // in this case, the output tensor is the last one in the graph
    return gf.getNodes().back();
}

int main() {
    simple_model model;
    init_model(model);

    ggml_context ctx;
    ggml_cgraph gf = build_graph(ctx, model);

    // perform computation
    ggml_tensor* result = compute(model, gf);

    // create a array to print result
    std::vector<float> out_data(result->nelements());

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, result->nbytes());

    // expected result:
    // [ 60.00 55.00 50.00 110.00
    //  90.00 54.00 54.00 126.00
    //  42.00 29.00 28.00 64.00 ]

    printf("mul mat (%d x %d) (transposed result):\n[", (int)result->ne[0], (int)result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");
    return 0;
}

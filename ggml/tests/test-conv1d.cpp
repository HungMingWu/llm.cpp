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

struct test_model {
    ggml_tensor* a;
    ggml_tensor* b;
    std::unique_ptr<ggml_backend> backend;
    std::unique_ptr<ggml_backend_buffer> buffer;
    ggml_context ctx;
};

void load_model(test_model& model, bool use_gpu = false) {
    // create data
    int K = 3, IC = 10, OC = 10;
    int IL = 8, N = 1;

    // Initialize adata
    std::vector<float> adata(K * IC * OC);
    for (int i = 0; i < K * IC * OC; i++) {
        adata[i] = 4.5f;
    }

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> hadata(K * IC * OC);
    from_float<ggml_fp16_t>(adata.data(), hadata.data(), K * IC * OC);

    // Initialize bdata
    std::vector<float> bdata(IL * IC * N);
    for (int i = 0; i < IL * IC * N; i++) {
        bdata[i] = 2.5f;
    }

    size_t buffer_size = 0;
    {
        buffer_size += K * IC * OC * ggml_type_size(GGML_TYPE_F16); // tensor a
        buffer_size += IL * IC * N * ggml_type_size(GGML_TYPE_F32); // tensor b
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size / 1024.f / 1024.f));

    ggml_log_set(ggml_log_callback_default);

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
    model.a = model.ctx.create(GGML_TYPE_F16, { K, IC, OC });
    model.b = model.ctx.create(GGML_TYPE_F32, { IL, IC, N });

    // create a allocator
    ggml_tallocr alloc(model.buffer.get());

    // alloc memory
    alloc.alloc(model.a);

    // load data to buffer
    ggml_backend_tensor_set(model.a, hadata.data(), 0, model.a->nbytes());

    // alloc memory
    alloc.alloc(model.b);

    ggml_backend_tensor_set(model.b, bdata.data(), 0, model.b->nbytes());
}

ggml_cgraph build_graph(test_model& model) {
    ggml_cgraph gf;

    int s0 = 1;
    int p0 = 1;
    int d0 = 1;

    // split conv1d in fundamental methods for test unit
    ggml_tensor* im2col_0 = ggml_im2col(&model.ctx, model.a, model.b, s0, 0, p0, 0, d0, 0, false, GGML_TYPE_F16);
    im2col_0->set_name("im2col_res");
    gf.build_forward_expand(im2col_0);

    ggml_tensor* conv1d_res = ggml_conv_1d(&model.ctx, model.a, model.b, s0, p0, d0);
    conv1d_res->set_name("conv1d_res");
    gf.build_forward_expand(conv1d_res);

    return gf;
}

void compute_graph(ggml_cgraph& gf, const test_model& model, ggml_gallocr_t allocr) {
    // allocate tensors
    allocr->alloc_graph(&gf);
    int n_threads = 1;

    if (auto cpu_backend = dynamic_cast<ggml_cpu_backend*>(model.backend.get())) {
        cpu_backend->set_n_threads(n_threads);
    }

    model.backend->graph_compute(&gf);

    //ggml_graph_print(gf);
}

int main(void)
{
    //ggml_time_init();

    test_model model;
    load_model(model, true);

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

    compute_graph(gf, model, &allocr);

    ggml_tensor* im2col_res = NULL;
    ggml_tensor* conv1d_res = NULL;

    for (auto &node : gf.getNodes()) {
        if (node->get_name() == "im2col_res") {
            im2col_res = node;
        }
        else if (node->get_name() == "conv1d_res") {
            conv1d_res = node;
        }
    }

    std::vector<uint16_t> im2col_data(im2col_res->nelements());
    std::vector<float> conv2d_data(conv1d_res->nelements());

    ggml_backend_tensor_get(im2col_res, im2col_data.data(), 0, im2col_res->nbytes());
    ggml_backend_tensor_get(conv1d_res, conv2d_data.data(), 0, conv1d_res->nbytes());

    const int n_conv1d_test = 80;
    const int n_im2col_test = 240;

    float expected_conv1d[n_conv1d_test] = {
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f,
        225.00f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 337.50f, 225.00f
    };
    // first im2col test

    uint16_t expected_im2col[n_conv1d_test] = {
        0, 16640, 16640, 0, 16640, 16640, 0, 16640,
        16640, 0, 16640, 16640, 0, 16640, 16640, 0,
        16640, 16640, 0, 16640, 16640, 0, 16640, 16640,
        0, 16640, 16640, 0, 16640, 16640, 16640, 16640,
        16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
        16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
        16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
        16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
        16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640,
        16640, 16640, 16640, 16640, 16640, 16640, 16640, 16640
    };

    printf("\nPerforming test:\n");

    bool passed = true;
    for (int i = 0; i < n_conv1d_test; i++) {
        if (
            im2col_data[i] != expected_im2col[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_im2col (%d): %s\n", (int)im2col_res->nelements(), passed && (im2col_res->nelements() == n_im2col_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    passed = true;
    for (int i = 0; i < n_conv1d_test; i++) {
        if (conv2d_data[i] != expected_conv1d[i]) {
            passed = false;
            break;
        }
    }

    printf("ggml_conv1d (%d): %s\n", (int)conv1d_res->nelements(), passed && (conv1d_res->nelements() == n_conv1d_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");

    return 0;
}

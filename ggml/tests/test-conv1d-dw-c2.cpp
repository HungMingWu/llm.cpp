#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <print>
#include <string_view>
#include <vector>

import ggml;

static void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

struct test_model {
    ggml_tensor* weight;
    ggml_tensor* input;
    std::unique_ptr<ggml_backend> backend;
    std::unique_ptr<ggml_backend_buffer> buffer;
    ggml_context ctx;
};

void load_model(test_model& model, bool use_gpu = false) {
    // create data
    int K = 3, IC = 2, OC = 2;
    int IL = 6, N = 1;

    // Initialize adata
    float weight_data[6] = { 10.0f, 20.0f, 30.0f, 0.1f, 0.2f, 0.3f };

    // Convert adata to fp16 format
    std::vector<ggml_fp16_t> h_weight_data(K * IC);
    for (size_t i = 0; i < K * IC; i++)
        h_weight_data[i] = fromFloat32<ggml_fp16_t>(weight_data[i]);

    // Initialize input data, 2 channels, 6 timesteps, 1 batch
    float input_data[12] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };

    size_t buffer_size = 0;
    {
        buffer_size += K * IC * ggml_type_size(GGML_TYPE_F16); // tensor weight
        buffer_size += IL * IC * N * ggml_type_size(GGML_TYPE_F32); // tensor input
        buffer_size += 1024; // overhead
    }

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int)sizeof(ggml_tensor));
    printf("%s: backend buffer size = %0.2f MB\n", __func__, (buffer_size / 1024.f / 1024.f));

    ggml_log_set(ggml_log_callback_default);

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

    model.buffer = model.backend->alloc_buffer(buffer_size);

    // create tensors
    // A Pytorch grouped Conv1d weight parameter is of shape (out_channels, input_channels/groups, kernel_size)
    model.weight = model.ctx.create(GGML_TYPE_F16, K, 1, IC);
    model.input = model.ctx.create(GGML_TYPE_F32, IL, IC, N);

    // create a allocator
    ggml_tallocr alloc(model.buffer.get());

    // alloc memory
    alloc.alloc(model.weight);

    // load data to buffer
    ggml_backend_tensor_set(model.weight, h_weight_data.data(), 0, model.weight->nbytes());

    // alloc memory
    alloc.alloc(model.input);
    ggml_backend_tensor_set(model.input, input_data, 0, model.input->nbytes());
}

ggml_cgraph build_graph(test_model& model) {
    ggml_cgraph gf;

    int s0 = 3;
    int p0 = 0;
    int d0 = 1;

    ggml_tensor* conv1d_dw_res = ggml_conv_1d_dw(&model.ctx, model.weight, model.input, s0, p0, d0);
    conv1d_dw_res->set_name("conv1d_dw_res");
    gf.build_forward_expand(conv1d_dw_res);

    return gf;
}

void compute_graph(const test_model& model, ggml_gallocr* allocr, ggml_cgraph& gf) {
    // allocate tensors
    allocr->alloc_graph(&gf);

    int n_threads = 1;

    if (auto cpu_backend = dynamic_cast<ggml_cpu_backend*>(model.backend.get())) {
        cpu_backend->set_n_threads(n_threads);
    }

    model.backend->graph_compute(&gf);
}

int main()
{
    test_model model;
    load_model(model, true);

    // create the worst case graph for memory usage estimation
    ggml_cgraph gf = build_graph(model);

    // calculate the temporaly memory required to compute
    ggml_gallocr allocr = [&] {
        ggml_gallocr allocr(model.backend->get_default_buffer_type());

        allocr.reserve(gf);
        size_t mem_size = allocr.get_buffer_size(0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);
        return allocr;
    }();

    compute_graph(model, &allocr, gf);

    ggml_tensor* conv1d_dw_res = NULL;

    for (auto node : gf.getNodes()) {
        if (node->get_name() == "conv1d_dw_res") {
            conv1d_dw_res = node;
        }
    }

    std::vector<float> conv2d_data(conv1d_dw_res->nelements());

    ggml_backend_tensor_get(conv1d_dw_res, conv2d_data.data(), 0, conv1d_dw_res->nbytes());

    const int n_conv1d_dw_test = 4;

    float expected_conv1d_dw[n_conv1d_dw_test] = {
        60.0f, 60.0f, 0.6f, 0.6f
    };

    printf("\nPerforming test:\n");

    bool passed = true;
    passed = true;
    for (int i = 0; i < n_conv1d_dw_test; i++) {
        if (std::abs(conv2d_data[i] - expected_conv1d_dw[i]) > 1e-4) {
            passed = false;
            break;
        }
    }

    std::println("ggml_conv1d ({}): {}", conv1d_dw_res->nelements(), passed && (conv1d_dw_res->nelements() == n_conv1d_dw_test) ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return 0;
}

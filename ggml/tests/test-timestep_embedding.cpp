#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <memory>
#include <print>
#include <span>
#include <vector>
#include <tuple>
#include "mdspan.hpp"

import ggml;

void set_timestep_embedding(
    ggml_tensor* timesteps,
    std::span<float> timesteps_span,
    ggml_tensor* embedding,
	const std::mdspan<float, std::dims<2>>& embedding_mdspan,
    int dim, int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [dim, N]
    int half = dim / 2;
    std::vector<float> freqs(half);
    for (int i = 0; i < half; ++i) {
        freqs[i] = (float)std::exp(-std::log(max_period) * i / half);
    }
    for (int i = 0; i < timesteps->ne[0]; ++i) {
        for (int j = 0; j < half; ++j) {
            float arg = timesteps_span[i] * freqs[j];
            embedding_mdspan[i, j] = std::cos(arg);
            embedding_mdspan[i, j + half] = std::sin(arg);
        }
        if (dim % 2 != 0) {
            embedding_mdspan[i, dim] = 0;
        }
    }
}

static bool equalsf(float v1, float v2) {
    if (fabs(v1 - v2) <= 0.00001) {
        return true;
    }
    return false;
}

std::tuple<ggml_tensor*, std::vector<float>> new_timestep_embedding(ggml_context* ctx,
    ggml_tensor* timesteps,
    std::span<float> timesteps_span,
    int dim,
    int max_period = 10000) {
    // timesteps: [N,]
    // embedding: [dim, N]
    int actual_dim = dim;
    ggml_tensor* embedding = ctx->create(GGML_TYPE_F32, actual_dim, timesteps->ne[0]);
	std::vector<float> embedding_data(embedding->nelements());
    std::mdspan embedding_mdspan(embedding_data.data(),
        embedding->ne[1], embedding->ne[0]);
    set_timestep_embedding(timesteps, timesteps_span, embedding, embedding_mdspan, dim, max_period);
    return { embedding, std::move(embedding_data) };
}

int main(int argc, const char** argv) {
    std::vector<float> ts = { 12, 24 };
    int dim = 15;
    int max_period = 10000;
    std::vector<float> expected_result;
    {
        ggml_context ctx;

        ggml_tensor* timesteps = ctx.create(GGML_TYPE_F32, ts.size());
        auto [embedding, embedding_result] = new_timestep_embedding(&ctx, timesteps, ts, dim, max_period);

        for (auto value : embedding_result) {
            std::print("{:.4f} ", value);
        }
        std::println();
		expected_result = std::move(embedding_result);
    }
    std::println("-----------------------------------");
    {
        [[maybe_unused]] bool use_gpu = true;

        std::unique_ptr<ggml_backend> backend;
        std::unique_ptr<ggml_backend_buffer> params_buffer;

#ifdef GGML_USE_CUDA
        if (use_gpu) {
            std::println(stderr, "{}: using CUDA backend", __func__);
            backend = ggml_backend_cuda_init(0);
            if (!backend) {
                std::println(stderr, "{}: ggml_backend_cuda_init() failed", __func__);
            }
        }
#endif

#ifdef GGML_USE_METAL
        if (use_gpu) {
            std::println(stderr, "{}: using Metal backend", __func__);
            backend = ggml_backend_metal_init();
            if (!backend) {
                std::println(stderr, "{}: ggml_backend_metal_init() failed", __func__);
            }
        }
#endif

        if (!backend) {
            // fallback to CPU backend
            backend = ggml_backend_cpu_init();
        }

        ggml_context ctx;

        ggml_tensor* timesteps = ctx.create(GGML_TYPE_F32, ts.size());

        params_buffer = backend->alloc_tensors(&ctx);

        // load data to buffer
        ggml_backend_tensor_set(timesteps, ts.data(), 0, timesteps->nbytes());

        ggml_tensor* t = ggml_timestep_embedding(&ctx, timesteps, dim, max_period);

        ggml_gallocr galloc(backend->get_default_buffer_type());

        ggml_cgraph graph;
        graph.build_forward_expand(t);

        galloc.alloc_graph(&graph);

        int n_threads = 4;

        if (auto cpu_backend = dynamic_cast<ggml_cpu_backend*>(backend.get())) {
            cpu_backend->set_n_threads(n_threads);
        }

        backend->graph_compute(&graph);

        std::vector<float> output(t->nelements());
        ggml_backend_tensor_get(t, output.data(), 0, t->nbytes());

        assert(t->nelements() == expected_result.size());

        for (int i = 0; i < t->nelements(); i++) {
            std::print("{:.4f} ", output[i]);
            assert(equalsf(output[i], expected_result[i]));
        }
        std::println();
    }

    return 0;
}

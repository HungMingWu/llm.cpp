#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

import ggml;

int main(int /*argc*/, const char** /*argv*/)
{
    [[maybe_unused]] bool use_gpu = true;

    std::unique_ptr<ggml_backend> backend;

#ifdef GGML_USE_CUDA
    if (use_gpu) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        backend = ggml_backend_cuda_init(0);
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (!backend) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        backend = ggml_backend_metal_init();
        if (!backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    const int num_tensors = 2;

    if (!backend) {
        // fallback to CPU backend
        backend = ggml_backend_cpu_init();
    }

    // create context
    ggml_context ctx;
    ggml_tensor* t = ggml_arange(&ctx, 0, 3, 1);

    assert(t->ne[0] == 3);

    ggml_gallocr galloc(backend->get_default_buffer_type());

    ggml_cgraph graph;
    graph.build_forward_expand(t);

    // allocate tensors
    galloc.alloc_graph(&graph);

#if 0
    int n_threads = 4;

    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
#endif

    backend->compute(&graph);

    float* output = new float[t->nelements()];
    ggml_backend_tensor_get(t, output, 0, t->nbytes());

    for (int i = 0; i < t->ne[0]; i++) {
        printf("%.2f ", output[i]);
    }
    printf("\n");

    assert(output[0] == 0);
    assert(output[1] == 1);
    assert(output[2] == 2);

    delete[] output;

    return 0;
}

module;
#include <vector>

export module test;
import ggml;

export {
    template <typename Func>
    std::vector<float> run_graph_in_cpu(ggml_context* ctx, ggml_cgraph& gf, Func func)
    {
        auto backend = ggml_backend_cpu_init();
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend.get());
        for (auto& tensor : ctx->tensors)
            func(tensor);
        ggml_gallocr allocr(backend->get_default_buffer_type());
        allocr.reserve(&gf);
        allocr.alloc_graph(&gf);
        backend->compute(&gf);

        auto& result = gf.nodes.back();
        std::vector<float> out_data(result->nelements());
        // bring the data from the backend memory
        ggml_backend_tensor_get(result, out_data.data(), 0, result->nbytes());
        return out_data;
    }

    template <typename Func>
    void run_graph_in_cpu1(ggml_context* ctx, ggml_cgraph& gf, Func func)
    {
        auto backend = ggml_backend_cpu_init();
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend.get());
        for (auto& tensor : ctx->tensors)
            func(tensor);
        ggml_gallocr allocr(backend->get_default_buffer_type());
        allocr.reserve(&gf);
        allocr.alloc_graph(&gf);
        backend->compute(&gf);
    }

    template <typename Func>
    std::vector<float> run_graph_in_cuda(ggml_context* ctx, ggml_cgraph& gf, Func func)
    {
        auto backend = ggml_backend_cuda_init(0);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend.get());
        for (auto& tensor : ctx->tensors)
            func(tensor);
        ggml_gallocr allocr(backend->get_default_buffer_type());
        allocr.reserve(&gf);
        allocr.alloc_graph(&gf);
        backend->compute(&gf);

        auto& result = gf.nodes.back();
        std::vector<float> out_data(result->nelements());
        // bring the data from the backend memory
        ggml_backend_tensor_get(result, out_data.data(), 0, result->nbytes());
        return out_data;
    }
}
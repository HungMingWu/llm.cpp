module;
#include <vector>

export module test;
import ggml;

export {
    template <typename T = float, typename Func>
    std::vector<T> run_graph_in_cpu(ggml_context* ctx, ggml_cgraph& gf, Func func)
    {
        auto backend = ggml_backend_cpu_init();
        auto buffer = backend->alloc_tensors(ctx);
        for (auto& tensor : ctx->tensors)
            func(tensor);
        ggml_gallocr allocr(backend->get_default_buffer_type());
        allocr.reserve(&gf);
        allocr.alloc_graph(&gf);
        backend->compute(&gf);

        auto& result = gf.nodes.back();
        std::vector<T> out_data(result->nelements());
        // bring the data from the backend memory
        ggml_backend_tensor_get(result, out_data.data(), 0, result->nbytes());
        return out_data;
    }

    template <typename Func, typename Func2>
    void run_graph_in_cpu1(ggml_context* ctx, ggml_cgraph& gf, Func func, Func2 finish)
    {
        auto backend = ggml_backend_cpu_init();
        auto buffer = backend->alloc_tensors(ctx);
        for (auto& tensor : ctx->tensors)
            func(tensor);
        ggml_gallocr allocr(backend->get_default_buffer_type());
        allocr.reserve(&gf);
        allocr.alloc_graph(&gf);
        backend->compute(&gf);
        finish();
    }

    template <typename Func>
    std::vector<float> run_graph_in_cuda(ggml_context* ctx, ggml_cgraph& gf, Func func)
    {
        auto backend = ggml_backend_cuda_init(0);
        auto buffer = backend->alloc_tensors(ctx);
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
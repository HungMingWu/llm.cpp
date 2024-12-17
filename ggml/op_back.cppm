export module ggml:op_back;
import :ds;

export {
    ggml_tensor* ggml_rms_norm_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b,
        float eps);

    ggml_tensor* ggml_silu_back(
        ggml_context* ctx,
        ggml_tensor* a,
        ggml_tensor* b);
}
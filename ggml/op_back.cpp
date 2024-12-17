module ggml;

ggml_tensor* ggml_rms_norm_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b,
    float eps) {
    ggml_tensor* result = ggml_dup_tensor(ctx, a);

    ggml_set_op_params(*result, &eps, sizeof(eps));

    result->op = GGML_OP_RMS_NORM_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}

ggml_tensor* ggml_silu_back(
    ggml_context* ctx,
    ggml_tensor* a,
    ggml_tensor* b) {
    ggml_tensor* result = ggml_dup_tensor(ctx, a);

    result->op = GGML_OP_SILU_BACK;
    result->src.push_back(a);
    result->src.push_back(b);

    return result;
}
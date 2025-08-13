module ggml:cpu.op;
import :ds;

import :cpu.op.diag;
import :cpu.op.norm;
import :cpu.op.unary;
import :cpu.op.win;

void ggml_compute_forward_argmax(ggml_tensor* dst);
void ggml_compute_forward_concat(const ggml_compute_params* params, ggml_tensor* dst);
void ggml_compute_forward_get_rel_pos(ggml_tensor* dst);
void ggml_compute_forward_get_rows_back(ggml_tensor* dst);
void ggml_compute_forward_leaky_relu(ggml_tensor* dst);
void ggml_compute_forward_mean(ggml_tensor* dst);
void ggml_compute_forward_pool_1d(ggml_tensor* dst);
void ggml_compute_forward_pool_2d(ggml_tensor* dst);
void ggml_compute_forward_pool_2d_back(ggml_tensor* dst);
void ggml_compute_forward_repeat(ggml_tensor* dst);
void ggml_compute_forward_repeat_back(ggml_tensor* dst);
void ggml_compute_forward_sum(ggml_tensor* dst);
void ggml_compute_forward_sum_rows(ggml_tensor* dst);
void ggml_compute_forward_upscale(const ggml_compute_params* params, ggml_tensor* dst);

// may parallelism after msvc module fix
void ggml_compute_forward_argsort(ggml_tensor* dst);
void ggml_compute_forward_scale(ggml_tensor* dst);
void ggml_compute_forward_silu_back(ggml_tensor* dst);
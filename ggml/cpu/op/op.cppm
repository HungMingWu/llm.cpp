module ggml:cpu.op;
import :ds;

void ggml_compute_forward_argmax(ggml_tensor* dst);
void ggml_compute_forward_diag(ggml_tensor* dst);
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
void ggml_compute_forward_win_part(ggml_tensor* dst);
void ggml_compute_forward_win_unpart(ggml_tensor* dst);

// may parallelism after msvc module fix
void ggml_compute_forward_argsort(ggml_tensor* dst);
void ggml_compute_forward_concat(ggml_tensor* dst);
void ggml_compute_forward_cos(ggml_tensor* dst);
void ggml_compute_forward_group_norm(ggml_tensor* dst);
void ggml_compute_forward_log(ggml_tensor * dst);
void ggml_compute_forward_norm(ggml_tensor* dst);
void ggml_compute_forward_scale(ggml_tensor* dst);
void ggml_compute_forward_sin(ggml_tensor* dst);
void ggml_compute_forward_silu_back(ggml_tensor* dst);
void ggml_compute_forward_rms_norm(ggml_tensor* dst);
void ggml_compute_forward_rms_norm_back(ggml_tensor* dst);
void ggml_compute_forward_sqr(ggml_tensor* dst);
void ggml_compute_forward_sqrt(ggml_tensor* dst);
void ggml_compute_forward_unary(ggml_tensor* dst);
void ggml_compute_forward_upscale(ggml_tensor* dst);
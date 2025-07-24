module ggml:cpu.op;
import :ds;

import :cpu.op.argsort;
import :cpu.op.diag;
import :cpu.op.leaky_relu;
import :cpu.op.norm;
import :cpu.op.pool;
import :cpu.op.relpos;
import :cpu.op.repeat_back;
import :cpu.op.scale;
import :cpu.op.sum_rows;
import :cpu.op.unary;
import :cpu.op.upscale;
import :cpu.op.win;

void ggml_compute_forward_concat(const ggml_compute_params* params, ggml_tensor* dst);
void ggml_compute_forward_mean(ggml_tensor* dst);
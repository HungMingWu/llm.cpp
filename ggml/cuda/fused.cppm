module;
#include <stdint.h>
#include "op/cuda_func.h"
#include "vendors/cuda.h"

module ggml:cuda.fused;
import :ds;

namespace fused
{
	bool should_mul_mat_vec_f(const ggml_tensor* tensor);
	bool should_mul_mat_vec_q(const ggml_tensor* tensor);
	bool ggml_cuda_topk_moe_fusion(const ggml_cgraph* cgraph, int node_idx, ggml_cuda_topk_moe_args& args);
	void add(cudaStream_t stream, ggml_tensor* dst, int n_fuse);
	void mul(cudaStream_t stream, ggml_tensor* dst, int n_fuse);
	void softcap(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* src);
	void rms_norm(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* mul_tensor);
	void rms_norm_add(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);
	void topk_moe(cudaStream_t stream,
		const ggml_tensor* logits,
		ggml_tensor* weights,
		ggml_tensor* ids,
		const ggml_tensor* clamp,
		const ggml_tensor* scale,
		const ggml_tensor* bias,
		const ggml_cuda_topk_moe_args& args);

	void moe_expert_reduce(cudaStream_t stream,
		const ggml_tensor* experts,
		const ggml_tensor* weights,
		ggml_tensor* dst);
}
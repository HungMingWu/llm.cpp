module;
#include <stdint.h>
#include "vendors/cuda.h"

module ggml:cuda.fused;
import :ds;

namespace fused
{
	bool should_mul_mat_vec_f(const ggml_tensor* tensor);
	bool should_mul_mat_vec_q(const ggml_tensor* tensor);

	std::initializer_list<enum ggml_op> ggml_cuda_topk_moe_ops(bool with_norm, bool delayed_softmax = false);

	bool ggml_cuda_can_fuse(const ggml_cgraph* cgraph, int node_idx, std::initializer_list<ggml_op> ops, std::initializer_list<ggml_unary_op> unary_ops);
	bool ggml_can_fuse(const struct ggml_cgraph* cgraph, int node_idx, std::initializer_list<enum ggml_op> ops);

	void add(cudaStream_t stream, ggml_tensor* dst, int n_fuse);
	void softcap(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* src);
	void rms_norm(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* mul_tensor);
	void rms_norm_add(cudaStream_t stream, ggml_tensor* dst, ggml_tensor* mul_tensor, ggml_tensor* add_tensor);
	void topk_moe(cudaStream_t stream,
		const ggml_tensor* logits,
		ggml_tensor* weights,
		ggml_tensor* ids,
		const bool with_norm,
		const bool delayed_softmax = false,
		ggml_tensor* clamp = nullptr);

	void moe_expert_reduce(cudaStream_t stream,
		const ggml_tensor* experts,
		const ggml_tensor* weights,
		ggml_tensor* dst);
}
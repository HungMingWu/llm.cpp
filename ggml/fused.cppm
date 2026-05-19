module;
#include <stdint.h>

module ggml:fused;
import :ds;

namespace fused
{
	bool should_use_topk_moe(const ggml_tensor* gating_op,
		const ggml_tensor* weights,
		const ggml_tensor* logits,
		const ggml_tensor* ids);

	bool ggml_cuda_can_fuse(const ggml_cgraph* cgraph, int node_idx, std::initializer_list<ggml_op> ops, std::initializer_list<ggml_unary_op> unary_ops);
	bool ggml_can_fuse(const ggml_cgraph* cgraph, int node_idx, const enum ggml_op* ops, int num_ops);
	bool ggml_can_fuse(const ggml_cgraph* cgraph, int node_idx, std::initializer_list<enum ggml_op> ops);
	bool ggml_can_fuse_subgraph(const ggml_cgraph* cgraph,
		int                                 start_idx,
		std::initializer_list<enum ggml_op> ops,
		std::initializer_list<int>          outputs = {});
	bool ggml_can_fuse_subgraph(const ggml_cgraph* cgraph,
		int                        node_idx,
		int                        count,
		const ggml_op* ops,
		const int* outputs,
		int                        num_outputs);

	bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph* cgraph,
		const int           node_idx,
		const int           node_count,
		const int* out_nodes,
		const int           out_count,
		const bool          is_topk_moe = false);
}
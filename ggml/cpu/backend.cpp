module;
#include <stdint.h>
#include <bit>
#include <exec/static_thread_pool.hpp>
#include <exec/async_scope.hpp>
#define GGML_ASSERT(...) 

module ggml;
import :cpu.ds;

void ggml_compute_forward(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	ggml_compute_params* params,
	bool use_ref,
	ggml_tensor* tensor);

int ggml_cpu_try_fuse_ops(
	exec::static_thread_pool& pool,
	exec::async_scope& scope,
	const ggml_cgraph* cgraph,
	const int node_n,
	bool use_ref);

static bool ggml_op_is_empty(enum ggml_op op) {
	switch (op) {
	case GGML_OP_NONE:
	case GGML_OP_RESHAPE:
	case GGML_OP_TRANSPOSE:
	case GGML_OP_VIEW:
	case GGML_OP_PERMUTE:
		return true;
	default:
		return false;
	}
}

enum ggml_status ggml_cpu_backend::graph_compute_impl(ggml_cgraph* cgraph)
{
	exec::static_thread_pool pool(n_threads);
	exec::async_scope scope;
	ggml_compute_params params = {
		/*.ith       =*/ 0, //state->ith,
		/*.nth       =*/ 1, //atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
	};

	for (int node_n = 0; node_n < cgraph->nodes.size(); node_n++) {
		auto node = cgraph->nodes[node_n];

		if (ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
			// skip NOPs
			continue;
		}

		if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
			continue;
		}

		// TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
		// Try fused ops, fall back to normal compute
		const int n_fused = ggml_cpu_try_fuse_ops(pool, scope, cgraph, node_n, use_ref);
		if (n_fused > 0) {
			node_n += n_fused;
		}
		else {
			ggml_compute_forward(pool, scope, &params, use_ref, node);
		}
#if 0
		if (state->ith == 0 && cplan->abort_callback &&
			cplan->abort_callback(cplan->abort_callback_data)) {
			tp->abort = true;
			tp->ec = GGML_STATUS_ABORTED;
		}
		state->threadpool->barrier();
#endif
		stdexec::sync_wait(scope.on_empty());
	}
	return {};
}
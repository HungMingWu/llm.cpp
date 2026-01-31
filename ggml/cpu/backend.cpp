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

	for (auto& node : cgraph->nodes) {

		if (ggml_op_is_empty(node->op) || ggml_is_empty(node)) {
			// skip NOPs
			continue;
		}

		if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
			continue;
		}

		ggml_compute_forward(pool, scope, &params, use_ref, node);
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
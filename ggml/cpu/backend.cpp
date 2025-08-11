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
	ggml_tensor* tensor);

enum ggml_status ggml_cpu_backend::graph_compute_impl(ggml_cgraph* cgraph)
{
	exec::static_thread_pool pool(8);
	exec::async_scope scope;
	ggml_compute_params params = {
		/*.ith       =*/ 0, //state->ith,
		/*.nth       =*/ 1, //atomic_load_explicit(&tp->n_threads_cur, memory_order_relaxed),
	};

	for (auto& node : cgraph->nodes) {
		ggml_compute_forward(pool, scope, &params, node);
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
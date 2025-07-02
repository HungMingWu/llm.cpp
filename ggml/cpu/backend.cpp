module;
#include <stdint.h>
#include <bit>
#define GGML_ASSERT(...) 

module ggml;
import :cpu.ds;

enum ggml_status ggml_cpu_backend::graph_compute_impl(ggml_cgraph* cgraph)
{
	ggml_cplan cplan = ggml_graph_plan(*cgraph, n_threads, threadpool);

	work_data.resize(cplan.work_size);
	cplan.work_data = work_data.data();

	cplan.abort_callback = abort_callback;
	cplan.abort_callback_data = abort_callback_data;

	return ggml_graph_compute(cgraph, cplan);
}

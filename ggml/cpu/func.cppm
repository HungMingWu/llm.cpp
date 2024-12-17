module;
#define GGML_ASSERT(...)

export module ggml:cpu.func;
import :ds;
import :cpu.ds;

export
{
	ggml_cplan ggml_graph_plan(
		const ggml_cgraph& cgraph,
		int   n_threads,
		ggml_threadpool* threadpool);

	ggml_status ggml_graph_compute(ggml_cgraph* cgraph, ggml_cplan& cplan);
}
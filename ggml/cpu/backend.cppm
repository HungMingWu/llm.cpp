module;
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <vector>

export module ggml:cpu.backend;
import :ds;
import :func;
import :cpu.registry;

struct ggml_cpu_backend : public ggml_backend {
protected:
	enum ggml_status graph_compute_impl(ggml_cgraph* cgraph) override;
public:
	using ggml_backend::ggml_backend;
	size_t n_threads = 1;// GGML_DEFAULT_N_THREADS;
	ggml_threadpool_t   threadpool = nullptr;

	std::vector<uint8_t> work_data;

	ggml_abort_callback abort_callback = nullptr;
	void* abort_callback_data = nullptr;

	const char* get_name() override {
		return "CPU";
	}

	ggml_backend_graph_plan_t graph_plan_create(const ggml_cgraph* cgraph) override {
#if 0
		struct ggml_backend_plan_cpu* cpu_plan = new ggml_backend_plan_cpu;

		cpu_plan->cplan = ggml_graph_plan(cgraph, n_threads, threadpool);
		cpu_plan->cgraph = *cgraph; // FIXME: deep copy

		if (cpu_plan->cplan.work_size > 0) {
			cpu_plan->cplan.work_data = new uint8_t[cpu_plan->cplan.work_size];
			if (cpu_plan->cplan.work_data == NULL) {
				delete cpu_plan;
				return NULL;
			}
		}

		cpu_plan->cplan.abort_callback = abort_callback;
		cpu_plan->cplan.abort_callback_data = abort_callback_data;

		return cpu_plan;
#else
		return {};
#endif
	}

	void graph_plan_free(ggml_backend_graph_plan_t plan) override
	{
#if 0
		delete[] cpu_plan->cplan.work_data;
		delete cpu_plan;
#endif
	}

	enum ggml_status graph_plan_compute(ggml_backend_graph_plan_t plan) override
	{
#if 0
		ggml_backend_plan_cpu* cpu_plan = (struct ggml_backend_plan_cpu*)plan;

		return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);
#else
		return {};
#endif
	}
};

export
{
	std::unique_ptr<ggml_backend> ggml_backend_cpu_init() {
		// initialize CPU backend now to avoid slowing the first graph computation
		//ggml_cpu_init();

		return std::make_unique<ggml_cpu_backend>(
			ggml_backend_cpu_reg()->get_devices()[0]
		);
	}
}

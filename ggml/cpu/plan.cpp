module;
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <new>

#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

module ggml;
import :ds;
import :tensor;
import :cpu.ds;
import :cpu.traits;
import :cpu.vec_dot;

#define GGML_N_TASKS_MAX (-1)

static int ggml_get_n_tasks(struct ggml_tensor* node, int n_threads) {
	int n_tasks = 0;

	if (ggml_is_empty(node)) {
		// no need to multi-thread a no-op
		n_tasks = 1;
		return n_tasks;
	}

	switch (node->op) {
	case GGML_OP_CPY:
	case GGML_OP_DUP:
	case GGML_OP_CONT:
	case GGML_OP_ADD:
	case GGML_OP_ADD1:
	case GGML_OP_ACC:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_SUB:
	case GGML_OP_SQR:
	case GGML_OP_SQRT:
	case GGML_OP_LOG:
	case GGML_OP_SIN:
	case GGML_OP_COS:
	case GGML_OP_SUM:
	case GGML_OP_SUM_ROWS:
	case GGML_OP_MEAN:
	case GGML_OP_ARGMAX:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_COUNT_EQUAL:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_REPEAT:
	case GGML_OP_REPEAT_BACK:
	case GGML_OP_LEAKY_RELU:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_UNARY:
		switch (ggml_get_unary_op(node)) {
		case GGML_UNARY_OP_ABS:
		case GGML_UNARY_OP_SGN:
		case GGML_UNARY_OP_NEG:
		case GGML_UNARY_OP_STEP:
		case GGML_UNARY_OP_TANH:
		case GGML_UNARY_OP_ELU:
		case GGML_UNARY_OP_RELU:
		case GGML_UNARY_OP_SIGMOID:
		case GGML_UNARY_OP_HARDSWISH:
		case GGML_UNARY_OP_HARDSIGMOID:
		case GGML_UNARY_OP_EXP:
		{
			n_tasks = 1;
		} break;

		case GGML_UNARY_OP_GELU:
		case GGML_UNARY_OP_GELU_ERF:
		case GGML_UNARY_OP_GELU_QUICK:
		case GGML_UNARY_OP_SILU:
		{
			n_tasks = n_threads;
		} break;
		default:
			GGML_ABORT("fatal error");
		}
		break;
	case GGML_OP_GLU:
		switch (ggml_get_glu_op(node)) {
		case GGML_GLU_OP_REGLU:
		case GGML_GLU_OP_GEGLU:
		case GGML_GLU_OP_SWIGLU:
		case GGML_GLU_OP_GEGLU_ERF:
		case GGML_GLU_OP_GEGLU_QUICK:
		{
			n_tasks = n_threads;
		} break;
		default:
			GGML_ABORT("fatal error");
		}
		break;
	case GGML_OP_SILU_BACK:
	case GGML_OP_MUL:
	case GGML_OP_DIV:
	case GGML_OP_NORM:
	case GGML_OP_RMS_NORM:
	case GGML_OP_RMS_NORM_BACK:
	case GGML_OP_L2_NORM:
	case GGML_OP_GROUP_NORM:
	case GGML_OP_CONCAT:
	case GGML_OP_MUL_MAT:
	case GGML_OP_MUL_MAT_ID:
	case GGML_OP_OUT_PROD:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_GET_ROWS:
	case GGML_OP_SET_ROWS:
	{
		// FIXME: get_rows can use additional threads, but the cost of launching additional threads
		// decreases performance with GPU offloading
		//n_tasks = n_threads;
		n_tasks = 1;
	} break;
	case GGML_OP_SCALE:
	case GGML_OP_SET:
	case GGML_OP_RESHAPE:
	case GGML_OP_VIEW:
	case GGML_OP_PERMUTE:
	case GGML_OP_TRANSPOSE:
	case GGML_OP_GET_ROWS_BACK:
	case GGML_OP_DIAG:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_DIAG_MASK_ZERO:
	case GGML_OP_DIAG_MASK_INF:
	case GGML_OP_SOFT_MAX_BACK:
	case GGML_OP_ROPE:
	case GGML_OP_ROPE_BACK:
	case GGML_OP_ADD_REL_POS:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_CLAMP:
	{
		n_tasks = 1; //TODO
	} break;
	case GGML_OP_SOFT_MAX:
	{
		n_tasks = std::min<int>(n_threads, ggml_nrows(node->src[0]));
	} break;
	case GGML_OP_IM2COL:
	case GGML_OP_IM2COL_BACK:
	case GGML_OP_CONV_2D_DW:
	case GGML_OP_CONV_TRANSPOSE_1D:
	case GGML_OP_CONV_TRANSPOSE_2D:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_POOL_1D:
	case GGML_OP_POOL_2D:
	case GGML_OP_POOL_2D_BACK:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_UPSCALE:
	case GGML_OP_PAD:
	case GGML_OP_PAD_REFLECT_1D:
	case GGML_OP_ROLL:
	case GGML_OP_ARANGE:
	case GGML_OP_TIMESTEP_EMBEDDING:
	case GGML_OP_ARGSORT:
	case GGML_OP_FLASH_ATTN_EXT:
	case GGML_OP_FLASH_ATTN_BACK:
	case GGML_OP_SSM_CONV:
	case GGML_OP_SSM_SCAN:
	case GGML_OP_RWKV_WKV6:
	case GGML_OP_GATED_LINEAR_ATTN:
	case GGML_OP_RWKV_WKV7:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_WIN_PART:
	case GGML_OP_WIN_UNPART:
	case GGML_OP_GET_REL_POS:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_CUSTOM:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_CROSS_ENTROPY_LOSS:
	case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
	case GGML_OP_OPT_STEP_ADAMW:
	{
		n_tasks = n_threads;
	} break;
	case GGML_OP_NONE:
	{
		n_tasks = 1;
	} break;
	case GGML_OP_COUNT:
	{
		GGML_ABORT("fatal error");
	}
	default:
	{
		fprintf(stderr, "%s: op not implemented: ", __func__);
		if (node->op < GGML_OP_COUNT) {
			fprintf(stderr, "%s\n", ggml_op_name(node->op));
		}
		else {
			fprintf(stderr, "%d\n", node->op);
		}
		GGML_ABORT("fatal error");
	}
	}

	assert(n_tasks > 0);

	return n_tasks;
}

bool ggml_cpu_extra_work_size(int n_threads, const ggml_tensor* op, size_t* size) {
	// TODO
#if 0
	for (auto extra : ggml_backend_cpu_get_extra_buffers_type()) {
		if (extra && extra->context) {
			auto buf_extra = (ggml::cpu::extra_buffer_type*)extra->context;
			auto tensor_traits = buf_extra->get_tensor_traits(op);
			if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
				return true;
			}
		}
	}
#endif
	return false;
}

ggml_cplan ggml_graph_plan(
	const ggml_cgraph& cgraph,
	int   n_threads,
	ggml_threadpool* threadpool)
{
	if (threadpool == nullptr) {
		//GGML_PRINT_DEBUG("Threadpool is not specified. Will create a disposable threadpool : n_threads %d\n", n_threads);
	}
	if (n_threads <= 0) {
		n_threads = threadpool ? threadpool->n_threads_max : GGML_DEFAULT_N_THREADS;
	}
	ggml_cplan cplan;

	int max_tasks = 1;
	size_t work_size = 0;

	// thread scheduling for the different operations + work buffer size estimation
	for (auto& node : cgraph.nodes) {
		const int n_tasks = ggml_get_n_tasks(node, n_threads);

		max_tasks = std::max(max_tasks, n_tasks);

		size_t cur = 0;
		if (!ggml_cpu_extra_work_size(n_threads, node, &cur)) {
			switch (node->op) {
#if 0
			case GGML_OP_ACC:
			{
				if (ggml_is_quantized(node->src[0]->type)) {
					cur = ggml_type_size(GGML_TYPE_F32) * node->src[1]->ne[0] * n_tasks;
				}
			} break;
#endif
			case GGML_OP_MUL_MAT:
			{
				const enum ggml_type vec_dot_type = type_traits_cpu[node->src[0]->type].vec_dot_type;

				if (node->src[1]->type != vec_dot_type) {
					cur = ggml_row_size(vec_dot_type, node->src[1]->nelements());
				}
			} break;
#if 0
			case GGML_OP_FLASH_ATTN_BACK:
			{
				const int64_t    D = node->src[0]->ne[0];
				const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
				const int64_t mxDn = MAX(D, ne11) * 2; // *2 because of S and SM in ggml_compute_forward_flash_attn_back
				if (node->src[1]->type == GGML_TYPE_F32) {
					cur = sizeof(float) * mxDn * n_tasks; // TODO: this can become (n_tasks-1)
					cur += sizeof(float) * mxDn * n_tasks; // this is overestimated by x2
				}
				else if (node->src[1]->type == GGML_TYPE_F16) {
					cur = sizeof(float) * mxDn * n_tasks; // TODO: this can become (n_tasks-1)
					cur += sizeof(float) * mxDn * n_tasks; // this is overestimated by x2
				}
				else if (node->src[1]->type == GGML_TYPE_BF16) {
					cur = sizeof(float) * mxDn * n_tasks; // TODO: this can become (n_tasks-1)
					cur += sizeof(float) * mxDn * n_tasks; // this is overestimated by x2
				}
			} break;
			case GGML_OP_COUNT:
			{
				GGML_ABORT("fatal error");
			}
#endif
			default:
				break;
			}
		}

		work_size = std::max(work_size, cur);
	}

	if (work_size > 0) {
		work_size += std::hardware_destructive_interference_size * (n_threads);
	}

	cplan.threadpool = threadpool;
	cplan.n_threads = std::min(max_tasks, n_threads);
	cplan.work_size = work_size;

	return cplan;
}

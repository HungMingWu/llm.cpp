module;
#include <assert.h>
#include <stdint.h>
#include <array>
#include <bit>
#include <string_view>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :ds;

static bool ggml_is_contiguous_n(const ggml_tensor* tensor, int n) {
	size_t next_nb = ggml_type_size(tensor->type);
	if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
		return false;
	}
	next_nb *= tensor->ne[0] / ggml_blck_size(tensor->type);
	for (int i = 1; i < GGML_MAX_DIMS; i++) {
		if (tensor->ne[i] != 1) {
			if (i > n) {
				if (tensor->nb[i] != next_nb) {
					return false;
				}
				next_nb *= tensor->ne[i];
			}
			else {
				// this dimension does not need to be contiguous
				next_nb = tensor->ne[i] * tensor->nb[i];
			}
		}
	}
	return true;
}

size_t ggml_tensor::nbytes() const {
	size_t nbytes;
	const size_t blck_size = ggml_blck_size(type);
	if (blck_size == 1) {
		nbytes = ggml_type_size(type);
		for (int i = 0; i < GGML_MAX_DIMS; ++i) {
			nbytes += (ne[i] - 1) * nb[i];
		}
	}
	else {
		nbytes = ne[0] * nb[0] / blck_size;
		for (int i = 1; i < GGML_MAX_DIMS; ++i) {
			nbytes += (ne[i] - 1) * nb[i];
		}
	}

	return nbytes;
}

int64_t ggml_tensor::nelements() const {
	return ne[0] * ne[1] * ne[2] * ne[3];
}

void ggml_tensor::set_flag(int32_t flag)
{
	if (flag == GGML_TENSOR_FLAG_LOSS) {
		GGML_ASSERT(ggml_is_scalar(this));
		GGML_ASSERT(type == GGML_TYPE_F32);
	}
	else if (flag == GGML_TENSOR_FLAG_PARAM) {
		GGML_ASSERT(op == GGML_OP_NONE);
	}
	flags |= flag;
}

bool ggml_is_contiguous_0(const ggml_tensor* tensor) {
	return ggml_is_contiguous_n(tensor, 0);
}

bool ggml_is_contiguous_1(const ggml_tensor* tensor) {
	return ggml_is_contiguous_n(tensor, 1);
}

bool ggml_is_contiguous_2(const ggml_tensor* tensor) {
	return ggml_is_contiguous_n(tensor, 2);
}

bool ggml_is_contiguous(const ggml_tensor* tensor) {
	return ggml_is_contiguous_0(tensor);
}

ggml_tensor* ggml_dup_tensor(ggml_context* ctx, const ggml_tensor* src) {
	return ctx->create(src->type, src->ne);
}

bool ggml_can_repeat(const ggml_tensor* t0, const ggml_tensor* t1) {
	return ggml_is_empty(t0) ? ggml_is_empty(t1) :
		(t1->ne[0] % t0->ne[0] == 0) &&
		(t1->ne[1] % t0->ne[1] == 0) &&
		(t1->ne[2] % t0->ne[2] == 0) &&
		(t1->ne[3] % t0->ne[3] == 0);
}

bool ggml_is_vector(const ggml_tensor* tensor) {
	return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

ggml_tensor* ggml_view_tensor(
	ggml_context* ctx,
	ggml_tensor* src) {
	ggml_tensor* result = ctx->create(src->type, src->ne, src, 0);
	result->set_name("{} (view)", src->name);

	for (int i = 0; i < GGML_MAX_DIMS; i++) {
		result->nb[i] = src->nb[i];
	}

	return result;
}

void ggml_flash_attn_ext_set_prec(
	ggml_tensor* a,
	enum ggml_prec       prec)
{
	GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
	// scale is on first pos, max_bias on second
	a->op_params[3] = std::bit_cast<int32_t>(prec);
}

enum ggml_prec ggml_flash_attn_ext_get_prec(
	const struct ggml_tensor* a) {
	GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);

	const int32_t prec_i32 = ggml_get_op_params_i32(a, 3);

	return (enum ggml_prec)prec_i32;
}

// change the precision of a matrix multiplication
// set to GGML_PREC_F32 for higher precision (useful for phi-2)
void ggml_mul_mat_set_prec(
	ggml_tensor* a,
	enum ggml_prec prec)
{
	GGML_ASSERT(a->op == GGML_OP_MUL_MAT);

	const int32_t prec_i32 = (int32_t)prec;

	a->op_params[0] = prec_i32;
}

void ggml_soft_max_add_sinks(
	ggml_tensor* a,
	ggml_tensor* sinks) {
	GGML_ASSERT(a->src.size() == 3);
	if (!sinks) {
		a->src[2] = NULL;
		return;
	}

	GGML_ASSERT(a->op == GGML_OP_SOFT_MAX);
	GGML_ASSERT(a->src[2] == NULL);
	GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
	GGML_ASSERT(sinks->type == GGML_TYPE_F32);

	a->src[2] = sinks;
}

void ggml_flash_attn_ext_add_sinks(
	ggml_tensor* a,
	ggml_tensor* sinks) {
	GGML_ASSERT(a->src.size() == 5);
	if (!sinks) {
		a->src[4] = NULL;
		return;
	}

	GGML_ASSERT(a->op == GGML_OP_FLASH_ATTN_EXT);
	GGML_ASSERT(a->src[4] == NULL);
	GGML_ASSERT(a->src[0]->ne[2] == sinks->ne[0]);
	GGML_ASSERT(sinks->type == GGML_TYPE_F32);

	a->src[4] = sinks;
}
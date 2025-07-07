module;
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <array>
#include <bit>
#define GGML_ASSERT(...)

export module ggml:tensor;
import :ds;
import :traits;

export
{
	int32_t ggml_get_op_params_i32(const ggml_tensor* tensor, uint32_t i) {
	        assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
		return ((const int32_t*)(tensor->op_params))[i];
	}

	void ggml_backend_tensor_memset(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size);

	int64_t ggml_nrows(const ggml_tensor* tensor) {
		return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
	}

	enum ggml_unary_op ggml_get_unary_op(const ggml_tensor* tensor) {
		GGML_ASSERT(tensor->op == GGML_OP_UNARY);
		return (enum ggml_unary_op)ggml_get_op_params_i32(tensor, 0);
	}

	bool ggml_is_empty(const ggml_tensor* tensor) {
		for (int i = 0; i < GGML_MAX_DIMS; ++i) {
			if (tensor->ne[i] == 0) {
				// empty if any dimension has no elements
				return true;
			}
		}
		return false;
	}

	bool ggml_is_transposed(const ggml_tensor* tensor) {
		return tensor->nb[0] > tensor->nb[1];
	}

	bool ggml_is_3d(const ggml_tensor* tensor) {
		return tensor->ne[3] == 1;
	}

	bool ggml_is_matrix(const ggml_tensor* tensor) {
		return tensor->ne[2] == 1 && tensor->ne[3] == 1;
	}

	bool ggml_is_contiguous_0(const ggml_tensor* tensor);
	bool ggml_is_contiguous_1(const ggml_tensor* tensor);
	bool ggml_is_contiguous_2(const ggml_tensor* tensor);
	bool ggml_is_contiguous(const ggml_tensor* tensor);

	bool ggml_is_scalar(const ggml_tensor* tensor) {
		return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
	}

	ggml_tensor* ggml_set_zero(ggml_tensor* tensor) {
		if (ggml_is_empty(tensor)) {
			return tensor;
		}
		if (tensor->buffer) {
			ggml_backend_tensor_memset(tensor, 0, 0, tensor->nbytes());
		}
		else {
			assert(false);
			//GGML_ASSERT(tensor->data);
			//memset(tensor->data, 0, tensor->nbytes());
		}
		return tensor;
	}

	bool ggml_are_same_layout(const ggml_tensor* a, const ggml_tensor* b) {
		if (a->type != b->type) {
			return false;
		}
		for (int i = 0; i < GGML_MAX_DIMS; i++) {
			if (a->ne[i] != b->ne[i]) {
				return false;
			}
			if (a->nb[i] != b->nb[i]) {
				return false;
			}
		}
		return true;
	}

	ggml_status ggml_backend_view_init(ggml_tensor* tensor) {
		GGML_ASSERT(tensor->buffer == NULL);
		GGML_ASSERT(tensor->view_src != NULL);
		GGML_ASSERT(tensor->view_src->buffer != NULL);
		GGML_ASSERT(tensor->view_src->data != NULL);

		tensor->buffer = tensor->view_src->buffer;
		tensor->data = (char*)tensor->view_src->data + tensor->view_offs;
		return tensor->buffer->init_tensor(tensor);
	}

	bool ggml_are_same_shape(const ggml_tensor* t0, const struct ggml_tensor* t1) {
		return t0->ne == t1->ne;
	}

	size_t ggml_element_size(const ggml_tensor* tensor) {
		return ggml_type_size(tensor->type);
	}

	bool ggml_is_contiguous_channels(const ggml_tensor* tensor) {
		return
			tensor->nb[0] > tensor->nb[2] &&
			tensor->nb[1] > tensor->nb[0] &&
			tensor->nb[2] == ggml_type_size(tensor->type);
	}

	bool ggml_is_contiguously_allocated(const ggml_tensor* tensor) {
		return tensor->nbytes() == tensor->nelements() * ggml_type_size(tensor->type) / ggml_blck_size(tensor->type);
	}

	int ggml_n_dims(const ggml_tensor* tensor) {
		for (int i = GGML_MAX_DIMS - 1; i >= 1; --i) {
			if (tensor->ne[i] > 1) {
				return i + 1;
			}
		}
		return 1;
	}

	// true if the elements in dimension 0 are contiguous, or there is just 1 block of elements
	bool ggml_is_contiguous_rows(const ggml_tensor* tensor)
	{
		return
			tensor->ne[0] == ggml_blck_size(tensor->type) ||
			tensor->nb[0] == ggml_type_size(tensor->type);
	}

	enum ggml_glu_op ggml_get_glu_op(const ggml_tensor* tensor)
	{
		GGML_ASSERT(tensor->op == GGML_OP_GLU);
		return std::bit_cast<ggml_glu_op>(tensor->op_params[0]);
	}

	ggml_tensor* ggml_dup_tensor(ggml_context* ctx, const ggml_tensor* src);

	// check if t1 can be represented as a repeatition of t0
	bool ggml_can_repeat(const ggml_tensor* t0, const ggml_tensor* t1);
}

module;
#include <assert.h>
#include <stdint.h>
#include <array>
#include <string_view>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

module ggml;
import :ds;

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

size_t ggml_backend_buffer_type::get_alloc_size(const ggml_tensor* tensor) { return tensor->nbytes(); }

module;
#include <math.h>
#include <string.h>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <assert.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)

export module ggml:func;
import :alloc;
import :ds;
import :log;
import :os;
import :tensor;
import :traits;

bool ggml_is_numa()
{
	return false;
}

bool ggml_backend_buffer_copy_tensor(const ggml_tensor* src, ggml_tensor* dst) {
	ggml_backend_buffer* dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
	return dst_buf->cpy_tensor(src, dst);
}

export
{
	constexpr size_t GGML_PAD(size_t x, size_t n) {
		return (x + n - 1) & ~(n - 1);
	}

	float* ggml_get_data_f32(const ggml_tensor* tensor) {
		assert(tensor->type == GGML_TYPE_F32);
		return (float*)(tensor->data);
	}
}

struct ggml_backend_reg_entry {
	ggml_backend_reg_t reg;
	dl_handle_ptr handle;
};

struct ggml_backend_registry {
	std::vector<ggml_backend_reg_entry> backends;
	std::vector<ggml_backend_device*> devices;

	ggml_backend_registry();
	~ggml_backend_registry();

	void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr);
	void register_device(ggml_backend_device* device);
	ggml_backend_reg_t load_backend(const std::filesystem::path& path, bool silent);
	void unload_backend(ggml_backend_reg_t reg, bool silent);
};

static ggml_backend_registry& get_reg() {
	static ggml_backend_registry reg;
	return reg;
}

export {
	std::span<ggml_backend_device*> backend_devs()
	{
		return get_reg().devices;
	}

	ggml_backend_device* ggml_backend_dev_by_type(enum ggml_backend_dev_type type)
	{
		for (auto dev : get_reg().devices) {
			if (dev->get_type() == type) {
				return dev;
			}
		}
		return nullptr;
	}

	ggml_backend_device*    ggml_backend_buft_get_device(ggml_backend_buffer_type* buft)
	{
		return {};
	}
	size_t  ggml_get_max_tensor_size(const struct ggml_context* ctx)
	{
		return {};
	}

	// implementation at cpu side
	ggml_backend_buffer_type* ggml_backend_cpu_buffer_type();

	void ggml_set_op_params(ggml_tensor &tensor, const void* params, size_t params_size) {
		assert(params_size <= GGML_MAX_OP_PARAMS);
		memcpy(tensor.op_params, params, params_size);
	}

	std::unique_ptr<ggml_backend> ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char* params) {
		ggml_backend_device* dev = ggml_backend_dev_by_type(type);
		if (!dev) {
			return nullptr;
		}
		return dev->init_backend(params);
	}

	 void ggml_backend_tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
		 GGML_ASSERT(tensor);

		 ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

		 if (size == 0) {
			 return;
		 }

		 GGML_ASSERT(buf != NULL && "tensor buffer not set");
		 GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
		 GGML_ASSERT(offset + size <= tensor->nbytes() && "tensor write out of bounds");

		 buf->set_tensor(tensor, data, offset, size);

	 }

	 void ggml_backend_tensor_get(const ggml_tensor* tensor, void* data, size_t offset, size_t size) {
		 GGML_ASSERT(tensor);
		 ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
		 if (size == 0) {
			 return;
		 }

		 GGML_ASSERT(buf != NULL && "tensor buffer not set");
		 GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
		 GGML_ASSERT(offset + size <= tensor->nbytes() && "tensor read out of bounds");

		 buf->get_tensor(tensor, data, offset, size);
	 }

	 void ggml_backend_tensor_memset(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
		 ggml_backend_buffer* buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

		 if (size == 0) {
			 return;
		 }

		 GGML_ASSERT(buf != NULL && "tensor buffer not set");
		 GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
		 GGML_ASSERT(offset + size <= tensor->nbytes() && "tensor write out of bounds");

		 buf->memset_tensor(tensor, value, offset, size);
	 }

	 void ggml_graph_dump_dot(const ggml_cgraph* gb, const ggml_cgraph* gf, const char* filename);

	 void* ggml_get_data(const ggml_tensor* tensor) {
		 return tensor->data;
	 }

	 const char* ggml_op_desc(const ggml_tensor* t) {
		 if (t->op == GGML_OP_UNARY) {
			 enum ggml_unary_op uop = ggml_get_unary_op(t);
			 return ggml_unary_op_name(uop);
		 }
		 if (t->op == GGML_OP_GLU) {
			 enum ggml_glu_op gop = ggml_get_glu_op(t);
			 return ggml_glu_op_name(gop);
		 }
		 return ggml_op_name(t->op);
	 }

	 bool ggml_backend_compare_graph_backend(ggml_backend* backend1, ggml_backend* backend2, ggml_cgraph* graph, ggml_backend_eval_callback callback, ggml_tensor* test_node);

	 bool ggml_quantize_requires_imatrix(enum ggml_type type);

	 size_t ggml_quantize_chunk(
		 enum ggml_type   type,
		 const float* src,
		 void* dst,
		 int64_t   start,
		 int64_t   nrows,
		 int64_t   n_per_row,
		 const float* imatrix);

	 ggml_tensor* ggml_graph_get_grad(const ggml_cgraph* cgraph, const ggml_tensor* node) {
		 auto it = cgraph->grads.find(node);
		 return (it != cgraph->grads.end()) ? it->second : nullptr;
	 }

	 struct ggml_tensor* ggml_graph_get_grad_acc(const struct ggml_cgraph* cgraph, const struct ggml_tensor* node) {
		 auto it = cgraph->grad_accs.find(node);
		 return (it != cgraph->grad_accs.end()) ? it->second : nullptr;
	 }

	 ggml_type ggml_ftype_to_ggml_type(ggml_ftype ftype);

	 void ggml_backend_tensor_copy(ggml_tensor* src, ggml_tensor* dst);

	 // creates a copy of the tensor with the same memory layout
	 ggml_tensor* ggml_dup_tensor_layout(ggml_context* ctx, const ggml_tensor* tensor);

	 bool ggml_is_view_op(enum ggml_op op) {
		 return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
	 }

	 const char* ggml_status_to_string(enum ggml_status status) {
		 switch (status) {
		 case GGML_STATUS_ALLOC_FAILED: return "GGML status: error (failed to allocate memory)";
		 case GGML_STATUS_FAILED:       return "GGML status: error (operation failed)";
		 case GGML_STATUS_SUCCESS:      return "GGML status: success";
		 case GGML_STATUS_ABORTED:      return "GGML status: warning (operation aborted)";
		 }

		 return "GGML status: unknown";
	 }

	 bool ggml_can_fuse(const ggml_cgraph* cgraph, int node_idx, const enum ggml_op* ops, int num_ops);

	 float ggml_softplus(float input) {
		 return (input > 20.0f) ? input : logf(1 + expf(input));
	 };

	 // Remove later
	 size_t ggml_backend_reg_count() {
		 return get_reg().backends.size();
	 }

	 ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
		 GGML_ASSERT(index < ggml_backend_reg_count());
		 return get_reg().backends[index].reg;
	 }

}

module;
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include "inplace_vector.hpp"

#include <assert.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

#define GGML_USE_CPU

export module ggml:func;
import :alloc;
import :ds;
import :log;
import :os;
import :tensor;
import :traits;
import :cpu.registry;

#ifdef GGML_USE_CUDA
import :cuda.registry;
#endif

bool ggml_is_numa()
{
	return false;
}

export {
ggml_tensor* ggml_dup_tensor(ggml_context* ctx, const ggml_tensor* src);
ggml_status ggml_backend_view_init(ggml_tensor* tensor);
}

bool ggml_backend_buffer_copy_tensor(const ggml_tensor* src, ggml_tensor* dst) {
	ggml_backend_buffer* dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
	return dst_buf->cpy_tensor(src, dst);
}

template <typename Iter>
static bool alloc_tensor_range(ggml_context* ctx,
	Iter first, Iter last,
	ggml_backend_buffer_type_t buft, size_t size,
	std::vector<std::unique_ptr<ggml_backend_buffer>> *buffers) {

	std::unique_ptr<ggml_backend_buffer> buffer = buft->alloc_buffer(size);

	if (buffer == NULL) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("{}: failed to allocate {} buffer of size {}", __func__, buft->get_name(), size);
#endif
		return false;
	}

	ggml_tallocr tallocr(buffer.get());

	for (auto it = first; it != last; ++it) {
		auto& t = *it;
		if (t->data == NULL) {
			if (t->view_src == NULL) {
				tallocr.alloc(t);
			}
			else if (t->buffer == NULL) {
				ggml_backend_view_init(t);
			}
		}
		else {
			if (t->view_src != NULL && t->buffer == NULL) {
				// view of a pre-allocated tensor
				ggml_backend_view_init(t);
			}
		}
	}

	buffers->push_back(std::move(buffer));
	return true;
}

void ggml_cpu_init(void) {
#if 0
	// needed to initialize f16 tables
	{
		struct ggml_init_params params = { 0, NULL, false };
		struct ggml_context* ctx = ggml_init(params);
		ggml_free(ctx);
	}

	ggml_critical_section_start();

	static bool is_first_call = true;

	if (is_first_call) {
		// initialize GELU, Quick GELU, SILU and EXP F32 tables
		{
			const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

			for (int i = 0; i < (1 << 16); ++i) {
				union {
					uint16_t u16;
					ggml_fp16_t fp16;
				} u = { i };
				float f = GGML_FP16_TO_FP32(u.fp16);
				ggml_table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
				ggml_table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
			}

			const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

			GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start) / 1000.0);
		}

#if defined(__ARM_ARCH)
		ggml_init_arm_arch_features();
#endif

		is_first_call = false;
	}

	ggml_critical_section_end();
#endif
}

export
{
	constexpr size_t GGML_PAD(size_t x, size_t n) {
		return (x + n - 1) & ~(n - 1);
	}

	std::unique_ptr<ggml_context> ggml_init()
	{
		// TODO
#if 0
		static bool is_first_call = true;

		ggml_critical_section_start();

		if (is_first_call) {
			// initialize time system (required on Windows)
			ggml_time_init();

			for (int i = 0; i < (1 << 16); ++i) {
				union {
					uint16_t u16;
					ggml_fp16_t fp16;
				} u = { i };
				ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
			}

			is_first_call = false;
		}

		ggml_critical_section_end();
#endif

		return std::make_unique<ggml_context>();
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
	std::vector<ggml_backend_dev_t> devices;

	ggml_backend_registry();
	~ggml_backend_registry();

	void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr);
	void register_device(ggml_backend_dev_t device);
	ggml_backend_reg_t load_backend(const std::wstring& path, bool silent);
	void unload_backend(ggml_backend_reg_t reg, bool silent);
};

static ggml_backend_registry& get_reg() {
	static ggml_backend_registry reg;
	return reg;
}

export {
	size_t ggml_backend_dev_count()
	{
		return get_reg().devices.size();
	}

	ggml_backend_dev_t ggml_backend_dev_get(size_t index)
	{
		GGML_ASSERT(index < ggml_backend_dev_count());
		return get_reg().devices[index];
	}

	ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type)
	{
		for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
			ggml_backend_dev_t dev = ggml_backend_dev_get(i);
			if (dev->get_type() == type) {
				return dev;
			}
		}
		return nullptr;
	}

	ggml_backend_dev_t    ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft)
	{
		return {};
	}
	size_t  ggml_get_max_tensor_size(const struct ggml_context* ctx)
	{
		return {};
	}

	// implementation at cpu side
	ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type();

	void ggml_set_op_params(ggml_tensor &tensor, const void* params, size_t params_size) {
		assert(params_size <= GGML_MAX_OP_PARAMS);
		memcpy(tensor.op_params, params, params_size);
	}

	int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
		return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
	}

	ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char* params) {
		ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
		if (!dev) {
			return nullptr;
		}
		return dev->init_backend(params);
	}

	struct ggml_tensor* ggml_cast(
		struct ggml_context* ctx,
		struct ggml_tensor* a,
		enum   ggml_type      type) {
		// TODO
		return nullptr;
	}

	ggml_tensor* ggml_tanh(
		struct ggml_context* ctx,
		struct ggml_tensor* a)
	{
		// TODO
		return nullptr;
	}

	struct ggml_tensor* ggml_set_1d(
		struct ggml_context* ctx,
		struct ggml_tensor* a,
		struct ggml_tensor* b,
		size_t                offset) // in bytes
	{
		// TODO
		return nullptr;
	}

	// conv_1d with padding = half
	// alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
	ggml_tensor* ggml_conv_1d_ph(
		struct ggml_context* ctx,
		struct ggml_tensor* a,  // convolution kernel
		struct ggml_tensor* b,  // data
		int                   s,  // stride
		int                   d) // dilation
	{
		// TODO
		return nullptr;
	}

	ggml_tensor* ggml_sigmoid(
		struct ggml_context* ctx,
		struct ggml_tensor* a)
	{
		// TODO
		return nullptr;
	}

	ggml_tensor* ggml_conv_1d_dw_ph(
		struct ggml_context* ctx,
		struct ggml_tensor* a,   // convolution kernel
		struct ggml_tensor* b,   // data
		int                   s0,  // stride
		int                   d0) // dilation
	{
		// TODO
		return nullptr;
	}

	ggml_tensor* ggml_relu(
		struct ggml_context* ctx,
		struct ggml_tensor* a)
	{
		// TODO
		return nullptr;
	}

	 // top k elements per row
	 ggml_tensor* ggml_top_k(
		 struct ggml_context* ctx,
		 struct ggml_tensor* a,
		 int                   k)
	 {
		 // TODO
		 return nullptr;
	 }

	 ggml_tensor* ggml_exp(
		 struct ggml_context* ctx,
		 struct ggml_tensor* a)
	 {
		 // TODO
		 return nullptr;
	 }

	 ggml_tensor* ggml_neg(
		 struct ggml_context* ctx,
		 struct ggml_tensor* a)
	 {
		 // TODO
		 return nullptr;
	 }

	 void ggml_flash_attn_ext_set_prec(
		 ggml_tensor* a,
		 enum ggml_prec       prec)
	 {
		 // TODO
	 }

	 // change the precision of a matrix multiplication
	 // set to GGML_PREC_F32 for higher precision (useful for phi-2)
	 void ggml_mul_mat_set_prec(
		 struct ggml_tensor* a,
		 enum ggml_prec       prec)
	 {
		 // TODO
	 }

	 struct multi_backend_buffer : public ggml_backend_buffer {
		 std::vector<std::unique_ptr<ggml_backend_buffer>> buffers;
	 protected:
		 void clear_impl(uint8_t value) override {
			 for (auto& buffer : buffers)
				 buffer->clear(value);
		 }
	 public:
		 multi_backend_buffer(
			 ggml_backend_buffer_type_t buft, size_t size, std::vector<std::unique_ptr<ggml_backend_buffer>> buffers)
			 : ggml_backend_buffer(buft, size), 
			   buffers(std::move(buffers))
		 {

		 }
	 };

	 std::unique_ptr<ggml_backend_buffer> ggml_backend_alloc_ctx_tensors_from_buft(ggml_context* ctx, ggml_backend_buffer_type_t buft) {
		 size_t alignment = buft->get_alignment();
		 size_t max_size = buft->get_max_size();

		 std::vector<std::unique_ptr<ggml_backend_buffer>> buffers;

		 size_t cur_buf_size = 0;
		 auto first = ctx->getTensors().begin();
		 for (auto it = ctx->getTensors().begin(); it != ctx->getTensors().end(); ++it) {
			 auto& t = *it;
			 size_t this_size = 0;
			 if (t->data == nullptr && t->view_src == nullptr) {
				 this_size = GGML_PAD(buft->get_alloc_size(t), alignment);
			 }
#if 0
			 if (this_size > max_size) {
				 GGML_LOG_ERROR("{}: tensor {} is too large to fit in a {} buffer (tensor size: {}, max buffer size: {})",
					 __func__, t->name,
					 ggml_backend_buft_name(buft),
					 this_size, max_size);
				 return nullptr;
			 }
#endif
			 if ((cur_buf_size + this_size) > max_size) {
				 // allocate tensors in the current buffer
				 if (!alloc_tensor_range(ctx, first, it, buft, cur_buf_size, &buffers)) {
					 return NULL;
				 }
				 first = it;
				 cur_buf_size = this_size;
			 }
			 else {
				 cur_buf_size += this_size;
			 }
		 }

		 // allocate remaining tensors
		 if (cur_buf_size > 0) {
			 if (!alloc_tensor_range(ctx, first, ctx->getTensors().end(), buft, cur_buf_size, &buffers)) {
				 return nullptr;
			 }
		 }

		 if (buffers.empty()) {
#ifndef NDEBUG
			 GGML_LOG_DEBUG("{}: all tensors in the context are already allocated", __func__);
#endif
			 return nullptr;
		 }

		 if (buffers.size() == 1) {
			 return std::move(buffers[0]);
		 }
		 else {
			 size_t total_size = 0;
			 for (const auto& buffer : buffers) {
				 total_size += buffer->get_size();
			 }
			 return std::make_unique<multi_backend_buffer>(buffers[0]->get_type(), total_size, std::move(buffers));
		 }
	 }

	 void ggml_backend_tensor_set(ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
		 GGML_ASSERT(tensor);

		 ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

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
		 ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
		 if (size == 0) {
			 return;
		 }

		 GGML_ASSERT(buf != NULL && "tensor buffer not set");
		 GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
		 GGML_ASSERT(offset + size <= tensor->nbytes() && "tensor read out of bounds");

		 buf->get_tensor(tensor, data, offset, size);
	 }

	 void ggml_backend_tensor_memset(ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
		 ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

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

	 void ggml_unravel_index(const ggml_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3) {
		 const int64_t ne2 = tensor->ne[2];
		 const int64_t ne1 = tensor->ne[1];
		 const int64_t ne0 = tensor->ne[0];

		 const int64_t i3_ = (i / (ne2 * ne1 * ne0));
		 const int64_t i2_ = (i - i3_ * ne2 * ne1 * ne0) / (ne1 * ne0);
		 const int64_t i1_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0) / ne0;
		 const int64_t i0_ = (i - i3_ * ne2 * ne1 * ne0 - i2_ * ne1 * ne0 - i1_ * ne0);

		 if (i0) {
			 *i0 = i0_;
		 }
		 if (i1) {
			 *i1 = i1_;
		 }
		 if (i2) {
			 *i2 = i2_;
		 }
		 if (i3) {
			 *i3 = i3_;
		 }
	 }

	 void ggml_set_i32_nd(const struct ggml_tensor* tensor, int i0, int i1, int i2, int i3, int32_t value) {
		 void* data = (char*)tensor->data + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
		 switch (tensor->type) {
		 case GGML_TYPE_I8:
		 {
			 ((int8_t*)(data))[0] = value;
		 } break;
		 case GGML_TYPE_I16:
		 {
			 ((int16_t*)(data))[0] = value;
		 } break;
		 case GGML_TYPE_I32:
		 {
			 ((int32_t*)(data))[0] = value;
		 } break;
		 case GGML_TYPE_F16:
		 {
			 ((ggml_fp16_t*)(data))[0] = fromFloat32<ggml_fp16_t>(value);
		 } break;
		 case GGML_TYPE_BF16:
		 {
			 ((ggml_bf16_t*)(data))[0] = fromFloat32<ggml_bf16_t>(value);
		 } break;
		 case GGML_TYPE_F32:
		 {
			 ((float*)(data))[0] = value;
		 } break;
		 default:
		 {
			 GGML_ABORT("fatal error");
		 }
		 }
	 }

	 void ggml_set_i32_1d(const struct ggml_tensor* tensor, int i, int32_t value) {
		 if (!ggml_is_contiguous(tensor)) {
			 int64_t id[4] = { 0, 0, 0, 0 };
			 ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
			 ggml_set_i32_nd(tensor, id[0], id[1], id[2], id[3], value);
			 return;
		 }
		 switch (tensor->type) {
		 case GGML_TYPE_I8:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
			 ((int8_t*)(tensor->data))[i] = value;
		 } break;
		 case GGML_TYPE_I16:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
			 ((int16_t*)(tensor->data))[i] = value;
		 } break;
		 case GGML_TYPE_I32:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
			 ((int32_t*)(tensor->data))[i] = value;
		 } break;
		 case GGML_TYPE_F16:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
			 ((ggml_fp16_t*)(tensor->data))[i] = fromFloat32<ggml_fp16_t>(value);
		 } break;
		 case GGML_TYPE_BF16:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
			 ((ggml_bf16_t*)(tensor->data))[i] = fromFloat32<ggml_bf16_t>(value);
		 } break;
		 case GGML_TYPE_F32:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(float));
			 ((float*)(tensor->data))[i] = value;
		 } break;
		 default:
		 {
			 GGML_ABORT("fatal error");
		 }
		 }
	 }

	 int32_t ggml_get_i32_nd(const struct ggml_tensor* tensor, int i0, int i1, int i2, int i3) {
		 void* data = (char*)tensor->data + i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
		 switch (tensor->type) {
		 case GGML_TYPE_I8:
			 return ((int8_t*)data)[0];
		 case GGML_TYPE_I16:
			 return ((int16_t*)data)[0];
		 case GGML_TYPE_I32:
			 return ((int32_t*)data)[0];
		 case GGML_TYPE_F16:
			 return toFloat32(((ggml_fp16_t*)data)[0]);
		 case GGML_TYPE_BF16:
			 return toFloat32(((ggml_bf16_t*)data)[0]);
		 case GGML_TYPE_F32:
			 return ((float*)data)[0];
		 default:
			 GGML_ABORT("fatal error");
		 }
	 }

	 int32_t ggml_get_i32_1d(const struct ggml_tensor* tensor, int i) {
		 if (!ggml_is_contiguous(tensor)) {
			 int64_t id[4] = { 0, 0, 0, 0 };
			 ggml_unravel_index(tensor, i, &id[0], &id[1], &id[2], &id[3]);
			 return ggml_get_i32_nd(tensor, id[0], id[1], id[2], id[3]);
		 }
		 switch (tensor->type) {
		 case GGML_TYPE_I8:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(int8_t));
			 return ((int8_t*)(tensor->data))[i];
		 }
		 case GGML_TYPE_I16:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(int16_t));
			 return ((int16_t*)(tensor->data))[i];
		 }
		 case GGML_TYPE_I32:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(int32_t));
			 return ((int32_t*)(tensor->data))[i];
		 }
		 case GGML_TYPE_F16:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
			 return toFloat32(((ggml_fp16_t*)(tensor->data))[i]);
		 }
		 case GGML_TYPE_BF16:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(ggml_bf16_t));
			 return toFloat32(((ggml_bf16_t*)(tensor->data))[i]);
		 }
		 case GGML_TYPE_F32:
		 {
			 GGML_ASSERT(tensor->nb[0] == sizeof(float));
			 return ((float*)(tensor->data))[i];
		 }
		 default:
		 {
			 GGML_ABORT("fatal error");
		 }
		 }
	 }

	 void ggml_vec_set_i8(const int n, int8_t* x, const int8_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
	 void ggml_vec_set_i16(const int n, int16_t* x, const int16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
	 void ggml_vec_set_i32(const int n, int32_t* x, const int32_t   v) { for (int i = 0; i < n; ++i) x[i] = v; }
	 void ggml_vec_set_f16(const int n, ggml_fp16_t* x, const int32_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
	 void ggml_vec_set_bf16(const int n, ggml_bf16_t* x, const ggml_bf16_t v) { for (int i = 0; i < n; ++i) x[i] = v; }
	 void ggml_vec_set_f32(const int n, float* x, const float   v) { for (int i = 0; i < n; ++i) x[i] = v; }

	 struct ggml_tensor* ggml_set_i32(struct ggml_tensor* tensor, int32_t value) {
		 const int n = ggml_nrows(tensor);
		 const int nc = tensor->ne[0];
		 const size_t n1 = tensor->nb[1];

		 char* const data = (char * const)tensor->data;

		 switch (tensor->type) {
		 case GGML_TYPE_I8:
		 {
			 assert(tensor->nb[0] == sizeof(int8_t));
			 for (int i = 0; i < n; i++) {
				 ggml_vec_set_i8(nc, (int8_t*)(data + i * n1), value);
			 }
		 } break;
		 case GGML_TYPE_I16:
		 {
			 assert(tensor->nb[0] == sizeof(int16_t));
			 for (int i = 0; i < n; i++) {
				 ggml_vec_set_i16(nc, (int16_t*)(data + i * n1), value);
			 }
		 } break;
		 case GGML_TYPE_I32:
		 {
			 assert(tensor->nb[0] == sizeof(int32_t));
			 for (int i = 0; i < n; i++) {
				 ggml_vec_set_i32(nc, (int32_t*)(data + i * n1), value);
			 }
		 } break;
		 case GGML_TYPE_F16:
		 {
			 assert(tensor->nb[0] == sizeof(ggml_fp16_t));
			 for (int i = 0; i < n; i++) {
				 ggml_vec_set_f16(nc, (ggml_fp16_t*)(data + i * n1), fromFloat32<ggml_fp16_t>(value));
			 }
		 } break;
		 case GGML_TYPE_BF16:
		 {
			 assert(tensor->nb[0] == sizeof(ggml_fp16_t));
			 for (int i = 0; i < n; i++) {
				 ggml_vec_set_bf16(nc, (ggml_bf16_t*)(data + i * n1), fromFloat32<ggml_bf16_t>(value));
			 }
		 } break;
		 case GGML_TYPE_F32:
		 {
			 assert(tensor->nb[0] == sizeof(float));
			 for (int i = 0; i < n; i++) {
				 ggml_vec_set_f32(nc, (float*)(data + i * n1), value);
			 }
		 } break;
		 default:
		 {
			 GGML_ABORT("fatal error");
		 }
		 }

		 return tensor;
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

	 bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, ggml_cgraph* graph, ggml_backend_eval_callback callback, ggml_tensor* test_node);

	 bool ggml_quantize_requires_imatrix(enum ggml_type type);

	 size_t ggml_quantize_chunk(
		 enum ggml_type   type,
		 const float* src,
		 void* dst,
		 int64_t   start,
		 int64_t   nrows,
		 int64_t   n_per_row,
		 const float* imatrix);

	 float ggml_get_f32_1d(const struct ggml_tensor* tensor, int i) {
		 switch (tensor->type) {
		 case GGML_TYPE_I8:
		 {
			 return ((int8_t*)(tensor->data))[i];
		 }
		 case GGML_TYPE_I16:
		 {
			 return ((int16_t*)(tensor->data))[i];
		 }
		 case GGML_TYPE_I32:
		 {
			 return ((int32_t*)(tensor->data))[i];
		 }
		 case GGML_TYPE_F16:
		 {
			 return 0;// GGML_FP16_TO_FP32(((ggml_fp16_t*)(tensor->data))[i]);
		 }
		 case GGML_TYPE_BF16:
		 {
			 return 0;// GGML_BF16_TO_FP32(((ggml_bf16_t*)(tensor->data))[i]);
		 }
		 case GGML_TYPE_F32:
		 {
			 return ((float*)(tensor->data))[i];
		 }
		 default:
		 {
			 GGML_ABORT("fatal error");
		 }
		 }
	 }

	 ggml_tensor* ggml_graph_get_grad(const ggml_cgraph* cgraph, const ggml_tensor* node) {
		 auto it = cgraph->grads.find(node);
		 return (it != cgraph->grads.end()) ? it->second : nullptr;
	 }

	 void ggml_quantize_init(ggml_type type);

	 ggml_type ggml_ftype_to_ggml_type(ggml_ftype ftype);

	 void ggml_backend_tensor_copy(ggml_tensor* src, ggml_tensor* dst);

	 // creates a copy of the tensor with the same memory layout
	 ggml_tensor* ggml_dup_tensor_layout(ggml_context* ctx, const ggml_tensor* tensor) {
		 ggml_tensor* dup = ggml_dup_tensor(ctx, tensor);
		 for (int i = 0; i < GGML_MAX_DIMS; i++) {
			 dup->nb[i] = tensor->nb[i];
		 }
		 return dup;
	 }

	 bool ggml_is_view_op(enum ggml_op op) {
		 return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
	 }

	 std::string utf16_to_utf8(const std::wstring& str);

	 // Remove later
	 size_t ggml_backend_reg_count() {
		 return get_reg().backends.size();
	 }

	 ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
		 GGML_ASSERT(index < ggml_backend_reg_count());
		 return get_reg().backends[index].reg;
	 }

}

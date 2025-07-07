module;
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <bit>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "inplace_vector.hpp"

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
	#    include <mach-o/dyld.h>
	#    include <dlfcn.h>
#else
	#    include <dlfcn.h>
	#    include <unistd.h>
#endif

#define GGML_LOG_DEBUG(...)
#define GGML_LOG_ERROR(...)
#define GGML_LOG_INFO(...)
#include <assert.h>
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_ABORT(...)

#define GGML_BACKEND_API_VERSION 1
#define GGML_USE_CPU

export module ggml:func;
import :alloc;
import :ds;
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

#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN 4
#else
#define GGML_MEM_ALIGN 16
#endif

#define GGML_UNUSED(x) (void)(x)

export {
void ggml_set_op_params(ggml_tensor& tensor, const void* params, size_t params_size);
ggml_tensor* ggml_dup_tensor(ggml_context* ctx, const ggml_tensor* src);
ggml_status ggml_backend_view_init(ggml_tensor* tensor);
ggml_tensor* ggml_dup_tensor_layout(ggml_context* ctx, const ggml_tensor* tensor);
}

bool ggml_backend_buffer_copy_tensor(const ggml_tensor* src, ggml_tensor* dst) {
	ggml_backend_buffer* dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
	return dst_buf->cpy_tensor(src, dst);
}

ggml_tensor* graph_copy_dup_tensor(std::unordered_map<ggml_tensor*, ggml_tensor*>& node_copies,
	ggml_context* ctx_allocated, ggml_context* ctx_unallocated, ggml_tensor* src) {

	GGML_ASSERT(src != nullptr);
	GGML_ASSERT(src->data && "graph must be allocated");

	auto it = node_copies.find(src);
	if (it != node_copies.end()) return it->second;

	ggml_tensor* dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
	if (src->view_src != nullptr) {
		dst->view_src = graph_copy_dup_tensor(node_copies, ctx_allocated, ctx_unallocated, src->view_src);
		dst->view_offs = src->view_offs;
	}
	dst->op = src->op;
	memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
	dst->name = src->name;

	// copy src
	for (auto s : src->src) {
		if (!s) dst->src.push_back(nullptr);
		else dst->src.push_back(graph_copy_dup_tensor(node_copies, ctx_allocated, ctx_unallocated, s));
	}

	node_copies[src] = dst;
	return dst;
}

static ggml_tensor* ggml_view_impl(
	ggml_context* ctx,
	ggml_tensor* a,
	std::initializer_list<int64_t> ne,
	size_t                offset) {

	ggml_tensor* result = ctx->create(a->type, ne, a, offset);
	result->set_name("{} (view)", a->name);

	ggml_set_op_params(*result, &offset, sizeof(offset));

	result->op = GGML_OP_VIEW;
	result->src.push_back(a);

	return result;
}

static int64_t ggml_calc_pool_output_size(int64_t ins, int ks, int s, float p) {
	return (ins + 2 * p - ks) / s + 1;
};

static int64_t ggml_calc_conv_transpose_1d_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
	return (ins - 1) * s - 2 * p + d * (ks - 1) + 1;
}

export {
std::unique_ptr<ggml_backend_buffer> ggml_backend_alloc_ctx_tensors(ggml_context* ctx, ggml_backend_t backend);
}

template <typename Iter>
static bool alloc_tensor_range(ggml_context* ctx,
	Iter first, Iter last,
	ggml_backend_buffer_type_t buft, size_t size,
	std::vector<std::unique_ptr<ggml_backend_buffer>> *buffers) {

	std::unique_ptr<ggml_backend_buffer> buffer = buft->alloc_buffer(size);

	if (buffer == NULL) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("%s: failed to allocate %s buffer of size %zu\n", __func__, ggml_backend_buft_name(buft), size);
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

static std::string utf16_to_utf8(const std::wstring& str) {
	std::string result;
	result.reserve(str.size() * 2);
	for (wchar_t wc : str) {
		if (wc <= 0x7F) {
			result.push_back(static_cast<char>(wc));
		}
		else if (wc <= 0x7FF) {
			result.push_back(0xC0 | ((wc >> 6) & 0x1F));
			result.push_back(0x80 | (wc & 0x3F));
		}
		else if (wc <= 0xFFFF) {
			result.push_back(0xE0 | ((wc >> 12) & 0x0F));
			result.push_back(0x80 | ((wc >> 6) & 0x3F));
			result.push_back(0x80 | (wc & 0x3F));
		}
		else {
			throw std::runtime_error("Character out of UTF-8 range");
		}
	}
	return result;
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

	ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, ggml_tensor* tensor, void* addr) {
		GGML_ASSERT(tensor->buffer == NULL);
		GGML_ASSERT(tensor->data == NULL);
		GGML_ASSERT(tensor->view_src == NULL);
		GGML_ASSERT(addr >= buffer->get_base());
		GGML_ASSERT((char*)addr + buffer->get_alloc_size(tensor) <=
			(char*)buffer->get_base() + buffer->get_size());

		tensor->buffer = buffer;
		tensor->data = addr;
		return buffer->init_tensor(tensor);
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

    constexpr size_t ggml_tensor_overhead() {
        return sizeof(ggml_object) + sizeof(ggml_tensor);
    }

	float* ggml_get_data_f32(const ggml_tensor* tensor) {
		assert(tensor->type == GGML_TYPE_F32);
		return (float*)(tensor->data);
	}
}

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
	void operator()(HMODULE handle) {
		FreeLibrary(handle);
	}
};

static dl_handle* dl_load_library(const std::wstring& path) {
	// suppress error dialogs for missing DLLs
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	HMODULE handle = LoadLibraryW(path.c_str());

	SetErrorMode(old_mode);

	return handle;
}

static void* dl_get_sym(dl_handle* handle, const char* name) {
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	void* p = (void*)GetProcAddress(handle, name);

	SetErrorMode(old_mode);

	return p;
}

#else

using dl_handle = void;

struct dl_handle_deleter {
	void operator()(void* handle) {
		dlclose(handle);
	}
};

static void* dl_load_library(const std::wstring& path) {
	dl_handle* handle = dlopen(utf16_to_utf8(path).c_str(), RTLD_NOW | RTLD_LOCAL);

	return handle;
}

static void* dl_get_sym(dl_handle* handle, const char* name) {
	return dlsym(handle, name);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

struct ggml_backend_reg_entry {
	ggml_backend_reg_t reg;
	dl_handle_ptr handle;
};

struct ggml_backend_registry {
	std::vector<ggml_backend_reg_entry> backends;
	std::vector<ggml_backend_dev_t> devices;

	ggml_backend_registry();
	~ggml_backend_registry();

	void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
		if (!reg) {
			return;
		}

#ifndef NDEBUG
		GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
			__func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
		backends.push_back({ reg, std::move(handle) });
		for (auto dev : reg->get_devices())
			register_device(dev);
	}

	void register_device(ggml_backend_dev_t device) {
#ifndef NDEBUG
		GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
		devices.push_back(device);
	}

	ggml_backend_reg_t load_backend(const std::wstring& path, bool silent) {
		dl_handle_ptr handle{ dl_load_library(path) };
		if (!handle) {
			if (!silent) {
				GGML_LOG_ERROR("%s: failed to load %s\n", __func__, utf16_to_utf8(path).c_str());
			}
			return nullptr;
		}

		auto score_fn = (ggml_backend_score_t)dl_get_sym(handle.get(), "ggml_backend_score");
		if (score_fn && score_fn() == 0) {
			if (!silent) {
				GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, utf16_to_utf8(path).c_str());
			}
			return nullptr;
		}

		auto backend_init_fn = (ggml_backend_init_t)dl_get_sym(handle.get(), "ggml_backend_init");
		if (!backend_init_fn) {
			if (!silent) {
				GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, utf16_to_utf8(path).c_str());
			}
			return nullptr;
		}

		ggml_backend_reg_t reg = backend_init_fn();
		if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
			if (!silent) {
				if (!reg) {
					GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n", __func__, utf16_to_utf8(path).c_str());
				}
				else {
					GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
						__func__, utf16_to_utf8(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
				}
			}
			return nullptr;
		}

		GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), utf16_to_utf8(path).c_str());

		register_backend(reg, std::move(handle));

		return reg;
	}

	void unload_backend(ggml_backend_reg_t reg, bool silent) {
		auto it = std::find_if(backends.begin(), backends.end(),
			[reg](const ggml_backend_reg_entry& entry) { return entry.reg == reg; });

		if (it == backends.end()) {
			if (!silent) {
				GGML_LOG_ERROR("%s: backend not found\n", __func__);
			}
			return;
		}

		if (!silent) {
			GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
		}

		// remove devices
		devices.erase(
			std::remove_if(devices.begin(), devices.end(),
				[reg](ggml_backend_dev_t dev) { return dev->get_backend_reg() == reg; }),
			devices.end());

		// remove backend
		backends.erase(it);
	}
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
	ggml_tensor* ggml_get_tensor(struct ggml_context* ctx, const char* name)
	{
		return nullptr;
	}
	size_t  ggml_get_max_tensor_size(const struct ggml_context* ctx)
	{
		return {};
	}
	ggml_backend_buffer_t         ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void* ptr, size_t size, size_t max_tensor_size)
	{
		return {};
	}

	void                          ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props* props)
	{

	}

	// implementation at cpu side
	ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type();

	// check if t1 can be represented as a repeatition of t0
	bool ggml_can_repeat(const ggml_tensor &t0, const ggml_tensor &t1) {
		return ggml_is_empty(&t0) ? ggml_is_empty(&t1) :
			(t1.ne[0] % t0.ne[0] == 0) &&
			(t1.ne[1] % t0.ne[1] == 0) &&
			(t1.ne[2] % t0.ne[2] == 0) &&
			(t1.ne[3] % t0.ne[3] == 0);
	}

	bool ggml_is_vector(const ggml_tensor &tensor) {
		return tensor.ne[1] == 1 && tensor.ne[2] == 1 && tensor.ne[3] == 1;
	}

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

	void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
		// clear is optional if the buffer is zero-sized
		if (buffer->get_size() == 0) {
			return;
		}

		buffer->clear(value);
	}

	size_t ggml_graph_overhead_custom(size_t size, bool grads) {
		// TODO
#if 0
		return GGML_OBJECT_SIZE + GGML_PAD(ggml_graph_nbytes(size, grads), GGML_MEM_ALIGN);
#else
		return 0;
#endif
	}

	ggml_cgraph* ggml_new_graph_custom(ggml_context* ctx, size_t size, bool grads) {
		// TODO
		return nullptr;
	}

	struct ggml_tensor* ggml_cast(
		struct ggml_context* ctx,
		struct ggml_tensor* a,
		enum   ggml_type      type) {
		// TODO
		return nullptr;
	}

	ggml_tensor* ggml_view_tensor(
		ggml_context* ctx,
		ggml_tensor* src) {
		ggml_tensor* result = ctx->create(src->type, { src->ne[0], src->ne[1], src->ne[2], src->ne[3] }, src, 0);
		result->set_name("{} (view)", src->name);

		for (int i = 0; i < GGML_MAX_DIMS; i++) {
			result->nb[i] = src->nb[i];
		}

		return result;
	}

	ggml_tensor* ggml_dup_tensor(ggml_context* ctx, const ggml_tensor* src) {
		return ctx->create(src->type, { src->ne[0], src->ne[1], src->ne[2], src->ne[3] });
	}

	ggml_tensor* ggml_conv_transpose_1d(
		ggml_context* ctx,
		ggml_tensor* a,
		ggml_tensor* b,
		int                   s0,
		int                   p0,
		int                   d0) {
		GGML_ASSERT(ggml_is_matrix(b));
		GGML_ASSERT(a->ne[2] == b->ne[1]);
		GGML_ASSERT(a->ne[3] == 1);

		GGML_ASSERT(p0 == 0);
		GGML_ASSERT(d0 == 1);

		ggml_tensor* result = ctx->create(GGML_TYPE_F32, {
			ggml_calc_conv_transpose_1d_output_size(b->ne[0], a->ne[0], s0, 0 /*p0*/, 1 /*d0*/),
			a->ne[1],
			b->ne[2],
			1
		});

		int32_t params[] = { s0, p0, d0 };
		ggml_set_op_params(*result, params, sizeof(params));

		result->op = GGML_OP_CONV_TRANSPOSE_1D;
		result->src.push_back(a);
		result->src.push_back(b);

		return result;
	}

	ggml_tensor* ggml_get_rows(
		ggml_context* ctx,
		ggml_tensor* a,  // data
		ggml_tensor* b) // row indices
	{
		GGML_ASSERT(a->ne[2] == b->ne[1]);
		GGML_ASSERT(b->ne[3] == 1);
		GGML_ASSERT(b->type == GGML_TYPE_I32);

		// TODO: implement non F32 return
		enum ggml_type type = GGML_TYPE_F32;
		if (a->type == GGML_TYPE_I32) {
			type = a->type;
		}
		ggml_tensor* result = ctx->create(type, { a->ne[0], b->ne[0], b->ne[1], b->ne[2] });

		result->op = GGML_OP_GET_ROWS;
		result->src.push_back(a);
		result->src.push_back(b);

		return result;
	}

	ggml_tensor* ggml_tanh(
		struct ggml_context* ctx,
		struct ggml_tensor* a)
	{
		// TODO
		return nullptr;
	}

	ggml_cgraph* ggml_new_graph(struct ggml_context* ctx) // size = GGML_DEFAULT_GRAPH_SIZE, grads = false
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
	 public:
		 multi_backend_buffer(
			 ggml_backend_buffer_type_t buft, size_t size, std::vector<std::unique_ptr<ggml_backend_buffer>> buffers)
			 : ggml_backend_buffer(buft, size), 
			   buffers(std::move(buffers))
		 {

		 }
		 void clear(uint8_t value) override {
			 for (auto& buffer : buffers)
				 buffer->clear(value);
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
				 GGML_LOG_ERROR("%s: tensor %s is too large to fit in a %s buffer (tensor size: %zu, max buffer size: %zu)\n",
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
			 GGML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
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

	 std::unique_ptr<ggml_backend_buffer> ggml_backend_alloc_ctx_tensors(ggml_context* ctx, ggml_backend_t backend) {
		 return ggml_backend_alloc_ctx_tensors_from_buft(ctx, backend->get_default_buffer_type());
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

	 int64_t ggml_calc_conv_transpose_output_size(int64_t ins, int64_t ks, int s, int p) {
		 return (ins - 1) * s - 2 * p + ks;
	 }

	 ggml_tensor* ggml_conv_transpose_2d_p0(
		 ggml_context* ctx,
		 ggml_tensor* a,
		 ggml_tensor* b,
		 int                   stride) {
		 GGML_ASSERT(a->ne[3] == b->ne[2]);

		 ggml_tensor* result = ctx->create(GGML_TYPE_F32, {
			 ggml_calc_conv_transpose_output_size(b->ne[0], a->ne[0], stride, 0 /*p0*/),
			 ggml_calc_conv_transpose_output_size(b->ne[1], a->ne[1], stride, 0 /*p1*/),
			 a->ne[2], b->ne[3]
		 });

		 result->op_params[0] = std::bit_cast<uint32_t>(stride);

		 result->op = GGML_OP_CONV_TRANSPOSE_2D;
		 result->src.push_back(a);
		 result->src.push_back(b);

		 return result;
	 }

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

	 ggml_tensor* ggml_view(
		 ggml_context* ctx,
		 ggml_tensor* a,
		 std::initializer_list<int64_t> ne,
		 std::initializer_list<size_t> nb,
		 size_t                offset) {

		 assert(nb.size() + 1 == ne.size());
		 ggml_tensor* result = ggml_view_impl(ctx, a, ne, offset);

		 if (nb.size() == 1) {
			 auto it = nb.begin();
			 result->nb[1] = *it;
			 result->nb[2] = result->nb[1] * result->ne[1];
			 result->nb[3] = result->nb[2];
		 }
		 else if (nb.size() == 2) {
			 auto it = nb.begin();
			 result->nb[1] = *it++;
			 result->nb[2] = *it++;
			 result->nb[3] = result->nb[2] * result->ne[2];
		 }
		 else if (nb.size() == 3) {
			 auto it = nb.begin();
			 result->nb[1] = *it++;
			 result->nb[2] = *it++;
			 result->nb[3] = *it++;
		 }

		 return result;
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

	 ggml_tensor* ggml_pool_1d(
		 ggml_context* ctx,
		 ggml_tensor* a,
		 enum ggml_op_pool     op,
		 int                   k0,
		 int                   s0,
		 int                   p0) {

		 ggml_tensor* result = ctx->create(GGML_TYPE_F32, {
			 ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
			 a->ne[1],
			 a->ne[2],
			 a->ne[3]
		  });

		 int32_t params[] = { op, k0, s0, p0 };
		 ggml_set_op_params(*result, params, sizeof(params));

		 result->op = GGML_OP_POOL_1D;
		 result->src.push_back(a);

		 return result;
	 }

	 ggml_tensor* ggml_pool_2d(
		 ggml_context* ctx,
		 ggml_tensor* a,
		 enum ggml_op_pool     op,
		 int                   k0,
		 int                   k1,
		 int                   s0,
		 int                   s1,
		 int32_t               p0,
		 int32_t               p1) {
		 ggml_tensor* result = ctx->create(GGML_TYPE_F32, {
			 ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
			 ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
			 a->ne[2],
			 a->ne[3]
		 });

		 int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
		 ggml_set_op_params(*result, params, sizeof(params));

		 result->op = GGML_OP_POOL_2D;
		 result->src.push_back(a);

		 return result;
	 }

	 ggml_tensor* ggml_pad_reflect_1d(
		 ggml_context* ctx,
		 ggml_tensor* a,
		 int                   p0,
		 int                   p1) {
		 GGML_ASSERT(p0 >= 0);
		 GGML_ASSERT(p1 >= 0);

		 GGML_ASSERT(p0 < a->ne[0]); // padding length on each size must be less than the
		 GGML_ASSERT(p1 < a->ne[0]); // existing length of the dimension being padded

		 GGML_ASSERT(ggml_is_contiguous(a));
		 GGML_ASSERT(a->type == GGML_TYPE_F32);

		 ggml_tensor* result = ctx->create(a->type, {
			 a->ne[0] + p0 + p1,
			 a->ne[1],
			 a->ne[2],
			 a->ne[3]
		 });

		 int32_t params[] = { p0, p1 };
		 ggml_set_op_params(*result, params, sizeof(params));

		 result->op = GGML_OP_PAD_REFLECT_1D;
		 result->src.push_back(a);

		 return result;
	 }

	 // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
	 // a: [OC¡AIC, KH, KW]
	 // b: [N, IC, IH, IW]
	 // result: [N, OH, OW, IC*KH*KW]
	 ggml_tensor* ggml_im2col(
		 ggml_context* ctx,
		 ggml_tensor* a,
		 ggml_tensor* b,
		 int                   s0,
		 int                   s1,
		 int                   p0,
		 int                   p1,
		 int                   d0,
		 int                   d1,
		 bool                  is_2D,
		 enum ggml_type        dst_type) {
		 if (is_2D) {
			 GGML_ASSERT(a->ne[2] == b->ne[2]);
		 }
		 else {
			 //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
			 GGML_ASSERT(b->ne[1] == a->ne[1]);
			 GGML_ASSERT(b->ne[3] == 1);
		 }

		 const int64_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
		 const int64_t OW = ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

		 GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
		 GGML_ASSERT((OW > 0) && "b too small compared to a");

		 ggml_tensor* result = ctx->create(dst_type, {
			 is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
			 OW,
			 is_2D ? OH : b->ne[2],
			 is_2D ? b->ne[3] : 1,
		 });
		 int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
		 ggml_set_op_params(*result, params, sizeof(params));

		 result->op = GGML_OP_IM2COL;
		 result->src.push_back(a);
		 result->src.push_back(b);

		 return result;
	 }

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

	 ggml_tensor* ggml_get_rows_back(
		 struct ggml_context* ctx,
		 struct ggml_tensor* a,
		 struct ggml_tensor* b,
		 struct ggml_tensor* c) {
		 GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(*b) && b->type == GGML_TYPE_I32);
		 GGML_ASSERT(ggml_is_matrix(c) && (a->ne[0] == c->ne[0]));

		 // TODO: implement non F32 return
		 //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
		 ggml_tensor* result = ctx->create(GGML_TYPE_F32, { c->ne[0], c->ne[1] });

		 result->op = GGML_OP_GET_ROWS_BACK;
		 result->src.push_back(a);
		 result->src.push_back(b);

		 return result;
	 }

	 ggml_tensor* ggml_argmax(
		 ggml_context* ctx,
		 ggml_tensor* a) {
		 GGML_ASSERT(ggml_is_matrix(a));
		 GGML_ASSERT(a->ne[0] <= INT32_MAX);

		 ggml_tensor* result = ctx->create(GGML_TYPE_I32, { a->ne[1] });

		 result->op = GGML_OP_ARGMAX;
		 result->src.push_back(a);

		 return result;
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

	 // Remove later
	 size_t ggml_backend_reg_count() {
		 return get_reg().backends.size();
	 }

	 ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
		 GGML_ASSERT(index < ggml_backend_reg_count());
		 return get_reg().backends[index].reg;
	 }

}

module;
#include <span>
#include <string>
#include <vector>

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

export module ggml:cpu.device;
import :ds;
import :utility;
import :traits;
import :cpu.buffer;
import :cpu.traits;

export
{
	struct ggml_backend_cpu_device : public ggml_backend_device {
		std::string description = "CPU";
		std::vector<ggml_backend_buffer_type_t> extra_bufts;

		ggml_backend_cpu_device(ggml_backend_reg_t reg) : ggml_backend_device(reg) {
#ifdef __APPLE__
			size_t len = 0;
			if (!sysctlbyname("machdep.cpu.brand_string", NULL, &len, NULL, 0)) {
				description.resize(len);
				sysctlbyname("machdep.cpu.brand_string", &description[0], &len, NULL, 0); // NOLINT
			}
#elif defined(__linux__)
			FILE* f = fopen("/proc/cpuinfo", "r");
			if (f) {
				char buf[1024];
				while (fgets(buf, sizeof(buf), f)) {
					if (strncmp(buf, "model name", 10) == 0) {
						char* p = strchr(buf, ':');
						if (p) {
							p++;
							while (std::isspace(*p)) {
								p++;
							}
							while (std::isspace(p[strlen(p) - 1])) {
								p[strlen(p) - 1] = '\0';
							}
							description = p;
							break;
						}
					}
				}
				fclose(f);
			}
#elif defined(_WIN32)
			HKEY hKey;
			if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
				TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
				0,
				KEY_READ,
				&hKey) == ERROR_SUCCESS) {
				DWORD cpu_brand_size = 0;
				if (RegQueryValueExA(hKey,
					TEXT("ProcessorNameString"),
					NULL,
					NULL,
					NULL,
					&cpu_brand_size) == ERROR_SUCCESS) {
					description.resize(cpu_brand_size);
					if (RegQueryValueExA(hKey,
						TEXT("ProcessorNameString"),
						NULL,
						NULL,
						(LPBYTE)&description[0], // NOLINT
						&cpu_brand_size) == ERROR_SUCCESS) {
						if (description.find('\0') != std::string::npos) {
							description.resize(description.find('\0'));
						}
					}
				}
				RegCloseKey(hKey);
			}
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
			if (ggml_backend_amx_buffer_type()) {
				extra_bufts.push_back(ggml_backend_amx_buffer_type());
			}
#endif

#ifdef GGML_USE_CPU_AARCH64
			if (ggml_backend_cpu_aarch64_buffer_type()) {
				extra_bufts.push_back(ggml_backend_cpu_aarch64_buffer_type());
			}
#endif

		}
		const char* get_name() override
		{
			return "CPU";
		}
		const char* get_description() override
		{
			return description.c_str();
		}
		void get_memory(size_t* free, size_t* total) override
		{
			// TODO
			*free = 0;
			*total = 0;
		}
		enum ggml_backend_dev_type get_type() override
		{
			return GGML_BACKEND_DEVICE_TYPE_CPU;
		}
		void get_props(struct ggml_backend_dev_props* props) override
		{
#if 0
			props->name = ggml_backend_cpu_device_get_name(dev);
			props->description = ggml_backend_cpu_device_get_description(dev);
			props->type = ggml_backend_cpu_device_get_type(dev);
			ggml_backend_cpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
			props->caps = {
				/* .async                 = */ false,
				/* .host_buffer           = */ false,
				/* .buffer_from_host_ptr  = */ true,
				/* .events                = */ false,
			};
#endif
		}
		ggml_backend_t init_backend(const char* params) override
		{
#if 0
			return ggml_backend_cpu_init();
#else
			return {};
#endif
		}
		ggml_backend_buffer_type_t get_buffer_type() override
		{
			static cpu_backend_buffer_type type;
			return &type;
		}
		ggml_backend_buffer_t buffer_from_host_ptr(void* ptr, size_t size, size_t max_tensor_size) override
		{
#if 0
			return ggml_backend_cpu_buffer_from_ptr(ptr, size);
#else
			return {};
#endif
		}

		bool supports_op(const ggml_tensor* op) override
		{
			const ggml_tensor* src0 = op->src[0];
			const ggml_tensor* src1 = op->src[1];

			if (is_one_of(op->op, GGML_OP_NONE, GGML_OP_RESHAPE, GGML_OP_VIEW,
				GGML_OP_PERMUTE, GGML_OP_TRANSPOSE)) {
				return true;
			}

			// check whether ettra buffer type support op
			for (auto extra : extra_bufts) {
				if (extra->supports_op(op))
					return true;
			}

			switch (op->op) {
			case GGML_OP_CPY:
				return is_not_one_of(op->type, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
					GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
					GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M); // missing type_traits.from_float
			case GGML_OP_MUL_MAT:
				return src1->type == GGML_TYPE_F32 || src1->type == ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
			case GGML_OP_ROPE_BACK:
				return op->src[2] == NULL && (op->op_params[2] & 4) == 0;
			case GGML_OP_IM2COL_BACK:
				return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32;
			case GGML_OP_OUT_PROD:
				return (src0->type == GGML_TYPE_F32 || ggml_is_quantized(src0->type)) && src1->type == GGML_TYPE_F32;
			default:
				return true;
			}
		}
		bool supports_buft(ggml_backend_buffer_type_t buft) override
		{
#if 0
			return ggml_backend_buft_is_host(buft) || ggml_backend_cpu_is_extra_buffer_type(buft);
#else
			return false;
#endif
		}
		std::span<const ggml_backend_buffer_type_t> get_extra_bufts() override
		{
			return extra_bufts;
		}
	};

}

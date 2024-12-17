module;
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#define GGML_ASSERT(...) assert(__VA_ARGS__)

export module ggml:gguf;
import :ds;
import :func;

export 
{
	constexpr size_t GGUF_DEFAULT_ALIGNMENT = 32;
	constexpr uint32_t GGUF_VERSION = 3;

	enum gguf_type {
		GGUF_TYPE_UINT8 = 0,
		GGUF_TYPE_INT8 = 1,
		GGUF_TYPE_UINT16 = 2,
		GGUF_TYPE_INT16 = 3,
		GGUF_TYPE_UINT32 = 4,
		GGUF_TYPE_INT32 = 5,
		GGUF_TYPE_FLOAT32 = 6,
		GGUF_TYPE_BOOL = 7,
		GGUF_TYPE_STRING = 8,
		GGUF_TYPE_ARRAY = 9,
		GGUF_TYPE_UINT64 = 10,
		GGUF_TYPE_INT64 = 11,
		GGUF_TYPE_FLOAT64 = 12,
		GGUF_TYPE_COUNT,       // marks the end of the enum
	};
	static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

	const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
		{GGUF_TYPE_UINT8,   sizeof(uint8_t)},
		{GGUF_TYPE_INT8,    sizeof(int8_t)},
		{GGUF_TYPE_UINT16,  sizeof(uint16_t)},
		{GGUF_TYPE_INT16,   sizeof(int16_t)},
		{GGUF_TYPE_UINT32,  sizeof(uint32_t)},
		{GGUF_TYPE_INT32,   sizeof(int32_t)},
		{GGUF_TYPE_FLOAT32, sizeof(float)},
		{GGUF_TYPE_BOOL,    sizeof(int8_t)},
		{GGUF_TYPE_STRING,  0}, // undefined
		{GGUF_TYPE_ARRAY,   0}, // undefined
		{GGUF_TYPE_UINT64,  sizeof(uint64_t)},
		{GGUF_TYPE_INT64,   sizeof(int64_t)},
		{GGUF_TYPE_FLOAT64, sizeof(double)},
	};

	size_t gguf_type_size(enum gguf_type type) {
		auto it = GGUF_TYPE_SIZE.find(type);
		return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
	}

	template <typename T>
	struct type_to_gguf_type;

	template <>
	struct type_to_gguf_type<uint8_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
	};

	template <>
	struct type_to_gguf_type<int8_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_INT8;
	};

	template <>
	struct type_to_gguf_type<uint16_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
	};

	template <>
	struct type_to_gguf_type<int16_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_INT16;
	};

	template <>
	struct type_to_gguf_type<uint32_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
	};

	template <>
	struct type_to_gguf_type<int32_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_INT32;
	};

	template <>
	struct type_to_gguf_type<float> {
		static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
	};

	template <>
	struct type_to_gguf_type<bool> {
		static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
	};

	template <>
	struct type_to_gguf_type<std::string> {
		static constexpr enum gguf_type value = GGUF_TYPE_STRING;
	};

	template <>
	struct type_to_gguf_type<uint64_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
	};

	template <>
	struct type_to_gguf_type<int64_t> {
		static constexpr enum gguf_type value = GGUF_TYPE_INT64;
	};

	template <>
	struct type_to_gguf_type<double> {
		static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
	};

	struct gguf_kv {
		std::string key;

		bool is_array;
		enum gguf_type type;

		std::vector<int8_t>      data;
		std::vector<std::string> data_string;

		template <typename T>
		gguf_kv(const std::string& key, const T value)
			: key(key), is_array(false), type(type_to_gguf_type<T>::value) {
			GGML_ASSERT(!key.empty());
			data.resize(sizeof(T));
			memcpy(data.data(), &value, sizeof(T));
		}

		template <typename T>
		gguf_kv(const std::string& key, const std::vector<T>& value)
			: key(key), is_array(true), type(type_to_gguf_type<T>::value) {
			GGML_ASSERT(!key.empty());
			data.resize(value.size() * sizeof(T));
			for (size_t i = 0; i < value.size(); ++i) {
				const T tmp = value[i];
				memcpy(data.data() + i * sizeof(T), &tmp, sizeof(T));
			}
		}

		gguf_kv(const std::string& key, const std::string& value)
			: key(key), is_array(false), type(GGUF_TYPE_STRING) {
			GGML_ASSERT(!key.empty());
			data_string.push_back(value);
		}

		gguf_kv(const std::string& key, const std::vector<std::string>& value)
			: key(key), is_array(true), type(GGUF_TYPE_STRING) {
			GGML_ASSERT(!key.empty());
			data_string = value;
		}

		const std::string& get_key() const {
			return key;
		}

		const enum gguf_type& get_type() const {
			return type;
		}

		size_t get_ne() const {
			if (type == GGUF_TYPE_STRING) {
				const size_t ne = data_string.size();
				GGML_ASSERT(is_array || ne == 1);
				return ne;
			}
			const size_t type_size = gguf_type_size(type);
			GGML_ASSERT(data.size() % type_size == 0);
			const size_t ne = data.size() / type_size;
			GGML_ASSERT(is_array || ne == 1);
			return ne;
		}

		template <typename T>
		const T& get_val(const size_t i = 0) const {
			GGML_ASSERT(type_to_gguf_type<T>::value == type);
			if constexpr (std::is_same<T, std::string>::value) {
				GGML_ASSERT(data_string.size() >= i + 1);
				return data_string[i];
			}
			const size_t type_size = gguf_type_size(type);
			GGML_ASSERT(data.size() % type_size == 0);
			GGML_ASSERT(data.size() >= (i + 1) * type_size);
			return reinterpret_cast<const T*>(data.data())[i];
		}

		void cast(const enum gguf_type new_type) {
			const size_t new_type_size = gguf_type_size(new_type);
			GGML_ASSERT(data.size() % new_type_size == 0);
			type = new_type;
		}
	};

	struct gguf_tensor_info {
		ggml_tensor t; // for holding the equivalent info
		uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
	};

    struct gguf_context {
        uint32_t version = GGUF_VERSION;

        std::vector<gguf_kv> kv;
        std::vector<gguf_tensor_info> info;

        size_t alignment = GGUF_DEFAULT_ALIGNMENT;
        size_t offset = 0; // offset of `data` from beginning of file
        size_t size = 0; // size of `data` in bytes

        void* data = nullptr;
	public:
		std::optional<size_t> find_key(std::string_view key) const;
		std::span<const gguf_tensor_info> get_infos() const { return info; }
		size_t get_data_offset() const { return offset; }
    };

    std::optional<gguf_context> gguf_init_from_file(const char* fname);

	void constructFrom(const gguf_context &ctx, ggml_context *out);
}

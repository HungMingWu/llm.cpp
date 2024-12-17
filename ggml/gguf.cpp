module;
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <optional>
#include <print>
#include <string_view>
#define GGML_ASSERT(...) assert(__VA_ARGS__)
#define GGML_LOG_ERROR(...)

module ggml;

struct gguf_reader {
    FILE* file;

    gguf_reader(FILE* file) : file(file) {}

    template <typename T>
    bool read(T& dst) const {
        return fread(&dst, 1, sizeof(dst), file) == sizeof(dst);
    }

    template <typename T>
    bool read(std::vector<T>& dst, const size_t n) const {
        dst.resize(n);
        for (size_t i = 0; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                if (!read(tmp)) {
                    return false;
                }
                dst[i] = tmp;
            }
            else {
                if (!read(dst[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool read(bool& dst) const {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum ggml_type& dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = ggml_type(tmp);
        return true;
    }

    bool read(enum gguf_type& dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = gguf_type(tmp);
        return true;
    }

    bool read(std::string& dst) const {
        uint64_t size = -1;
        if (!read(size)) {
            return false;
        }
        dst.resize(size);
        return fread(dst.data(), 1, dst.length(), file) == dst.length();
    }

    bool read(void* dst, const size_t size) const {
        return fread(dst, 1, size, file) == size;
    }
};

static constexpr std::array<char, 4> GGUF_MAGIC{ 'G', 'G', 'U', 'F' };

template<typename T>
bool gguf_read_emplace_helper(const gguf_reader& gr, std::vector<struct gguf_kv>& kv, const std::string& key, const bool is_array, const size_t n) {
    if (is_array) {
        std::vector<T> value;
        try {
            if (!gr.read(value, n)) {
                return false;
            }
        }
        catch (std::length_error&) {
            GGML_LOG_ERROR("{}: encountered length_error while reading value for key '{}'", __func__, key);
            return false;
        }
        catch (std::bad_alloc&) {
            GGML_LOG_ERROR("{}: encountered bad_alloc error while reading value for key '{}'", __func__, key);
            return false;
        }
        kv.emplace_back(key, value);
    }
    else {
        T value;
        if (!gr.read(value)) {
            return false;
        }
        kv.emplace_back(key, value);
    }
    return true;
}
static constexpr std::string_view GGUF_KEY_GENERAL_ALIGNMENT{ "general.alignment" };

uint32_t gguf_get_val_u32(const gguf_context &ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < ctx.kv.size());
    GGML_ASSERT(ctx.kv[key_id].get_ne() == 1);
    return ctx.kv[key_id].get_val<uint32_t>();
}

std::optional<gguf_context> gguf_init_from_file_impl(FILE* file) {
    const gguf_reader gr(file);
    gguf_context ctx;

    bool ok = true;

    // file magic
    {
        std::vector<char> magic;
        ok = ok && gr.read(magic, 4);

        if (!ok) {
            GGML_LOG_ERROR("{}: failed to read magic", __func__);
            return std::nullopt;
        }

        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                GGML_LOG_ERROR("{}: invalid magic characters: '{}{}{}{}', expected 'GGUF'", __func__, magic[0], magic[1], magic[2], magic[3]);
                return std::nullopt;
            }
        }
    }

    // header
    int64_t n_kv = 0;
    int64_t n_tensors = 0;

    if (ok && gr.read(ctx.version)) {
        if (ok && ctx.version == 0) {
            GGML_LOG_ERROR("%s: bad GGUF version: %" PRIu32 "\n", __func__, ctx.version);
            ok = false;
        }

        /*
         * bit layout is different when reading non-native endian models.
         * assuming that the GGUF version is 3, the non-native endian model
         * would read it as 0x30000000. we can use the AND operation against
         * the last 4 hexadecimal digits to check if the model is the same
         * endianness as the host system.
        */
        if (ok && (ctx.version & 0x0000FFFF) == 0x00000000) {
            GGML_LOG_ERROR("%s: failed to load model: this GGUF file version %" PRIu32 " is extremely large, is there a mismatch between the host and model endianness?\n", __func__, ctx->version);
            ok = false;
        }

        if (ok && ctx.version == 1) {
            GGML_LOG_ERROR("%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        if (ok && ctx.version > GGUF_VERSION) {
            GGML_LOG_ERROR("%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    }
    else {
        ok = false;
    }

    if (ok && gr.read(n_kv)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_kv < 0 || n_kv > int64_t(SIZE_MAX / sizeof(gguf_kv))) {
            GGML_LOG_ERROR("{}: number of key value pairs is {} but must be in [0, {}]",
                __func__, n_kv, SIZE_MAX / sizeof(gguf_kv));
            ok = false;
        }
    }
    else {
        ok = false;
    }

    if (!ok) {
        GGML_LOG_ERROR("{}: failed to read header", __func__);
        return std::nullopt;
    }

    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n = 1;

            try {
                ok = ok && gr.read(key);
            }
            catch (std::length_error&) {
                GGML_LOG_ERROR("{}: encountered length_error while reading key {}", __func__, i);
                ok = false;
            }
            catch (std::bad_alloc&) {
                GGML_LOG_ERROR("{}: encountered bad_alloc error while reading key {}", __func__, i);
                ok = false;
            }
            for (size_t j = 0; ok && j < ctx.kv.size(); ++j) {
                if (key == ctx.kv[j].key) {
                    GGML_LOG_ERROR("{}: duplicate key '{}' for tensors {} and {}", __func__, key, j, i);
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }

            ok = ok && gr.read(type);
            if (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }
            if (!ok) {
                break;
            }

            switch (type) {
            case GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<float>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<bool>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<double>(gr, ctx.kv, key, is_array, n); break;
            case GGUF_TYPE_ARRAY:
            default:
            {
                GGML_LOG_ERROR("{}: key '{}' has invalid GGUF type {}", __func__, key, static_cast<int>(type));
                ok = false;
            } break;
            }
        }

        if (!ok) {
            GGML_LOG_ERROR("{}: failed to read key-value pairs", __func__);
            return std::nullopt;
        }
        GGML_ASSERT(int64_t(ctx.kv.size()) == n_kv);

        const std::optional<size_t> alignment_idx = ctx.find_key(GGUF_KEY_GENERAL_ALIGNMENT);
        ctx.alignment = !alignment_idx.has_value() ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx.value());

        if (ctx.alignment == 0 || (ctx.alignment & (ctx.alignment - 1)) != 0) {
            GGML_LOG_ERROR("{}: alignment {} is not a power of 2", __func__, ctx.alignment);
            return std::nullopt;
        }
    }

    // read the tensor info
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            try {
                ok = ok && gr.read(name);
            }
            catch (std::length_error&) {
                GGML_LOG_ERROR("{}: encountered length_error while reading tensor name {}", __func__, i);
                ok = false;
            }
            catch (std::bad_alloc&) {
                GGML_LOG_ERROR("{} encountered bad_alloc error while reading tensor name {}", __func__, i);
                ok = false;
            }
            if (name.length() >= GGML_MAX_NAME) {
                GGML_LOG_ERROR("{}: tensor name {} is too long: {} >= {}", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                break;
            }
            info.t.set_name(name);

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (info.t.name == ctx.info[j].t.name) {
                    GGML_LOG_ERROR("{}: duplicate tensor name '{}' for tensors {} and {}", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }

        // tensor shape
        {
            uint32_t n_dims = -1;
            ok = ok && gr.read(n_dims);
            if (n_dims > GGML_MAX_DIMS) {
                GGML_LOG_ERROR("{}: tensor '{}' has invalid number of dimensions: {} > {}",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                break;
            }
            for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                if (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                // check that all ne are non-negative
                if (info.t.ne[j] < 0) {
                    GGML_LOG_ERROR("{}: tensor '{}' dimension {} has invalid number of elements: {} < 0",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    break;
                }
            }

            // check that the total number of elements is representable
            if (ok && ((INT64_MAX / info.t.ne[1] <= info.t.ne[0]) ||
                (INT64_MAX / info.t.ne[2] <= info.t.ne[0] * info.t.ne[1]) ||
                (INT64_MAX / info.t.ne[3] <= info.t.ne[0] * info.t.ne[1] * info.t.ne[2]))) {

                GGML_LOG_ERROR("{}: total number of elements in tensor '{}' with shape "
                    "({}, {}, {}, {}) is >= {}",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("{}: tensor '{}' has invalid ggml type {} ({})",
                    __func__, info.t.name, static_cast<int>(info.t.type), ggml_type_name(info.t.type));
                ok = false;
                break;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                GGML_LOG_ERROR("{}: tensor '{}' of type {} ({}) has {} elements per row, "
                    "not a multiple of block size ({})",
                    __func__, info.t.name, (int)info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0] * (info.t.ne[0] / blck_size);
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1] * info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx.info.push_back(info);
    }

    if (!ok) {
        GGML_LOG_ERROR("{}: failed to read tensor info", __func__);
        return std::nullopt;
    }
    GGML_ASSERT(int64_t(ctx.info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if (fseek(file, GGML_PAD(ftell(file), ctx.alignment), SEEK_SET) != 0) {
        GGML_LOG_ERROR("{}: failed to seek to beginning of data section", __func__);
        return std::nullopt;
    }

    // store the current file offset - this is where the data section starts
    ctx.offset = ftell(file);

    // compute the total size of the data section, taking into account the alignment
    {
        ctx.size = 0;
        for (size_t i = 0; i < ctx.info.size(); ++i) {
            const gguf_tensor_info& ti = ctx.info[i];
            if (ti.offset != ctx.size) {
                GGML_LOG_ERROR("{}: tensor '{}' has offset {}, expected {}",
                    __func__, ti.t.name, ti.offset, ctx.size);
                GGML_LOG_ERROR("{}: failed to read tensor data", __func__);
                return std::nullopt;
            }
            ctx.size += GGML_PAD(ti.t.nbytes(), ctx.alignment);
        }
    }

    return ctx;
}

std::optional<gguf_context> gguf_init_from_file(const char* fname)
{
    FILE* file = fopen(fname, "rb");

    if (!file) {
        GGML_LOG_ERROR("{}: failed to open GGUF file '{}'", __func__, fname);
        return std::nullopt;
    }

    std::optional<gguf_context> result = gguf_init_from_file_impl(file);
    fclose(file);
    return result;
}

std::optional<size_t> gguf_context::find_key(std::string_view key) const
{
	for (size_t i = 0; i < kv.size(); i++) {
		if (kv[i].get_key() == key) {
			return i;
		}
	}

    return std::nullopt;
}

void constructFrom(const gguf_context &ctx, ggml_context* out)
{
    // create the tensors
    for (const auto& info : ctx.info) {
        const auto& ne = info.t.ne;
        ggml_tensor* cur = out->create(info.t.type, { ne[0], ne[1], ne[2], ne[3] });
        cur->set_name(info.t.name);
    }
}

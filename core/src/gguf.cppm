module;
#include <stdio.h>
#include <stdlib.h>
#include <array>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

module llm:gguf;
import :Binaryreader;
import :ds;
import :Util;
import ggml;

namespace gguf
{
    void gguf_tensor_info_sanitize(struct gguf_tensor_info* info) {
        if (info->type < 0 || info->type >= GGML_TYPE_COUNT) {
            throw make_format_runtime_error("invalid type ({})", static_cast<int>(info->type));
        }

        if (info->name.length() >= GGML_MAX_NAME) {
            throw make_format_runtime_error("tensor '{}' name is too long", info->name);
        }

        for (uint32_t i = 0; i < info->n_dims; ++i) {
            if (info->ne[i] <= 0) {
                throw make_format_runtime_error("invalid number of elements ({})", info->ne[i]);
            }
        }

        // prevent overflow for total number of elements
        if (INT64_MAX / info->ne[1] <= info->ne[0]) {
            throw make_format_runtime_error("invalid number of elements ({})", info->ne[1]);
        }

        if (INT64_MAX / info->ne[2] <= info->ne[0] * info->ne[1]) {
            throw make_format_runtime_error("invalid number of elements ({})", info->ne[2]);
        }

        if (INT64_MAX / info->ne[3] <= info->ne[0] * info->ne[1] * info->ne[2]) {
            throw make_format_runtime_error("invalid number of elements ({})", info->ne[3]);
        }
    }

    template <typename T>
    T gguf_get_val(const gguf_context &ctx, int key_id) {
        return std::get<T>(ctx.kv[key_id].value);
    }

    int gguf_get_n_tensors(const struct gguf_context &ctx) {
        return ctx.header.n_tensors;
    }

    // return -1 if tensor not found
    int gguf_find_tensor(const gguf_context& ctx, std::string_view name) {
        const int n_tensors = gguf_get_n_tensors(ctx);
        for (int i = 0; i < n_tensors; ++i) {
            if (name == ctx.get_tensor_name(i)) {
                return i;
            }
        }

        return -1;
    }

	gguf_context* gguf_init_from_file(const std::filesystem::path& fname, gguf_init_params params) {
        std::ifstream file(fname, std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            //fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
            return NULL;
        }

        BinaryReader reader(std::move(file));

        auto ctx = std::make_unique<gguf_context>();

        // read the header
        try 
        {
            reader
                >> ctx->header.magic
                >> ctx->header.version
                >> ctx->header.n_tensors
                >> ctx->header.n_kv;

            if (ctx->header.version == 1) {
                throw std::runtime_error("GGUFv1 is no longer supported. please use a more up-to-date version");
            }

            if (ctx->header.magic != GGUF_MAGIC) {
                throw make_format_runtime_error("invalid magic characters '{}{}{}{}'", 
                        ctx->header.magic[0],
                        ctx->header.magic[1],
                        ctx->header.magic[2],
                        ctx->header.magic[3]);
            }

            // sanity-checks to prevent from integer/buffer overflows

            const bool ok =
                (ctx->header.n_tensors < std::numeric_limits<size_t>::max() / sizeof(gguf_tensor_info)) &&
                (ctx->header.n_tensors < (std::numeric_limits<size_t>::max() / 2) / ggml_tensor_overhead()) &&
                (ctx->header.n_kv < (std::numeric_limits<size_t>::max() / 2) / sizeof(struct gguf_kv));

            if (!ok) {
                throw std::exception{};
            }
        }
        catch (const std::runtime_error& error) {
            fprintf(stderr, "%s: %s\n", __func__, error.what());
            return nullptr;
        }
        catch (const std::exception&) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            return nullptr;
        }

        // read the kv pairs
        try {
            const uint64_t n_kv = ctx->header.n_kv;
            ctx->kv.resize(n_kv);
            for (auto& kv : ctx->kv) {
                reader >> kv.key >> kv.value;
            }
        }
        catch (const std::runtime_error& error) {
            printf("%s: %s\n", __func__, error.what());
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            return nullptr;
        }

        // read the tensor infos
        if (ctx->header.n_tensors > 0) {
            try {
                ctx->infos.resize(ctx->header.n_tensors);

                for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
                    struct gguf_tensor_info* info = &ctx->infos[i];

                    for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                        info->ne[j] = 1;
                    }

                    reader >> info->name >> info->n_dims;

                    if (info->n_dims > GGML_MAX_DIMS) {
                        throw make_format_runtime_error("invalid number of dimensions ({})", info->n_dims);
                    }

                    for (uint32_t j = 0; j < info->n_dims; ++j) {
                        reader >> info->ne[j];
                    }

                    reader >> info->type >> info->offset;

                    gguf_tensor_info_sanitize(info);

                    // make sure there is no duplicated tensor names
                    for (uint64_t j = 0; j < i; ++j) {
                        if (info->name == ctx->infos[j].name) {
                            throw make_format_runtime_error("duplicated tensor name {}", info->name);
                        }
                    }
                }
            }
            catch (const std::runtime_error& err) {
                fprintf(stderr, "%s: %s\n", __func__, err.what());
                fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                return nullptr;
            }
        }

        int alignment_idx = ctx->find_key("general.alignment");
        if (alignment_idx != -1) {
            ctx->alignment = gguf_get_val<uint32_t>(*ctx, alignment_idx);
        }

        // offset from start of file
        size_t offset = reader.getOffset();

        // we require the data section to be aligned, so take into account any padding
        {
            const size_t offset_pad = offset % ctx->alignment;

            if (offset_pad != 0) {
                offset += ctx->alignment - offset_pad;
                reader.setOffset(offset);
            }
        }

        // store the current file offset - this is where the data section starts
        ctx->offset = offset;

        // compute the total size of the data section, taking into account the alignment
        try {
            for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
                struct gguf_tensor_info* info = &ctx->infos[i];

                const int64_t ne =
                    (int64_t)info->ne[0] *
                    (int64_t)info->ne[1] *
                    (int64_t)info->ne[2] *
                    (int64_t)info->ne[3];

                if (ggml_blck_size(info->type) == 0) {
                    // this tensor type support have been removed:
                    throw make_format_runtime_error("tensor '{}' of type {}: {}",
                        info->name, (int)info->type, ggml_type_name(info->type));
                }

                if (ne % ggml_blck_size(info->type) != 0) {
                    throw make_format_runtime_error(
                        "tensor '{}' of type {} ({}) number of elements ({}) is not a multiple of block size ({})",
                        info->name, (int)info->type, ggml_type_name(info->type), ne, ggml_blck_size(info->type));
                }
                const size_t size_cur = ggml_row_size(info->type, ne);

                ctx->size += GGML_PAD(size_cur, ctx->alignment);

            }
        }
        catch (const std::runtime_error& error) {
            fprintf(stderr, "%s: %s\n", __func__, error.what());
            return nullptr;
        }

        return ctx.release();
	}
}

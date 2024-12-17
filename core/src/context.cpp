module;
#include <stdio.h>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <variant>

module llm;
import :ds;
import :Util;

size_t gguf_context::get_data_offset() const {
    return offset;
}

size_t gguf_context::get_tensor_offset(size_t idx) const {
    return infos[idx].offset;
}

std::string_view gguf_context::get_tensor_name(size_t idx) const {
    return infos[idx].name;
}

int gguf_context::find_key(std::string_view key) const {
    auto it = std::ranges::find(kv, key, &gguf_kv::key);
    return (it == kv.end()) ? -1 : (it - kv.begin());
}

int gguf_context::get_n_kv() const {
    return header.n_kv;
}

int gguf_context::get_version() const {
    return header.version;
}
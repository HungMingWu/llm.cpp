module;
#include <exception>
#include <format>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

module llm:Util;
import :ds;

template <typename... Args>
auto make_format_runtime_error(const std::format_string<Args...> fmt, Args&& ...args) {
	return std::runtime_error(std::vformat(fmt.get(), std::make_format_args(args...)));
}

void replace_all(std::string& s, const std::string& search, const std::string& replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

struct strVisitor {
    template <typename T>
    static std::string toString(const T& value) {
        if constexpr (std::is_same_v<T, std::string>) {
            return value;
        }
        else if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, bool_value>) {
            return value ? "true" : "false";
        }
        else {
            return std::to_string(value);
        }
    }

    template <typename T>
    std::string operator()(const T& value) {
        return toString(value);
    }

    template <typename T>
    std::string operator()(const std::vector<T>& vec) {
        std::string s = "[";
        for (size_t i = 0; i < vec.size(); i++) {
            const auto& v = vec[i];
            std::string itemStr = toString(v);
            if constexpr (std::is_same_v<T, std::string>) {
                // escape quotes
                replace_all(itemStr, "\\", "\\\\");
                replace_all(itemStr, "\"", "\\\"");
                itemStr = std::format(R"("{}")", itemStr);
            }
            s += itemStr;
            if (i < vec.size() - 1) {
                s += ", ";
            }
        }
        s += "]";
        return s;
    }
};

std::string toString(const gguf_value& value)
{
    return std::visit(strVisitor{}, value);
}

struct vecVisitor {
    template <typename T>
    size_t operator()(const T& value) {
        throw std::logic_error("Not a vector type");
    }

    template <typename T>
    size_t operator()(const std::vector<T>& vec) {
        return vec.size();
    }
};

size_t getArrSize(const gguf_value &value)
{
    return std::visit(vecVisitor{}, value);
}
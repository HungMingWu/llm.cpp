#pragma once
#include <format>
#include <functional>
#include <string_view>

enum ggml_log_level {
    GGML_LOG_LEVEL_NONE = 0,
    GGML_LOG_LEVEL_DEBUG = 1,
    GGML_LOG_LEVEL_INFO = 2,
    GGML_LOG_LEVEL_WARN = 3,
    GGML_LOG_LEVEL_ERROR = 4,
    GGML_LOG_LEVEL_CONT = 5, // continue previous log
};

void log_internal(ggml_log_level level, std::string_view output);

template <typename ...Args>
void log_template(ggml_log_level level, std::format_string<Args...> fmt, Args... args)
{
    std::string output = std::vformat(fmt.get(), std::make_format_args(args...));
    log_internal(level, output);
}

// TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
using ggml_log_callback = std::function<void(ggml_log_level, std::string_view)>;

// Set callback for all future logging events.
// If this is not called, or NULL is supplied, everything is output on stderr.
void ggml_log_set_internal(ggml_log_callback ggml_log_set);

template <typename ...Args>
constexpr void GGML_LOG_INFO1(std::format_string<Args...> fmt, Args... args)
{
    return log_template(GGML_LOG_LEVEL_INFO, fmt, std::forward<Args>(args)...);
}
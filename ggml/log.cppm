module;
#include <functional>
#include <print>
#include <string_view>

export module ggml:log;

export
{
    enum ggml_log_level : int {
        GGML_LOG_LEVEL_NONE = 0,
        GGML_LOG_LEVEL_DEBUG = 1,
        GGML_LOG_LEVEL_INFO = 2,
        GGML_LOG_LEVEL_WARN = 3,
        GGML_LOG_LEVEL_ERROR = 4,
        GGML_LOG_LEVEL_CONT = 5, // continue previous log
    };

    void log(ggml_log_level level, std::string_view output);
    using ggml_log_callback = std::function<void(ggml_log_level, std::string_view)>;

    template <typename ...Args>
    void log_template(ggml_log_level level, std::format_string<Args...> fmt, Args... args)
    {
        std::string output = std::vformat(fmt.get(), std::make_format_args(args...));
        log(level, output);
    }

    template <typename ...Args>
    constexpr void GGML_LOG_DEBUG(std::format_string<Args...> fmt, Args... args)
    {
        return log_template(GGML_LOG_LEVEL_DEBUG, fmt, std::forward<Args>(args)...);
    }

    template <typename ...Args>
    constexpr void GGML_LOG_INFO(std::format_string<Args...> fmt, Args... args)
    {
        return log_template(GGML_LOG_LEVEL_INFO, fmt, std::forward<Args>(args)...);
    }

    template <typename ...Args>
    constexpr void GGML_LOG_ERROR(std::format_string<Args...> fmt, Args... args)
    {
        return log_template(GGML_LOG_LEVEL_ERROR, fmt, std::forward<Args>(args)...);
    }

    void ggml_log_set(ggml_log_callback ggml_log_set);
}

module;
#include <string_view>
#include <print>

module ggml;

void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

struct Logger {
    ggml_log_callback logger = ggml_log_callback_default;
};

Logger& get_logger() {
    static Logger logger;
    return logger;
}

void log(ggml_log_level level, std::string_view output)
{
    get_logger().logger(level, output);
}

void ggml_log_set(ggml_log_callback ggml_log_set)
{
    get_logger().logger = ggml_log_set ? ggml_log_set : ggml_log_callback_default;
}

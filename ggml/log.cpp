module;
#include <string_view>
#include <print>

module ggml;

void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

ggml_log_callback g_logger = ggml_log_callback_default;

void log(ggml_log_level level, std::string_view output)
{
    g_logger(level, output);
}

void ggml_log_set(ggml_log_callback ggml_log_set)
{
	g_logger = ggml_log_set ? ggml_log_set : ggml_log_callback_default;
}

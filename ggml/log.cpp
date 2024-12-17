#include <string_view>
#include <print>

#include "log.h"

static void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

static ggml_log_callback g_logger = ggml_log_callback_default;

void ggml_log_set_internal(ggml_log_callback log_callback)
{
    g_logger = log_callback ? log_callback : ggml_log_callback_default;
}

void log_internal(ggml_log_level level, std::string_view output)
{
    g_logger(level, output);
}
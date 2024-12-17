module;
#include "log.h"

export module ggml:log;

export
{
	using ggml_log_level = ::ggml_log_level;
	void ggml_log_set(ggml_log_callback ggml_log_set)
	{
		ggml_log_set_internal(ggml_log_set);
	}
}
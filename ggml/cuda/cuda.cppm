module;
#include <memory>

export module ggml:cuda;
import :ds;

export
{
	std::unique_ptr<ggml_backend> ggml_backend_cuda_init(int device);
}
export module ggml:rpc;
import :ds;

export
{
	ggml_backend_device* ggml_backend_rpc_add_device(const char* endpoint);

}
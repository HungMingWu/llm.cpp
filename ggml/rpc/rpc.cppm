export module ggml:rpc;
import :ds;

export
{
	ggml_backend_reg* ggml_backend_rpc_add_server(const char* endpoint);

}
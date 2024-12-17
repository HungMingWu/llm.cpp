#include <filesystem>

import llm;

int main(int argc, char* argv[])
{
    common_params params;
    params.model = "D:\\stories260K.gguf";
    auto result = common_init_from_params(params);
    return 0;
}

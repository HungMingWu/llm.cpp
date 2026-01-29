#include "common.h"
#include <initializer_list>
#include <string>

static constexpr std::initializer_list<int> cuda_arch_list {__CUDA_ARCH_LIST__};

int ggml_cuda_highest_compiled_arch(const int arch) {
    int result = -1;
    for (auto& cur : cuda_arch_list) {
        if (result <= arch && result > cur)
            result = cur;
    }
    return result;
}

const char* get_arch_list_names()
{
    static std::string arch_list_names = [=]() {
        std::string result;
        bool first = true;
        for (auto& arch : cuda_arch_list) {
            if (!first) {
                result += ",";
			}
			result += std::to_string(arch);
            first = false;
        }
        return result;
    }();
    return arch_list_names.c_str();
}
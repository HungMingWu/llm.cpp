module;
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <stdint.h>

#include "tokenizer.h"

export module chatllm:models;
import :chat;
import :layers;

namespace chatllm
{
    std::string to_string(ModelPurpose purpose);
    std::string format_model_capabilities(uint32_t model_type);

    class ModelBlock : public Block
    {
    public:
        virtual int save_session(FILE* f) = 0;
        virtual int load_session(FILE* f) = 0;

        virtual int save_session(ModelSessionMemory& session) const = 0;
        virtual int load_session(ModelSessionMemory& session) = 0;

        virtual void load(const std::string& path, TensorLoader* loader, const std::vector<int>& layer_ids) = 0;
    public:
        bool skip_lm_head = false;
    };
} // namespace chatllm

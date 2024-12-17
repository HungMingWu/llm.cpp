module;
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <regex>
#include <string>
#include <functional>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <numbers>
#include <unordered_map>

#include "basics.h"
#include "vision_process.h"
#include "JSON.h"

json::JSON json::JSON::_null = json::JSON();

module chatllm;
import ggml;
import :models.base;
import :models.qwen;

namespace chatllm
{
    static ForwardContext* dbg_ctx = nullptr;
    static std::unordered_map<ggml::tensor*, std::string> inspected_set;
    static ggml::tensor* dbg_w = nullptr;

    void set_dbg_ctx(ForwardContext* c)
    {
        dbg_ctx = c;
    }

    void print_tensor_shape(const char* info, ggml::tensor* tensor)
    {
        printf("%s: shape of %s (%p): [%zd, %zd, %zd, %zd] [%zd, %zd, %zd, %zd]\n",
            info, tensor->name.c_str(), tensor->data,
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
    }

    void print_tensor(ggml::tensor* tensor, int offset, const bool full)
    {
        const int PRINT_CNT = 64;
        print_tensor_shape("\n", tensor);

        std::vector<uint8_t > data;
        data.resize(ggml::nbytes(tensor));
        Backend::read_tensor_data(tensor, data.data());

        switch (tensor->type)
        {
        case GGML_TYPE_F32:
        {
            float* p = (float*)data.data();
            const size_t n = ggml::nbytes(tensor) / sizeof(float);
            for (size_t i = 0; i < n; i++)
            {
                if (!full && ((PRINT_CNT < i) && (i < n - PRINT_CNT))) continue;
                float t = p[i];
                //t = ggml_fp16_to_fp32(ggml_fp32_to_fp16(t));
                printf("[%3d] = %+3.18f\n", (int)i, t);
                //printf("[%3d] = %08x\n", (int)i, *(uint32_t *)(p + i));
            }
        }
        break;
        case GGML_TYPE_F16:
        {
            ggml_fp16_t* p = (ggml_fp16_t*)data.data();
            const size_t n = ggml::nbytes(tensor) / sizeof(ggml_fp16_t);
            for (size_t i = 0; i < n; i++)
            {
                if (!full && ((PRINT_CNT < i) && (i < n - PRINT_CNT))) continue;

                printf("[%3d] = %+3.18f\n", (int)i, toFloat32(p[i]));
            }
        }
        break;
        case GGML_TYPE_Q8_0:
        {
#define QK8_0 32
            typedef struct {
                ggml_fp16_t d;       // delta
                int8_t  qs[QK8_0]; // quants
            } block_q8_0;

            char* pp = (char*)data.data();
            for (size_t i = 0; i < ggml::nbytes(tensor) / sizeof(block_q8_0); i++)
            {
                block_q8_0* p = (block_q8_0*)(pp + i * sizeof(block_q8_0));
                float scale = toFloat32(p->d);

                printf("[%3d] =", (int)i * QK8_0);

                for (int j = 0; j < QK8_0; j++)
                {
                    printf(" %+3.15f", p->qs[j] * scale);
                }
                printf("\n");
            }
        }
        break;
        default:
        {
            char* p = (char*)data.data();
            p += offset;
            for (size_t i = 0; i < ggml::nbytes(tensor); i++)
            {
                if ((i & 0xf) == 0) printf("\n%05d: ", (int)i);
                printf("%5d", p[i]);
            }
        }
        break;
        }

        printf("\n");
        exit(-1);
    }

    static bool need_observe_tensor_evaluation_callback(ggml::tensor* tensor, void* user_data)
    {
        return inspected_set.find(tensor) != inspected_set.end();
    }

    static bool observe_tensor_evaluation_callback(ggml::tensor* tensor, void* user_data)
    {
        auto it = inspected_set.find(tensor);
        if (it == inspected_set.end()) return true;

        if (dbg_w)
        {
            printf("\n--------------- dbg_w ----------------------\n");
            print_tensor(dbg_w);

            dbg_w = nullptr;
        }

        printf("\n--------------- %s ----------------------\n", it->second.c_str());
        bool full = true;
        print_tensor(tensor, 0, full);

        return true;
    }

    void dump_weight_tensor(ggml::tensor* tensor)
    {
        dbg_w = tensor;
    }

    void inspect_tensor(ggml::tensor* tensor, const char* format, ...)
    { //return;
        if (nullptr == dbg_ctx) return;
        if (tensor == nullptr) return;

        std::string tag;

        va_list args;
        va_start(args, format);
        int size = vsnprintf(nullptr, 0, format, args) + 1; // +1 for the null terminator
        va_end(args);

        if (size > 0)
        {
            std::unique_ptr<char[]> buffer(new char[size]);

            va_start(args, format);
            vsnprintf(buffer.get(), size, format, args);
            va_end(args);

            tag = buffer.get();
        }

        inspected_set[tensor] = tag;

        dbg_ctx->get_backend_context()->set_eval_observe_callback(need_observe_tensor_evaluation_callback, observe_tensor_evaluation_callback, nullptr);
    }

    ChatModelAccessPoints get_chat_model_access_points(ModelType model_type)
    {
        if (get_model_purpose(model_type) != ModelPurpose::Chat) return 0;

        switch (model_type)
        {
        case MODEL_TYPE_LLAMA_MULTI:
            return ChatModelAccessPoint::Text;
        default:
            break;
        }

        ChatModelAccessPoints tag = model_type >> 24;
        return (tag << 1) | 1;
    }

    static std::string format_access_points(ChatModelAccessPoints bitmap)
    {
        const static std::vector<std::string> names({
            "Text", "Image Input", "Image Output", "Audio Input", "Audio Output", "Video Input", "Video Output"
            });

        std::vector<std::string> aps;
        for (size_t i = 0; i < names.size(); i++)
            if (bitmap & (1 << i))
                aps.push_back(names[i]);
        return utils::join(aps, ", ");
    }

    std::string to_string(ModelPurpose purpose)
    {
        switch (purpose)
        {
        case ModelPurpose::TextEmbedding:
            return "Text Embedding";
        case ModelPurpose::Ranker:
            return "Ranker";
        case ModelPurpose::Chat:
            return "Chat";
        case ModelPurpose::TTS:
            return "TTS";
        case ModelPurpose::ASR:
            return "ASR";
        default:
            CHATLLM_THROW << "unknown model purpose: " << purpose;
            return "???";
        }
    }

    std::string format_model_capabilities(ModelType model_type)
    {
        if (get_model_purpose(model_type) != ModelPurpose::Chat)
        {
            return to_string(get_model_purpose(model_type));
        }
        ChatModelAccessPoints bitmap = get_chat_model_access_points(model_type);
        return format_access_points(bitmap);
    }

    std::string format_model_capabilities(uint32_t model_type)
    {
        return format_model_capabilities((ModelType)model_type);
    }

    std::string to_string(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_CHATGLM:
            return "ChatGLM";
        case MODEL_TYPE_CHATGLM2:
            return "ChatGLM2";
        case MODEL_TYPE_CHATGLM3:
            return "ChatGLM3";
        case MODEL_TYPE_GLM4:
            return "GLM-4";
        case MODEL_TYPE_CODEGEEX2:
            return "CodeGeeX2";
        case MODEL_TYPE_CODEGEEX4:
            return "CodeGeeX4";
        case MODEL_TYPE_CHARACTERGLM:
            return "CharacterGLM";
        case MODEL_TYPE_INTERNLM:
        case MODEL_TYPE_INTERNLM2:
        case MODEL_TYPE_INTERNLM2_1:
        case MODEL_TYPE_INTERNLM3:
            return "InternLM";
        case MODEL_TYPE_LLAMA2:
        case MODEL_TYPE_LLAMA2PLUS:
            return "LlaMA2";
        case MODEL_TYPE_MEGREZ:
            return "Megrez";
        case MODEL_TYPE_FALCON3:
            return "Falcon3";
        case MODEL_TYPE_REKA_FLASH3:
            return "Reka-Flash-3";
        case MODEL_TYPE_CODELLAMA:
            return "CodeLlaMa";
        case MODEL_TYPE_BAICHUAN:
        case MODEL_TYPE_BAICHUANLLAMA:
            return "Baichuan";
        case MODEL_TYPE_BAICHUAN_M1:
            return "Baichuan-M1";
        case MODEL_TYPE_DEEPSEEK:
            return "DeepSeek-LLM";
        case MODEL_TYPE_DEEPSEEK_CODER:
            return "DeepSeek-Coder";
        case MODEL_TYPE_CODEFUSE_DEEPSEEK:
            return "CodeFuse-DeepSeek";
        case MODEL_TYPE_NUMINAMATH:
            return "NumiaMath";
        case MODEL_TYPE_DEEPSEEK_V2:
        case MODEL_TYPE_DEEPSEEK_V2_LIGHT:
            return "DeepSeek-V2";
        case MODEL_TYPE_DEEPSEEK_V3:
        case MODEL_TYPE_DEEPSEEK_V3_LIGHT:
            return "DeepSeek-V3";
        case MODEL_TYPE_DEEPSEEK_V1_MoE:
            return "DeepSeek-V1-MoE";
        case MODEL_TYPE_GIGACHAT:
            return "GigaChat";
        case MODEL_TYPE_YI:
            return "Yi";
        case MODEL_TYPE_MAP_NEO:
            return "MAP-Neo";
        case MODEL_TYPE_PHI2:
        case MODEL_TYPE_PHI2_V2:
            return "Phi-2";
        case MODEL_TYPE_PHI3:
        case MODEL_TYPE_PHI3_SU:
        case MODEL_TYPE_PHI3_SU2:
        case MODEL_TYPE_PHI3_SU3:
            return "Phi-3";
        case MODEL_TYPE_PHI3_MOE:
            return "Phi-3.5 MoE";
        case MODEL_TYPE_PHI4:
        case MODEL_TYPE_PHI4_MINI:
            return "Phi-4";
        case MODEL_TYPE_DOLPHINPHI2:
        case MODEL_TYPE_DOLPHINPHI2_V2:
            return "Dolphin Phi-2";
        case MODEL_TYPE_WIZARDCODER:
            return "WizardCoder";
        case MODEL_TYPE_WIZARDLM:
            return "WizardLM";
        case MODEL_TYPE_WIZARDMATH:
            return "WizardMath";
        case MODEL_TYPE_MISTRAL:
        case MODEL_TYPE_MISTRAL2:
            return "Mistral";
        case MODEL_TYPE_DEEPHERMES3_MISTRAL:
            return "DeepHermes-3-Mistral";
        case MODEL_TYPE_MIXTRAL:
            return "Mixtral MoE";
        case MODEL_TYPE_OPENCHAT:
            return "OpenChat";
        case MODEL_TYPE_NEURALBEAGLE:
            return "NeuralBeagle";
        case MODEL_TYPE_STARLING:
            return "Starling";
        case MODEL_TYPE_WIZARDLM2_MOE:
            return "WizardLM-2-MoE";
        case MODEL_TYPE_QWEN:
            return "QWen";
        case MODEL_TYPE_QWEN2:
        case MODEL_TYPE_QWEN2TIE:
            return "QWen2";
        case MODEL_TYPE_QWEN2MoE:
            return "QWen2-MoE";
        case MODEL_TYPE_MARCO_O1:
            return "Marco-o1";
        case MODEL_TYPE_QWQ:
            return "QwQ";
        case MODEL_TYPE_TIGERBOT:
            return "TigerBot";
        case MODEL_TYPE_BLUELM:
            return "BlueLM";
        case MODEL_TYPE_STABLELM:
            return "StableLM";
        case MODEL_TYPE_ORION:
            return "Orion";
        case MODEL_TYPE_MINICPM:
        case MODEL_TYPE_MINICPM2:
            return "MiniCPM";
        case MODEL_TYPE_MINICPM3:
            return "MiniCPM3";
        case MODEL_TYPE_MINICPM_MoE:
            return "MiniCPM-MoE";
        case MODEL_TYPE_PERSIMMON:
            return "Persimmon";
        case MODEL_TYPE_FUYU:
            return "Fuyu";
        case MODEL_TYPE_GEMMA:
            return "Gemma";
        case MODEL_TYPE_GEMMA2:
            return "Gemma-2";
        case MODEL_TYPE_GEMMA3:
            return "Gemma-3";
        case MODEL_TYPE_COHERE_COMMAND_R:
            return "Command-R";
        case MODEL_TYPE_COHERE_COMMAND_R7B:
            return "Command-R7B";
        case MODEL_TYPE_COHERE_AYA_23:
            return "Aya-23";
        case MODEL_TYPE_GROK_1:
            return "Grok-1";
        case MODEL_TYPE_ZHINAO:
            return "Zhinao";
        case MODEL_TYPE_LLAMA3:
            return "LlaMA3";
        case MODEL_TYPE_LLAMA3_1:
            return "LlaMA3.1";
        case MODEL_TYPE_LLAMA3_2:
            return "LlaMA3.2";
        case MODEL_TYPE_EXAONE:
            return "EXAONE";
        case MODEL_TYPE_BCE_Embedding:
            return "BCE-Embedding";
        case MODEL_TYPE_BCE_ReRanker:
            return "BCE-ReRanker";
        case MODEL_TYPE_BGE_M3:
            return "BGE-M3";
        case MODEL_TYPE_BGE_ReRanker_M3:
            return "BGE-ReRanker-M3";
        case MODEL_TYPE_STARCODER2:
            return "StarCoder2";
        case MODEL_TYPE_XVERSE:
            return "XVERSE";
        case MODEL_TYPE_INDEX:
            return "Index";
        case MODEL_TYPE_OLMoE:
            return "OLMoE";
        case MODEL_TYPE_OLMo2:
            return "OLM-2";
        case MODEL_TYPE_LLAMA_MULTI:
            return "LlaMA-Multi";
        case MODEL_TYPE_SMOLLM:
            return "SmolLM";
        case MODEL_TYPE_LLAMA3_GROQ_TOOL:
            return "LlaMA-Groq-Tool-Use";
        case MODEL_TYPE_ALPHAGEO_LM:
            return "AlphaGeometry-LM";
        case MODEL_TYPE_GRANITE_MoE:
            return "Granite-MoE";
        case MODEL_TYPE_GRANITE:
            return "Granite";
        case MODEL_TYPE_TELECHAT2:
            return "TeleChat2";
        case MODEL_TYPE_HUNYUAN_DENSE:
            return "HuanYuan";
        case MODEL_TYPE_READERLM2:
            return "ReaderLM-v2";
        case MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN:
            return "DeepSeek-R1-Distill-QWen";
        case MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA:
            return "DeepSeek-R1-Distill-LlaMA";
        case MODEL_TYPE_AQUILA2:
            return "Aquila2";
        case MODEL_TYPE_MiniCPM_Embedding_Light:
            return "MiniCPM-Embedding-Light";
        case MODEL_TYPE_MiniCPM_ReRanker_Light:
            return "MiniCPM-ReRanker-Light";
        case MODEL_TYPE_MOONLIGHT:
            return "Moonlight";
        case MODEL_TYPE_INSTELLA:
            return "Instella";
        case MODEL_TYPE_DECILM:
            return "DeciLM";
        case MODEL_TYPE_SOLARPRO:
            return "Solar-Pro";
        case MODEL_TYPE_BAILINGMOE:
            return "Bailing";
        default:
            return "???";
        }
    }

    std::string to_native_string(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_INTERNLM:
        case MODEL_TYPE_INTERNLM2:
        case MODEL_TYPE_INTERNLM2_1:
        case MODEL_TYPE_INTERNLM3:
            return "书生·浦语";
        case MODEL_TYPE_BAICHUAN:
        case MODEL_TYPE_BAICHUANLLAMA:
        case MODEL_TYPE_BAICHUAN_M1:
            return "百川";
        case MODEL_TYPE_PHI2:
        case MODEL_TYPE_PHI2_V2:
        case MODEL_TYPE_PHI3:
        case MODEL_TYPE_PHI3_SU:
        case MODEL_TYPE_PHI3_SU2:
        case MODEL_TYPE_PHI3_SU3:
        case MODEL_TYPE_PHI3_MOE:
        case MODEL_TYPE_PHI4:
            return "Φ";
        case MODEL_TYPE_QWEN:
        case MODEL_TYPE_QWEN2:
        case MODEL_TYPE_QWEN2TIE:
        case MODEL_TYPE_QWEN2MoE:
        case MODEL_TYPE_QWQ:
            return "通义千问";
        case MODEL_TYPE_TIGERBOT:
            return "虎博";
        case MODEL_TYPE_BLUELM:
            return "蓝心";
        case MODEL_TYPE_NEURALBEAGLE:
            return "🐶";
        case MODEL_TYPE_COHERE_COMMAND_R:
            return "⌘-R";
        case MODEL_TYPE_ZHINAO:
            return "360智脑";
        case MODEL_TYPE_XVERSE:
            return "元象";
        case MODEL_TYPE_MEGREZ:
            return "无穹天权";
        case MODEL_TYPE_TELECHAT2:
            return "星辰";
        case MODEL_TYPE_ORION:
            return "猎户星空";
        case MODEL_TYPE_HUNYUAN_DENSE:
            return "混元";
        default:
            return "";
        }
    }

    static std::string regex_replace(const std::string& input, const std::regex& regex,
        std::function<std::string(const std::smatch&)> format)
    {
        std::ostringstream oss;
        size_t last_index = 0;
        for (auto it = std::sregex_iterator(input.begin(), input.end(), regex); it != std::sregex_iterator(); it++)
        {
            oss << it->prefix() << format(*it);
            last_index = it->position() + it->length();
        }
        oss << input.substr(last_index);
        return oss.str();
    }

    template <class Config, class Embedding, class LayerBlock, class FinalBlock, typename... _Types> class EmbeddingModel : public Model
        <Config, Embedding, FinalBlock, LayerBlock, _Types...>
    {
    public:
        typedef Model<Config, Embedding, FinalBlock, LayerBlock, _Types...> Base;
        typedef HeterogeneousModel BaseBase;

        EmbeddingModel() = default;

        EmbeddingModel(InitContext* ctx, const Config& config, _Types... layer_args)
            : Base(ctx, config, nullptr, std::forward<_Types>(layer_args)...)
        {
            Base::set_final_steps(std::make_unique<EmbeddingPoolingFinalSteps>());
        }
    };

#if 0
    namespace glm
    {
#include "../models/chatglm.cpp"
    }

    namespace codegeex
    {
#include "../models/codegeex.cpp"
    }

    namespace internlm
    {
#include "../models/internlm.cpp"
    }

    namespace llama
    {
#include "../models/llama.cpp"
    }

    namespace codellama
    {
#include "../models/codellama.cpp"
    }

    namespace deepseek
    {
#include "../models/deepseek.cpp"
    }

    namespace deepseek_coder
    {
#include "../models/deepseek_coder.cpp"
    }

    namespace baichuan
    {
#include "../models/baichuan.cpp"
    }

    namespace yi
    {
#include "../models/yi.cpp"
    }
    namespace phi
    {
#include "../models/phi.cpp"
    }

    namespace mistral
    {
#include "../models/mistral.cpp"
    }

    namespace openchat
    {
#include "../models/openchat.cpp"
    }

    namespace starling
    {
#include "../models/starling.cpp"
    }

    namespace wizard
    {
#include "../models/wizard.cpp"
    }

    namespace qwen
    {
#include "../models/qwen.cpp"
    }

    namespace tigerbot
    {
#include "../models/tigerbot.cpp"
    }

    namespace bluelm
    {
#include "../models/bluelm.cpp"
    }

    namespace dolphinphi2
    {
#include "../models/dolphinphi2.cpp"
    }

    namespace stablelm
    {
#include "../models/stablelm.cpp"
    }

    namespace neuralbeagle
    {
#include "../models/neuralbeagle.cpp"
    }

    namespace bce
    {
#include "../models/bce.cpp"
    }

    namespace bge
    {
#include "../models/bge.cpp"
    }

    namespace orion
    {
#include "../models/orion.cpp"
    }

    namespace minicpm
    {
#include "../models/minicpm.cpp"
    }

    namespace adept
    {
#include "../models/adept.cpp"
    }

    namespace gemma
    {
#include "../models/gemma.cpp"
    }

    namespace codefuse
    {
#include "../models/codefuse.cpp"
    }

    namespace characterglm
    {
#include "../models/characterglm.cpp"
    }

    namespace cohere
    {
#include "../models/cohere.cpp"
    }

    namespace grok
    {
#include "../models/grok.cpp"
    }

    namespace zhinao
    {
#include "../models/zhinao.cpp"
    }

    namespace starcoder
    {
#include "../models/starcoder.cpp"
    }

    namespace m_a_p
    {
#include "../models/m_a_p.cpp"
    }

    namespace xverse
    {
#include "../models/xverse.cpp"
    }

    namespace index
    {
#include "../models/index.cpp"
    }

    namespace numinamath
    {
#include "../models/numinamath.cpp"
    }

    namespace smollm
    {
#include "../models/smollm.cpp"
    }

    namespace groq
    {
#include "../models/groq.cpp"
    }

    namespace allenai
    {
#include "../models/allenai.cpp"
    }

    namespace alphageo
    {
#include "../models/alphageo.cpp"
    }

    namespace granite
    {
#include "../models/granite.cpp"
    }

    namespace megrez
    {
#include "../models/megrez.cpp"
    }

    namespace falcon
    {
#include "../models/falcon.cpp"
    }

    namespace exaone
    {
#include "../models/exaone.cpp"
    }

    namespace telechat
    {
#include "../models/telechat.cpp"
    }

    namespace jina
    {
#include "../models/jina.cpp"
    }

    namespace hunyuan
    {
#include "../models/hunyuan.cpp"
    }

    namespace moonshot
    {
#include "../models/moonshot.cpp"
    }

    namespace instella
    {
#include "../models/instella.cpp"
    }

    namespace reka
    {
#include "../models/reka.cpp"
    }

    namespace hermes
    {
#include "../models/hermes.cpp"
    }

    namespace decilm
    {
#include "../models/decilm.cpp"
    }

    namespace solar
    {
#include "../models/solar.cpp"
    }

    namespace gigachat
    {
#include "../models/gigachat.cpp"
    }

    namespace aquila
    {
#include "../models/aquila.cpp"
    }

    namespace bailing
    {
#include "../models/bailing.cpp"
    }

    namespace kimi
    {
#include "../models/kimi.cpp"
    }

    namespace apriel
    {
#include "../models/apriel.cpp"
    }

    namespace orpheus
    {
#include "../models/orpheus.cpp"
    }

    namespace oute
    {
#include "../models/oute.cpp"
    }
#endif
    template <class Config>
    void load_config(ModelLoader& loader, Config& config, const ModelObject::extra_args& args)
    {
        if (0 == loader.offset_config)
            loader.offset_config = loader.tell();
        else
            loader.seek(loader.offset_config, SEEK_SET);

        // load config
        config = loader.read_basic<Config>();
        if (args.max_length > 0)
            config.max_length = args.max_length;
        if (args.re_quantize >= 0)
            config.dtype = (ggml::type)args.re_quantize;

        loader.offset_tokenizer = loader.tell();
    }

    template <class Config, class Tokenizer>
    Tokenizer* load_tokenizer(ModelLoader& loader, Config& config)
    {
        loader.seek(loader.offset_tokenizer, SEEK_SET);

        // load tokenizer
        Tokenizer* tokenizer = new Tokenizer(config);
        tokenizer->load(loader.get_reader(), config.vocab_size);
        tokenizer->load_config(loader.meta_json);
        loader.load_all_tensors();

        return tokenizer;
    }

    static void parse_slice(std::vector<int>& values, const std::string& s, int num_hidden_layers)
    {
        int spec[3] = { 0, num_hidden_layers, 1 };
        int index = 0;
        std::string t(s);
        if (t.size() > 0) index = 1;

        while ((t.size() > 0) && (index <= 3))
        {
            size_t pos = t.find_first_of(':');
            std::string part = t.substr(0, pos);
            if (part.size() > 0)
                spec[index - 1] = atoi(part.c_str());
            if (pos == std::string::npos) break;
            index++;
            t = t.substr(pos + 1);
        }

        if (index < 1) return;

        if (index == 1)
        {
            values.push_back(spec[0]);
            return;
        }

        if (spec[2] == 0) return;
        if (spec[0] < 0) spec[0] += num_hidden_layers;
        if (spec[1] < 0) spec[1] += num_hidden_layers;

        if (spec[2] > 0)
        {
            for (int i = spec[0]; i < spec[1]; i += spec[2])
            {
                values.push_back(i);
            }
        }
        else
        {
            for (int i = spec[0]; i > spec[1]; i += spec[2])
                values.push_back(i);
        }
    }

    static int parse_int_lists(std::vector<int>& values, const std::string& s, int num_hidden_layers)
    {
        const static std::regex r(R""([\r\n]+)"");
        std::string t(s);
        while (t.size() > 0)
        {
            size_t pos = t.find_first_of(',');
            parse_slice(values, t.substr(0, pos), num_hidden_layers);
            if (pos == std::string::npos) break;
            t = t.substr(pos + 1);
        }
        return 0;
    }

    template <class Config, class ConditionalGeneration>
    ConditionalGeneration* load_model(ModelLoader& loader, Config& config, const ModelObject::extra_args& args)
    {
        std::vector<int> layers;
        if (args.layer_spec.size() > 0)
        {
            parse_int_lists(layers, args.layer_spec, config.num_hidden_layers);
            config.num_hidden_layers = (int)layers.size();
        }

        RuntimeConfig rt_config(args.moe_on_cpu, args.n_threads, args.batch_size, (ggml::type)args.cache_type);
        rt_config.model_gpu_layers = args.model_n_gpu_layers;

        // load model
        ConditionalGeneration* model = new ConditionalGeneration(config, rt_config);
        model->set_type(loader.model_type);
        model->set_names(loader.model_name, loader.model_native_name);
        if (layers.size() > 0)
            model->set_layer_ids(layers);

        loader.push_allocator_manager(model->get_alloc_manager());
        model->load_more(loader.meta_json);
        model->load(loader);

        return model;
    }

    template <class Config, class ConditionalGeneration>
    ConditionalGeneration* load_model(ModelLoader& loader, const ModelObject::extra_args& args)
    {
        Config config;

        load_config<Config>(loader, config, args);

        return load_model<Config, ConditionalGeneration>(loader, config, args);
    }

    template <class Config, class Tokenizer, class ConditionalGeneration>
    bool load_model(ModelLoader& loader, ModelFactory::Result& result, const ModelObject::extra_args& args)
    {
        // load config
        Config config;

        load_config<Config>(loader, config, args);

        // load tokenizer
        result.tokenizer = std::unique_ptr<BaseTokenizer>(load_tokenizer<Config, Tokenizer>(loader, config));

#if (0)
        // test tokenizer
        std::vector<int> ids = result.tokenizer->encode("\nAlice:");
        for (auto x : ids) std::cout << x << ", ";
        std::cout << std::endl;

        //ids = {0,1,2,195,196};
        std::cout << result.tokenizer->decode(ids) << std::endl;
        exit(-1);
#endif
        // load model
        result.model = std::unique_ptr<AbstractModel>(load_model<Config, ConditionalGeneration>(loader, config, args));

        result.model->set_tokenizer(result.tokenizer.get());

        return true;
    }

    static void load_file_header(ModelLoader& loader)
    {
        // load magic
        loader.seek(0, SEEK_SET);
        std::string magic = loader.read_string(4);

        if (magic == "ggml")
        {
            loader.ff = ModelLoader::FileFormat::GGML;
        }
        else if (magic == "ggmm")
        {
            loader.ff = ModelLoader::FileFormat::GGMM;
            const uint32_t GGMM_VER = 1;
            uint32_t ver = loader.read_basic<uint32_t>();
            CHATLLM_CHECK(GGMM_VER == ver) << "GGMM file version error: " << ver;

            loader.ggml_header = loader.read_basic<ModelLoader::GGMMHeader>();
            if ((int64_t)loader.ggml_header.offset_config > loader.tell())
            {
                loader.meta = loader.read_string(loader.ggml_header.offset_config - loader.tell());
                size_t last_non_null = loader.meta.find_last_not_of('\0');
                if (last_non_null != std::string::npos)
                    loader.meta.erase(last_non_null + 1);
                else;
                loader.meta_json = json::JSON::Load(loader.meta);
                auto name = loader.meta_json["model_name"];
                auto native = loader.meta_json["model_native_name"];
                if (name.IsString())
                    loader.model_name = name.ToString();
                if (native.IsString())
                    loader.model_native_name = native.ToString();
            }
            loader.seek(loader.ggml_header.offset_config, 0);
        }
        else
        {
            CHATLLM_CHECK(false) << "model file is broken (bad magic): " << magic;
        }

        loader.model_type = loader.read_basic<int>();
        loader.version = loader.read_basic<int>();

        if (loader.model_name.size() < 1)
            loader.model_name = to_string((ModelType(loader.model_type)));
        if (loader.model_native_name.size() < 1)
            loader.model_native_name = to_native_string((ModelType(loader.model_type)));
    }

    std::string ModelFactory::load_info(ModelLoader& loader)
    {
        load_file_header(loader);
        BaseConfig config = loader.read_basic<BaseConfig>();
        std::ostringstream oss;
        auto model_type = (ModelType)(loader.model_type);
        auto purpose = get_model_purpose(model_type);
        oss << "Model name  : " << loader.model_name;
        if (loader.model_native_name.size() > 0)
            oss << " (" << loader.model_native_name << ")";
        oss << " (" << std::hex << std::setw(8) << std::setfill('0') << model_type << ")" << std::dec;
        oss << std::endl;

        oss << "Model type  : " << to_string(purpose);
        if (ModelPurpose::Chat == purpose)
            oss << " {" << format_access_points(get_chat_model_access_points(model_type)) << "}";
        oss << std::endl;

        oss << "File version: " << loader.version << " (" << ModelLoader::ff_to_str(loader.ff) << ")" << std::endl
            << "Quantization: " << ggml::type_to_str(config.dtype) << std::endl;

        oss << std::endl
            << "vocab_size          : " << config.vocab_size << std::endl
            << "hidden_size         : " << config.hidden_size << std::endl
            << "num_attention_heads : " << config.num_attention_heads << std::endl
            << "num_hidden_layers   : " << config.num_hidden_layers << std::endl
            << "intermediate_size   : " << config.intermediate_size << std::endl
            << "max_length          : " << config.max_length << std::endl << std::endl

            << "bos_token_id        : " << config.bos_token_id << std::endl
            << "eos_token_id        : " << config.eos_token_id << std::endl
            << "pad_token_id        : " << config.pad_token_id << std::endl
            << "sep_token_id        : " << config.sep_token_id << std::endl;

        if (loader.meta.size() > 0)
        {
            ggml::log(GGML_LOG_LEVEL_INFO, "meta: %s", loader.meta.c_str());
        }

        return oss.str();
    }

    bool ModelFactory::load(ModelLoader& loader, Result& result, const ModelObject::extra_args& args)
    {
        load_file_header(loader);
        return ModelFactory::load(loader.model_type, loader.version, loader, result, args);
    }

#define ALL_MODELS  \
        CASE(CHATGLM,               glm::v1, 1)                 \
        CASE(CHATGLM2,              glm::v2, 1)                 \
        CASE(CHATGLM3,              glm::v3, 1)                 \
        CASE(CODEGEEX2,             codegeex::v2, 1)            \
        CASE(CHARACTERGLM,          characterglm, 1)            \
        CASE(GLM4,                  glm::v4, 1)                 \
        CASE(CODEGEEX4,             codegeex::v4, 1)            \
        CASE(GLM4_0414,             glm::glm4_0414, 1)          \
                                                                \
        CASE(INTERNLM,              internlm::v1, 1)            \
        CASE(INTERNLM2,             internlm::v2, 1)            \
        CASE(INTERNLM2_1,           internlm::v2_1, 1)          \
        CASE(INTERNLM3,             internlm::v3, 1)            \
                                                                \
        CASE(LLAMA2,                llama::v2, 1)               \
        CASE(LLAMA3,                llama::v3, 1)               \
        CASE(CODELLAMA,             codellama, 1)               \
        CASE(LLAMA2PLUS,            llama::v2_plus, 1)          \
        CASE(LLAMA_MULTI,           llama::multi, 1)            \
        CASE(LLAMA3_1,              llama::v3_1, 1)             \
        CASE(LLAMA3_2,              llama::v3_2, 1)             \
        CASE(MEGREZ,                megrez::chat, 1)            \
        CASE(FALCON3,               falcon::v3, 1)              \
        CASE(REKA_FLASH3,           reka::flash, 1)             \
        CASE(EXAONE,                exaone, 1)                  \
        CASE(DEEPSEEK_R1_DISTILL_LLAMA, llama::ds_r1_distill, 1)\
        CASE(LLAMA4,                llama::v4, 1)               \
                                                                \
        CASE(DEEPSEEK,              deepseek::v1, 1)            \
        CASE(DEEPSEEK_CODER,        deepseek_coder, 1)          \
        CASE(CODEFUSE_DEEPSEEK,     codefuse::deepseek, 1)      \
        CASE(NUMINAMATH,            numinamath, 1)              \
        CASE(DEEPSEEK_V2_LIGHT,     deepseek::v2_light, 1)      \
        CASE(DEEPSEEK_V2,           deepseek::v2, 1)            \
        CASE(DEEPSEEK_V3_LIGHT,     deepseek::v3_light, 1)      \
        CASE(DEEPSEEK_V1_MoE,       deepseek::v1_moe, 1)        \
        CASE(BAILINGMOE,            bailing::moe, 1)            \
        CASE(XVERSEMOE,             xverse::moe, 1)             \
                                                                \
        CASE(BAICHUANLLAMA,         baichuan::_7b, 1)           \
        CASE(BAICHUAN,              baichuan::larger, 1)        \
        CASE(BAICHUAN_M1,           baichuan::m1, 1)            \
                                                                \
        CASE(YI,                    yi, 1)                      \
        CASE(MAP_NEO,               m_a_p::neo, 1)              \
                                                                \
        CASE(PHI2,                  phi::v2::v1, 1)             \
        CASE(PHI2_V2,               phi::v2::v2, 1)             \
        CASE(PHI3,                  phi::v3, 1)                 \
        CASE(PHI3_SU,               phi::v3_su, 1)              \
        CASE(PHI3_SU2,              phi::v3_su2, 1)             \
        CASE(PHI3_SU3,              phi::v3_su3, 1)             \
        CASE(PHI3_MOE,              phi::v3_moe, 1)             \
        CASE(PHI4,                  phi::v4, 1)                 \
        CASE(PHI4_MINI,             phi::v4_mini, 1)            \
                                                                \
        CASE(WIZARDCODER,           wizard::coder, 1)           \
        CASE(WIZARDLM,              wizard::lm, 1)              \
        CASE(WIZARDMATH,            wizard::math, 1)            \
                                                                \
        CASE(MISTRAL,               mistral::mistral, 1)        \
        CASE(OPENCHAT,              openchat, 1)                \
        CASE(MIXTRAL,               mistral::mixtral, 1)        \
        CASE(MISTRAL2,              mistral::mistral2, 1)       \
        CASE(DEEPHERMES3_MISTRAL,   hermes::_mistral, 1)        \
                                                                \
        CASE(QWEN,                  qwen::v1, 2)                \
        CASE(QWEN2,                 qwen::v2, 1)                \
        CASE(QWEN2MoE,              qwen::v2_moe, 1)            \
        CASE(QWEN2TIE,              qwen::v2_tie, 1)            \
        CASE(MARCO_O1,              qwen::marco_o1, 1)          \
        CASE(QWQ,                   qwen::qwq, 1)               \
        CASE(READERLM2,             jina::readerlm, 1)          \
        CASE(DEEPSEEK_R1_DISTILL_QWEN, qwen::ds_r1_distill, 1)  \
        CASE(DEEPSEEK_R1_DISTILL_QWEN3,qwen::ds_r1_distill_v3, 1)\
        CASE(AQUILA2,               aquila::v2, 1)              \
        CASE(QWEN2_5_VL,            qwen::v2_5_vl, 1)           \
        CASE(QWEN3,                 qwen::v3, 1)                \
                                                                \
        CASE(TIGERBOT,              tigerbot, 1)                \
                                                                \
        CASE(BLUELM,                bluelm, 1)                  \
                                                                \
        CASE(DOLPHINPHI2,           dolphinphi2::v1, 1)         \
                                                                \
        CASE(STABLELM,              stablelm, 1)                \
                                                                \
        CASE(NEURALBEAGLE,          neuralbeagle, 1)            \
        CASE(STARLING,              starling, 1)                \
        CASE(WIZARDLM2_MOE,         wizard::moe, 1)             \
                                                                \
        CASE(ORION,                 orion, 1)                   \
                                                                \
        CASE(MINICPM,               minicpm::v1, 1)             \
        CASE(MINICPM2,              minicpm::v2, 1)             \
        CASE(MINICPM_MoE,           minicpm::moe, 1)            \
        CASE(MINICPM3,              minicpm::v3, 1)             \
                                                                \
        CASE(PERSIMMON,             adept::persimmon, 1)        \
        CASE(FUYU,                  adept::fuyu, 1)             \
                                                                \
        CASE(GEMMA,                 gemma::v1, 1)               \
        CASE(GEMMA2,                gemma::v2, 2)               \
        CASE(GEMMA3,                gemma::v3, 1)               \
        CASE(GEMMA3Vis,             gemma::v3, 1)               \
                                                                \
        CASE(COHERE_COMMAND_R,      cohere::command_r, 1)       \
        CASE(COHERE_AYA_23,         cohere::aya_23, 1)          \
        CASE(COHERE_COMMAND_R7B,    cohere::v2, 1)              \
                                                                \
        CASE(GROK_1,                grok::v1, 1)                \
                                                                \
        CASE(ZHINAO,                zhinao, 1)                  \
                                                                \
        CASE(STARCODER2,            starcoder::v2, 1)           \
                                                                \
        CASE(XVERSE,                xverse::dense, 1)           \
                                                                \
        CASE(INDEX,                 index, 1)                   \
                                                                \
        CASE(SMOLLM,                smollm, 1)                  \
        CASE(LLAMA3_GROQ_TOOL,      groq, 1)                    \
                                                                \
        CASE(OLMoE,                 allenai::moe, 1)            \
        CASE(OLMo2,                 allenai::dense, 1)          \
                                                                \
        CASE(ALPHAGEO_LM,           alphageo, 1)                \
                                                                \
        CASE(GRANITE_MoE,           granite::moe, 1)            \
        CASE(GRANITE,               granite::dense, 1)          \
                                                                \
        CASE(TELECHAT2,             telechat::v2, 1)            \
                                                                \
        CASE(HUNYUAN_DENSE,         hunyuan::dense, 1)          \
                                                                \
        CASE(MOONLIGHT,             moonshot::moonlight, 1)     \
                                                                \
        CASE(INSTELLA,              instella, 1)                \
                                                                \
        CASE(DECILM,                decilm, 1)                  \
                                                                \
        CASE(SOLARPRO,              solar::pro, 1)              \
                                                                \
        CASE(GIGACHAT,              gigachat, 1)                \
                                                                \
        CASE(KIMI_VL,               kimi::vl, 1)                \
                                                                \
        CASE(APRIEL,                apriel, 1)                  \
                                                                \
        CASE(BCE_Embedding,         bce::embedding, 1)          \
        CASE(BCE_ReRanker,          bce::ranker, 1)             \
        CASE(BGE_M3,                bge::embedding, 1)          \
        CASE(BGE_ReRanker_M3,       bge::ranker, 1)             \
        CASE(MiniCPM_Embedding_Light,   minicpm::emb_light, 1)  \
        CASE(MiniCPM_ReRanker_Light,    minicpm::ranker_light, 1)\
        CASE(ORPHEUS_TTS,               orpheus::tts, 1)        \
        CASE(OUTE_TTS_LLAMA,            oute::tts_llama, 1)     \
        CASE(OUTE_TTS_QWEN3,            oute::tts_qwen3, 1)


    AbstractModel* ModelFactory::load_model_again(ModelLoader& loader, const ModelObject::extra_args& args)
    {
        int a = 1;
        // TO FIX
#if 0
        int model_type = loader.model_type;
        int version = loader.version;

#define CASE(TYPE, ns, ver)         \
            case MODEL_TYPE_ ##TYPE:        \
            {                               \
                CHATLLM_CHECK(version == ver) << "only support version " #ver " for now but got " << version;   \
                return load_model<ns::Config,                                                                   \
                                  ns::ConditionalGeneration>(loader, args);                                     \
            }

        switch ((ModelType)model_type)
        {
            ALL_MODELS
        default:
            CHATLLM_THROW << "invalid model type " << model_type;
            return nullptr;
        }

#undef CASE
#endif
        return nullptr;
    }

    bool ModelFactory::load(int model_type, int version, ModelLoader& loader, Result& result, const ModelObject::extra_args& args)
    {
#define CASE(TYPE, ns, ver)         \
            case MODEL_TYPE_ ##TYPE:        \
            {                               \
                CHATLLM_CHECK(version == ver) << "only support version " #ver " for now but got " << version;   \
                return load_model<ns::Config,                                                                   \
                                  ns::Tokenizer,                                                                \
                                  ns::ConditionalGeneration>(loader, result, args);                             \
            }

        switch ((ModelType)model_type)
        {
            // TO FIX
#if 0
            ALL_MODELS
#endif
        case MODEL_TYPE_QWEN2TIE:
            {
                CHATLLM_CHECK(version == 1) << "only support version 1 for now but got " << version;
                return load_model<qwen::v2_tie::Config,
                    qwen::v2_tie::Tokenizer,
                    qwen::v2_tie::ConditionalGeneration>(loader, result, args);
            }
        default:
            CHATLLM_THROW << "invalid model type " << model_type;
            return false;
        }

#undef CASE
    }

} // namespace chatllm

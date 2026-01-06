module;
#include <algorithm>
#include <cmath>
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
#include <print>
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
        clear_inspected_tensors();
    }

    void unset_dbg_ctx(ForwardContext* c)
    {
        if (c == dbg_ctx)
            dbg_ctx = nullptr;
    }

    ForwardContext::~ForwardContext()
    {
        unset_dbg_ctx(this);
    }

    void print_tensor_shape(const char* info, ggml::tensor* tensor)
    {
        std::println("{}: {} shape of {} ({}): [{}, {}, {}, {}] [{}, {}, {}, {}]",
            info, ggml::type_to_str(ggml::type_of(tensor)).c_str(),
            tensor->name, tensor->data,
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
            bool flag = false;
            for (size_t i = 0; i < n; i++)
            {
                if (std::isnan(p[i]) || std::isinf(p[i]))
                {
                    flag = true;
                    break;
                }
            }
            //if (!flag) break;
            for (size_t i = 0; i < n; i++)
            {
                if (!full && ((PRINT_CNT < i) && (i < n - PRINT_CNT))) continue;
                float t = p[i];
                //t = ggml_fp16_to_fp32(ggml_fp32_to_fp16(t));
                std::println("[{:3}] = {:+3.18f}", i, t);
                //printf("[%3d] = %08x\n", (int)i, *(uint32_t *)(p + i));
            }
            if (flag) exit(-1);
        }
        break;
        case GGML_TYPE_F16:
        {
            ggml_fp16_t* p = (ggml_fp16_t*)data.data();
            const size_t n = ggml::nbytes(tensor) / sizeof(ggml_fp16_t);
            for (size_t i = 0; i < n; i++)
            {
                if (!full && ((PRINT_CNT < i) && (i < n - PRINT_CNT))) continue;

                std::println("[{:3}] = {:+3.18f}", i, toFloat32(p[i]));
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
    }

    static bool need_observe_tensor_evaluation_callback(ggml::tensor* tensor)
    {
        return inspected_set.find(tensor) != inspected_set.end();
    }

    static bool observe_tensor_evaluation_callback(ggml::tensor* tensor)
    {
        auto it = inspected_set.find(tensor);
        if (it == inspected_set.end()) return true;

        if (dbg_w)
        {
            std::print("\n--------------- dbg_w");
            print_tensor(dbg_w);

            dbg_w = nullptr;
        }

        std::print("\n--------------- {}", it->second);
        bool full = true;
        print_tensor(tensor, 0, full);

        return true;
    }

    void dump_weight_tensor(ggml::tensor* tensor)
    {
        dbg_w = tensor;
    }

    void clear_inspected_tensors(void)
    {
        inspected_set.clear();
    }

    void inspect_tensor(ggml::tensor* tensor, std::string tag)
    { //return;
        if (nullptr == dbg_ctx) return;
        if (tensor == nullptr) return;

        if (!ggml_is_contiguous(tensor))
        {
            tensor = ggml::cont(dbg_ctx, tensor);
            ggml::build_forward_expand(dbg_ctx, tensor);
        }

        // if (strstr(tag.c_str(), "gen_vision_model.decoder.mid.0") == nullptr) return;

        inspected_set[tensor] = std::move(tag);

        dbg_ctx->get_backend_context()->set_eval_observe_callback(need_observe_tensor_evaluation_callback, observe_tensor_evaluation_callback);
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

    DynamicBlock::DynamicBlock() : Block(), _loaded(false)
    {
    }

    bool DynamicBlock::is_loaded(void) const
    {
        return _loaded;
    }

    BaseMediaProjectedEmbeddingGeneration::BaseMediaProjectedEmbeddingGeneration(const RuntimeConfig& runtime_config)
        : _ctx(&backend_context),
        n_threads(runtime_config.n_threads),
        max_embedding_num(-1)
    {
    }

    bool BaseMediaProjectedEmbeddingGeneration::load(ModelLoader& loader)
    {
        if (model.get()) model->load("vision_model.", &loader);
        return true;
    }

    bool BaseMediaProjectedEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON& config)
    {
        return true;
    }

    void BaseMediaProjectedEmbeddingGeneration::generate(const GenerationConfig& gen_config, BaseTokenizer* tok, ggml::type dtype, std::vector<uint8_t>& buf)
    {
        if ((model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!model->is_loaded()) return;

        for (auto& media : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, media, buf);
        }
    }

    void BaseMediaProjectedEmbeddingGeneration::write_media_tensor(ggml::tensor* media_emb, const BaseTokenizer::MediaAsEmbeddingVector& media)
    {
        Backend::write_tensor_data(media_emb, media.data.data(), 0, media.data.size() * sizeof(media.data[0]));
    }

    bool BaseMediaProjectedEmbeddingGeneration::run_model(const GenerationConfig& gen_config, BaseTokenizer* tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector& media, std::vector<uint8_t>& buf)
    {
        ForwardContext ctx(&backend_context);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor* media_emb = make_media_tensor(&ctx, media);

        set_dbg_ctx(&ctx);

        auto r = model->forward(&ctx, media_emb);

        if (ggml::type_of(r) != dtype)
        {
            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
            ggml::tensor* t = ggml::new_tensor_like(&ctx, dtype, r);
            r = ggml::cpy(&ctx, r, t);
        }

        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        write_media_tensor(media_emb, media);

        ctx.compute();

        size_t offset = buf.size();
        buf.resize(offset + ggml::nbytes(r));
        Backend::read_tensor_data(r, buf.data() + offset);
        ctx.reset();

        return true;
    }

    static void load_file_header(ModelLoader& loader)
    {
        // load magic
        loader.seek(0, SEEK_SET);
        std::string magic = loader.read_string(4);
        bool is_ggml = false;

        if (magic == "ggml")
        {
            loader.ff = ModelLoader::FileFormat::GGML;
            is_ggml = true;
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

        if (is_ggml)
        {
            loader.model_name = to_string((ModelType(loader.model_type)));
            loader.model_native_name = to_native_string((ModelType(loader.model_type)));
        }
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
        \

    class ModelLoadRegistry
    {
    public:
        static void reg(int model_type, BaseImplModelLoader* loader);
        static BaseImplModelLoader* get_loader(int model_type);
    protected:
        ModelLoadRegistry() {}
        ModelLoadRegistry(const ModelLoadRegistry&) = delete;
        static ModelLoadRegistry* get();
        std::unordered_map<int, BaseImplModelLoader*> loaders;
    };

    ModelLoadRegistry* ModelLoadRegistry::get()
    {
        static ModelLoadRegistry* obj = new ModelLoadRegistry();
        return obj;
    }

    void ModelLoadRegistry::reg(int model_type, BaseImplModelLoader* loader)
    {
        get()->loaders.insert(std::pair(model_type, loader));
    }

    BaseImplModelLoader* ModelLoadRegistry::get_loader(int model_type)
    {
        auto obj = get();
        auto v = obj->loaders.find(model_type);
        return v != obj->loaders.end() ? v->second : nullptr;
    }

    BaseImplModelLoader::BaseImplModelLoader(int model_type, int version)
        : version(version)
    {
        ModelLoadRegistry::reg(model_type, this);
    }

    std::unique_ptr<AbstractModel> ModelFactory::load_model_again(ModelLoader& loader, const ModelObject::extra_args& args)
    {
        int model_type = loader.model_type;
        int version = loader.version;

        auto _loader = ModelLoadRegistry::get_loader(model_type);
        CHATLLM_CHECK(_loader != nullptr) << "invalid model type " << model_type;
        CHATLLM_CHECK(version == _loader->version) << "only support version " << _loader->version << " for now but got " << version;
        return _loader->load_model(loader, args);
    }

    bool ModelFactory::load(int model_type, int version, ModelLoader& loader, Result& result, const ModelObject::extra_args& args)
    {
        auto _loader = ModelLoadRegistry::get_loader(model_type);
        CHATLLM_CHECK(_loader != nullptr) << "invalid model type 0x" << std::hex << model_type;
        CHATLLM_CHECK(version == _loader->version) << "only support version " << _loader->version << " for now but got " << version;
        return _loader->load_model(loader, result, args);
    }

} // namespace chatllm

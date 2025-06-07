module;
#include <random>
#include "../basics.h"
#include "../tokenizer.h"

#define MAKE_TYPE_TAG(v)            (((uint32_t)(v) >> 1) << 24)
#define MODEL_TYPE_TAG_ChatImageIn                              MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::ImageInput)
#define MODEL_TYPE_TAG_ChatImageInVideoIn                       MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::ImageInput + ChatModelAccessPoint::VideoInput)
#define MODEL_TYPE_TAG_ChatImageInVideoInAudioInAudioOut        MAKE_TYPE_TAG(ChatModelAccessPoint::Text + ChatModelAccessPoint::ImageInput + ChatModelAccessPoint::VideoInput + ChatModelAccessPoint::AudioInput + ChatModelAccessPoint::AudioOutput)

module chatllm:models.base;
import :chat;
import :models;
import :layers;

namespace chatllm
{
    class ForwardContext;
    ForwardContext* dbg_ctx = nullptr;

    enum ModelType
    {
        MODEL_TYPE_CHATGLM = 1,
        MODEL_TYPE_CHATGLM2 = 2,
        MODEL_TYPE_CHATGLM3 = 3,
        MODEL_TYPE_CODEGEEX2 = 4,
        MODEL_TYPE_CHARACTERGLM = 5,
        MODEL_TYPE_GLM4 = 6,
        MODEL_TYPE_CODEGEEX4 = 7,
        MODEL_TYPE_GLM4_0414 = 8,

        MODEL_TYPE_INTERNLM = 0x100,
        MODEL_TYPE_INTERNLM2 = 0x101, // extended model, supporting 7B & 20B
        MODEL_TYPE_INTERNLM2_1 = 0x102,
        MODEL_TYPE_INTERNLM3 = 0x103,

        MODEL_TYPE_LLAMA2 = 0x150,
        MODEL_TYPE_CODELLAMA = 0x151,
        MODEL_TYPE_WIZARDCODER = 0x152,
        MODEL_TYPE_WIZARDLM = 0x153,
        MODEL_TYPE_WIZARDMATH = 0x154,
        MODEL_TYPE_TIGERBOT = 0x155,
        MODEL_TYPE_LLAMA2PLUS = 0x156,
        MODEL_TYPE_MEGREZ = 0x157,
        MODEL_TYPE_FALCON3 = 0x158,
        MODEL_TYPE_REKA_FLASH3 = 0x159,

        MODEL_TYPE_BAICHUANLLAMA = 0x200,
        MODEL_TYPE_BAICHUAN = 0x201,
        MODEL_TYPE_BAICHUAN_M1 = 0x202,

        MODEL_TYPE_DEEPSEEK = 0x300,
        MODEL_TYPE_DEEPSEEK_CODER = 0x301,
        MODEL_TYPE_CODEFUSE_DEEPSEEK = 0x302,
        MODEL_TYPE_NUMINAMATH = 0x303,
        MODEL_TYPE_DEEPSEEK_V2_LIGHT = 0x320,
        MODEL_TYPE_DEEPSEEK_V2 = 0x321,
        MODEL_TYPE_DEEPSEEK_V3_LIGHT = 0x322,   // DOES NOT EXIST
        MODEL_TYPE_DEEPSEEK_V3 = 0x323,
        MODEL_TYPE_DEEPSEEK_V1_MoE = 0x324,
        MODEL_TYPE_GIGACHAT = 0x325,
        MODEL_TYPE_BAILINGMOE = 0x326,
        MODEL_TYPE_XVERSEMOE = 0x327,

        MODEL_TYPE_YI = 0x400,
        MODEL_TYPE_MAP_NEO = 0x401,

        MODEL_TYPE_PHI2 = 0x500,
        MODEL_TYPE_PHI2_V2 = 0x501,
        MODEL_TYPE_PHI3 = 0x520,
        MODEL_TYPE_PHI3_SU = 0x521,
        MODEL_TYPE_PHI3_SU2 = 0x522,
        MODEL_TYPE_PHI3_SU3 = 0x523,
        MODEL_TYPE_PHI3_MOE = 0x530,
        MODEL_TYPE_PHI4 = 0x531,
        MODEL_TYPE_PHI4_MINI = 0x532,

        MODEL_TYPE_DOLPHINPHI2 = 0x510,
        MODEL_TYPE_DOLPHINPHI2_V2 = 0x511,

        MODEL_TYPE_MISTRAL = 0x600,
        MODEL_TYPE_MIXTRAL = 0x601,
        MODEL_TYPE_OPENCHAT = 0x602,
        MODEL_TYPE_NEURALBEAGLE = 0x603,
        MODEL_TYPE_STARLING = 0x604,
        MODEL_TYPE_WIZARDLM2_MOE = 0x605,
        MODEL_TYPE_MISTRAL2 = 0x606,
        MODEL_TYPE_DEEPHERMES3_MISTRAL = 0x607,

        MODEL_TYPE_QWEN = 0x700,
        MODEL_TYPE_QWEN2 = 0x710,
        MODEL_TYPE_QWEN2TIE = 0x711,
        MODEL_TYPE_QWEN2MoE = 0x750,
        MODEL_TYPE_MARCO_O1 = 0x751,
        MODEL_TYPE_QWQ = 0x752,
        MODEL_TYPE_READERLM2 = 0x753,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN = 0x754,
        MODEL_TYPE_QWEN3 = 0x755,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_QWEN3 = 0x756,

        MODEL_TYPE_BLUELM = 0x800,

        MODEL_TYPE_STABLELM = 0x900,

        MODEL_TYPE_ORION = 0x1000,

        MODEL_TYPE_MINICPM = 0x1100,
        MODEL_TYPE_MINICPM2 = 0x1101,
        MODEL_TYPE_MINICPM_MoE = 0x1102,
        MODEL_TYPE_MINICPM3 = 0x1110,

        MODEL_TYPE_PERSIMMON = 0x1200,
        MODEL_TYPE_FUYU = 0x1201,

        MODEL_TYPE_GEMMA = 0x1300,
        MODEL_TYPE_GEMMA2 = 0x1301,
        MODEL_TYPE_GEMMA3 = 0x1302,

        MODEL_TYPE_COHERE_COMMAND_R = 0x1400,
        MODEL_TYPE_COHERE_AYA_23 = 0x1401,
        MODEL_TYPE_COHERE_COMMAND_R7B = 0x1402,

        MODEL_TYPE_GROK_1 = 0x1500,

        MODEL_TYPE_ZHINAO = 0x1600,

        MODEL_TYPE_LLAMA3 = 0x1700,
        MODEL_TYPE_SMOLLM = 0x1701,
        MODEL_TYPE_LLAMA3_GROQ_TOOL = 0x1702,
        MODEL_TYPE_LLAMA3_1 = 0x1703,
        MODEL_TYPE_LLAMA3_2 = 0x1704,
        MODEL_TYPE_EXAONE = 0x1705,
        MODEL_TYPE_DEEPSEEK_R1_DISTILL_LLAMA = 0x1706,
        MODEL_TYPE_AQUILA2 = 0x1707,

        MODEL_TYPE_STARCODER2 = 0x1800,

        MODEL_TYPE_XVERSE = 0x1900,

        MODEL_TYPE_INDEX = 0x1a00,

        MODEL_TYPE_OLMoE = 0x1b00,
        MODEL_TYPE_OLMo2 = 0x1b01,

        MODEL_TYPE_ALPHAGEO_LM = 0x1c00,

        MODEL_TYPE_GRANITE_MoE = 0x1d00,
        MODEL_TYPE_GRANITE = 0x1d01,

        MODEL_TYPE_TELECHAT2 = 0x1e00,

        MODEL_TYPE_HUNYUAN_DENSE = 0x1f00,

        MODEL_TYPE_MOONLIGHT = 0x2000,

        MODEL_TYPE_INSTELLA = 0x2100,

        MODEL_TYPE_DECILM = 0x2200,

        MODEL_TYPE_SOLARPRO = 0x2300,

        MODEL_TYPE_APRIEL = 0x2400,

        MODEL_TYPE_BCE_Embedding = 0x10000100,
        MODEL_TYPE_BCE_ReRanker = 0x10000101,
        MODEL_TYPE_BGE_M3 = 0x10000102,
        MODEL_TYPE_BGE_ReRanker_M3 = 0x10000103,
        MODEL_TYPE_MiniCPM_Embedding_Light = 0x10000104,
        MODEL_TYPE_MiniCPM_ReRanker_Light = 0x10000105,
        MODEL_TYPE_ORPHEUS_TTS = 0x10000106,
        MODEL_TYPE_OUTE_TTS_LLAMA = 0x10000107,
        MODEL_TYPE_OUTE_TTS_QWEN3 = 0x10000108,

        MODEL_TYPE_LLAMA_MULTI = 0x20000001,

        MODEL_TYPE_LLAMA4 = MODEL_TYPE_TAG_ChatImageIn + 0x0000001,
        MODEL_TYPE_GEMMA3Vis = MODEL_TYPE_TAG_ChatImageIn + 0x0000011,

        MODEL_TYPE_QWEN2_5_VL = MODEL_TYPE_TAG_ChatImageInVideoIn + 0x0000001,

        MODEL_TYPE_KIMI_VL = MODEL_TYPE_TAG_ChatImageInVideoIn + 0x0000100,
    };

    class ModelBlock : public Block
    {
    public:
        virtual int save_session(FILE* f) = 0;
        virtual int load_session(FILE* f) = 0;

        virtual int save_session(ModelSessionMemory& session) const = 0;
        virtual int load_session(ModelSessionMemory& session) = 0;

        virtual void load(const std::string& path, TensorLoader* loader, const std::vector<int>& layer_ids) = 0;
    };

    struct RuntimeConfig
    {
        bool moe_on_cpu;
        int n_threads;
        int batch_input_size;
        ggml::type cache_type;
        std::map<std::string, std::string> model_gpu_layers;
        RuntimeConfig(bool moe_on_cpu, int n_threads, int batch_input_size, ggml::type cache_type) :
            moe_on_cpu(moe_on_cpu), n_threads(n_threads), batch_input_size(batch_input_size), cache_type(cache_type)
        {
        }
    };

    class ForwardContext : public ComputeContext
    {
    public:
        ForwardContext(BackendContext* backend_context) : ComputeContext(backend_context)
        {
        }

        struct ggml_context* get_ctx() override { return gctx.get(); }
        ggml_cgraph* get_cgraph(void) override { return &gf; }

    public:
        GGMLContext gctx;
        ggml_cgraph gf;
    };

    ModelPurpose get_model_purpose(ModelType model_type)
    {
        switch (model_type)
        {
        case MODEL_TYPE_BCE_Embedding:
        case MODEL_TYPE_BGE_M3:
        case MODEL_TYPE_MiniCPM_Embedding_Light:
            return ModelPurpose::TextEmbedding;
        case MODEL_TYPE_BCE_ReRanker:
        case MODEL_TYPE_BGE_ReRanker_M3:
        case MODEL_TYPE_MiniCPM_ReRanker_Light:
            return ModelPurpose::Ranker;
        case MODEL_TYPE_ORPHEUS_TTS:
        case MODEL_TYPE_OUTE_TTS_LLAMA:
        case MODEL_TYPE_OUTE_TTS_QWEN3:
            return ModelPurpose::TTS;
        default:
            return ModelPurpose::Chat;
        }
    }

    class Sampler
    {
    public:
        static const int ABORT = -1;
        virtual ~Sampler() = default;
    public:
        virtual void seed(int x)
        {
            gen.seed((unsigned int)x);
        }

        virtual void reset() {}

        virtual int sampling(float* logits, const int vocab_size) = 0;
    protected:
        std::mt19937 gen;
    };

    class NonGreedySampler : public Sampler
    {
    public:
        NonGreedySampler(float temperature, float presence_penalty, int top_k)
            : inv_temp(0.0f), inv_presence_penalty(0.0f), presence_penalty(presence_penalty), top_k(top_k)
        {
            temp_en = fabs(temperature - 1.0f) > 1e-5f;
            if (temp_en) inv_temp = 1.f / temperature;

            presence_penalty_en = fabs(presence_penalty - 1.0f) > 1e-5f;
            if (presence_penalty_en) inv_presence_penalty = 1.0f / presence_penalty;
        }

        void reset() override
        {
            g.clear();
        }

        int sampling(float* logits, const int vocab_size) override
        {
            g.resize(vocab_size, 0);
            token_scores.resize(vocab_size);

            if (temp_en)
            {
                for (int i = 0; i < vocab_size; i++)
                    logits[i] *= inv_temp;
            }

            if (presence_penalty_en)
            {
                for (int i = 0; i < vocab_size; i++)
                {
                    if (g[i] > 0)
                        logits[i] *= logits[i] > 0 ? inv_presence_penalty : presence_penalty;
                }
            }

            for (int i = 0; i < vocab_size; i++)
            {
                token_scores[i] = { .id = i, .score = logits[i] };
            }

            // top_k sampling
            if (0 < top_k && top_k < (int)token_scores.size())
            {
                std::nth_element(token_scores.begin(), token_scores.begin() + top_k, token_scores.end(),
                    std::greater<TokenIdScore>());
                token_scores.resize(top_k);
            }

            do_sampling(logits, vocab_size);

            if (token_scores.size() < 1)
                return ABORT;

            // sample next token
            for (size_t i = 0; i < token_scores.size(); i++)
            {
                logits[i] = token_scores[i].score;
            }

            std::discrete_distribution<> dist(logits, logits + token_scores.size());
            int next_token_id = token_scores[dist(gen)].id;

            g[next_token_id] += 1;
            return next_token_id;
        }

    protected:
        struct TokenIdScore
        {
            int id;
            float score;

            bool operator<(const TokenIdScore& other) const { return score < other.score; }
            bool operator>(const TokenIdScore& other) const { return score > other.score; }
        };

        void sampling_softmax_inplace(TokenIdScore* first, TokenIdScore* last)
        {
            float max_score = std::max_element(first, last)->score;
            float sum = 0.f;
            for (TokenIdScore* p = first; p != last; p++)
            {
                float s = std::exp(p->score - max_score);
                p->score = s;
                sum += s;
            }
            float inv_sum = 1.f / sum;
            for (TokenIdScore* p = first; p != last; p++)
            {
                p->score *= inv_sum;
            }
        }

        virtual void do_sampling(float* logits, const int vocab_size) = 0;
        bool temp_en;
        bool presence_penalty_en;
        float inv_temp;
        float inv_presence_penalty;
        float presence_penalty;
        int top_k;
        std::vector<TokenIdScore> token_scores;
        std::vector<int> g;
    };

    class TopPSampler : public NonGreedySampler
    {
    public:
        TopPSampler(float temperature, float presence_penalty, int top_k, float top_p)
            : NonGreedySampler(temperature, presence_penalty, top_k), top_p(top_p)
        {
        }

    protected:
        void do_sampling(float* next_token_logits, const int vocab_size) override
        {
            // top_p sampling
            if (0.f < top_p && top_p < 1.f)
            {
                std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>()); // hot code!
                sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());

                float cumsum = 0.f;
                for (size_t i = 0; i < token_scores.size(); i++)
                {
                    cumsum += token_scores[i].score;
                    if (cumsum >= top_p)
                    {
                        token_scores.resize(i + 1);
                        break;
                    }
                }
            }

            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        }

    protected:
        const float top_p;
    };

    // Reference:
    // https://www.trentonbricken.com/Tail-Free-Sampling/#tail-free-sampling-algorithm
    class FreeTailSampler : public NonGreedySampler
    {
    public:
        FreeTailSampler(float temperature, float presence_penalty, int top_k, float z)
            : NonGreedySampler(temperature, presence_penalty, top_k), z(z)
        {
        }

    protected:

        void do_sampling(float* next_token_logits, const int vocab_size) override
        {
            if (token_scores.size() < 3) return;

            sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
            std::sort(token_scores.begin(), token_scores.end(), std::greater<TokenIdScore>()); // hot code!

            snd_d.resize(token_scores.size() - 2);
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                snd_d[i] = token_scores[i].score + token_scores[i + 2].score - 2 * token_scores[i + 1].score;
            }

            // abs, then norm
            float sum = 1e-6f;
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                snd_d[i] = fabs(snd_d[i]);
                sum += snd_d[i];
            }
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                snd_d[i] /= sum;
            }

            float cdf = 0.0;
            for (size_t i = 0; i < snd_d.size(); i++)
            {
                cdf += snd_d[i];
                if (cdf > z)
                {
                    token_scores.resize(i + 1);
                    break;
                }
            }
        }

    protected:
        const float z;
        std::vector<float> snd_d;
    };

    class GreedySampler : public Sampler
    {
    public:
        int sampling(float* logits, const int vocab_size) override
        {
            return (int)(std::max_element(logits, logits + vocab_size) - logits);
        }
    };

    class SamplerFactory
    {
    public:
        static Sampler* Create(const GenerationConfig& gen_config, int seed)
        {
            Sampler* r = nullptr;
            if (gen_config.do_sample)
            {
                if (gen_config.sampling == "top_p")
                    r = new TopPSampler(gen_config.temperature, gen_config.presence_penalty, gen_config.top_k, gen_config.top_p);
                else if (gen_config.sampling == "tfs")
                    r = new FreeTailSampler(gen_config.temperature, gen_config.presence_penalty, gen_config.top_k, gen_config.tfs_z);
                else if (gen_config.sampling != "greedy")
                    CHATLLM_CHECK(false) << "unknown sampling algorithm: " << gen_config.sampling;
            }

            if (nullptr == r)
                r = new GreedySampler();

            r->seed(seed);
            return r;
        }
    };

    class BaseModelForConditionalGeneration : public BaseModel
    {
    public:
        BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, const RuntimeConfig& runtime_config, size_t GRAPH_SIZE = 4096)
            : BaseModel(model_type, get_model_purpose(model_type)),
            transformer(nullptr),
            GRAPH_SIZE(GRAPH_SIZE),
            batch_input(runtime_config.batch_input_size), logit_scale(-1.0f),
            w_ctx_(&backend_context),
            config_(config)
        {
            w_ctx_.cache_dtype = runtime_config.cache_type;
            prepare(runtime_config);
            for (int i = 0; i < config.num_hidden_layers; i++)
                layer_ids.push_back(i);
        }

        virtual ~BaseModelForConditionalGeneration() = default;

        void set_layer_ids(const std::vector<int>& ids) override
        {
            CHATLLM_CHECK((int)ids.size() == config_.num_hidden_layers) << "length(layer_ids) must be " << config_.num_hidden_layers;
            layer_ids.clear();
            for (auto x : ids)
                layer_ids.push_back(x);
        }

        int get_max_length(void) override
        {
            return config_.max_length;
        }

        void shift_memory(int keep) override
        {
            if (keep >= n_past) return;

            transformer->shift_cache(n_past - keep, n_past);
            BaseModel::shift_memory(keep);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            return transformer->get_param_num(effective_only);
        }

        std::vector<int> generate(const std::vector<int>& input_ids, const GenerationConfig& gen_config,
            const bool continuous,
            bool& completed,
            ModelPerfInfo* performance,
            int gen_max_tokens,
            BaseStreamer* streamer = nullptr)
        {
            CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
                << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
                << config_.max_length << ")";

            //for (int i = 0; i < (int)input_ids.size(); i++)
            //    printf("%d, ", input_ids[i]);
            //printf("\nn_past = %d, %d\n\n", n_past, continuous);

            std::unique_ptr<Sampler> sampler = std::unique_ptr<Sampler>(SamplerFactory::Create(gen_config, _seed));

            aborted = false;

            std::vector<int> curr_input_ids(input_ids);

            std::vector<int> output_ids;
            output_ids.reserve(gen_config.max_length);

            if (!continuous)
            {
                n_past = 0;
                n_past_offset = 0;
            }

            completed = false;

            transformer->set_ctx((int)input_ids.size());
            int next_output_idx = 0;

            if (gen_max_tokens > 0)
                gen_max_tokens = n_past + (int)curr_input_ids.size() + gen_max_tokens;

            bool first_call = true;

            if (performance)
                performance->Reset();

            before_generate(gen_config);

            while (!aborted && !completed && (n_past + (int)curr_input_ids.size() < gen_config.max_length))
            {
                std::vector<float> lm_logits;
                if (!generate_next_token(curr_input_ids, gen_config, lm_logits))
                {
                    ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
                    aborted = true;
                    break;
                }

                if (first_call)
                {
                    if (performance)
                        performance->Accumulate(ModelPerfInfo::Type::Prompt, curr_input_ids.size());
                    first_call = false;
                }

                //#define DISABLE_CACHE
#ifndef DISABLE_CACHE
                n_past += (int)curr_input_ids.size();
                curr_input_ids.clear();
#endif
                float* logits = lm_logits.data();
                const size_t tok_num = lm_logits.size() / config_.vocab_size;

                for (size_t tok_idx = 0; (tok_idx < tok_num) && !aborted; tok_idx++, logits += config_.vocab_size)
                {
                    int next_token_id = sampler->sampling(logits, config_.vocab_size);

                    //printf("\n>>next = %d<<\n", next_token_id);
                    //fflush(stdout);
                    //exit(-1);

                    if (next_token_id == Sampler::ABORT)
                    {
                        aborted = true;
                        break;
                    }

                    curr_input_ids.push_back(next_token_id);

                    int pop_output = 0;
                    int keep_idx = 0;
                    output_ids.push_back(next_token_id);

                    if (is_output_terminated(output_ids, keep_idx, pop_output))
                    {
                        while (pop_output-- > 0)
                            output_ids.pop_back();
                        keep_idx = (int)output_ids.size();
                        completed = true;
                    }

                    if (streamer)
                    {
                        if (keep_idx > (int)output_ids.size())
                            keep_idx = (int)output_ids.size();
                        for (; next_output_idx < keep_idx; next_output_idx++)
                            streamer->put({ output_ids[next_output_idx] });
                    }

                    if ((gen_max_tokens > 0) && ((n_past + (int)curr_input_ids.size() >= gen_max_tokens)))
                    {
                        aborted = true;
                        break;
                    }
                }
            }

            if (aborted && !completed)
                completed = true;

            if (performance)
                performance->Accumulate(ModelPerfInfo::Type::Generation, output_ids.size() - curr_input_ids.size());

            after_generate();

            //printf("\nn_past = %d\n", n_past);
            return output_ids;
        }

        void text_embedding(const GenerationConfig& gen_config, const std::vector<int>& input_ids,
            std::vector<float>& embedding) override
        {
            auto r = run_model(input_ids, gen_config, 0, embedding);
            if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
        }

        float qa_rank(const GenerationConfig& gen_config, const std::vector<int>& input_ids) override
        {
            std::vector<float> output;
            auto r = run_model(input_ids, gen_config, 0, output);
            if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
            CHATLLM_CHECK(output.size() == 1) << "ouput must be scaler";

            return output[0];
        }

        bool generate_next_token(const std::vector<int>& input_ids, const GenerationConfig& gen_config, std::vector<float>& lm_logits) override
        {
            int batch = batch_input > 1 ? batch_input : 1;

            const int* p = input_ids.data();
            int remain = (int)input_ids.size();
            int past = n_past + n_past_offset;

            for (; (remain > batch) && !aborted; p += batch, remain -= batch, past += batch)
            {
                if (!run_model(p, batch, gen_config, past, lm_logits))
                    return false;
            }

            return run_model(p, remain, gen_config, past, lm_logits);
        }

        int save_session(FILE* f) const override
        {
            int r = BaseModel::save_session(f);
            if (r != 0)
                return r;
            return transformer->save_session(f);
        }

        int load_session(FILE* f) override
        {
            int r = BaseModel::load_session(f);
            if (r != 0) return r;
            return transformer->load_session(f);
        }

        int save_session(ModelSessionMemory& session) const override
        {
            int r = BaseModel::save_session(session);
            if (r != 0)
                return r;
            return transformer->save_session(session);
        }

        int load_session(ModelSessionMemory& session) override
        {
            int r = BaseModel::load_session(session);
            if (r != 0) return r;
            return transformer->load_session(session);
        }

        void prepare(const RuntimeConfig& rt_config)
        {
            w_ctx_.user_options.moe_on_cpu = rt_config.moe_on_cpu;
            backend_context.init(rt_config.model_gpu_layers, "main", config_.num_hidden_layers, GRAPH_SIZE, rt_config.n_threads);
        }

        LayerAllocatorManager* get_alloc_manager(void) override
        {
            return &backend_context.layer_allocators;
        }

        void load(ModelLoader& loader) override
        {
            transformer->load("model.", &loader, layer_ids);
        }

    protected:
        virtual void before_generate(const GenerationConfig& gen_config)
        {
        }

        virtual void after_generate(void)
        {
            tokenizer->media_emb.clear();
        }

        virtual void do_build_graph(ForwardContext& ctc, const std::vector<int>& input_ids,
            const GenerationConfig& gen_config,
            int past)
        {

        }

        virtual bool before_initial_run(const int ids_count,
            const GenerationConfig& gen_config,
            int past)
        {
            //printf("before_initial_run 1\n");
            //backend_context.show_buffer_sizes();

            ForwardContext ctx(&backend_context);
            ctx.gctx = GGMLContext();

            ggml::tensor* input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);

            ggml::tensor* r = transformer->forward(&ctx, input_ids_tensor, past);

            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);

            ggml::build_forward_expand(&ctx, r);

            bool s = ctx.reserve_memory();

            //printf("before_initial_run 2\n");
            //backend_context.show_buffer_sizes();

            return s;
        }

        bool run_model(const std::vector<int>& input_ids,
            const GenerationConfig& gen_config,
            int past,
            std::vector<float>& output)
        {
            return run_model(input_ids.data(), (int)input_ids.size(), gen_config, past, output);
        }

        virtual bool run_model(const int* input_ids, const int ids_count,
            const GenerationConfig& gen_config,
            int past,
            std::vector<float>& output)
        {
            if (!initial_run)
            {
                initial_run = true;
                int past = gen_config.max_length - ids_count;
                if (past < 0) past = 0;
                if (!before_initial_run(ids_count, gen_config, past))
                    return false;
            }

            ForwardContext ctx(&backend_context);
            ctx.user_options = w_ctx_.user_options;

            ctx.gctx = GGMLContext();

            dbg_ctx = &ctx;

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
            ggml::tensor* input_ids_tensor = ggml::new_tensor_1d(&ctx, GGML_TYPE_I32, ids_count);

            ggml::tensor* r = transformer->forward(&ctx, input_ids_tensor, past);

            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale);

            ggml::build_forward_expand(&ctx, r);

            CHATLLM_CHECK(r->type == GGML_TYPE_F32) << "output type must be float: " << r->type;

            output.resize(ggml::nbytes(r) / sizeof(output[0]));

            if (!ctx.allocate()) return false;

            Backend::write_tensor_data(input_ids_tensor, input_ids);

            if (gen_config.dump_dot.size() > 0)
            {
                backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
                exit(-1);
            }

            ctx.compute();

            Backend::read_tensor_data(r, output.data());

            ctx.reset();

            return true;
        }

        virtual bool is_output_terminated(const std::vector<int>& output_ids, int& keep_idx, int& pop_output)
        {
            if (output_ids.size() < 1)
                return false;

            int last_tok_id = output_ids[output_ids.size() - 1];

            if (tokenizer->is_terminate_token_id(last_tok_id))
            {
                pop_output = 1;
                return true;
            }
            else
            {
                keep_idx = (int)output_ids.size();
                return false;
            }
        }

        bool match_output_sequence(const std::vector<int>& output_ids, const std::vector<int>& pattern)
        {
            if (output_ids.size() < pattern.size())
                return false;

            auto x0 = output_ids.begin() + output_ids.size() - pattern.size();
            auto x1 = pattern.begin();

            for (size_t i = 0; i < pattern.size(); i++, x0++, x1++)
            {
                if (*x0 != *x1)
                    return false;
            }
            return true;
        }

        template <class T> T* get_typed_transformer(void) const
        {
            return dynamic_cast<T*>(transformer);
        }

    protected:
        ModelBlock* transformer;
        const size_t GRAPH_SIZE;
        int batch_input;
        float logit_scale;
        std::vector<int> layer_ids;
        BackendContext backend_context;
        InitContext w_ctx_; // weight context
    private:
        BaseConfig config_;
        bool initial_run = false;
    };

    template <class Config, class Embedding, class FinalNorm> class HeterogeneousModel : public ModelBlock
    {
    public:
        HeterogeneousModel() = default;

        HeterogeneousModel(InitContext* ctx, const Config& config, bool lm_head_bias, std::function<Block* (InitContext*, int)> create_layer)
            : HeterogeneousModel(ctx, config, new Linear(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size, config.vocab_size, lm_head_bias), create_layer)
        {
        }

        HeterogeneousModel(InitContext* ctx, const Config& config, Block* lm_head, std::function<Block* (InitContext*, int)> create_layer)
            : config(config),
            word_embeddings(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog), config.vocab_size, config.hidden_size, config.max_length),
            final_layernorm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.hidden_size),
            lm_head(lm_head), logits_pp(nullptr),
            cache_size(0)
        {
            layers.reserve(config.num_hidden_layers);
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                ctx->move_to_layer(layer_id);
                auto layer = create_layer(ctx, layer_id);
                layers.emplace_back(layer);

                layer->set_id(layer_id);
                cache_size += layer->get_cache_size();

                auto allocator = ctx->get_allocator();
                auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
                layer->set_cache_buffer(buf);
            }
        }

        ggml::tensor* forward(ComputeContext* ctx, ggml::tensor* input_ids, int n_past) override
        {
            before_forward(ctx, input_ids, n_past);

            ctx->move_to_layer(LayerAllocatorManager::Prolog);
            ggml::tensor* hidden_states = word_embeddings.forward(ctx, input_ids);
            for (auto& layer : layers)
            {
                ctx->move_to_layer(layer->get_id());
                hidden_states = layer->forward(ctx, hidden_states, n_past);
            }

            ctx->move_to_layer(LayerAllocatorManager::Epilog);
            return final_steps(ctx, input_ids, hidden_states);
        }

        void set_ctx(int n_ctx) override
        {
            for (auto& layer : layers)
                layer->set_ctx(n_ctx);
        }

        void shift_cache(int shift, int total) override
        {
            for (auto& layer : layers)
                layer->shift_cache(shift, total);
        }

        int64_t get_param_num(bool effective_only) const override
        {
            int64_t r = 0;
            r += word_embeddings.get_param_num(effective_only);
            r += get_param_num_of_layers(effective_only);
            r += final_layernorm.get_param_num(effective_only);
            if (lm_head)
                r += lm_head->get_param_num(effective_only);
            if (logits_pp)
                r += logits_pp->get_param_num(effective_only);
            return r;
        }

        Block* get_layer(int index)
        {
            return layers[index];
        }

        int save_session(FILE* f) override
        {
            struct state state = { .cache_size = cache_size };
            if (fwrite(&state, sizeof(state), 1, f) != 1)
                return -1;

            std::vector<uint8_t> buffer;

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                auto layer = layers[layer_id];
                buffer.resize(layer->get_cache_size());
                size_t size = layer->read_cache_data(buffer.data(), buffer.size());
                if (size != buffer.size())
                    return -4;
                if (fwrite(buffer.data(), 1, size, f) != size)
                    return -3;
            }

            return 0;
        }

        int load_session(FILE* f) override
        {
            struct state state = { 0 };
            if (fread(&state, sizeof(state), 1, f) != 1)
                return -10;
            if (state.cache_size != cache_size)
                return -1;

            std::vector<uint8_t> buffer;

            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                auto layer = layers[layer_id];
                buffer.resize(layer->get_cache_size());
                if (fread(buffer.data(), 1, buffer.size(), f) != buffer.size())
                    return -4;
                size_t size = layer->write_cache_data(buffer.data(), buffer.size());
                if (size != buffer.size())
                    return -3;
            }

            return 0;
        }

        int save_session(ModelSessionMemory& session) const override
        {
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                auto layer = layers[layer_id];
                const size_t size = layer->get_cache_size();
                void* buf = session.prepare_buffer(layer_id, size);
                if (layer->read_cache_data(buf, size) != size)
                    return -1;
            }

            return 0;
        }

        int load_session(ModelSessionMemory& session) override
        {
            for (int layer_id = 0; layer_id < config.num_hidden_layers; layer_id++)
            {
                auto layer = layers[layer_id];
                size_t size = 0;
                void* buf = session.get_buffer(layer_id, &size);
                if (size != layer->get_cache_size()) return -1;
                if (layer->write_cache_data(buf, size) != size)
                    return -3;
            }

            return 0;
        }

        void load(const std::string& path, TensorLoader* loader, const std::vector<int>& layer_ids) override
        {
            word_embeddings.load(path + "embed_tokens.", loader);
            final_layernorm.load(path + "norm.", loader);
            if (lm_head)
                lm_head->load("lm_head.", loader);

            for (int i = 0; i < config.num_hidden_layers; i++)
            {
                std::string layer_prefix = path + "layers." + std::to_string(layer_ids[i]) + '.';
                layers[i]->load(layer_prefix, loader);
            }
        }

    private:
        struct state
        {
            size_t cache_size;
        };
    protected:
        virtual void before_forward(ComputeContext* ctx, ggml::tensor* input_ids, int n_past) {}

        virtual int64_t get_param_num_of_layers(bool effective_only) const
        {
            int64_t r = 0;
            for (auto& layer : layers)
                r += layer->get_param_num(effective_only);
            return r;
        }

        ggml::tensor* final_steps(ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
        {
            hidden_states = ggml::view_2d(ctx, hidden_states, config.hidden_size, 1,
                config.hidden_size * ggml::element_size(hidden_states),
                (input_ids->ne[0] - 1) * config.hidden_size * ggml::element_size(hidden_states));

            ggml::tensor* transformer_outputs = final_layernorm.forward(ctx, hidden_states);

            transformer_outputs =
                ggml::view_1d(ctx, transformer_outputs, config.hidden_size, 0);

            ggml::tensor* lm_logits = lm_head ? lm_head->forward(ctx, transformer_outputs)
                : word_embeddings.forward(ctx, transformer_outputs);

            if (logits_pp)
                lm_logits = logits_pp->forward(ctx, lm_logits);
            return lm_logits;
        }
    public:
        Config config;
        Embedding word_embeddings;
        FinalNorm final_layernorm;
        Block* lm_head;
        Block* logits_pp;
    protected:
        // std::vector<std::unique_ptr<Block>> layers;
        std::vector<Block*> layers;
        size_t cache_size;
    };

    template <class Config, class Embedding, class FinalNorm, class LayerBlock, typename... _Types> class Model :
        public HeterogeneousModel<Config, Embedding, FinalNorm>
    {
    private:
        typedef HeterogeneousModel<Config, Embedding, FinalNorm> Base;
    protected:
        class Accessor
        {
            friend Model;
        protected:
            Accessor() : m(nullptr) {}
        public:
            LayerBlock& operator[](int index)
            {
                if (nullptr == m)
                {
                    uintptr_t offset = (uintptr_t) & (((Model*)(nullptr))->layers);
                    m = (Model*)(uintptr_t(this) - offset);
                }
                return *(dynamic_cast<LayerBlock*>((m->Base::layers)[index])); // .get()));
            }
        private:
            Model* m;
        };
    public:
        Model() = default;
        Model(InitContext* ctx, const Config& config, bool lm_head_bias, _Types... layer_args)
            : Model(ctx, config, new Linear(ctx, config.hidden_size, config.vocab_size, lm_head_bias), std::forward<_Types>(layer_args)...)
        {
        }

        Model(InitContext* ctx, const Config& config, Block* lm_head, _Types... layer_args)
            : HeterogeneousModel<Config, Embedding, FinalNorm>(ctx, config, lm_head,
                [&](InitContext* ctx, int layer_index) {
                    return new LayerBlock(ctx, std::forward<_Types>(layer_args)...);
                })
        {
        }

    protected:
        int64_t get_param_num_of_layers(bool effective_only) const override
        {
            int64_t r = 0;
            if (Base::layers.size() > 0)
                r += Base::layers[0]->get_param_num(effective_only) * Base::layers.size();
            return r;
        }
    public:
        Accessor layers;
    };
}
module;
#include <string.h>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <string>
#include "../basics.h"

module chatllm;
import :models.base;

namespace chatllm
{
    class NonGreedySampler : public Sampler
    {
    public:
        NonGreedySampler(const GenerationConfig& gen_config, float temperature, int top_k)
            : Sampler(gen_config),
            inv_temp(0.0f), top_k(top_k)
        {
            temp_en = fabs(temperature - 1.0f) > 1e-5f;
            if (temp_en) inv_temp = 1.f / temperature;
        }

        int sampling(std::span<float> logits, float* confidence_level) override
        {
            token_scores.resize(logits.size());

            if (temp_en)
            {
                std::ranges::for_each(logits, [this](auto& logit) { logit *= inv_temp; });
            }

            penalty.process(logits);

            for (int i = 0; i < logits.size(); i++)
            {
                token_scores[i] = { .id = i, .score = logits[i] };
            }

            // top_k sampling
            if (0 < top_k && top_k < (int)token_scores.size())
            {
                std::ranges::nth_element(token_scores, token_scores.begin() + top_k,
                    std::greater<TokenIdScore>());
                token_scores.resize(top_k);
            }

            do_sampling(logits);

            if (token_scores.size() < 1)
                return ABORT;

            // sample next token
            for (size_t i = 0; i < token_scores.size(); i++)
            {
                logits[i] = token_scores[i].score;
            }

            std::discrete_distribution<> dist(logits.data(), logits.data() + token_scores.size());
            auto pos = dist(gen);
            int next_token_id = token_scores[pos].id;

            penalty.accept_choice(next_token_id);
            if (confidence_level) *confidence_level = token_scores[pos].score;

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

        void sampling_softmax_inplace(std::span<TokenIdScore> samples)
        {
            float max_score = std::max_element(samples.begin(), samples.end())->score;
            float sum = 0.f;
            for (auto& p : samples)
            {
                float s = std::exp(p.score - max_score);
                p.score = s;
                sum += s;
            }
            float inv_sum = 1.f / sum;
            for (auto& p : samples)
            {
                p.score *= inv_sum;
            }
        }

        virtual void do_sampling(std::span<float> logits) = 0;
        bool temp_en;
        float inv_temp;
        int top_k;
        std::vector<TokenIdScore> token_scores;
    };

    class TopPSampler : public NonGreedySampler
    {
    public:
        TopPSampler(const GenerationConfig& gen_config, float temperature, int top_k, float top_p)
            : NonGreedySampler(gen_config, temperature, top_k), top_p(top_p)
        {
        }

    protected:
        void do_sampling(std::span<float> next_token_logits) override
        {
            // top_p sampling
            if (0.f < top_p && top_p < 1.f)
            {
                std::ranges::sort(token_scores, std::greater<TokenIdScore>()); // hot code!
                sampling_softmax_inplace(token_scores);

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

            sampling_softmax_inplace(token_scores);
        }

    protected:
        const float top_p;
    };

    // Reference:
    // https://www.trentonbricken.com/Tail-Free-Sampling/#tail-free-sampling-algorithm
    class FreeTailSampler : public NonGreedySampler
    {
    public:
        FreeTailSampler(const GenerationConfig& gen_config, float temperature, int top_k, float z)
            : NonGreedySampler(gen_config, temperature, top_k), z(z)
        {
        }

    protected:

        void do_sampling(std::span<float> next_token_logits) override
        {
            if (token_scores.size() < 3) return;

            sampling_softmax_inplace(token_scores);
            std::ranges::sort(token_scores, std::greater<TokenIdScore>()); // hot code!

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
        int sampling(std::span<float> logits, float* confidence_level) override
        {
            int r = (int)(std::ranges::max_element(logits) - logits.begin());
            if (confidence_level)
            {
                float max_score = *std::ranges::max_element(logits);
                float sum = 0.f;
                for (auto logit : logits)
                {
                    sum += expf(logit - max_score);
                }
                *confidence_level = expf(logits[r] - max_score) / sum;
            }
            return r;
        }
    };

    std::unique_ptr<Sampler> SamplerFactory::Create(const GenerationConfig& gen_config)
    {
        auto r = [&]() -> std::unique_ptr<Sampler>  {
            if (gen_config.do_sample)
            {
                if (gen_config.sampling == "top_p")
                    return std::make_unique<TopPSampler>(gen_config, gen_config.temperature, gen_config.top_k, gen_config.top_p);
                else if (gen_config.sampling == "tfs")
                    return std::make_unique<FreeTailSampler>(gen_config, gen_config.temperature, gen_config.top_k, gen_config.tfs_z);
                else if (gen_config.sampling != "greedy")
                    CHATLLM_CHECK(false) << "unknown sampling algorithm: " << gen_config.sampling;
            }
            return std::make_unique<GreedySampler>();
        }();
        r->seed(gen_config.get_seed());
        return r;
    }

    BaseModelForConditionalGeneration::BaseModelForConditionalGeneration(ModelType model_type, BaseConfig config, const RuntimeConfig& runtime_config)
        : BaseModel(model_type, get_model_purpose(model_type)),
        transformer(nullptr),
        batch_input(runtime_config.batch_input_size), logit_scale(-1.0f),
        w_ctx_(&backend_context),
        config_(config)
    {
        w_ctx_.cache_dtype = runtime_config.cache_type;
        prepare(runtime_config);
        for (int i = 0; i < config.num_hidden_layers; i++)
            layer_ids.push_back(i);
    }

    void BaseModelForConditionalGeneration::set_layer_ids(std::span<const int> ids)
    {
        CHATLLM_CHECK((int)ids.size() == config_.num_hidden_layers) << "length(layer_ids) must be " << config_.num_hidden_layers;
        layer_ids = std::vector<int> { ids.begin(), ids.end() };
    }

    int BaseModelForConditionalGeneration::get_max_length(void)
    {
        return config_.max_length;
    }

    void BaseModelForConditionalGeneration::shift_memory(int keep)
    {
        if (keep >= n_past) return;

        transformer->shift_cache(n_past - keep, n_past);
        BaseModel::shift_memory(keep);
    }

    int64_t BaseModelForConditionalGeneration::get_param_num(bool effective_only) const
    {
        return transformer->get_param_num(effective_only);
    }

    AbstractModel::generate_result BaseModelForConditionalGeneration::generate(std::span<const int> input_ids, const GenerationConfig& gen_config,
        const bool continuous,
        ModelPerfInfo* performance,
        BaseStreamer* streamer)
    {
        CHATLLM_CHECK(gen_config.max_length <= config_.max_length)
            << "requested max_length (" << gen_config.max_length << ") is larger than model's max_length ("
            << config_.max_length << ")";

        //for (int i = 0; i < (int)input_ids.size(); i++)
        //    printf("%d, ", input_ids[i]);
        //printf("\nn_past = %d, %d\n\n", n_past, continuous);

        std::unique_ptr<Sampler> sampler = SamplerFactory::Create(gen_config);

        aborted = false;

        std::vector<int> curr_input_ids(input_ids.begin(), input_ids.end());

        std::vector<int> output_ids;
        output_ids.reserve(gen_config.max_length);

        if (!continuous)
        {
            n_past = 0;
            n_past_offset = 0;
        }

        bool completed = false;

        transformer->set_ctx((int)input_ids.size());
        int next_output_idx = 0;

        int gen_max_tokens = gen_config.max_new_tokens;
        if (gen_max_tokens > 0)
            gen_max_tokens = n_past + (int)curr_input_ids.size() + gen_max_tokens;

        bool first_call = true;

        if ((auto_output_prefix.size() > 0) && streamer)
            streamer->put(auto_output_prefix);

        if (performance)
            performance->Reset();

        before_generate(gen_config);

        #if (0)
        for (auto i : curr_input_ids)
            printf("%d, ", i);
        printf("\n");
        #endif

        while (!aborted && !completed && (n_past + (int)curr_input_ids.size() < gen_config.max_length))
        {
            std::vector<float> lm_logits;
            const int last_n_past = n_past;
            if (!generate_next_token(curr_input_ids, gen_config, lm_logits))
            {
                ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
                aborted = true;
                break;
            }

            if (lm_logits.size() == 0)
            {
                int num = n_past > last_n_past ? n_past - last_n_past : 0;
                performance->Accumulate(ModelPerfInfo::Type::Generation, num);
                completed = true;
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
            std::span<float> logits = lm_logits;
            const size_t tok_num = lm_logits.size() / config_.vocab_size;

            for (size_t tok_idx = 0; (tok_idx < tok_num) && !aborted; tok_idx++, logits = logits.subspan(config_.vocab_size))
            {
                int next_token_id = sampler->sampling(logits.first(config_.vocab_size));

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
                    auto subspan = std::span{ output_ids }.subspan(next_output_idx, keep_idx - next_output_idx);
                    streamer->put(subspan);
					next_output_idx = keep_idx;
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
        {
            size_t num = output_ids.size() > curr_input_ids.size() ? output_ids.size() - curr_input_ids.size() : 0;
            performance->Accumulate(ModelPerfInfo::Type::Generation, num);
        }

        after_generate();

        //printf("\nn_past = %d\n", n_past);
        return { std::move(output_ids), completed };
    }

    std::vector<float> BaseModelForConditionalGeneration::embedding(const GenerationConfig& gen_config, const std::vector<int>& input_ids)
    {
        before_generate(gen_config);
        std::vector<float> embedding;
        auto r = run_model(input_ids, gen_config, 0, embedding);
        if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
        return embedding;
    }

    float BaseModelForConditionalGeneration::qa_rank(const GenerationConfig& gen_config, const std::vector<int>& input_ids)
    {
        std::vector<float> output;
        before_generate(gen_config);
        auto r = run_model(input_ids, gen_config, 0, output);
        if (!r) ggml::log(GGML_LOG_LEVEL_ERROR, "Out of memory");
        CHATLLM_CHECK(output.size() == 1) << "ouput must be scaler";

        return output[0];
    }

    bool BaseModelForConditionalGeneration::generate_next_token(std::span<const int> input_ids, const GenerationConfig& gen_config, std::vector<float>& lm_logits)
    {
        int batch = batch_input > 1 ? batch_input : 1;
        int past = n_past + n_past_offset;

        for (auto chunk : input_ids | std::views::chunk(batch))
        {
            if (!run_model(chunk, gen_config, past, lm_logits, 1))
                return false;
            past += batch;
        }
        return true;
    }


    int BaseModelForConditionalGeneration::save_session(FILE* f) const
    {
        int r = BaseModel::save_session(f);
        if (r != 0)
            return r;
        return transformer->save_session(f);
    }

    int BaseModelForConditionalGeneration::load_session(FILE* f)
    {
        int r = BaseModel::load_session(f);
        if (r != 0) return r;
        return transformer->load_session(f);
    }

    int BaseModelForConditionalGeneration::save_session(ModelSessionMemory& session) const
    {
        int r = BaseModel::save_session(session);
        if (r != 0)
            return r;
        return transformer->save_session(session);
    }

    int BaseModelForConditionalGeneration::load_session(ModelSessionMemory& session)
    {
        int r = BaseModel::load_session(session);
        if (r != 0) return r;
        return transformer->load_session(session);
    }

    void BaseModelForConditionalGeneration::prepare(const RuntimeConfig& rt_config)
    {
        w_ctx_.user_options.moe_on_cpu = rt_config.moe_on_cpu;
        backend_context.init(rt_config.model_gpu_layers, "main", config_.num_hidden_layers, rt_config.n_threads);
    }

    LayerAllocatorManager& BaseModelForConditionalGeneration::get_alloc_manager()
    {
        return backend_context.layer_allocators;
    }

    void BaseModelForConditionalGeneration::load(ModelLoader& loader)
    {
        transformer->load("model.", &loader, layer_ids);
    }

    void BaseModelForConditionalGeneration::before_generate(const GenerationConfig& gen_config)
    {
    }

    void BaseModelForConditionalGeneration::after_generate(void)
    {
        tokenizer->media_emb.clear();
    }

    void BaseModelForConditionalGeneration::do_build_graph(ForwardContext& ctc, const std::vector<int>& input_ids,
        const GenerationConfig& gen_config,
        int past)
    {

    }

    bool BaseModelForConditionalGeneration::run_model(std::span<const int> input_ids,
        const GenerationConfig& gen_config,
        int past,
        std::vector<float>& output, const int batch_size,
        std::function<ggml::tensor* (ComputeContext*, ggml::tensor*)> func_epilog)
    {
        ForwardContext ctx(&backend_context);
        ctx.user_options = w_ctx_.user_options;

        set_dbg_ctx(&ctx);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor* input_ids_tensor = ctx.new_tensor(GGML_TYPE_I32, { batch_size, (int64_t)input_ids.size() });

        ggml::tensor* r = transformer->forward(&ctx, input_ids_tensor, past);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);

        if (func_epilog)
        {
            r = func_epilog(&ctx, r);
        }
        else
        {
            if (logit_scale > 0)
                r = ggml::scale(&ctx, r, logit_scale, false);
        }

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK((r->type == GGML_TYPE_F32) || (r->type == GGML_TYPE_I32)) << "output type must be float/int32: " << r->type;

        output.resize(ggml::nbytes(r) / sizeof(output[0]));

        if (!ctx.allocate()) return false;

        Backend::write_tensor_data(input_ids_tensor, input_ids.data());

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

    bool BaseModelForConditionalGeneration::is_output_terminated(const std::vector<int>& output_ids, int& keep_idx, int& pop_output)
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

    bool BaseModelForConditionalGeneration::match_output_sequence(const std::vector<int>& output_ids, const std::vector<int>& pattern)
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

    HeterogeneousModel::HeterogeneousModel(InitContext* ctx, int num_hidden_layers, int hidden_size,
        std::unique_ptr<Block> word_embeddings, std::unique_ptr<Block> final_layernorm,
        std::unique_ptr<Linear> lm_head, std::function<Block* (InitContext*, int)> create_layer)
        : hidden_size(hidden_size),
        word_embeddings(std::move(word_embeddings)),
        final_layernorm(std::move(final_layernorm)),
        lm_head(std::move(lm_head)),
        logits_pp(nullptr),
        cache_size(0),
        final_steps(std::make_unique<LMFinalSteps>())
    {
        layers.reserve(num_hidden_layers);
        for (int layer_id = 0; layer_id < num_hidden_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = create_layer(ctx, layer_id);
            layers.emplace_back(layer);

            layer->set_id(layer_id);
            cache_size += layer->get_cache_size();

            auto allocator = ctx->get_allocator();
            auto& buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
            layer->set_cache_buffer(buf);
        }
    }

    HeterogeneousModel::~HeterogeneousModel()
    {
        for (auto b : layers) delete b;
    }

    ggml::tensor* HeterogeneousModel::forward(ComputeContext* ctx, ggml::tensor* input_ids, int n_past)
    {
        before_forward(ctx, input_ids, n_past);

        ctx->move_to_layer(LayerAllocatorManager::Prolog);
        ggml::tensor* hidden_states = custom_embedding ? custom_embedding(ctx, input_ids) : word_embeddings->forward(ctx, input_ids);
        for (auto& layer : layers)
        {
            ctx->move_to_layer(layer->get_id());
            if (layer_preprocess.get())
            {
                auto t = layer_preprocess->forward(this, ctx, hidden_states, layer->get_id());
                if (t) hidden_states = t;
            }

            hidden_states = layer->forward(ctx, hidden_states, n_past);
        }

        last_hidden_state = hidden_states;

        ctx->move_to_layer(LayerAllocatorManager::Epilog);
        return final_steps->forward(this, ctx, input_ids, hidden_states);
    }

    void HeterogeneousModel::set_ctx(int n_ctx)
    {
        for (auto& layer : layers)
            layer->set_ctx(n_ctx);
    }

    void HeterogeneousModel::shift_cache(int shift, int total)
    {
        for (auto& layer : layers)
            layer->shift_cache(shift, total);
    }

    int64_t HeterogeneousModel::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += word_embeddings->get_param_num(effective_only);
        r += get_param_num_of_layers(effective_only);
        r += final_layernorm->get_param_num(effective_only);
        if (lm_head)
            r += lm_head->get_param_num(effective_only);
        if (logits_pp)
            r += logits_pp->get_param_num(effective_only);
        return r;
    }

    Block* HeterogeneousModel::get_layer(int index)
    {
        return layers[index];
    }

    int HeterogeneousModel::get_layer_num(void) const
    {
        return (int)layers.size();
    }

    void HeterogeneousModel::set_final_steps(std::unique_ptr<ModelFinalSteps> final_steps)
    {
        this->final_steps = std::move(final_steps);
    }

    ModelFinalSteps* HeterogeneousModel::get_final_steps()
    {
        return final_steps.get();
    }

    void HeterogeneousModel::set_layer_preprocess(std::unique_ptr<ModelLayerInputPreprocess> layer_preprocess)
    {
        this->layer_preprocess = std::move(layer_preprocess);
    }

    ModelLayerInputPreprocess* HeterogeneousModel::get_layer_preprocess()
    {
        return layer_preprocess.get();
    }

    int HeterogeneousModel::save_session(FILE* f)
    {
        struct state state = { .cache_size = cache_size };
        if (fwrite(&state, sizeof(state), 1, f) != 1)
            return -1;

        std::vector<std::byte> buffer;

        for (const auto &layer : layers)
        {
            buffer.resize(layer->get_cache_size());
            size_t size = layer->read_cache_data(buffer);
            if (size != buffer.size())
                return -4;
            if (fwrite(buffer.data(), 1, size, f) != size)
                return -3;
        }

        return 0;
    }

    int HeterogeneousModel::load_session(FILE* f)
    {
        struct state state = { 0 };
        if (fread(&state, sizeof(state), 1, f) != 1)
            return -10;
        if (state.cache_size != cache_size)
            return -1;

        std::vector<std::byte> buffer;

        for (const auto &layer : layers)
        {
            buffer.resize(layer->get_cache_size());
            if (fread(buffer.data(), 1, buffer.size(), f) != buffer.size())
                return -4;
            size_t size = layer->write_cache_data(buffer);
            if (size != buffer.size())
                return -3;
        }

        return 0;
    }

    int HeterogeneousModel::save_session(ModelSessionMemory& session) const
    {
        for (size_t layer_id = 0; layer_id < layers.size(); layer_id++)
        {
            const auto &layer = layers[layer_id];
            const size_t size = layer->get_cache_size();
            std::span<std::byte> buf = session.prepare_buffer(layer_id, size);
            if (layer->read_cache_data(buf) != size)
                return -1;
        }

        return 0;
    }

    int HeterogeneousModel::load_session(ModelSessionMemory& session)
    {
        for (size_t layer_id = 0; layer_id < layers.size(); layer_id++)
        {
            const auto &layer = layers[layer_id];
            std::span<std::byte> buf = session.get_buffer(layer_id);
            if (buf.size() != layer->get_cache_size()) return -1;
            if (layer->write_cache_data(buf) != buf.size())
                return -3;
        }

        return 0;
    }

    void HeterogeneousModel::load(const std::string& path, TensorLoader* loader, const std::vector<int>& layer_ids)
    {
        word_embeddings->load(path + "embed_tokens.", loader);
        final_layernorm->load(path + "norm.", loader);
        if (lm_head)
            lm_head->load("lm_head.", loader);

        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string layer_prefix = path + "layers." + std::to_string(layer_ids[i]) + '.';
            layers[i]->load(layer_prefix, loader);
        }
    }

    int64_t HeterogeneousModel::get_param_num_of_layers(bool effective_only) const
    {
        int64_t r = 0;
        for (auto& layer : layers)
            r += layer->get_param_num(effective_only);
        return r;
    }

    void HeterogeneousModel::reserve_batch_size(int size)
    {
        ModelBlock::reserve_batch_size(size);
        for (auto& layer : layers)
            layer->reserve_batch_size(size);
    }

    ggml::tensor* LMFinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        const int qlen = ggml::get_dim(hidden_states, 1);
        const int batch = ggml::get_dim(hidden_states, 2);
        const int last_n = qlen >= this->last_n ? this->last_n : qlen;
        order = nullptr;

        if (disable_head) return hidden_states;

        hidden_states = ctx->view(hidden_states, { batch, last_n, model->hidden_size },
            { ggml::row_size(hidden_states) * qlen, ggml::row_size(hidden_states) },
            (qlen - last_n) * ggml::row_size(hidden_states));

        ggml::tensor* transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);

        // now, this is continous
        transformer_outputs = ctx->reshape(transformer_outputs, { last_n * batch, ggml::get_dim(transformer_outputs, 0) });

        ggml::set_output(transformer_outputs);
        model->last_hidden_state = transformer_outputs;
        if (model->skip_lm_head)
            return transformer_outputs;

        ggml::tensor* lm_logits = model->lm_head ? model->lm_head->forward(ctx, transformer_outputs)
            : model->word_embeddings->forward(ctx, transformer_outputs);

        if (model->logits_pp)
            lm_logits = model->logits_pp->forward(ctx, lm_logits);

        if (do_orderring)
            order = ggml::ordering(ctx, lm_logits, true);

        if ((last_n > 1) && (batch > 1))
        {
            lm_logits = ctx->reshape(lm_logits, { batch, last_n, ggml::get_dim(lm_logits, 0) });
            if (order)
                order = ctx->reshape(order, { batch, last_n, ggml::get_dim(order, 0) });
        }

        if (order)
        {
            ggml::set_output(order);
            ggml::build_forward_expand(ctx, order);
        }

        return lm_logits;
    }

    void LMFinalSteps::set_read_last_n(int n)
    {
        last_n = n >= 1 ? n : 1;
    }

    void LMFinalSteps::set_do_orderring(bool flag)
    {
        do_orderring = flag;
    }
    ggml::tensor* LMFinalSteps::get_orderring_result(void)
    {
        return order;
    }

    ggml::tensor* EmbeddingPoolingFinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        ggml::tensor* transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);

        return transformer_outputs;
    }

    ggml::tensor* EmbeddingLastTokenFinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        hidden_states = ctx->view(hidden_states, { 1, model->hidden_size },
            { ggml::row_size(hidden_states) },
            (ggml::get_dim(input_ids, 0) - 1) * ggml::row_size(hidden_states));
        ggml::tensor* transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);
        transformer_outputs = ggml::simple_norm(ctx, transformer_outputs, 1e-5f);
        return transformer_outputs;
    }

    TensorGraphEvaluator::TensorGraphEvaluator(const RuntimeConfig& runtime_config, const std::string model_id, int max_layers)
        : n_threads(runtime_config.n_threads)
    {
        model_gpu_layers = BackendContext::get_ngl_of_model(runtime_config.model_gpu_layers, model_id);
        backend_context.init(model_gpu_layers, max_layers, n_threads);
    }

    bool TensorGraphEvaluator::evaluate(const GenerationConfig& gen_config,
        std::function<ggml::tensor* (ComputeContext* ctx)> make_graph,
        std::function<void(ComputeContext* ctx)> write_input_data,
        ggml::type expected_result_dtype,
        std::vector<int64_t>& result_shape,
        std::vector<uint8_t>& result_buf)
    {
        ForwardContext ctx(&backend_context);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        set_dbg_ctx(&ctx);

        ggml::tensor* r = make_graph(&ctx);

        if (ggml::type_of(r) != expected_result_dtype)
        {
            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
            ggml::tensor* t = ggml::new_tensor_like(&ctx, expected_result_dtype, r);
            r = ggml::cpy(&ctx, r, t);
        }

        ggml::get_shape(r, result_shape);

        ggml::set_output(r);
        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        write_input_data(&ctx);

        ctx.compute();

        set_dbg_ctx(nullptr);

        size_t offset = result_buf.size();
        result_buf.resize(offset + ggml::nbytes(r));
        Backend::read_tensor_data(r, result_buf.data() + offset);
        ctx.reset();

        return true;
    }
}

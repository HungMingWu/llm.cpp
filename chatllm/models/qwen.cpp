﻿module;
#include <sstream>
#include "../audio_process.h"
#include "../basics.h"
#include "../tokenizer.h"
#include "../JSON.h"

module chatllm;
import :models.qwen;

namespace chatllm::qwen::v1
{
    ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const Config& config)
        : Tokenizer(config, &_chat_encoder)
    {
    }

    Tokenizer::Tokenizer(const BaseConfig& config, BaseHistoryEncoder* encoder,
        BaseHistoryEncoder* qa_encoder,
        BaseHistoryEncoder* completion_encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
    {
        sys_prompt = "You are a helpful assistant.";
    }

    size_t Tokenizer::do_load(tokenizer::DataReader* buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
            );
        return tp->Load(buffer, n_vocab);
    }

    size_t Tokenizer::load(tokenizer::DataReader* buffer, int n_vocab)
    {
        size_t size = do_load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        // for QAnything
        pad_token_id = eos_token_id = bos_token_id = tp->PieceToId("<|endoftext|>");
        im_start_token_id = tp->PieceToId("<|im_start|>");
        im_end_token_id = tp->PieceToId("<|im_end|>");

        if (im_end_token_id < 0)
        {
            // QWen v1
            pad_token_id = eos_token_id = bos_token_id = tp->GetPieceSize() + 0;
            im_start_token_id = eos_token_id + 1;
            im_end_token_id = eos_token_id + 2;
        }

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        if (im_end_token_id >= 0)
            terminate_ids.insert(im_end_token_id);

        return size;
    }

    void Tokenizer::encode(const std::string& text, std::vector<int>& ids, bool add_im_start, bool add_im_end, bool add_nl) const
    {
        if (add_im_start)
            ids.push_back(im_start_token_id);
        BaseTokenizer::encode(text, ids);
        if (add_im_end)
            ids.push_back(im_end_token_id);
        if (add_nl)
            ids.push_back(nl_token_id);
    }

    void Tokenizer::encode(const std::string& text, std::vector<int>& ids) const
    {
        encode(text, ids, false, false, false);
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string& ai, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);

        append_ai_opening(round_idx, ids);

        tok->encode(ai, ids, false, true, true);
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);

        tok->encode("system", ids, true, false, true);
        tok->encode(tok->get_system_prompt(), ids, false, true, true);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string& user, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        std::ostringstream oss_prompt;

        tok->encode("user", ids, true, false, true);
        tok->encode(user, ids, false, true, true);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        tok->encode("assistant", ids, true, false, true);
    }

    void ChatHistoryEncoder::append_user_opening(int round_idx, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        tok->encode("user", ids, true, false, true);
    }

    bool Tokenizer::is_special_id(int id) const
    {
        return (id == pad_token_id) || (id == im_start_token_id) || (id == im_end_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config, ModelType type)
        : BaseModelForConditionalGeneration(type, config, runtime_config), config(config)
    {
        w_ctx_.dtype = config.dtype;

        // TODO: support of `use_dynamic_ntk`
        transformer = new ModelClass(&w_ctx_, config, false,
            config.hidden_size, config.num_attention_heads,
            config.intermediate_size, config.max_length);

        bool use_dynamic_ntk = (config.flags & 1) != 0;
        bool use_logn_attn = (config.flags & 2) != 0;

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto& layer = get_typed_transformer<ModelClass>()->layers[i];
            auto att = dynamic_cast<QWenSelfAttention*>(&layer.attention);
            att->config(config.rope_dim, config.rotary_emb_base, config.seq_length,
                use_dynamic_ntk, use_logn_attn);
        }
    }

    void ConditionalGeneration::load(ModelLoader& loader)
    {
        auto transformer = get_typed_transformer<ModelClass>();
        transformer->word_embeddings->load("transformer.wte.", &loader);
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            std::string layer_prefix = "transformer.h." + std::to_string(layer_ids[i]) + '.';

            loader.read_tensor(layer_prefix + "attn.k_proj.weight", transformer->layers[i].attention.k_proj.weight);
            loader.read_tensor(layer_prefix + "attn.k_proj.bias", transformer->layers[i].attention.k_proj.bias);
            loader.read_tensor(layer_prefix + "attn.q_proj.weight", transformer->layers[i].attention.q_proj.weight);
            loader.read_tensor(layer_prefix + "attn.q_proj.bias", transformer->layers[i].attention.q_proj.bias);
            loader.read_tensor(layer_prefix + "attn.v_proj.weight", transformer->layers[i].attention.v_proj.weight);
            loader.read_tensor(layer_prefix + "attn.v_proj.bias", transformer->layers[i].attention.v_proj.bias);
            loader.read_tensor(layer_prefix + "attn.c_proj.weight", transformer->layers[i].attention.o_proj.weight);

            loader.read_tensor(layer_prefix + "ln_1.weight", transformer->layers[i].input_layernorm.weight);
            loader.read_tensor(layer_prefix + "ln_2.weight", transformer->layers[i].post_attention_layernorm.weight);

            loader.read_tensor(layer_prefix + "mlp.c_proj.weight", transformer->layers[i].mlp.down_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.w1.weight", transformer->layers[i].mlp.up_proj.weight);
            loader.read_tensor(layer_prefix + "mlp.w2.weight", transformer->layers[i].mlp.gate_proj.weight);
        }
        transformer->final_layernorm->load("transformer.ln_f.", &loader);
        loader.read_tensor("lm_head.weight", transformer->lm_head->weight);
    }
}

namespace chatllm::qwen::v2
{
    Tokenizer::Tokenizer(const BaseConfig& config)
        : Tokenizer(config, &v1::_chat_encoder)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader* buffer, int n_vocab)
    {
        size_t r = v1::Tokenizer::load(buffer, n_vocab);

        im_start_token_id = tp->PieceToId("<|im_start|>");
        im_end_token_id = tp->PieceToId("<|im_end|>");
        bos_token_id = pad_token_id = eos_token_id = im_start_token_id - 1;

        std::vector<int> ids;
        tp->Encode("\n", &ids);
        nl_token_id = ids[0];

        if (im_end_token_id >= 0)
            terminate_ids.insert(im_end_token_id);

        return r;
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config, ModelType type, bool tie_embeddings)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 2),
        config(config), tie_embeddings(tie_embeddings)
    {
        w_ctx_.dtype = config.dtype;

        if (tie_embeddings)
        {
            transformer = new ModelClass(&w_ctx_, config, nullptr,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads,
                config.max_length);
        }
        else
        {
            transformer = new ModelClass(&w_ctx_, config, false,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.num_key_value_heads,
                config.max_length);
        }


        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto& layer = get_typed_transformer<ModelClass>()->layers[i];
            layer.attention.freq_base = config.rope_theta;
        }
    }
}

namespace chatllm::qwen::v2_tie
{

}

namespace chatllm::qwen::v2_moe
{
    namespace experts_60
    {
        const int NUM_EXPERTS = 60;
        const int EXPERTS_PER_TOK = 4;

        // make it easy to test with different number of experts.
#define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

        typedef ClassConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    namespace experts_64
    {
        const int NUM_EXPERTS = 64;
        const int EXPERTS_PER_TOK = 8;

        // make it easy to test with different number of experts.
#define EFFECTIVE_EXPERTS_PER_TOK       EXPERTS_PER_TOK

        typedef ClassConditionalGeneration<NUM_EXPERTS, EXPERTS_PER_TOK, EFFECTIVE_EXPERTS_PER_TOK> ConditionalGeneration;
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config) : ModelProxy()
    {
        switch (config.num_experts)
        {
        case experts_60::NUM_EXPERTS:
            set_proxy_model(experts_60::ConditionalGeneration::create(config, runtime_config));
            break;
        case experts_64::NUM_EXPERTS:
            set_proxy_model(experts_64::ConditionalGeneration::create(config, runtime_config));
            break;
        default:
            CHATLLM_CHECK(false) << "unsupported MoE param: num_experts = " << config.num_experts;
            break;
        }
    }
}

namespace chatllm::qwen::audio_tower
{
    AudioSelfAttention::AudioSelfAttention(InitContext* ctx, int hidden_size, int num_attention_heads, int max_length)
        : BaseCachelessAttention(ctx, hidden_size, num_attention_heads, num_attention_heads, max_length, true, true)
    {
        causal = false;
    }

    AudioTransformer::AudioTransformer(InitContext* ctx, const Config& config, int lm_hidden_size)
        : embed_positions(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
            ggml::type::GGML_TYPE_F32,
            config.max_source_positions, config.d_model),
        conv1(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
            config.num_mel_bins, config.d_model, 3, 1, 1),
        conv2(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Prolog),
            config.d_model, config.d_model, 3, 2, 1),
        layer_norm(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog), config.d_model),
        multi_modal_projector(LayerMover(ctx, LayerAllocatorManager::MiscLayer::Epilog),
            config.d_model, lm_hidden_size),
        loaded(false)
    {
        BlockParams::OverrideKProjBiased k_proj_biased(false);

        for (int layer_id = 0; layer_id < config.encoder_layers; layer_id++)
        {
            ctx->move_to_layer(layer_id);
            auto layer = new LayerBlock(ctx, config.d_model, config.encoder_attention_heads, config.encoder_ffn_dim, config.max_source_positions);
            layer->set_id(layer_id);
            layers.emplace_back(layer);
        }
    }

    int64_t AudioTransformer::get_param_num(bool effective_only) const
    {
        int64_t r = 0;
        r += embed_positions.get_param_num(effective_only);
        r += layer_norm.get_param_num(effective_only);
        r += conv1.get_param_num(effective_only);
        r += conv2.get_param_num(effective_only);
        r += multi_modal_projector.get_param_num(effective_only);
        for (size_t i = 0; i < layers.size(); i++)
            r += layers[i]->get_param_num(effective_only);
        return r;
    }

    void AudioTransformer::load(const std::string& path, TensorLoader* loader)
    {
        if (!loader->has_tensor(path + "embed_positions.weight")) return;

        embed_positions.load(path + "embed_positions.", loader);
        layer_norm.load(path + "layer_norm.", loader);
        multi_modal_projector.load("multi_modal_projector.linear.", loader);
        conv1.load(path + "conv1.", loader);
        conv2.load(path + "conv2.", loader);

        for (size_t i = 0; i < layers.size(); i++)
        {
            std::string block_path = path + "layers." + std::to_string(i) + ".";
            layers[i]->load(block_path, loader);
        }
        loaded = true;
    }

    ggml::tensor* AudioTransformer::forward(ComputeContext* ctx, ggml::tensor* input)
    {
        auto output = conv1.forward(ctx, input);
        output = ggml::act(ctx, ActFunc::GELU, output);
        output = conv2.forward(ctx, output);
        output = ggml::act(ctx, ActFunc::GELU, output);
        output = ggml::permute(ctx, output, 1, 0, 2, 3);
        output = ggml::cont(ctx, output);
        output = ggml::add(ctx, output, embed_positions.weight);

        for (size_t i = 0; i < layers.size(); i++)
        {
            output = layers[i]->forward(ctx, output, 0);
        }
        output = ggml::permute(ctx, output, 1, 0, 2, 3);
        output = ggml::avg_pool_1d(ctx, output, 2, 2);
        output = ggml::permute(ctx, output, 1, 0, 2, 3);
        output = layer_norm.forward(ctx, output);
        output = multi_modal_projector.forward(ctx, output);
        return output;
    }

    bool AudioTransformer::is_loaded(void) const
    {
        return loaded;
    }

    AudioEmbeddingGeneration::AudioEmbeddingGeneration(const RuntimeConfig& runtime_config, size_t GRAPH_SIZE)
        :
        GRAPH_SIZE(GRAPH_SIZE), _ctx(&backend_context),
        n_threads(runtime_config.n_threads)
    {
        _ctx.cache_dtype = runtime_config.cache_type;
        model_gpu_layers = BackendContext::get_ngl_of_model(runtime_config.model_gpu_layers, "aud");
    }

    bool AudioEmbeddingGeneration::load(ModelLoader& loader)
    {
        if (model.get())
        {
            loader.push_allocator_manager(&backend_context.layer_allocators);
            model->load("audio.", &loader);
            loader.pop_allocator_manager();
            return model->is_loaded();
        }
        else
            return false;
    }

    bool AudioEmbeddingGeneration::load_more(ggml::type dtype, int lm_hidden_size, const json::JSON& json_config)
    {
        const auto _cfg = json_config["config.json"]["audio_config"];
        if (!_cfg.IsObject()) return false;

        config.dtype = dtype;

        config.num_mel_bins = (int)_cfg["num_mel_bins"].ToInt();
        config.encoder_layers = (int)_cfg["encoder_layers"].ToInt();
        config.encoder_attention_heads = (int)_cfg["encoder_attention_heads"].ToInt();
        config.encoder_ffn_dim = (int)_cfg["encoder_ffn_dim"].ToInt();
        config.d_model = (int)_cfg["d_model"].ToInt();
        config.scale_embedding = (int)_cfg["scale_embedding"].ToInt();
        config.max_source_positions = (int)_cfg["max_source_positions"].ToInt();

        config.audio_token_index = (int)json_config["config.json"]["audio_token_index"].ToInt();

        auto pp_cfg = json_config["preprocessor_config.json"];
        if (!pp_cfg.IsObject()) return false;

        config.chunk_length = (int)pp_cfg["chunk_length"].ToInt();
        config.feature_size = (int)pp_cfg["feature_size"].ToInt();
        config.hop_length = (int)pp_cfg["hop_length"].ToInt();
        config.n_fft = (int)pp_cfg["n_fft"].ToInt();
        config.n_samples = (int)pp_cfg["n_samples"].ToInt();
        config.nb_max_frames = (int)pp_cfg["nb_max_frames"].ToInt();
        config.sampling_rate = (int)pp_cfg["sampling_rate"].ToInt();

        _ctx.dtype = dtype;
        backend_context.init(model_gpu_layers, config.encoder_layers, GRAPH_SIZE, n_threads);

        model.reset(new AudioTransformer(&_ctx, config, lm_hidden_size));

        return true;
    }

    void AudioEmbeddingGeneration::generate(const GenerationConfig& gen_config, BaseTokenizer* tok, ggml::type dtype, std::vector<uint8_t>& buf)
    {
        if ((model.get() == nullptr) || (tok->media_emb.size() < 1)) return;
        if (!model->is_loaded()) return;

        for (auto& media : tok->media_emb)
        {
            run_model(gen_config, tok, dtype, media, buf);
        }
    }

    bool AudioEmbeddingGeneration::run_model(const GenerationConfig& gen_config, BaseTokenizer* tok, ggml::type dtype, const BaseTokenizer::MediaAsEmbeddingVector& audio, std::vector<uint8_t>& buf)
    {
        ForwardContext ctx(&backend_context);

        ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Prolog);
        ggml::tensor* media_emb = ggml::new_tensor_2d(&ctx, ggml::type::GGML_TYPE_F32, config.max_source_positions * 2, config.feature_size);

        set_dbg_ctx(&ctx);

        auto r = model->forward(&ctx, media_emb);

        if (ggml::type_of(r) != dtype)
        {
            ctx.move_to_layer(LayerAllocatorManager::MiscLayer::Epilog);
            ggml::tensor* t = ggml::new_tensor_4d(&ctx, dtype, ggml::get_dim(r, 0), ggml::get_dim(r, 1), ggml::get_dim(r, 2), ggml::get_dim(r, 3));
            r = ggml::cpy(&ctx, r, t);
        }

        ggml::build_forward_expand(&ctx, r);

        CHATLLM_CHECK(ctx.allocate()) << "failed to allocate memory";

        if (gen_config.dump_dot.size() > 0)
        {
            backend_context.dump_graph(ctx.get_cgraph(), gen_config.dump_dot.c_str());
            exit(-1);
        }

        Backend::write_tensor_data(media_emb, audio.data.data(), 0, audio.data.size() * sizeof(audio.data[0]));

        ctx.compute();

        size_t offset = buf.size();
        buf.resize(offset + ggml::nbytes(r));
        Backend::read_tensor_data(r, buf.data() + offset);
        ctx.reset();

        return true;
    }
}

namespace chatllm::qwen::v2_audio
{
    typedef v2::Config Config;
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig& config)
        : v2::Tokenizer(config, &_chat_encoder),
        audio_bos_token_id(-1),
        audio_eos_token_id(-1)
    {
    }

    void Tokenizer::inject_audio_ids(std::vector<int>& ids, const int  ids_to_inject_start, const int ids_to_inject_count)
    {
        if (audio_bos_token_id < 0)
        {
            audio_bos_token_id = tp->PieceToId("<|audio_bos|>");
            audio_eos_token_id = tp->PieceToId("<|audio_eos|>");
        }
        ids.push_back(audio_bos_token_id);
        for (int i = 0; i < ids_to_inject_count; i++)
            ids.push_back(i + ids_to_inject_start);
        ids.push_back(audio_eos_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config, ModelType type)
        : ExtendEmbedding(4096),
        Base(config, runtime_config, type),
        audio(runtime_config)
    {
        delete pad_arg;
        pad_arg = nullptr;
    }

    bool ConditionalGeneration::load_more(const json::JSON& config)
    {
        Base::load_more(config);
        bool r = audio.load_more(this->config.dtype, this->config.hidden_size, config);
        if (r)
        {
            _chat_encoder.aud_config = &audio.config;
        }
        return r;
    }

    void ConditionalGeneration::load(ModelLoader& loader)
    {
        Base::load(loader);

        loader.add_tensor_name_translations({
            {".fc1.",                   ".fc2."},
            {".fc0.",                   ".fc1."},
            });

        _chat_encoder.aud_loaded = audio.load(loader);
    }

    void ConditionalGeneration::before_generate(const GenerationConfig& gen_config)
    {
        std::vector<uint8_t> buf;
        auto emb = dynamic_cast<Embedding*>(dynamic_cast<ModelClass*>(transformer)->word_embeddings.get());
        audio.generate(gen_config, dynamic_cast<Tokenizer*>(tokenizer), ggml::type_of(emb->weight), buf);
        if (buf.size() < 1) return;

        size_t offset = emb->get_base_nbytes();
        Backend::write_tensor_data(emb->weight, buf.data(), offset, buf.size());
    }

    void ChatHistoryEncoder::append_user(int round_idx, const Content& user, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        std::ostringstream oss_prompt;

        tok->encode("user", ids, true, false, true);

        for (auto& piece : user.pieces)
        {
            if (piece.type == ContentPiece::Type::Text)
            {
                tok->encode(piece.content, ids);
            }
            else if (piece.type == ContentPiece::Type::Audio)
            {
                CHATLLM_CHECK(aud_loaded) << "Audio model not loaded";

                std::vector<float>          pcm_samples;
                std::vector<audio::mel>     mel_chunks;

                if (!audio::load(piece.content.c_str(), pcm_samples, aud_config->sampling_rate)) continue;

                audio::mel_spectrogram(pcm_samples.data(), pcm_samples.size(),
                    aud_config->n_samples,
                    aud_config->sampling_rate,
                    aud_config->feature_size,
                    aud_config->n_fft,
                    aud_config->hop_length,
                    mel_chunks);

                auto& mel = mel_chunks[0];
                CHATLLM_CHECK(mel.n_len == aud_config->max_source_positions * 2);

                tok->media_emb.push_back({ .emb_vec_number = aud_config->max_source_positions / 2, .data = {} });

                auto& media = tok->media_emb.back();
                media.data = std::move(mel.data);

                const int id_start = tok->get_image_total_emb_vectors() - media.emb_vec_number + tok->vocab_size;
                tok->inject_audio_ids(ids, id_start, media.emb_vec_number);
            }
            else
            {
                CHATLLM_THROW << "Unsupported content type: " << (int)piece.type;
            }
        }

        ids.push_back(tok->im_end_token_id);
        ids.push_back(tok->nl_token_id);
    }
}

namespace chatllm::qwen::marco_o1
{
    typedef v2::Config Config;

    Tokenizer::Tokenizer(const BaseConfig& config)
        : v2::Tokenizer(config)
    {
        sys_prompt = "\n你是一个经过良好训练的AI助手，你的名字是Marco-o1.由阿里国际数字商业集团的AI Business创造.\n        \n## 重要！！！！！\n当你回答问题时，你的思考应该在<Thought>内完成，<Output>内输出你的结果。\n<Thought>应该尽可能是英文，但是有2个特例，一个是对原文中的引用，另一个是是数学应该使用markdown格式，<Output>内的输出需要遵循用户输入的语言。\n        ";
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config)
        : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_MARCO_O1)
    {
    }
}

namespace chatllm::qwen::qwq
{
    typedef v2::Config Config;

    Tokenizer::Tokenizer(const BaseConfig& config)
        : v2::Tokenizer(config)
    {
        sys_prompt = "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.";
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config)
        : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_QWQ)
    {
    }
}

namespace chatllm::qwen::ds_r1_distill
{
    static ChatHistoryEncoder _chat_encoder;

    Tokenizer::Tokenizer(const BaseConfig& config)
        : Tokenizer(config, &_chat_encoder)
    {
    }

    Tokenizer::Tokenizer(const BaseConfig& config, BaseHistoryEncoder* encoder,
        BaseHistoryEncoder* qa_encoder,
        BaseHistoryEncoder* completion_encoder)
        : BaseTokenizer::BaseTokenizer(config, encoder, qa_encoder, completion_encoder)
    {
        sys_prompt = "";
    }

    size_t Tokenizer::load(tokenizer::DataReader* buffer, int n_vocab)
    {
        tp = new tokenizer::BPEProcessor2(
            {
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            }
            );
        size_t size = tp->Load(buffer, n_vocab);
        tp->EnableReturnSpecialToken(true);

        user_token_id = tp->PieceToId("<｜User｜>");
        assistant_token_id = tp->PieceToId("<｜Assistant｜>");

        std::vector<int> ids;
        tp->Encode("\n", &ids);

        nl_token_id = -1;

        if (ids.size() == 1)
            nl_token_id = ids[0];

        bos_token_id = tp->PieceToId("<｜begin▁of▁sentence｜>");
        eos_token_id = tp->PieceToId("<｜end▁of▁sentence｜>");

        return size;
    }

    void ChatHistoryEncoder::append_sys_prompt(std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        ids.push_back(tok->bos_token_id);
        if (tok->get_system_prompt().size() > 0)
        {
            tok->encode(tok->get_system_prompt(), ids);
        }
    }

    void ChatHistoryEncoder::append_ai(int round_idx, const std::string& ai, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        ids.push_back(tok->assistant_token_id);
        tok->encode(ai, ids);
        ids.push_back(tok->eos_token_id);
    }

    void ChatHistoryEncoder::append_user(int round_idx, const std::string& user, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        ids.push_back(tok->user_token_id);
        tok->encode(user, ids);
    }

    void ChatHistoryEncoder::append_ai_opening(int round_idx, std::vector<int>& ids) const
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        ids.push_back(tok->assistant_token_id);
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config,
        ModelType type)
        : v2::ConditionalGeneration(config, runtime_config, type, config.tie != 0)
    {
    }
}

namespace chatllm::qwen::v2_5_vl
{
    Tokenizer::Tokenizer(const BaseConfig& config) : v2::Tokenizer(config)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader* buffer, int n_vocab)
    {
        size_t r = v2::Tokenizer::load(buffer, n_vocab);

        return r;
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config)
        : v2::ConditionalGeneration(config, runtime_config, ModelType::MODEL_TYPE_QWEN2, config.tie_word_embeddings != 0)
    {
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            auto& layer = get_typed_transformer<ModelClass>()->layers[i];
        }
    }
}

namespace chatllm::qwen::v3
{
    typedef QWen3MoEBlock<128, 8> QWen3MoEBlock128_8;

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config, ModelType type, const bool skip_lm_head, int extra_tensors)
        : BaseModelForConditionalGeneration(type, config, runtime_config, 4096 * 4),
        config(config)
    {
        w_ctx_.dtype = config.dtype;

        if (skip_lm_head || config.tie_word_embeddings)
        {
            transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                nullptr,
                [&](InitContext* ctx, int layer_index) {
                    return create_layer(ctx, layer_index);
                });
        }
        else
        {
            transformer = new ModelClass(&w_ctx_, config.num_hidden_layers, config.hidden_size,
                create_embedding<Embedding>(&w_ctx_, config),
                create_final_norm<RMSNorm>(&w_ctx_, config),
                create_lm_head(&w_ctx_, config, false),
                [&](InitContext* ctx, int layer_index) {
                    return create_layer(ctx, layer_index);
                });
        }

        if (config.yarn_scaling_factor > 0.0)
            ggml::log(GGML_LOG_LEVEL_WARN, "TODO: YaRN (yarn_scaling_factor = %f) not implemented", config.yarn_scaling_factor);

        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (config.layer_is_sparse[i])
            {
                auto layer = (QWen3MoEBlock128_8*)get_typed_transformer<ModelClass>()->get_layer(i);
                layer->attention.freq_base = config.rope_theta;
                layer->mlp.norm_topk_prob = config.norm_topk_prob != 0;
            }
            else
            {
                auto layer = (QWen3Block*)get_typed_transformer<ModelClass>()->get_layer(i);
                layer->attention.freq_base = config.rope_theta;
            }
        }
    }

    int ConditionalGeneration::get_sparse_layer_num()
    {
        int num = 0;
        for (int i = 0; i < config.num_hidden_layers; i++)
        {
            if (config.layer_is_sparse[i])
                num++;
        }
        return num;
    }

    Block* ConditionalGeneration::create_layer(InitContext* ctx, int layer_index)
    {
        if (config.layer_is_sparse[layer_index])
        {
            if ((config.num_experts_per_tok == 8) && (config.num_experts == 128))
            {
                return new QWen3MoEBlock128_8(ctx, config.hidden_size, config.num_attention_heads,
                    config.intermediate_size, config.moe_intermediate_size,
                    config.num_key_value_heads, config.head_dim, config.max_length);
            }
            else
            {
                CHATLLM_CHECK(false) << "unsupported MoE param";
                return nullptr;
            }
        }
        else
        {
            return new QWen3Block(ctx, config.hidden_size, config.num_attention_heads,
                config.intermediate_size,
                config.num_key_value_heads, config.head_dim, config.max_length);
        }
    }
}

namespace chatllm::qwen::ds_r1_distill_v3
{
    Tokenizer::Tokenizer(BaseConfig config)
        : ds_r1_distill::Tokenizer(config)
    {
        std::time_t now = std::time(nullptr);
        std::tm* timeinfo = std::localtime(&now);
        char buffer[1000];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d, %A", timeinfo);
        std::string date_str(buffer);
        sys_prompt = "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是" + date_str + "。";
    }
}

namespace chatllm::qwen::v3_emb
{
    Tokenizer::Tokenizer(const BaseConfig& config) : v3::Tokenizer(config)
    {
        task = "Given a web search query, retrieve relevant passages that answer the query";
    }

    std::vector<int> Tokenizer::encode_embedding(const std::string& text, EmbeddingPurpose purpose) const
    {
        std::vector<int> ids;
        std::ostringstream oss;
        switch (purpose)
        {
        case EmbeddingPurpose::Query:
            oss << "Instruct: " << task << "\nQuery:" << text;
            BaseTokenizer::encode(oss.str(), ids);
            break;

        default:
            BaseTokenizer::encode(text, ids);
            break;
        }
        ids.push_back(eos_token_id);
        return ids;
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config, ModelType type, const bool skip_lm_head, int extra_tensors)
        : v3::ConditionalGeneration(config, runtime_config, type, skip_lm_head, extra_tensors)
    {
        dynamic_cast<HeterogeneousModel*>(transformer)->set_final_steps(std::make_unique<EmbeddingLastTokenFinalSteps>());
    }

    void ConditionalGeneration::set_additional_args(const std::map<std::string, std::string>& args)
    {
        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        tok->task = utils::get_opt(args, "task", tok->task);
    }
}

namespace chatllm::qwen::v3_ranker
{
    Tokenizer::Tokenizer(const BaseConfig& config)
        : v3_emb::Tokenizer(config)
    {
    }

    size_t Tokenizer::load(tokenizer::DataReader* buffer, int n_vocab)
    {
        size_t size = v3_emb::Tokenizer::load(buffer, n_vocab);

        yes_token_id = tp->PieceToId("yes");
        no_token_id = tp->PieceToId("no");

        return size;
    }

    void Tokenizer::encode_qa(const std::string& q, const std::string& a, std::vector<int>& ids) const
    {
        std::ostringstream oss;
        oss << "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        oss << "<Instruct>: " << task << "\n<Query>: " << q << "\n<Document>: " << a;
        oss << "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        BaseTokenizer::encode(oss.str(), ids);
    }

    class FinalSteps : public LMFinalSteps
    {
    public:
        ggml::tensor* forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states) override;
    public:
        ggml::tensor* yes_no_ids;
    };

    ggml::tensor* FinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        ggml::tensor* logits = LMFinalSteps::forward(model, ctx, input_ids, hidden_states);
        logits = ggml::reshape_2d(ctx, logits, 1, ggml::get_dim(logits, 0));
        logits = ggml::get_rows(ctx, logits, yes_no_ids);
        logits = ggml::reshape_1d(ctx, logits, 2);
        logits = ggml::soft_max(ctx, logits);
        logits = ggml::view_1d(ctx, logits, 1, 0);
        return logits;
    }

    ConditionalGeneration::ConditionalGeneration(const Config& config, const RuntimeConfig& runtime_config)
        : v3_emb::ConditionalGeneration(config, runtime_config, MODEL_TYPE_QWEN3_ReRanker, false, 1)
    {
        dynamic_cast<HeterogeneousModel*>(transformer)->set_final_steps(std::make_unique<FinalSteps>());

        FinalSteps* steps = dynamic_cast<FinalSteps*>(dynamic_cast<HeterogeneousModel*>(transformer)->get_final_steps());
        steps->yes_no_ids = ggml::new_tensor_1d(&w_ctx_, ggml::type::GGML_TYPE_I32, 2);
        w_ctx_.get_allocator()->alloc(steps->yes_no_ids);
        yes_no_ids = steps->yes_no_ids;
    }

    void ConditionalGeneration::set_tokenizer(BaseTokenizer* tokenizer)
    {
        v3::ConditionalGeneration::set_tokenizer(tokenizer);

        Tokenizer* tok = dynamic_cast<Tokenizer*>(tokenizer);
        int ids[2];
        ids[0] = tok->yes_token_id;
        ids[1] = tok->no_token_id;
        Backend::write_tensor_data(yes_no_ids, ids, 0, sizeof(ids));
    }
}
module;
#include <functional>
#include <memory>
#include <string>

module chatllm;
import :models.base;

namespace chatllm
{
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
            auto buf = allocator->alloc(layer->get_cache_size(), BackendBufAllocator::Usage::Matrix);
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
        ggml::tensor* hidden_states = word_embeddings->forward(ctx, input_ids);
        for (auto& layer : layers)
        {
            ctx->move_to_layer(layer->get_id());
            hidden_states = layer->forward(ctx, hidden_states, n_past);
        }

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

    void HeterogeneousModel::set_final_steps(std::unique_ptr<ModelFinalSteps> final_steps)
    {
        this->final_steps = std::move(final_steps);
    }

    ModelFinalSteps* HeterogeneousModel::get_final_steps()
    {
        return final_steps.get();
    }

    int HeterogeneousModel::save_session(FILE* f)
    {
        struct state state = { .cache_size = cache_size };
        if (fwrite(&state, sizeof(state), 1, f) != 1)
            return -1;

        std::vector<uint8_t> buffer;

        for (const auto &layer : layers)
        {
            buffer.resize(layer->get_cache_size());
            size_t size = layer->read_cache_data(buffer.data(), buffer.size());
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

        std::vector<uint8_t> buffer;

        for (const auto &layer : layers)
        {
            buffer.resize(layer->get_cache_size());
            if (fread(buffer.data(), 1, buffer.size(), f) != buffer.size())
                return -4;
            size_t size = layer->write_cache_data(buffer.data(), buffer.size());
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
            void* buf = session.prepare_buffer(layer_id, size);
            if (layer->read_cache_data(buf, size) != size)
                return -1;
        }

        return 0;
    }

    int HeterogeneousModel::load_session(ModelSessionMemory& session)
    {
        for (size_t layer_id = 0; layer_id < layers.size(); layer_id++)
        {
            const auto &layer = layers[layer_id];
            size_t size = 0;
            void* buf = session.get_buffer(layer_id, &size);
            if (size != layer->get_cache_size()) return -1;
            if (layer->write_cache_data(buf, size) != size)
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

    ggml::tensor* LMFinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        hidden_states = ggml::view_2d(ctx, hidden_states, model->hidden_size, 1,
            ggml::row_size(hidden_states),
            (ggml::get_dim(input_ids, 0) - 1) * ggml::row_size(hidden_states));

        ggml::tensor* transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);

        transformer_outputs =
            ggml::view_1d(ctx, transformer_outputs, model->hidden_size, 0);

        ggml::tensor* lm_logits = model->lm_head ? model->lm_head->forward(ctx, transformer_outputs)
            : model->word_embeddings->forward(ctx, transformer_outputs);

        if (model->logits_pp)
            lm_logits = model->logits_pp->forward(ctx, lm_logits);
        return lm_logits;
    }

    ggml::tensor* EmbeddingPoolingFinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        ggml::tensor* transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);

        return transformer_outputs;
    }

    ggml::tensor* EmbeddingLastTokenFinalSteps::forward(HeterogeneousModel* model, ComputeContext* ctx, ggml::tensor* input_ids, ggml::tensor* hidden_states)
    {
        hidden_states = ggml::view_2d(ctx, hidden_states, model->hidden_size, 1,
            ggml::row_size(hidden_states),
            (ggml::get_dim(input_ids, 0) - 1) * ggml::row_size(hidden_states));
        ggml::tensor* transformer_outputs = model->final_layernorm->forward(ctx, hidden_states);
        transformer_outputs = ggml::simple_norm(ctx, transformer_outputs, 1e-5f);
        return transformer_outputs;
    }
}

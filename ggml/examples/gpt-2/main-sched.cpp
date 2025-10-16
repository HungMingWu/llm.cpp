#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <print>
#include <random>
#include <string>
#include <vector>

import ggml;
import gpt.common;

static void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx = 1024;
    int32_t n_embd = 768;
    int32_t n_head = 12;
    int32_t n_layer = 12;
    int32_t ftype = 1;
    float   eps = 1e-5f;
};

struct gpt2_layer {
    // normalization
    ggml_tensor* ln_1_g;
    ggml_tensor* ln_1_b;

    ggml_tensor* ln_2_g;
    ggml_tensor* ln_2_b;

    // attention
    ggml_tensor* c_attn_attn_w;
    ggml_tensor* c_attn_attn_b;

    ggml_tensor* c_attn_proj_w;
    ggml_tensor* c_attn_proj_b;

    // mlp
    ggml_tensor* c_mlp_fc_w;
    ggml_tensor* c_mlp_fc_b;

    ggml_tensor* c_mlp_proj_w;
    ggml_tensor* c_mlp_proj_b;
};

struct gpt2_model {
    gpt2_hparams hparams;

    // normalization
    ggml_tensor* ln_f_g;
    ggml_tensor* ln_f_b;

    ggml_tensor* wte;     // position embedding
    ggml_tensor* wpe;     //    token embedding
    ggml_tensor* lm_head; // language model head

    std::vector<gpt2_layer> layers;

    // key + value memory
    ggml_tensor* memory_k;
    ggml_tensor* memory_v;

    //
    ggml_context ctx_w;

    std::vector<std::unique_ptr<ggml_backend>> backends;
    std::vector<ggml_backend*> backends_view;
    std::vector<ggml_backend_buffer_type*> backend_buft;
    std::vector<std::unique_ptr<ggml_backend_buffer>> buffers_w;
    std::unique_ptr<ggml_backend_buffer> buffer_kv;
    std::unique_ptr<ggml_backend_buffer> buffer_input;

    std::map<std::string, ggml_tensor*> tensors;

    // inputs/constants
    ggml_tensor* embd;
    ggml_tensor* position;
};

void init_backends(gpt2_model& model, const gpt_params& params) {
    std::unique_ptr<ggml_backend> gpu_backend;

    ggml_log_set(ggml_log_callback_default);

    // initialize the backends
#ifdef GGML_USE_CUDA
    if (params.n_gpu_layers > 0) {
        std::println(stderr, "{}: using CUDA backend", __func__);
        gpu_backend = ggml_backend_cuda_init(0);
        if (!gpu_backend) {
            std::println(stderr, "{}: ggml_backend_cuda_init() failed", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (params.n_gpu_layers > 0) {
        std::println(stderr, "{}: using Metal backend", __func__);
        gpu_backend = ggml_backend_metal_init();
        if (!gpu_backend) {
            std::println(stderr, "{}: ggml_backend_metal_init() failed", __func__);
        }
    }
#endif
    if (gpu_backend) {
        model.backends.push_back(std::move(gpu_backend));
        model.backends_view.push_back(model.backends.back().get());
        model.backend_buft.push_back(model.backends.back()->get_default_buffer_type());
    }

#ifdef GGML_USE_BLAS
    ggml_backend* blas_backend = ggml_backend_blas_init();
    if (!blas_backend) {
        std::println(stderr, "{}: failed to initialize BLAS backend", __func__);
    }
    else {
        ggml_backend_blas_set_n_threads(blas_backend, params.n_threads);
        model.backends.push_back(blas_backend);
        model.backends_view.push_back(model.backends.back().get());
        model.backend_buft.push_back(model.backends.back()->get_default_buffer_type());
    }
#endif

    // always add the CPU backend as a fallback
    std::unique_ptr<ggml_cpu_backend> cpu_backend = ggml_backend_cpu_init();
    cpu_backend->set_n_threads(params.n_threads);
    model.backends.push_back(std::move(cpu_backend));
    model.backends_view.push_back(model.backends.back().get());
    model.backend_buft.push_back(model.backends.back()->get_default_buffer_type());
}

// load the model's weights from a file
bool gpt2_model_load(const std::string& fname, gpt2_model& model, gpt_vocab& vocab, const gpt_params& params) {
    std::println("{}: loading model from '{}'", __func__, fname);

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        std::println(stderr, "{}: failed to open '{}'", __func__, fname);
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char*)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            std::println(stderr, "{}: invalid model file '{}' (bad magic)", __func__, fname);
            return false;
        }
    }

    // load hparams
    {
        auto& hparams = model.hparams;

        fin.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char*)&hparams.n_ctx, sizeof(hparams.n_ctx));
        fin.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char*)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char*)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        std::println("{}: n_vocab = {}", __func__, hparams.n_vocab);
        std::println("{}: n_ctx   = {}", __func__, hparams.n_ctx);
        std::println("{}: n_embd  = {}", __func__, hparams.n_embd);
        std::println("{}: n_head  = {}", __func__, hparams.n_head);
        std::println("{}: n_layer = {}", __func__, hparams.n_layer);
        std::println("{}: ftype   = {}", __func__, hparams.ftype);
        std::println("{}: qntvr   = {}", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char*)&n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            std::println(stderr, "{}: invalid model file '{}' (bad vocab size {} != {})",
                __func__, fname, n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char*)&len, sizeof(len));

            buf.resize(len);
            fin.read((char*)buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype)(model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        std::println(stderr, "{}: invalid model file '{}' (bad ftype value {})",
            __func__, fname, model.hparams.ftype);
        return false;
    }

    auto& ctx = model.ctx_w;

    // create tensors for the weights
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.ln_f_g = ctx.create(GGML_TYPE_F32, { n_embd });
        model.ln_f_b = ctx.create(GGML_TYPE_F32, { n_embd });

        model.wte = ctx.create(wtype, { n_embd, n_vocab });
        model.wpe = ctx.create(GGML_TYPE_F32, { n_embd, n_ctx });
        model.lm_head = ctx.create(wtype, { n_embd, n_vocab });

        // map by name
        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wte"] = model.wte;
        model.tensors["model/wpe"] = model.wpe;
        model.tensors["model/lm_head"] = model.lm_head;

        for (int i = 0; i < n_layer; ++i) {
            auto& layer = model.layers[i];

            layer.ln_1_g = ctx.create(GGML_TYPE_F32, { n_embd });
            layer.ln_1_b = ctx.create(GGML_TYPE_F32, { n_embd });

            layer.ln_2_g = ctx.create(GGML_TYPE_F32, { n_embd });
            layer.ln_2_b = ctx.create(GGML_TYPE_F32, { n_embd });

            layer.c_attn_attn_w = ctx.create(wtype, { n_embd, 3 * n_embd });
            layer.c_attn_attn_b = ctx.create(GGML_TYPE_F32, { 3 * n_embd });

            layer.c_attn_proj_w = ctx.create(wtype, { n_embd, n_embd });
            layer.c_attn_proj_b = ctx.create(GGML_TYPE_F32, { n_embd });

            layer.c_mlp_fc_w = ctx.create(wtype, { n_embd, 4 * n_embd });
            layer.c_mlp_fc_b = ctx.create(GGML_TYPE_F32, { 4 * n_embd });

            layer.c_mlp_proj_w = ctx.create(wtype, { 4 * n_embd, n_embd });
            layer.c_mlp_proj_b = ctx.create(GGML_TYPE_F32, { n_embd });

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"] = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"] = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"] = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"] = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"] = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"] = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"] = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"] = layer.c_mlp_proj_b;
        }
    }

    // assign tensors to backends
    init_backends(model, params);
    auto &backend_gpu = model.backends.front();
    auto &backend_cpu = model.backends.back();
    std::map<std::string, ggml_backend*> tensor_backends;
    {
        const int i_gpu_first_layer = model.hparams.n_layer - params.n_gpu_layers;
        for (auto it : model.tensors) {
            const std::string& name = it.first;
            // input tensors
            if (name == "model/wte" || name == "model/wpe") {
                if (params.n_gpu_layers > model.hparams.n_layer) {
                    tensor_backends[name] = backend_gpu.get();
                }
                else {
                    tensor_backends[name] = backend_cpu.get();
                }
            }
            // output tensors
            if (name == "model/ln_f/g" || name == "model/ln_f/b" || name == "model/lm_head") {
                if (params.n_gpu_layers > 0) {
                    tensor_backends[name] = backend_gpu.get();
                }
                else {
                    tensor_backends[name] = backend_cpu.get();
                }
            }
            // layer tensors
            if (name.substr(0, 7) == "model/h") {
                // parse layer number
                int layer = std::stoi(name.substr(7, 2));
                if (layer >= i_gpu_first_layer) {
                    tensor_backends[name] = backend_gpu.get();
                }
                else {
                    tensor_backends[name] = backend_cpu.get();
                }
            }
        }
    }

    // allocate buffers
    std::map<ggml_backend*, ggml_tallocr> backend_buffers;
    for (auto &backend : model.backends) {
        // compute the size of the buffer
        size_t size = 0;
        for (auto it : model.tensors) {
            if (tensor_backends[it.first] == backend.get()) {
                size += it.second->nbytes() + 512;
            }
        }
        if (size > 0) {
            std::println("{}: {:8} buffer size = {:8.2f} MB", __func__, backend->get_name(), size / 1024.0 / 1024.0);
            // allocate the buffer
            std::unique_ptr<ggml_backend_buffer> buffer = backend->alloc_buffer(size);
            buffer->setUsage(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            // create an allocator for the buffer to allocate the tensors
            backend_buffers.insert(std::make_pair(backend.get(), ggml_tallocr(buffer.get())));

            model.buffers_w.push_back(std::move(buffer));
        }
        else {
            model.buffers_w.push_back(NULL);
        }
    }

    // allocate key + value memory
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;

        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.memory_k = ctx.create(GGML_TYPE_F32, { n_elements });
        model.memory_v = ctx.create(GGML_TYPE_F32, { n_elements });

        model.memory_k->set_name("model/memory_k");
        model.memory_v->set_name("model/memory_v");

        const size_t memory_size = model.memory_k->nbytes() + model.memory_v->nbytes();

        std::println("{}: memory size = {:8.2f} MB, n_mem = {}", __func__, memory_size / 1024.0 / 1024.0, n_mem);

        // create a backend buffer (can be in host or device memory)
        auto& backend_kv = params.n_gpu_layers >= hparams.n_layer / 2 ? backend_gpu : backend_cpu;
        std::println("{}: backend_kv = {}", __func__, backend_kv->get_name());
        model.buffer_kv = backend_kv->alloc_buffer(memory_size + 512 * 2);

        // allocate the tensors into the backend buffer
        {
            ggml_tallocr alloc(model.buffer_kv.get());

            // this updates the pointers in the tensors to point to the correct location in the buffer
            // this is necessary since the ggml_context is .no_alloc == true
            // note that the buffer can actually be a device buffer, depending on the backend
            alloc.alloc(model.memory_k);
            alloc.alloc(model.memory_v);
        }
    }

    // load weights
    {
        size_t total_size = 0;

        bool has_lm_head = false;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char*>(&length), sizeof(length));
            fin.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end()) {
                std::println(stderr, "{}: unknown tensor '{}' in model file", __func__, name);
                return false;
            }

            auto tensor = model.tensors[name];
            tensor->set_name(name);
            if (tensor->nelements() != nelements) {
                std::println(stderr, "{}: tensor '{}' has wrong size in model file", __func__, name);
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                std::println(stderr, "{}: tensor '{}' has wrong shape in model file: got [{}, {}], expected [{}, {}]",
                    __func__, name, tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                std::println("{:24} - [{:5}, {:5}], type = {:6}, {:6.2f} MB, {:9} bytes", name, ne[0], ne[1], ggml_type_name(ggml_type(ttype)), tensor->nbytes() / 1024.0 / 1024.0, tensor->nbytes());
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != tensor->nbytes()) {
                std::println(stderr, "{}: tensor '{}' has wrong size in model file: got {}, expected {}",
                    __func__, name, tensor->nbytes(), nelements * bpe);
                return false;
            }

            // allocate the tensor
            ggml_backend* backend = tensor_backends[name];
            auto &alloc = backend_buffers.find(backend)->second;
            alloc.alloc(tensor);
            //std::println("{}: [{:5.5}] {}", __func__, backends->get_name(), name.c_str());

            if (0 //ggml_backend_is_cpu(backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(backend)
#endif
                ) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char*>(tensor->data), tensor->nbytes());
            }
            else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(tensor->nbytes());
                fin.read(read_buf.data(), tensor->nbytes());
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, tensor->nbytes());
            }

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                auto &alloc_head = backend_buffers.find(tensor_backends["model/lm_head"])->second;
                alloc_head.alloc(model.lm_head);
                //std::println("{}: [{:5.5}] {} (copied)", __func__, tensor_backends["model/lm_head"]->get_name(), "model/lm_head");
                ggml_backend_tensor_copy(tensor, model.lm_head);
                total_size += model.lm_head->nbytes();
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += tensor->nbytes();
        }
        std::println("{}: model size  = {:8.2f} MB", __func__, total_size / 1024.0 / 1024.0);
    }

    fin.close();

    // allocate input tensors
    {
        model.embd = ctx.create(GGML_TYPE_I32, { model.hparams.n_ctx });
        model.position = ctx.create(GGML_TYPE_I32, { model.hparams.n_ctx });

        model.embd->set_name("in/embd");
        model.position->set_name("in/position");

        // add input tensors to cpu backend
        size_t input_size = model.embd->nbytes() + model.position->nbytes();

        // FIXME: use cpu backend after sched impl
        auto &backend_input = params.n_gpu_layers >= model.hparams.n_layer ? backend_gpu : backend_cpu;
        model.buffer_input = backend_input->alloc_buffer(input_size + 512 * 3);
        std::println("{}: backend_in = {} ({} bytes)", __func__, backend_input->get_name(), input_size);

        // allocate the tensors into the backend buffer
        ggml_tallocr alloc(model.buffer_input.get());
        alloc.alloc(model.embd);
        alloc.alloc(model.position);
    }

    return true;
}

// build the computation graph
ggml_cgraph gpt2_graph(
    ggml_context &ctx,
    const gpt2_model& model,
    const int n_past,
    const std::vector<gpt_vocab::id>& embd_inp) {
    const int N = embd_inp.size();

    const auto& hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;

    ggml_cgraph gf;

    ggml_tensor* embd = ggml_view(&ctx, model.embd, { N }, {}, 0);

    // set inputs
    // TODO: move to gpt2_eval
    ggml_backend_tensor_set(model.embd, embd_inp.data(), 0, N * ggml_element_size(embd));

    ggml_tensor* position = ggml_view(&ctx, model.position, { N }, {}, 0);
    for (int i = 0; i < N; ++i) {
        int32_t v = n_past + i;
        ggml_backend_tensor_set(model.position, &v, i * sizeof(int32_t), sizeof(v));
    }

    const float KQ_scale = 1.0f / sqrtf(float(model.hparams.n_embd) / model.hparams.n_head);

    // wte + wpe
    ggml_tensor* inpL =
        ggml_add(&ctx,
            ggml_get_rows(&ctx, model.wte, embd),
            ggml_get_rows(&ctx, model.wpe, position), false);
    inpL->set_name("inpL");
    inpL->src[0]->set_name("wte");
    inpL->src[1]->set_name("wpe");

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(&ctx, inpL, hparams.eps, false);
            cur->set_name("l{}.norm", il);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(&ctx,
                ggml_mul(&ctx,
                    cur,
                    model.layers[il].ln_1_g, false),
                model.layers[il].ln_1_b, false);
            cur->set_name("l{}.ln_1_b", il);
            cur->src[0]->set_name("l{}.ln_1_g", il);
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_mul_mat(&ctx,
                model.layers[il].c_attn_attn_w,
                cur);
            cur->set_name("l{}.attn_w", il);

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_attn_attn_b, false);
            cur->set_name("l{}.attn_b", il);
        }

        // self-attention
        {
            ggml_tensor* Qcur = ggml_view(&ctx, cur, { n_embd, N }, { cur->nb[1] }, 0 * sizeof(float) * n_embd);
            ggml_tensor* Kcur = ggml_view(&ctx, cur, { n_embd, N }, { cur->nb[1] }, 1 * sizeof(float) * n_embd);
            ggml_tensor* Vcur = ggml_view(&ctx, cur, { n_embd, N }, { cur->nb[1] }, 2 * sizeof(float) * n_embd);

            Qcur->set_name("l{}.Qcur", il);
            Kcur->set_name("l{}.Kcur", il);
            Vcur->set_name("l{}.Vcur", il);

            // store key and value to memory
            if (N >= 1) {
                ggml_tensor* k = ggml_view(&ctx, model.memory_k, { N * n_embd }, {}, (ggml_element_size(model.memory_k) * n_embd) * (il * n_ctx + n_past));
                ggml_tensor* v = ggml_view(&ctx, model.memory_v, { N * n_embd }, {}, (ggml_element_size(model.memory_v) * n_embd) * (il * n_ctx + n_past));

                gf.build_forward_expand(ggml_cpy(&ctx, Kcur, k));
                gf.build_forward_expand(ggml_cpy(&ctx, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            ggml_tensor* Q =
                ggml_permute(&ctx,
                    ggml_cont(&ctx, Qcur, { n_embd / n_head, n_head, N }),
                    0, 2, 1, 3);
            Q->set_name("l{}.Q", il);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            ggml_tensor* K =
                ggml_permute(&ctx,
                    ggml_reshape(&ctx,
                        ggml_view(&ctx, model.memory_k, { (n_past + N) * n_embd }, {}, il * n_ctx * ggml_element_size(model.memory_k) * n_embd),
                        { n_embd / n_head, n_head, n_past + N }),
                    0, 2, 1, 3);
            K->set_name("l{}.K", il);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(&ctx0,
            //            ggml_permute(&ctx0,
            //                ggml_reshape(&ctx0,
            //                    ggml_view(&ctx0, model.memory_v, { (n_past + N) * n_embd }, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    { n_embd / n_head, n_head, n_past + N }),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(&ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(&ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            ggml_tensor* KQ = ggml_mul_mat(&ctx, K, Q);
            KQ->set_name("l{}.KQ", il);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            ggml_tensor* KQ_scaled = ggml_scale(&ctx, KQ, KQ_scale, false);
            KQ_scaled->set_name("l{}.KQ_scaled", il);

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            ggml_tensor* KQ_masked = ggml_diag_mask_inf(&ctx, KQ_scaled, n_past, false);
            KQ_masked->set_name("l{}.KQ_masked", il);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            ggml_tensor* KQ_soft_max = ggml_soft_max(&ctx, KQ_masked, false);
            KQ_soft_max->set_name("l{}.KQ_soft_max", il);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            ggml_tensor* V_trans =
                ggml_cont(&ctx,
                    ggml_permute(&ctx,
                        ggml_reshape(&ctx,
                            ggml_view(&ctx, model.memory_v, { (n_past + N) * n_embd }, {}, il * n_ctx * ggml_element_size(model.memory_v) * n_embd),
                            { n_embd / n_head, n_head, n_past + N }),
                        1, 2, 0, 3),
                    { n_past + N, n_embd / n_head, n_head });

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            ggml_tensor* KQV = ggml_mul_mat(&ctx, V_trans, KQ_soft_max);
            KQV->set_name("l{}.KQV", il);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            ggml_tensor* KQV_merged = ggml_permute(&ctx, KQV, 0, 2, 1, 3);
            KQV_merged->set_name("l{}.KQV_merged", il);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cont(&ctx, KQV_merged, { n_embd, N });
            cur->set_name("l{}.KQV_merged_contiguous", il);
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(&ctx,
                model.layers[il].c_attn_proj_w,
                cur);
            cur->set_name("l{}.attn_proj_w", il);

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_attn_proj_b, false);
            cur->set_name("l{}.attn_proj_b", il);
        }

        // add the input
        cur = ggml_add(&ctx, cur, inpL, false);
        cur->set_name("l{}.add", il);

        ggml_tensor* inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(&ctx, inpFF, hparams.eps, false);
                cur->set_name("l{}.FFnorm", il);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(&ctx,
                    ggml_mul(&ctx,
                        cur,
                        model.layers[il].ln_2_g, false),
                    model.layers[il].ln_2_b, false);
                cur->set_name("l{}.ln_2_b", il);
                cur->src[0]->set_name("l{}.ln_2_g", il);
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(&ctx,
                model.layers[il].c_mlp_fc_w,
                cur);
            cur->set_name("l{}.mlp_fc_w", il);

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_mlp_fc_b, false);
            cur->set_name("l{}.mlp_fc_b", il);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(&ctx, cur, false);
            cur->set_name("l{}.gelu", il);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(&ctx,
                model.layers[il].c_mlp_proj_w,
                cur);
            cur->set_name("l{}.mlp_proj_w", il);

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_mlp_proj_b, false);
            cur->set_name("l{}.mlp_proj_b", il);
        }

        // input for next layer
        inpL = ggml_add(&ctx, cur, inpFF, false);
        inpL->set_name("l{}.add2", il);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(&ctx, inpL, hparams.eps, false);
        inpL->set_name("out_norm");

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(&ctx,
            ggml_mul(&ctx,
                inpL,
                model.ln_f_g, false),
            model.ln_f_b, false);
        inpL->set_name("out_ln_f_b");
        inpL->src[0]->set_name("out_ln_f_g");
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(&ctx, model.lm_head, inpL);
    inpL->set_name("out_lm_head");

    // logits -> probs
    //inpL = ggml_soft_max(&ctx0, inpL, false);

    gf.build_forward_expand(inpL);

    return gf;
}

// evaluate the transformer
//
//   - model:     the model
//   - sched:     the backend scheduler
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt2_eval(
    const gpt2_model& model,
    ggml_backend_sched* sched,
    const int n_past,
    const std::vector<gpt_vocab::id>& embd_inp,
    std::vector<float>& embd_w) {
    const int N = embd_inp.size();

    const auto& hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;
    ggml_context ctx;

    ggml_cgraph gf = gpt2_graph(ctx, model, n_past, embd_inp);

    // run the computation
    sched->reset();

    sched->graph_compute(gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // in this case, the output tensor is the last one in the graph
    ggml_tensor* inpL = gf.getNodes().back();

    //embd_w.resize(n_vocab*N);
    //ggml_backend_tensor_get(inpL, embd_w.data(), 0, sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    ggml_backend_tensor_get(inpL, embd_w.data(), (n_vocab * (N - 1)) * sizeof(float), sizeof(float) * n_vocab);

    return true;
}

int main(int argc, char** argv) {
    Stopwatch main_sw;
    gpt_params params;
    params.model = "models/gpt-2-117M/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    std::println("{}: seed = {}", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gpt2_model model;

    // load the model
    {
        Stopwatch start_sw;
        if (!gpt2_model_load(params.model, model, vocab, params)) {
            std::println(stderr, "{}: failed to load model from '{}'", __func__, params.model);
            return 1;
        }
        t_load_us = start_sw.get_elapsed();

        test_gpt_tokenizer(vocab, params.token_test);
    }

    // create the backend scheduler
    // the scheduler handles the allocation of the compute buffers and the scheduling of the computation between the different backends
    std::unique_ptr<ggml_backend_sched> sched;
    {
        // initialize the scheduler
        sched = std::make_unique<ggml_backend_sched>(model.backends_view, model.backend_buft, false, true);

        // create the worst case graph for memory usage estimation
        int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
        int n_past = model.hparams.n_ctx - n_tokens;
        ggml_context ctx;
        ggml_cgraph gf = gpt2_graph(ctx, model, n_past, std::vector<gpt_vocab::id>(n_tokens, 0));

        sched->reserve(&gf);


        // compute the required memory
        size_t mem_size = 0;
        for (size_t i = 0; i < model.backends.size(); i++) {
            size_t size = sched->get_buffer_size(model.backends[i].get());
            if (size > 0) {
                mem_size += size;
                std::println("{}: {:8} compute buffer size = {:8.2f} MB", __func__, model.backends[i]->get_name(), size / 1024.0 / 1024.0);
                //std::println("{}: {:8} compute buffer size = {} bytes", __func__, model.backends[i]->get_name(), size);
            }
        }

        std::println("{}: total compute buffer size: {:.2f} MB", __func__, mem_size / 1024.0 / 1024.0);
    }

    int n_past = 0;

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

    std::println("{}: prompt: '{}'", __func__, params.prompt);
    std::print("{}: number of tokens in prompt = {}, first 8 tokens: ", __func__, embd_inp.size());
    for (int i = 0; i < std::min(8, (int)embd_inp.size()); i++) {
        std::print("{} ", embd_inp[i]);
    }
    std::print("\n\n");

    // submit the input prompt token-by-token
    // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
    std::vector<gpt_vocab::id> embd;

    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            Stopwatch predict_sw;
            if (!gpt2_eval(model, sched.get(), n_past, embd, logits)) {
                std::println("Failed to predict");
                return 1;
            }
            t_predict_us += predict_sw.get_elapsed();
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                Stopwatch samplet_sw;
                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
                t_sample_us += samplet_sw.get_elapsed();
            }

            // add it to the context
            embd.push_back(id);
        }
        else {
            // if here, it means we are still processing the input prompt
            for (size_t k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (int32_t(embd.size()) >= params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            std::print("{}", vocab.id_to_token[id]);
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 50256) {
            break;
        }
    }

    // report timing
    {
        std::print("\n\n");
        std::println("{}:     load time = {:8.2f} ms", __func__, t_load_us / 1000.0f);
        std::println("{}:   sample time = {:8.2f} ms", __func__, t_sample_us / 1000.0f);
        std::println("{}:  predict time = {:8.2f} ms / {:.2f} ms per token", __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
        std::println("{}:    total time = {:8.2f} ms", __func__, main_sw.get_elapsed() / 1000.0f);
    }

    return 0;
}

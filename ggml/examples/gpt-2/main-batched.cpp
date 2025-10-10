#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <print>
#include <random>
#include <set>
#include <string>
#include <vector>

#define GGML_ASSERT(...) assert(__VA_ARGS__)

import ggml;
import gpt.common;

static void ggml_log_callback_default(ggml_log_level, std::string_view text) {
    std::println("{}", text);
}

using gpt2_pos = int32_t;
using gpt2_seq_id = int32_t;

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

struct gpt2_kv_cell {
    gpt2_pos pos = -1;
    gpt2_pos delta = 0;

    std::set<gpt2_seq_id> seq_id;

    bool has_seq_id(const gpt2_seq_id& id) const {
        return seq_id.find(id) != seq_id.end();
    }
};

struct gpt2_kv_cache {
    // key + value memory
    ggml_tensor* k;
    ggml_tensor* v;
    //

    uint32_t head = 0;
    uint32_t size = 0;

    // computed before each graph build
    uint32_t n = 0;

    std::vector<gpt2_kv_cell> cells;

    std::unique_ptr<ggml_backend_buffer> buffer;

    void seq_cp(gpt2_seq_id seq_id_src, gpt2_seq_id seq_id_dst,
        gpt2_pos p0, gpt2_pos p1);
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

    gpt2_kv_cache kv_cache;

    ggml_context ctx_w;

    std::unique_ptr<ggml_backend> backend;

    std::unique_ptr<ggml_backend_buffer> buffer_w;

    std::map<std::string, ggml_tensor*> tensors;
};

// Input data for gpt2_decode
// A gpt2_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
// - seq_id : the sequence to which the respective token belongs
// - logits : if zero, the logits for the respective token will not be output
//
struct gpt2_batch {
    int32_t n_tokens = -1;

    std::vector<gpt_vocab::id> token;
    std::vector<float> embd;
    std::vector<gpt2_pos> pos;
    std::vector<gpt2_seq_id> seq_id;
    std::vector<int8_t> logits;
public:
    gpt2_batch(int32_t n_tokens, int32_t embd);
};

// load the model's weights from a file
bool gpt2_model_load(const std::string& fname, gpt2_model& model, gpt_vocab& vocab, int n_ctx, int n_gpu_layers) {
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
            __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto& ctx = model.ctx_w;

    size_t buffer_size = 0;

    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        buffer_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_g
        buffer_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_b

        buffer_size += ggml_row_size(wtype, n_vocab * n_embd); // wte
        buffer_size += ggml_row_size(GGML_TYPE_F32, n_ctx * n_embd); // wpe
        buffer_size += ggml_row_size(wtype, n_vocab * n_embd); // lm_head

        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_g
        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_b

        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_g
        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_b

        buffer_size += n_layer * (ggml_row_size(wtype, 3 * n_embd * n_embd)); // c_attn_attn_w
        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, 3 * n_embd));        // c_attn_attn_b

        buffer_size += n_layer * (ggml_row_size(wtype, n_embd * n_embd));   // c_attn_proj_w
        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd));          // c_attn_proj_b

        buffer_size += n_layer * (ggml_row_size(wtype, 4 * n_embd * n_embd)); // c_mlp_fc_w
        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, 4 * n_embd));        // c_mlp_fc_b

        buffer_size += n_layer * (ggml_row_size(wtype, 4 * n_embd * n_embd)); // c_mlp_proj_w
        buffer_size += n_layer * (ggml_row_size(GGML_TYPE_F32, 4 * n_embd));        // c_mlp_proj_b

        buffer_size += (6 + 12 * n_layer) * 128; // alignment overhead

        std::println("{}: ggml tensor size    = {} bytes", __func__, sizeof(ggml_tensor));
        std::println("{}: backend buffer size = {:6.2f} MB", __func__, buffer_size / (1024.0 * 1024.0));
    }

    ggml_log_set(ggml_log_callback_default);

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        std::println(stderr, "{}: using CUDA backend", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            std::println(stderr, "{}: ggml_backend_cuda_init() failed", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        std::println(stderr, "{}: using Metal backend", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            std::println(stderr, "{}: ggml_backend_metal_init() failed", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        std::println(stderr, "{}: using CPU backend", __func__);
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        std::println(stderr, "{}: ggml_backend_cpu_init() failed", __func__);
        return false;
    }

    // allocate weights buffer
    model.buffer_w = model.backend->alloc_buffer(buffer_size);

    // prepare memory for the weights
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

    // override the default training context with the user-provided
    model.hparams.n_ctx = n_ctx;

    // key + value memory
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;

        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.kv_cache.k = ctx.create(GGML_TYPE_F32, { n_elements });
        model.kv_cache.v = ctx.create(GGML_TYPE_F32, { n_elements });

        model.kv_cache.head = 0;
        model.kv_cache.size = n_ctx;

        model.kv_cache.cells.resize(n_ctx);

        const size_t memory_size = model.kv_cache.k->nbytes() + model.kv_cache.v->nbytes();

        std::println("{}: memory size = {:8.2f} MB, n_mem = {}", __func__, memory_size / 1024.0 / 1024.0, n_mem);

        // create a backend buffer (can be in host or device memory)
        model.kv_cache.buffer = model.backend->alloc_buffer(memory_size + 256);

        // allocate the tensors into the backend buffer
        {
            ggml_tallocr alloc(model.kv_cache.buffer.get());

            // this updates the pointers in the tensors to point to the correct location in the buffer
            // this is necessary since the ggml_context is .no_alloc == true
            // note that the buffer can actually be a device buffer, depending on the backend
            alloc.alloc(model.kv_cache.k);
            alloc.alloc(model.kv_cache.v);
        }
    }

    // load weights
    {
        ggml_tallocr alloc(model.buffer_w.get());

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
                std::println("{:24} - [{:5}, {:5}], type = {:6}, {:6.2} MB, {:9} bytes", name, ne[0], ne[1], ggml_type_name(ggml_type(ttype)), tensor->nbytes() / 1024.0 / 1024.0, tensor->nbytes());
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != tensor->nbytes()) {
                std::println(stderr, "{}: tensor '{}' has wrong size in model file: got {}, expected {}",
                    __func__, name, tensor->nbytes(), nelements * bpe);
                return false;
            }

            alloc.alloc(tensor);

            if (0 //ggml_backend_is_cpu(model.backend)
#ifdef GGML_USE_METAL
                || ggml_backend_is_metal(model.backend)
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
                //ggml_tallocr_alloc(alloc, model.lm_head);
                //ggml_backend_tensor_copy(tensor, model.lm_head);
                model.lm_head = tensor;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += tensor->nbytes();
        }

        std::println("{}: model size  = {:8.2f} MB", __func__, total_size / 1024.0 / 1024.0);
    }

    fin.close();

    return true;
}

// build the computation graph
ggml_cgraph gpt2_graph(
    gpt2_model& model,
    const  gpt2_batch& batch,
    bool    measure) {
    const auto& hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;

    const auto& kv_cache = model.kv_cache;

    const int32_t n_tokens = batch.n_tokens;
    const int32_t n_kv = measure ? n_ctx : kv_cache.n;
    const int32_t kv_head = measure ? n_ctx - n_tokens : kv_cache.head;

    auto& ctx = model.ctx_w;

    ggml_cgraph gf;

    ggml_tensor* inpL;
    if (!batch.token.empty()) {
        ggml_tensor* inp_tokens = ctx.create(GGML_TYPE_I32, { n_tokens });
        inp_tokens->set_name("inp_tokens");
        inp_tokens->set_flag(GGML_TENSOR_FLAG_INPUT);

        ggml_tensor* position = ctx.create(GGML_TYPE_I32, { n_tokens });
        position->set_name("position");
        position->set_flag(GGML_TENSOR_FLAG_INPUT);

        // wte + wpe
        inpL =
            ggml_add(&ctx,
                ggml_get_rows(&ctx, model.wte, inp_tokens),
                ggml_get_rows(&ctx, model.wpe, position));
    }
    else {
        GGML_ASSERT(!batch.embd.empty());

        inpL = ctx.create(GGML_TYPE_F32, { n_embd, n_tokens });
        inpL->set_name("embd");
        inpL->set_flag(GGML_TENSOR_FLAG_INPUT);
    }

    // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
    ggml_tensor* KQ_mask = ctx.create(GGML_TYPE_F32, { n_kv, n_tokens, 1 });
    KQ_mask->set_name("KQ_mask");
    KQ_mask->set_flag(GGML_TENSOR_FLAG_INPUT);


    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor* cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(&ctx, inpL, hparams.eps);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(&ctx,
                ggml_mul(&ctx,
                    cur,
                    model.layers[il].ln_1_g),
                model.layers[il].ln_1_b);
        }

        // attn
        // [2304,        768] - model.layers[il].c_attn_attn_w
        // [2304,          1] - model.layers[il].c_attn_attn_b
        // [ 768,   n_tokens] - cur (in)
        // [2304,   n_tokens] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, n_tokens]
        {
            cur = ggml_mul_mat(&ctx,
                model.layers[il].c_attn_attn_w,
                cur);

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_attn_attn_b);
        }

        // self-attention
        {
            ggml_tensor* Qcur = ggml_view(&ctx, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * n_embd);
            ggml_tensor* Kcur = ggml_view(&ctx, cur, { n_embd, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * n_embd);
            ggml_tensor* Vcur = ggml_view(&ctx, cur, { n_embd, n_tokens }, { cur->nb[1] }, 2 * sizeof(float) * n_embd);

            // store key and value to memory
            if (n_tokens >= 1) {
                ggml_tensor* k = ggml_view(&ctx, model.kv_cache.k, { n_tokens * n_embd }, {}, (ggml_element_size(model.kv_cache.k) * n_embd) * (il * n_ctx + kv_head));
                ggml_tensor* v = ggml_view(&ctx, model.kv_cache.v, { n_tokens * n_embd }, {}, (ggml_element_size(model.kv_cache.v) * n_embd) * (il * n_ctx + kv_head));

                gf.build_forward_expand(ggml_cpy(&ctx, Kcur, k));
                gf.build_forward_expand(ggml_cpy(&ctx, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            ggml_tensor* Q =
                ggml_permute(&ctx,
                    ggml_cont(&ctx,
                        Qcur,
                        { n_embd / n_head, n_head, n_tokens }),
                    0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_kv).permute(0, 2, 1, 3)
            // [64, n_kv, 12]
            ggml_tensor* K =
                ggml_permute(&ctx,
                    ggml_reshape(&ctx,
                        ggml_view(&ctx, model.kv_cache.k, { n_kv * n_embd }, {}, il * n_ctx * ggml_element_size(model.kv_cache.k) * n_embd),
                        { n_embd / n_head, n_head, n_kv }),
                    0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(&ctx0,
            //            ggml_permute(&ctx0,
            //                ggml_reshape(&ctx0,
            //                    ggml_view(&ctx0, model.kv_cache.v, { n_kv * n_embd }, il*n_ctx*ggml_element_size(model.kv_cache.v)*n_embd),
            //                    { n_embd/n_head, n_head, n_kv }),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(&ctx0, GGML_TYPE_F32, n_kv, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(&ctx0, Q, K, V, true);

            // K * Q
            // [n_kv, n_tokens, 12]
            ggml_tensor* KQ = ggml_mul_mat(&ctx, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_kv, n_tokens, 12]
            ggml_tensor* KQ_scaled =
                ggml_scale(&ctx,
                    KQ,
                    1.0f / sqrtf(float(n_embd) / n_head));

            // KQ_masked = mask_past(KQ_scaled)
            // [n_kv, n_tokens, 12]
            ggml_tensor* KQ_masked = ggml_add(&ctx, KQ_scaled, KQ_mask);

            // KQ = soft_max(KQ_masked)
            // [n_kv, N, 12]
            ggml_tensor* KQ_soft_max = ggml_soft_max(&ctx, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_kv).permute(1, 2, 0, 3).contiguous()
            // [n_kv, 64, 12]
            ggml_tensor* V_trans =
                ggml_cont(&ctx,
                    ggml_permute(&ctx,
                        ggml_reshape(&ctx,
                            ggml_view(&ctx, model.kv_cache.v, { n_kv * n_embd }, {}, il * n_ctx * ggml_element_size(model.kv_cache.v) * n_embd),
                            { n_embd / n_head, n_head, n_kv }),
                        1, 2, 0, 3),
                    { n_kv, n_embd / n_head, n_head });

            // KQV = transpose(V) * KQ_soft_max
            // [64, n_tokens, 12]
            ggml_tensor* KQV = ggml_mul_mat(&ctx, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, n_tokens]
            ggml_tensor* KQV_merged = ggml_permute(&ctx, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, n_tokens]
            cur = ggml_cont(&ctx, KQV_merged, { n_embd, n_tokens });
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

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_attn_proj_b);
        }

        // add the input
        cur = ggml_add(&ctx, cur, inpL);

        ggml_tensor* inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(&ctx, inpFF, hparams.eps);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(&ctx,
                    ggml_mul(&ctx,
                        cur,
                        model.layers[il].ln_2_g),
                    model.layers[il].ln_2_b);
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

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_mlp_fc_b);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(&ctx, cur);

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

            cur = ggml_add(&ctx,
                cur,
                model.layers[il].c_mlp_proj_b);
        }

        // input for next layer
        inpL = ggml_add(&ctx, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(&ctx, inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(&ctx,
            ggml_mul(&ctx,
                inpL,
                model.ln_f_g),
            model.ln_f_b);
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(&ctx, model.lm_head, inpL);

    // logits -> probs
    //inpL = ggml_soft_max(&ctx0, inpL);

    gf.build_forward_expand(inpL);

    return gf;
}

void gpt2_kv_cache::seq_cp(
    gpt2_seq_id seq_id_src,
    gpt2_seq_id seq_id_dst,
    gpt2_pos p0,
    gpt2_pos p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<gpt2_pos>::max();

    for (auto &cell : cells) {
        if (cell.has_seq_id(seq_id_src) && cell.pos >= p0 && cell.pos < p1) {
            cell.seq_id.insert(seq_id_dst);
        }
    }
}

gpt2_batch::gpt2_batch(int32_t n_tokens, int32_t embd) :
    embd(embd ? n_tokens * embd : 0), token(embd ? 0 : n_tokens),
    pos(n_tokens), seq_id(n_tokens), logits(n_tokens) {
}

// Positive return values does not mean a fatal error, but rather a warning.
//   0 - success
// < 0 - error
int gpt2_decode(
    struct gpt2_model& model,
    ggml_gallocr*       allocr,
    struct gpt2_batch    batch,
    int                  n_threads,
    std::vector<float>& logits) {
    const int32_t n_tokens = batch.n_tokens;
    const auto& hparams = model.hparams;
    const int     n_vocab = hparams.n_vocab;

    if (n_tokens == 0) {
        std::println("{}: n_tokens == 0", __func__);
        return -1;
    }

    GGML_ASSERT((batch.token.empty() && !batch.embd.empty()) || (!batch.token.empty() && batch.embd.empty()));

    auto& cache = model.kv_cache;

    for (int i = 0; i < n_tokens; i++) {
        cache.cells[cache.head + i].pos = batch.pos[i];
        cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i]);
    }

    cache.n = cache.head + n_tokens;

    ggml_cgraph gf = gpt2_graph(model, batch, false);

    // allocate tensors
    allocr->alloc_graph(&gf);

    // set the graph inputs
    if (!batch.token.empty()) {
        ggml_tensor* inp_tokens = gf.get_tensor("inp_tokens");
        ggml_backend_tensor_set(inp_tokens, batch.token.data(), 0, n_tokens * ggml_element_size(inp_tokens));

        ggml_tensor* position = gf.get_tensor("position");
        for (int i = 0; i < n_tokens; ++i) {
            int32_t v = batch.pos[i];
            ggml_backend_tensor_set(position, &v, i * sizeof(int32_t), sizeof(v));
        }
    }
    else {
        ggml_tensor* embd = gf.get_tensor("embd");
        ggml_backend_tensor_set(embd, batch.embd.data(), 0, n_tokens * hparams.n_embd * ggml_element_size(embd));
    }

    {
        ggml_tensor* KQ_mask = gf.get_tensor("KQ_mask");
        const auto& kv_cache = model.kv_cache;
        const int32_t n_tokens = batch.n_tokens;
        const int32_t n_kv = kv_cache.n;

        std::vector<float> data_buf(n_kv * n_tokens);
        const float neg_inf_v = -INFINITY;

        for (int h = 0; h < 1; ++h) {
            int h_offset = h * (n_kv * n_tokens);
            for (int j = 0; j < n_tokens; ++j) {
                const gpt2_pos    pos = batch.pos[j];
                const gpt2_seq_id seq_id = batch.seq_id[j];

                for (int i = 0; i < n_kv; ++i) {
                    if (!kv_cache.cells[i].has_seq_id(seq_id) || kv_cache.cells[i].pos > pos) {
                        data_buf[h_offset + j * n_kv + i] = neg_inf_v;
                    }
                }
            }
        }

        ggml_backend_tensor_set(KQ_mask, data_buf.data(), 0, data_buf.size() * sizeof(float));
    }

    // run the computation
    if (auto cpu_backend = dynamic_cast<ggml_cpu_backend*>(model.backend.get())) {
        cpu_backend->set_n_threads(n_threads);
    }

    model.backend->graph_compute(&gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // in this case, the output tensor is the last one in the graph
    ggml_tensor* inpL = gf.getNodes().back();

    if (!batch.logits.empty()) {
        // return logits for all tokens
        logits.resize(n_vocab * n_tokens);
        for (int32_t i = 0; i < n_tokens; i++) {
            if (batch.logits[i] == 0) {
                continue;
            }
            ggml_backend_tensor_get(inpL, logits.data() + n_vocab * i, n_vocab * i * sizeof(float), sizeof(float) * n_vocab);
        }
    }
    else {
        // return result just for the last token
        logits.resize(n_vocab);
        ggml_backend_tensor_get(inpL, logits.data(), (n_vocab * (n_tokens - 1)) * sizeof(float), sizeof(float) * n_vocab);
    }

    // update the kv ring buffer
    cache.head += n_tokens;

    // ensure kv cache head points to a valid index.
    if (cache.head >= cache.size) {
        std::println("{}: cache.head >= cache.size", __func__);
        return -2;
    }

    return 0;
}

int main(int argc, char** argv) {
    Stopwatch main_sw;
    gpt_params params;

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
        if (!gpt2_model_load(params.model, model, vocab, params.n_ctx, params.n_gpu_layers)) {
            std::println(stderr, "{}: failed to load model from '{}'", __func__, params.model);
            return 1;
        }
        t_load_us = start_sw.get_elapsed();

        test_gpt_tokenizer(vocab, params.token_test);
    }

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = gpt_tokenize(vocab, params.prompt);

    const int n_parallel = params.n_parallel;
    const int n_batch_max = std::max(embd_inp.size(), (size_t)n_parallel);

    // create a gpt2_batch
    // we use this object to submit token data for decoding
    gpt2_batch batch(n_batch_max, 0);

    // create the worst case graph for memory usage estimation
    batch.n_tokens = n_batch_max;
    ggml_cgraph gf = gpt2_graph(model, batch, true);

    // prepare required memory and allocate the compute buffer
    ggml_gallocr allocr = [&] {
        // create an allocator to measure the memory usage
        ggml_gallocr allocr(model.backend->get_default_buffer_type());

        // pre-allocate the compute buffer for the worst case (optional)
        allocr.reserve(&gf);
        size_t mem_size = allocr.get_buffer_size(0);
        std::println(stderr, "{}: compute buffer size: {:.2} MB", __func__, mem_size / 1024.0 / 1024.0);
        return allocr;
    }();

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // evaluate the initial prompt
    batch.n_tokens = embd_inp.size();

    for (int32_t i = 0; i < batch.n_tokens; i++) {
        batch.token[i] = embd_inp[i];
        batch.pos[i] = i;
        batch.seq_id[i] = 0;
        batch.logits[i] = false;
    }

    // gpt2_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (gpt2_decode(model, &allocr, batch, params.n_threads, logits) != 0) {
        std::println("{}: gpt2_decode() failed", __func__);
        return 1;
    }

    // assign the system KV cache to all parallel sequences
    // this way, the parallel sequences will "reuse" the prompt tokens without having to copy them
    for (int32_t i = 1; i < n_parallel; ++i) {
        model.kv_cache.seq_cp(0, i, 0, batch.n_tokens);
    }

    if (n_parallel > 1) {
        std::println("\n\n{}: generating {} sequences ...", __func__, n_parallel);
    }

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

    std::println("{}: prompt: '{}'", __func__, params.prompt);
    std::print("{}: number of tokens in prompt = {}, first 8 tokens: ", __func__, embd_inp);
    for (int i = 0; i < std::min(8, (int)embd_inp.size()); i++) {
        std::print("{} ", embd_inp[i]);
    }
    std::print("\n\n");

    std::vector<gpt_vocab::token> streams(n_parallel);

    // remember the batch index of the last token for each parallel sequence
    // we need this to determine which logits to sample from
    std::vector<int32_t> i_batch(n_parallel, batch.n_tokens - 1);

    int n_cur = batch.n_tokens;
    int n_len = batch.n_tokens + params.n_predict;
    int n_decoded = 0;

    const int   n_vocab = model.hparams.n_vocab;
    const int   top_k = params.top_k;
    const float top_p = params.top_p;
    const float temp = params.temp;

    while (n_cur < n_len) {
        batch.n_tokens = 0;

        for (int32_t i = 0; i < n_parallel; ++i) {
            if (i_batch[i] < 0) {
                // the stream has already finished
                continue;
            }

            auto* logits_i = logits.data() + i_batch[i] * n_vocab;

            gpt_vocab::id id = 0;
            {
                Stopwatch sample_sw;
                id = gpt_sample_top_k_top_p(vocab, logits_i, top_k, top_p, temp, rng);
                t_sample_us += sample_sw.get_elapsed();
            }

            // is it an end of stream? -> mark the stream as finished
            if ((!params.ignore_eos && id == 50256) || n_cur == n_len - 1) {
                i_batch[i] = -1;
                std::println();
                if (n_parallel > 1) {
                    std::print("{}: stream {} finished at n_cur = {}", __func__, i, n_cur);
                }

                continue;
            }

            auto& token = vocab.id_to_token[id];
            if (n_parallel == 1) {
                std::print("{}", token);
                fflush(stdout);
            }

            streams[i] += token;

            // push this new token for next evaluation
            batch.token[batch.n_tokens] = id;
            batch.pos[batch.n_tokens] = n_cur;
            batch.seq_id[batch.n_tokens] = i;
            batch.logits[batch.n_tokens] = true;

            i_batch[i] = batch.n_tokens;

            batch.n_tokens += 1;

            n_decoded += 1;
        }

        // all streams are finished
        if (batch.n_tokens == 0) {
            break;
        }

        n_cur += 1;

        {
            Stopwatch predict_sw;
            // evaluate the current batch with the transformer model
            int ret_code = gpt2_decode(model, &allocr, batch, params.n_threads, logits);
            if (ret_code != 0) {
                std::println(stderr, "{} : failed to eval, return code {}", __func__, ret_code);
                return 1;
            }
            t_predict_us += predict_sw.get_elapsed();
        }
    }

    if (n_parallel > 1) {
        std::println();

        for (int32_t i = 0; i < n_parallel; ++i) {
            std::print("sequence {}:\n\n{}{}\n\n", i, params.prompt, streams[i]);
        }
    }

    // report timing
    {
        std::print("\n\n");
        std::println("{}:     n_decoded = {:8}", __func__, n_decoded);
        std::println("{}:     load time = {:8.2} ms", __func__, t_load_us / 1000.0f);
        std::println("{}:   sample time = {:8.2} ms", __func__, t_sample_us / 1000.0f);
        std::println("{}:  predict time = {:8.2} ms", __func__, t_predict_us / 1000.0f);
        std::println("{}:    total time = {:8.2} ms", __func__, main_sw.get_elapsed() / 1000.0f);
    }

    return 0;
}

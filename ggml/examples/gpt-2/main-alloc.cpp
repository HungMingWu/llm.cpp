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

    std::unique_ptr<ggml_backend> backend;
    std::map<std::string, ggml_tensor*> tensors;
};

// load the model's weights from a file
bool gpt2_model_load(const std::string& fname, gpt2_model& model, gpt_vocab& vocab) {
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

    size_t ctx_size = 0;

    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_g
        ctx_size += ggml_row_size(GGML_TYPE_F32, n_embd); // ln_f_b

        ctx_size += ggml_row_size(wtype, n_vocab * n_embd); // wte
        ctx_size += ggml_row_size(GGML_TYPE_F32, n_ctx * n_embd); // wpe
        ctx_size += ggml_row_size(wtype, n_vocab * n_embd); // lm_head

        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_g
        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_1_b

        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_g
        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd)); // ln_2_b

        ctx_size += n_layer * (ggml_row_size(wtype, 3 * n_embd * n_embd)); // c_attn_attn_w
        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, 3 * n_embd));        // c_attn_attn_b

        ctx_size += n_layer * (ggml_row_size(wtype, n_embd * n_embd));   // c_attn_proj_w
        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, n_embd));          // c_attn_proj_b

        ctx_size += n_layer * (ggml_row_size(wtype, 4 * n_embd * n_embd)); // c_mlp_fc_w
        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, 4 * n_embd));        // c_mlp_fc_b

        ctx_size += n_layer * (ggml_row_size(wtype, 4 * n_embd * n_embd)); // c_mlp_proj_w
        ctx_size += n_layer * (ggml_row_size(GGML_TYPE_F32, 4 * n_embd));        // c_mlp_proj_b

        ctx_size += n_ctx * n_layer * ggml_row_size(GGML_TYPE_F32, n_embd); // memory_k
        ctx_size += n_ctx * n_layer * ggml_row_size(GGML_TYPE_F32, n_embd); // memory_v

        ctx_size += (6 + 12 * n_layer) * 512; // object overhead

        std::println("{}: ggml tensor size    = {} bytes", __func__, sizeof(ggml_tensor));
        std::println("{}: ggml ctx size = {:6.2} MB", __func__, ctx_size / (1024.0 * 1024.0));
    }

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

    // key + value memory
    {
        const auto& hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;

        const int n_mem = n_layer * n_ctx;
        const int n_elements = n_embd * n_mem;

        model.memory_k = ctx.create(GGML_TYPE_F32, { n_elements });
        model.memory_v = ctx.create(GGML_TYPE_F32, { n_elements });

        const size_t memory_size = model.memory_k->nbytes() + model.memory_v->nbytes();

        std::println("{}: memory size = {:8.2} MB, n_mem = {}", __func__, memory_size / 1024.0 / 1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;

        bool has_lm_head = false;

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
                    __func__, name, tensor->nbytes(), nelements* bpe);
                return false;
            }

            fin.read(reinterpret_cast<char*>(tensor->data), tensor->nbytes());

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                memcpy(model.lm_head->data, tensor->data, tensor->nbytes());
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += tensor->nbytes();
        }

        std::println("{}: model size  = {:8.2} MB", __func__, total_size / 1024.0 / 1024.0);
    }

    fin.close();

    return true;
}

// build the computation graph
ggml_cgraph gpt2_graph(
    const gpt2_model& model,
    const int n_past,
    const int n_tokens) {
    const int N = n_tokens;

    const auto& hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;

    ggml_context ctx;

    ggml_cgraph gf;

    ggml_tensor* embd = ctx.create(GGML_TYPE_I32, { N });
    // at this point, the tensor data is not allocated yet and cannot be set
    // we will find the tensor after the graph is allocated by its name, and set the data then
    embd->set_name("embd");
    // setting a tensor as an input will ensure that it is allocated at the beginning of the graph
    // this is important to ensure that the input tensors are not overwritten before they are used
    embd->set_flag(GGML_TENSOR_FLAG_INPUT);

    ggml_tensor* position = ctx.create(GGML_TYPE_I32, { N });
    position->set_name("position");
    position->set_flag(GGML_TENSOR_FLAG_INPUT);

    // wte + wpe
    ggml_tensor* inpL =
        ggml_add(&ctx,
            ggml_get_rows(&ctx, model.wte, embd),
            ggml_get_rows(&ctx, model.wpe, position));

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
                    ggml_repeat(&ctx, model.layers[il].ln_1_g, cur),
                    cur),
                ggml_repeat(&ctx, model.layers[il].ln_1_b, cur));
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

            cur = ggml_add(&ctx,
                ggml_repeat(&ctx, model.layers[il].c_attn_attn_b, cur),
                cur);
        }

        // self-attention
        {
            ggml_tensor* Qcur = ggml_view(&ctx, cur, { n_embd, N }, { cur->nb[1] }, 0 * sizeof(float) * n_embd);
            ggml_tensor* Kcur = ggml_view(&ctx, cur, { n_embd, N }, { cur->nb[1] }, 1 * sizeof(float) * n_embd);
            ggml_tensor* Vcur = ggml_view(&ctx, cur, { n_embd, N }, { cur->nb[1] }, 2 * sizeof(float) * n_embd);

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

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            ggml_tensor* K =
                ggml_permute(&ctx,
                    ggml_reshape(&ctx,
                        ggml_view(&ctx, model.memory_k, { (n_past + N) * n_embd }, {}, il * n_ctx * ggml_element_size(model.memory_k) * n_embd),
                        { n_embd / n_head, n_head, n_past + N }),
                    0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(&ctx0,
            //            ggml_permute(&ctx0,
            //                ggml_reshape_3d(&ctx0,
            //                    ggml_view_1d(&ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(&ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(&ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            ggml_tensor* KQ = ggml_mul_mat(&ctx, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            ggml_tensor* KQ_scaled =
                ggml_scale(&ctx,
                    KQ,
                    1.0f / sqrtf(float(n_embd) / n_head));

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            ggml_tensor* KQ_masked = ggml_diag_mask_inf(&ctx, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            ggml_tensor* KQ_soft_max = ggml_soft_max(&ctx, KQ_masked);

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

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            ggml_tensor* KQV_merged = ggml_permute(&ctx, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cont(&ctx, KQV_merged, { n_embd, N });
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
                ggml_repeat(&ctx, model.layers[il].c_attn_proj_b, cur),
                cur);
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
                        ggml_repeat(&ctx, model.layers[il].ln_2_g, cur),
                        cur),
                    ggml_repeat(&ctx, model.layers[il].ln_2_b, cur));
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
                ggml_repeat(&ctx, model.layers[il].c_mlp_fc_b, cur),
                cur);

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
                ggml_repeat(&ctx, model.layers[il].c_mlp_proj_b, cur),
                cur);
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
                ggml_repeat(&ctx, model.ln_f_g, inpL),
                inpL),
            ggml_repeat(&ctx, model.ln_f_b, inpL));
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(&ctx, model.lm_head, inpL);
    inpL->set_name("logits");
    // setting a tensor as the output will ensure that it is not overwritten by subsequent operations
    inpL->set_flag(GGML_TENSOR_FLAG_OUTPUT);

    // logits -> probs
    //inpL = ggml_soft_max(&ctx0, inpL);

    gf.build_forward_expand(inpL);

    return gf;
}

// evaluate the transformer
//
//   - model:     the model
//   - allocr:    ggml_gallocr to use to allocate the compute buffer
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt2_eval(
    const gpt2_model& model,
    ggml_gallocr* allocr,
    const int n_threads,
    const int n_past,
    const std::vector<gpt_vocab::id>& embd_inp,
    std::vector<float>& embd_w) {
    const int N = embd_inp.size();

    const auto& hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;

    ggml_cgraph gf = gpt2_graph(model, n_past, embd_inp.size());

    // allocate the graph tensors
    allocr->alloc_graph(&gf);

    // set the graph inputs
    ggml_tensor* embd = gf.get_tensor("embd");
    memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

    ggml_tensor* position = gf.get_tensor("position");
    for (int i = 0; i < N; ++i) {
        ((int32_t*)position->data)[i] = n_past + i;
    }

#if 0
    // run the computation
    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    static std::vector<uint8_t> work_buffer;
    work_buffer.resize(plan.work_size);
    plan.work_data = work_buffer.data();
    ggml_graph_compute(gf, &plan);
#endif

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    // get the graph outputs
    ggml_tensor* logits = gf.get_tensor("logits");

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(logits), sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float*)ggml_get_data(logits) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

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
        Stopwatch load_sw;
        if (!gpt2_model_load(params.model, model, vocab)) {
            std::println(stderr, "{}: failed to load model from '{}'", __func__, params.model);
            return 1;
        }
        t_load_us = load_sw.get_elapsed();

        test_gpt_tokenizer(vocab, params.token_test);
    }

    // create the worst case graph for memory usage estimation
    int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
    int n_past1 = model.hparams.n_ctx - n_tokens;
    ggml_cgraph gf = gpt2_graph(model, n_past1, n_tokens);

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

    int n_past = 0;

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int)embd_inp.size());

    std::println("{}: prompt: '{}'", __func__, params.prompt);
    std::print("{}: number of tokens in prompt = {}, first 8 tokens: ", __func__, embd_inp);
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
            if (!gpt2_eval(model, &allocr, params.n_threads, n_past, embd, logits)) {
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
                Stopwatch sample_sw;
                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
                t_sample_us += sample_sw.get_elapsed();
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
        std::println("{}:     load time = {:8.2} ms", __func__, t_load_us / 1000.0f);
        std::println("{}:   sample time = {:8.2} ms", __func__, t_sample_us / 1000.0f);
        std::println("{}:  predict time = {:8.2} ms / {:.2} ms per token", __func__, t_predict_us / 1000.0f, t_predict_us / 1000.0f / n_past);
        std::println("{}:    total time = {:8.2} ms", __func__, main_sw.get_elapsed() / 1000.0f);
    }

    return 0;
}

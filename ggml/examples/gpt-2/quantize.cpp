#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <print>
#include <string>
#include <vector>
#include <regex>

import ggml;
import gpt.common;

static constexpr int32_t GGML_QNT_VERSION = 2; // bump this on quantization format changes

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx = 1024;
    int32_t n_embd = 768;
    int32_t n_head = 12;
    int32_t n_layer = 12;
    int32_t ftype = 1;
};

// quantize a model
bool gpt2_model_quantize(const std::string& fname_inp, const std::string& fname_out, ggml_ftype ftype) {
    gpt_vocab vocab;

    std::println("{}: loading model from '{}'", __func__, fname_inp);

    auto finp = std::ifstream(fname_inp, std::ios::binary);
    if (!finp) {
        std::println(stderr, "{}: failed to open '{}' for reading", __func__, fname_inp);
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        std::println(stderr, "{}: failed to open '{}' for writing", __func__, fname_out);
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        finp.read((char*)&magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            std::println(stderr, "{}: invalid model file '{}' (bad magic)", __func__, fname_inp);
            return false;
        }

        fout.write((char*)&magic, sizeof(magic));
    }

    gpt2_hparams hparams;

    // load hparams
    {
        finp.read((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
        finp.read((char*)&hparams.n_ctx, sizeof(hparams.n_ctx));
        finp.read((char*)&hparams.n_embd, sizeof(hparams.n_embd));
        finp.read((char*)&hparams.n_head, sizeof(hparams.n_head));
        finp.read((char*)&hparams.n_layer, sizeof(hparams.n_layer));
        finp.read((char*)&hparams.ftype, sizeof(hparams.ftype));

        const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
        const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;

        std::println("{}: n_vocab     = {}", __func__, hparams.n_vocab);
        std::println("{}: n_ctx       = {}", __func__, hparams.n_ctx);
        std::println("{}: n_embd      = {}", __func__, hparams.n_embd);
        std::println("{}: n_head      = {}", __func__, hparams.n_head);
        std::println("{}: n_layer     = {}", __func__, hparams.n_layer);
        std::println("{}: ftype (src) = {}", __func__, hparams.ftype);
        std::println("{}: qntvr (src) = {}", __func__, qntvr_src);
        std::println("{}: ftype (dst) = {}", __func__, ftype_dst);
        std::println("{}: qntvr (dst) = {}", __func__, GGML_QNT_VERSION);

        fout.write((char*)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fout.write((char*)&hparams.n_ctx, sizeof(hparams.n_ctx));
        fout.write((char*)&hparams.n_embd, sizeof(hparams.n_embd));
        fout.write((char*)&hparams.n_head, sizeof(hparams.n_head));
        fout.write((char*)&hparams.n_layer, sizeof(hparams.n_layer));
        fout.write((char*)&ftype_dst, sizeof(ftype_dst));
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        finp.read((char*)&n_vocab, sizeof(n_vocab));
        fout.write((char*)&n_vocab, sizeof(n_vocab));

        if (n_vocab != hparams.n_vocab) {
            std::println(stderr, "{}: invalid model file '{}' (bad vocab size {} != {})",
                __func__, fname_inp, n_vocab, hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            finp.read((char*)&len, sizeof(len));
            fout.write((char*)&len, sizeof(len));

            word.resize(len);
            finp.read((char*)word.data(), len);
            fout.write((char*)word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> to_quant = {
        "model/wte",
        "model/lm_head",
        "model/h.*/attn/c_attn/w",
        "model/h.*/attn/c_proj/w",
        "model/h.*/mlp/c_fc/w",
        "model/h.*/mlp/c_proj/w",
    };

    if (!ggml_common_quantize_0(finp, fout, ftype, to_quant, {})) {
        std::println(stderr, "{}: failed to quantize model '{}'", __func__, fname_inp);
        return false;
    }

    finp.close();
    fout.close();

    return true;
}

// usage:
//  ./gpt-2-quantize models/gpt-2-117M/ggml-model.bin models/gpt-2-117M/ggml-model-quant.bin type
//
int main(int argc, char** argv) {
    if (argc != 4) {
        std::println(stderr, "usage: {} model-f32.bin model-quant.bin type", argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    Stopwatch main_sw;
    int64_t t_quantize_us = 0;

    // load the model
    {
        Stopwatch quantize_sw;
        if (!gpt2_model_quantize(fname_inp, fname_out, ggml_ftype(ftype))) {
            std::println(stderr, "{}: failed to quantize model from '{}'", __func__, fname_inp);
            return 1;
        }
        t_quantize_us = quantize_sw.get_elapsed();
    }

    // report timing
    {
        std::println();
        std::println("{}: quantize time = {:8.2} ms", __func__, t_quantize_us / 1000.0f);
        std::println("{}:    total time = {:8.2} ms", __func__, main_sw.get_elapsed() / 1000.0f);
    }

    return 0;
}

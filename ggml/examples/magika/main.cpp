#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <numeric>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

import ggml;

static const char* magika_labels[] = {
    "ai",                 "apk",                "appleplist",         "asm",                "asp",
    "batch",              "bmp",                "bzip",               "c",                  "cab",
    "cat",                "chm",                "coff",               "crx",                "cs",
    "css",                "csv",                "deb",                "dex",                "dmg",
    "doc",                "docx",               "elf",                "emf",                "eml",
    "epub",               "flac",               "gif",                "go",                 "gzip",
    "hlp",                "html",               "ico",                "ini",                "internetshortcut",
    "iso",                "jar",                "java",               "javabytecode",       "javascript",
    "jpeg",               "json",               "latex",              "lisp",               "lnk",
    "m3u",                "macho",              "makefile",           "markdown",           "mht",
    "mp3",                "mp4",                "mscompress",         "msi",                "mum",
    "odex",               "odp",                "ods",                "odt",                "ogg",
    "outlook",            "pcap",               "pdf",                "pebin",              "pem",
    "perl",               "php",                "png",                "postscript",         "powershell",
    "ppt",                "pptx",               "python",             "pythonbytecode",     "rar",
    "rdf",                "rpm",                "rst",                "rtf",                "ruby",
    "rust",               "scala",              "sevenzip",           "shell",              "smali",
    "sql",                "squashfs",           "svg",                "swf",                "symlinktext",
    "tar",                "tga",                "tiff",               "torrent",            "ttf",
    "txt",                "unknown",            "vba",                "wav",                "webm",
    "webp",               "winregistry",        "wmf",                "xar",                "xls",
    "xlsb",               "xlsx",               "xml",                "xpi",                "xz",
    "yaml",               "zip",                "zlibstream"
};

struct magika_hparams {
    const int block_size = 4096;
    const int beg_size = 512;
    const int mid_size = 512;
    const int end_size = 512;
    const int min_file_size_for_dl = 16;
    const int n_label = 113;
    const float f_norm_eps = 0.001f;
    const int padding_token = 256;
};

struct magika_model {
    magika_hparams hparams;

    ggml_tensor* dense_w;
    ggml_tensor* dense_b;

    ggml_tensor* layer_norm_gamma;
    ggml_tensor* layer_norm_beta;

    ggml_tensor* dense_1_w;
    ggml_tensor* dense_1_b;

    ggml_tensor* dense_2_w;
    ggml_tensor* dense_2_b;

    ggml_tensor* layer_norm_1_gamma;
    ggml_tensor* layer_norm_1_beta;

    ggml_tensor* target_label_w;
    ggml_tensor* target_label_b;

    std::unique_ptr<ggml_backend> backend = ggml_backend_cpu_init();
    std::unique_ptr<ggml_backend_buffer> buf_w = nullptr;
    ggml_context ctx_w;
};

ggml_tensor* checked_get_tensor(ggml_context &ctx, const char* name) {
    ggml_tensor* tensor = ctx.find(name);
    if (!tensor) {
        std::println(stderr, "{}: tensor '{}' not found", __func__, name);
        throw std::runtime_error("ggml_get_tensor() failed");
    }
    return tensor;
}

bool magika_model_load(const std::string& fname, magika_model& model) {
    auto& ctx = model.ctx_w;
    
    std::optional<gguf_context> ctx_gguf = gguf_init_from_file(fname.c_str());
    if (!ctx_gguf) {
        std::println(stderr, "{}: gguf_init_from_file() failed", __func__);
        return false;
    }

    constructFrom(ctx_gguf.value(), &ctx);

    model.buf_w = model.backend->alloc_tensors(&ctx);
    if (!model.buf_w) {
        std::println(stderr, "{}: ggml_backend_alloc_ctx_tensors() failed", __func__);
        return false;
    }

    try {
        model.dense_w = checked_get_tensor(ctx, "dense/kernel:0");
        model.dense_b = checked_get_tensor(ctx, "dense/bias:0");

        model.layer_norm_gamma = checked_get_tensor(ctx, "layer_normalization/gamma:0");
        model.layer_norm_beta = checked_get_tensor(ctx, "layer_normalization/beta:0");

        model.dense_1_w = checked_get_tensor(ctx, "dense_1/kernel:0");
        model.dense_1_b = checked_get_tensor(ctx, "dense_1/bias:0");

        model.dense_2_w = checked_get_tensor(ctx, "dense_2/kernel:0");
        model.dense_2_b = checked_get_tensor(ctx, "dense_2/bias:0");

        model.layer_norm_1_gamma = checked_get_tensor(ctx, "layer_normalization_1/gamma:0");
        model.layer_norm_1_beta = checked_get_tensor(ctx, "layer_normalization_1/beta:0");

        model.target_label_w = checked_get_tensor(ctx, "target_label/kernel:0");
        model.target_label_b = checked_get_tensor(ctx, "target_label/bias:0");
    }
    catch (const std::exception& e) {
        std::println(stderr, "{}: {}", __func__, e.what());
        return false;
    }

    std::ifstream f(fname, std::ios::binary);
    if (f.fail()) {
        std::println(stderr, "{}: fopen() failed", __func__);
        return false;
    }

    try {
        for (const auto& info : ctx_gguf->get_infos()) {
            ggml_tensor* tensor = ctx.find(info.t.name);
            size_t offs = ctx_gguf->get_data_offset() + info.offset;

            //std::println("{:-30}: [{:3}, {:3}, {:3}, {:3}] {}",
            //    name,
            //    tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
            //    ggml_type_name(tensor->type));

            std::vector<uint8_t> buf(tensor->nbytes());
            f.seekg(offs, std::ifstream::beg);
            f.read(reinterpret_cast<char*>(buf.data()), buf.size());

            ggml_backend_tensor_set(tensor, buf.data(), 0, buf.size());
        }
	}
	catch (const std::exception& e) {
		std::println(stderr, "{}: {}", __func__, e.what());
		return false;
	}

    return true;
}

ggml_cgraph magika_graph(
    const magika_model& model,
    const int n_files) {

    const auto& hparams = model.hparams;

    ggml_context ctx;

    ggml_cgraph gf;

    ggml_tensor* input = ctx.create(GGML_TYPE_F32, { 257, 1536, n_files }); // one-hot
    input->set_name("input");
    input->set_flag(GGML_TENSOR_FLAG_INPUT);

    struct ggml_tensor* cur;

    // dense
    cur = ggml_mul_mat(&ctx, model.dense_w, input);
    cur = ggml_add(&ctx, cur, model.dense_b); // [128, 1536, n_files]
    cur = ggml_gelu(&ctx, cur);

    // reshape
    cur = ggml_reshape(&ctx, cur, { 512, 384, n_files }); // [384, 512, n_files]
    cur = ggml_cont(&ctx, ggml_transpose(&ctx, cur));

    // layer normalization
    cur = ggml_norm(&ctx, cur, hparams.f_norm_eps);
    cur = ggml_mul(&ctx, cur, model.layer_norm_gamma); // [384, 512, n_files]
    cur = ggml_add(&ctx, cur, model.layer_norm_beta);  // [384, 512, n_files]

    // dense_1
    cur = ggml_cont(&ctx, ggml_transpose(&ctx, cur));
    cur = ggml_mul_mat(&ctx, model.dense_1_w, cur);
    cur = ggml_add(&ctx, cur, model.dense_1_b); // [256, 384, n_files]
    cur = ggml_gelu(&ctx, cur);

    // dense_2
    cur = ggml_mul_mat(&ctx, model.dense_2_w, cur);
    cur = ggml_add(&ctx, cur, model.dense_2_b); // [256, 384, n_files]
    cur = ggml_gelu(&ctx, cur);

    // global_max_pooling1d
    cur = ggml_cont(&ctx, ggml_transpose(&ctx, cur)); // [384, 256, n_files]
    cur = ggml_pool_1d(&ctx, cur, GGML_OP_POOL_MAX, 384, 384, 0); // [1, 256, n_files]
    cur = ggml_reshape(&ctx, cur, { 256, n_files }); // [256, n_files]

    // layer normalization 1
    cur = ggml_norm(&ctx, cur, hparams.f_norm_eps);
    cur = ggml_mul(&ctx, cur, model.layer_norm_1_gamma); // [256, n_files]
    cur = ggml_add(&ctx, cur, model.layer_norm_1_beta);  // [256, n_files]

    // target_label
    cur = ggml_mul_mat(&ctx, model.target_label_w, cur);
    cur = ggml_add(&ctx, cur, model.target_label_b); // [n_label, n_files]
    cur = ggml_soft_max(&ctx, cur); // [n_label, n_files]
    cur->set_name("target_label_probs");
    cur->set_flag(GGML_TENSOR_FLAG_OUTPUT);

    gf.build_forward_expand(cur);

    return gf;
}

bool magika_eval(
    struct magika_model& model,
    const std::vector<std::string>& fnames) {

    const auto& hparams = model.hparams;

    ggml_gallocr alloc(model.backend->get_default_buffer_type());

    ggml_cgraph gf = magika_graph(model, fnames.size());

    if (!alloc.alloc_graph(&gf)) {
        std::println(stderr, "{}: ggml_gallocr_alloc_graph() failed", __func__);
        return false;
    }

    ggml_tensor* input = gf.get_tensor("input");

    for (size_t i = 0; i < fnames.size(); i++) {
        FILE* f = fopen(fnames[i].c_str(), "rb");
        if (!f) {
            std::println(stderr, "{}: fopen() failed", __func__);
            return false;
        }
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);

        // the buffer is padded with the padding_token if the file is smaller than the block size
        std::vector<int> buf(1536, hparams.padding_token);
        std::vector<uint8_t> read_buf(std::max(hparams.beg_size, std::max(hparams.mid_size, hparams.end_size)));

        // read beg
        fseek(f, 0, SEEK_SET);
        int n_read = fread(read_buf.data(), 1, hparams.beg_size, f);
        for (int j = 0; j < n_read; j++) {
            // pad at the end
            buf[j] = read_buf[j];
        }

        // read mid
        long mid_offs = std::max(0L, (fsize - hparams.mid_size) / 2);
        fseek(f, mid_offs, SEEK_SET);
        n_read = fread(read_buf.data(), 1, hparams.mid_size, f);
        for (int j = 0; j < n_read; j++) {
            // pad at both ends
            long mid_idx = hparams.beg_size + (hparams.mid_size / 2) - n_read / 2 + j;
            buf[mid_idx] = read_buf[j];
        }

        // read end
        long end_offs = std::max(0L, fsize - hparams.end_size);
        fseek(f, end_offs, SEEK_SET);
        n_read = fread(read_buf.data(), 1, hparams.end_size, f);
        for (int j = 0; j < n_read; j++) {
            // pad at the beginning
            int end_idx = hparams.beg_size + hparams.mid_size + hparams.end_size - n_read + j;
            buf[end_idx] = read_buf[j];
        }

        fclose(f);

        const size_t inp_bytes = hparams.beg_size + hparams.mid_size + hparams.end_size;

        // convert to one-hot
        std::vector<float> one_hot(257 * inp_bytes);
        for (size_t j = 0; j < inp_bytes; j++) {
            one_hot[257 * j + buf[j]] = 1.0f;
        }

        ggml_backend_tensor_set(input, one_hot.data(), 257 * inp_bytes * i * sizeof(float), 257 * inp_bytes * sizeof(float));
    }

    if (model.backend->graph_compute(&gf) != GGML_STATUS_SUCCESS) {
        std::println(stderr, "%s: ggml_backend_graph_compute() failed", __func__);
        return false;
    }

    ggml_tensor* target_label_probs = gf.get_tensor("target_label_probs");

    // print probabilities for the top labels of each file
    for (size_t i = 0; i < fnames.size(); i++) {
        std::vector<float> probs(hparams.n_label);
        ggml_backend_tensor_get(target_label_probs, probs.data(), hparams.n_label * i * sizeof(float), hparams.n_label * sizeof(float));

        // sort the probabilities
        std::vector<int> idx(hparams.n_label);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&probs](int i1, int i2) { return probs[i1] > probs[i2]; });

        // print the top labels
        const int top_n = 5;
        printf("%-30s: ", fnames[i].c_str());
        for (int j = 0; j < top_n; j++) {
            printf("%s (%.2f%%) ", magika_labels[idx[j]], probs[idx[j]] * 100);
        }
        std::println();
    }

    return true;
}

int main(int argc, const char** argv) {
    if (argc < 3) {
        std::println(stderr, "usage: {} <model> <file1> [<file2> ...]", argv[0]);
        return 1;
    }

    const char* model_fname = argv[1];
    std::vector<std::string> fnames;
    for (int i = 2; i < argc; i++) {
        fnames.push_back(argv[i]);
    }

    magika_model model;
    if (!magika_model_load(model_fname, model)) {
        std::println(stderr, "magika_model_load() failed");
        return 1;
    }

    magika_eval(model, fnames);

    return 0;
}

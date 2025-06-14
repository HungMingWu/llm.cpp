module;
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <print>
#include <random>
#include <regex>
#include <sstream>

module gpt.common;

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
    {"q2_k", GGML_FTYPE_MOSTLY_Q2_K},
    {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},
    {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    {"q5_k", GGML_FTYPE_MOSTLY_Q5_K},
    {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
};

bool ggml_common_quantize_0(
    std::ifstream& finp,
    std::ofstream& fout,
    const ggml_ftype ftype,
    const std::vector<std::string>& to_quant,
    const std::vector<std::string>& to_skip)
{
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
    case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
    case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
    case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
    case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
    case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
    case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
    case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
    case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
    case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
    case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
    case GGML_FTYPE_UNKNOWN:
    case GGML_FTYPE_ALL_F32:
    case GGML_FTYPE_MOSTLY_F16:
    case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
    case GGML_FTYPE_MOSTLY_IQ2_XXS:
    case GGML_FTYPE_MOSTLY_IQ2_XS:
    case GGML_FTYPE_MOSTLY_IQ2_S:
    case GGML_FTYPE_MOSTLY_IQ3_XXS:
    case GGML_FTYPE_MOSTLY_IQ3_S:
    case GGML_FTYPE_MOSTLY_IQ1_S:
    case GGML_FTYPE_MOSTLY_IQ4_NL:
    case GGML_FTYPE_MOSTLY_IQ4_XS:
    case GGML_FTYPE_MOSTLY_IQ1_M:
    case GGML_FTYPE_MOSTLY_BF16:
    {
        std::println(stderr, "{}: invalid model type {}", __func__, static_cast<int>(ftype));
        return false;
    }
    };

    if (!ggml_is_quantized(qtype)) {
        std::println(stderr, "{}: invalid quantization type {} ({})", __func__,static_cast<int>(qtype), ggml_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char*>(&length), sizeof(length));
        finp.read(reinterpret_cast<char*>(&ttype), sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read(&name[0], length);

        std::println("{:64} - [{:5}, {:5}, {:5}], type = {:6} ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type)ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto& s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto& s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                std::println(stderr, "{}: unsupported ttype {} ({}) for integer quantization", __func__, ttype, ggml_type_name((ggml_type)ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char*>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = toFloat32(data_f16[i]);
                }
            }
            else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char*>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        }
        else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements * bpe);
            finp.read(reinterpret_cast<char*>(data_u8.data()), nelements * bpe);
        }

        fout.write(reinterpret_cast<char*>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char*>(&length), sizeof(length));
        fout.write(reinterpret_cast<char*>(&ttype), sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char*>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            switch ((ggml_type)ttype) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            {
                cur_size = ggml_quantize_chunk((ggml_type)ttype, data_f32.data(), work.data(), 0, nelements / ne[0], ne[0], nullptr);
            } break;
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_I8:
            case GGML_TYPE_I16:
            case GGML_TYPE_I32:
            case GGML_TYPE_I64:
            case GGML_TYPE_F64:
            case GGML_TYPE_Q8_1:
            case GGML_TYPE_Q8_K:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_BF16:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:
            case GGML_TYPE_COUNT:
            {
                std::println(stderr, "{}: unsupported quantization type {} ({})", __func__, ttype, ggml_type_name((ggml_type)ttype));
                return false;
            }
            }

            fout.write(reinterpret_cast<char*>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB\n", nelements * sizeof(float) / 1024.0 / 1024.0, cur_size / 1024.0 / 1024.0);
        }
        else {
            std::println("size = {:8.3} MB", data_u8.size() / 1024.0 / 1024.0);
            fout.write(reinterpret_cast<char*>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    std::println("{}: model size  = {:8.2} MB", __func__, total_size_org / 1024.0 / 1024.0);
    std::println("{}: quant size  = {:8.2} MB | ftype = {} ({})", __func__, total_size_new / 1024.0 / 1024.0, static_cast<int>(ftype), ggml_type_name(qtype));

    return true;
}

enum ggml_ftype ggml_parse_ftype(const char* str) {
    enum ggml_ftype ftype;
    if (str[0] == 'q') {
        const auto it = GGML_FTYPE_MAP.find(str);
        if (it == GGML_FTYPE_MAP.end()) {
            std::println(stderr, "{}: unknown ftype '{}'", __func__, str);
            return GGML_FTYPE_UNKNOWN;
        }
        ftype = it->second;
    }
    else {
        ftype = (enum ggml_ftype)atoi(str);
    }

    return ftype;
}

void ggml_print_ftypes(FILE* fp)
{
    for (const auto &[first, second] : GGML_FTYPE_MAP) {
        std::println(fp, "  type = \"{}\" or {}", first, static_cast<int>(second));
    }
}

static std::vector<gpt_vocab::id> parse_tokens_from_string(const std::string& input, char delimiter) {
    std::vector<gpt_vocab::id> output;
    std::stringstream ss(input);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        output.push_back(std::stoi(token));
    }

    return output;
}

static std::map<std::string, std::vector<gpt_vocab::id>> extract_tests_from_file(const std::string& fpath_test) {
    if (fpath_test.empty()) {
        std::println(stderr, "{} : No test file found.", __func__);
        return std::map<std::string, std::vector<gpt_vocab::id>>();
    }

    std::map<std::string, std::vector<gpt_vocab::id>> tests;

    auto fin = std::ifstream(fpath_test, std::ios_base::in);
    const char* delimeter = " => ";
    const char del_tok = ',';
    std::string line;
    while (std::getline(fin, line)) {
        size_t delimiterPos = line.find(delimeter);
        if (delimiterPos != std::string::npos) {
            std::string text = line.substr(0, delimiterPos);
            std::string s_tokens = line.substr(delimiterPos + std::strlen(delimeter));
            tests[text] = parse_tokens_from_string(s_tokens, del_tok);
        }
    }
    return tests;
}

void test_gpt_tokenizer(gpt_vocab& vocab, const std::string& fpath_test) {
    std::map<std::string, std::vector<gpt_vocab::id>> tests = extract_tests_from_file(fpath_test);

    size_t n_fails = 0;

    for (const auto& test : tests) {
        std::vector<gpt_vocab::id> tokens = gpt_tokenize(vocab, test.first);

        if (tokens != test.second) {
            n_fails++;

            // print out failure cases
            std::println(stderr, "{} : failed test: '{}'", __func__, test.first);
            std::println(stderr, "{} : tokens in hf:   ", __func__);
            for (const auto& t : test.second) {
                std::print(stderr, "{}({}), ", vocab.id_to_token[t], t);
            }
            std::println(stderr);
            std::print(stderr, "{} : tokens in ggml: ", __func__);
            for (const auto& t : tokens) {
                std::print(stderr, "{}({}), ", vocab.id_to_token[t], t);
            }
            std::println(stderr);
        }
    }

    std::println(stderr, "{} : {} tests failed out of {} tests.", __func__, n_fails, tests.size());
}

void gpt_split_words(std::string str, std::vector<std::string>& words) {
    const std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::regex re(pattern);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }
}

std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab& vocab, const std::string& text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;

        // Generate the subpattern from the special_tokens vector if it's not empty
        if (!vocab.special_tokens.empty()) {
            const std::regex escape(R"([\[\\\^\$\.\|\?\*\+\(\)\{\}])");
            std::string special_tokens_subpattern;
            for (const auto& token : vocab.special_tokens) {
                if (!special_tokens_subpattern.empty()) {
                    special_tokens_subpattern += "|";
                }
                special_tokens_subpattern += std::regex_replace(token, escape, R"(\$&)");
            }

            std::regex re(special_tokens_subpattern);
            std::smatch m;
            // Split the text by special tokens.
            while (std::regex_search(str, m, re)) {
                // Split the substrings in-between special tokens into words.
                gpt_split_words(m.prefix(), words);
                // Add matched special tokens as words.
                for (auto x : m) {
                    words.push_back(x);
                }
                str = m.suffix();
            }
            // Remaining text without special tokens will be handled below.
        }

        gpt_split_words(str, words);
    }

    // find the longest token that forms each word in words:
    std::vector<gpt_vocab::id> tokens;
    for (const auto& word : words) {
        for (int i = 0; i < (int)word.size(); ) {
            for (int j = word.size() - 1; j >= i; j--) {
                auto cand = word.substr(i, j - i + 1);
                auto it = vocab.token_to_id.find(cand);
                if (it != vocab.token_to_id.end()) { // word.substr(i, j-i+1) in vocab
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                }
                else if (j == i) { // word.substr(i, 1) has no matching
                    std::println(stderr, "{}: unknown token '{}'", __func__, word.substr(i, 1));
                    i++;
                }
            }
        }
    }

    return tokens;
}

gpt_vocab::id gpt_sample_top_k_top_p(
    const gpt_vocab& vocab,
    const float* logits,
    int    top_k,
    double top_p,
    double temp,
    std::mt19937& rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, gpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0 / temp;
        for (int i = 0; i < n_logits; ++i) {
            logits_id.push_back(std::make_pair(logits[i] * scale, i));
        }
    }

    // find the top K tokens
    std::partial_sort(
        logits_id.begin(),
        logits_id.begin() + top_k, logits_id.end(),
        [](const std::pair<double, gpt_vocab::id>& a, const std::pair<double, gpt_vocab::id>& b) {
            return a.first > b.first;
        });

    logits_id.resize(top_k);

    double maxl = -INFINITY;
    for (const auto& kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto& kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto& p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                top_k = i + 1;
                probs.resize(top_k);
                logits_id.resize(top_k);
                break;
            }
        }

        cumsum = 1.0 / cumsum;
        for (int i = 0; i < (int)probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    //std::println();
    //for (int i = 0; i < (int) probs.size(); i++) {
    //    std::println("{}: '{}' {}", i, vocab.id_to_token.at(logits_id[i].second), probs[i]);
    //}
    //exit(0);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}

void gpt_print_usage(int /*argc*/, char** argv, const gpt_params& params) {
    std::println(stderr, "usage: {} [options]", argv[0]);
    std::println(stderr);
    std::println(stderr, "options:");
    std::println(stderr, "  -h, --help            show this help message and exit");
    std::println(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)");
    std::println(stderr, "  -t N, --threads N     number of threads to use during computation (default: {})", params.n_threads);
    std::println(stderr, "  -p PROMPT, --prompt PROMPT");
    std::println(stderr, "                        prompt to start generation with (default: random)");
    std::println(stderr, "  -f FNAME, --file FNAME");
    std::println(stderr, "                        load prompt from a file");
    std::println(stderr, "  -tt TOKEN_TEST, --token_test TOKEN_TEST");
    std::println(stderr, "                        test tokenization");
    std::println(stderr, "  -n N, --n_predict N   number of tokens to predict (default: {})", params.n_predict);
    std::println(stderr, "  --top_k N             top-k sampling (default: {})", params.top_k);
    std::println(stderr, "  --top_p N             top-p sampling (default: {:.1})", params.top_p);
    std::println(stderr, "  --temp N              temperature (default: {:.1})", params.temp);
    std::println(stderr, "  --repeat-last-n N     last n tokens to consider for penalize (default: {}, 0 = disabled)", params.repeat_last_n);
    std::println(stderr, "  --repeat-penalty N    penalize repeat sequence of tokens (default: {:.2}, 1.0 = disabled)", (double)params.repeat_penalty);
    std::println(stderr, "  -b N, --batch_size N  batch size for prompt processing (default: {})", params.n_batch);
    std::println(stderr, "  -c N, --context N     context / KV cache size (default: {})", params.n_ctx);
    std::println(stderr, "  --ignore-eos          ignore EOS token during generation");
    std::println(stderr, "  -ngl N, --gpu-layers N  number of layers to offload to GPU on supported models (default: {})", params.n_gpu_layers);
    std::println(stderr, "  -m FNAME, --model FNAME");
    std::println(stderr, "                        model path (default: {})", params.model);
    std::println(stderr);
}

// Function to check if the next argument exists
static std::string get_next_arg(int& i, int argc, char** argv, const std::string& flag, gpt_params& params) {
    if (i + 1 < argc && argv[i + 1][0] != '-') {
        return argv[++i];
    }
    else {
        std::println(stderr, "error: {} requires one argument.", flag);
        gpt_print_usage(argc, argv, params);
        exit(0);
    }
}

bool gpt_params_parse(int argc, char** argv, gpt_params& params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-p" || arg == "--prompt") {
            params.prompt = get_next_arg(i, argc, argv, arg, params);
        }
        else if (arg == "-n" || arg == "--n_predict") {
            params.n_predict = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-np" || arg == "--n_parallel") {
            params.n_parallel = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "--top_k") {
            params.top_k = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "--top_p") {
            params.top_p = std::stof(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "--temp") {
            params.temp = std::stof(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "--repeat-last-n") {
            params.repeat_last_n = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "--repeat-penalty") {
            params.repeat_penalty = std::stof(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-b" || arg == "--batch_size") {
            params.n_batch = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-c" || arg == "--context") {
            params.n_ctx = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
            params.n_gpu_layers = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "--ignore-eos") {
            params.ignore_eos = true;
        }
        else if (arg == "-m" || arg == "--model") {
            params.model = get_next_arg(i, argc, argv, arg, params);
        }
        else if (arg == "-i" || arg == "--interactive") {
            params.interactive = true;
        }
        else if (arg == "-ip" || arg == "--interactive-port") {
            params.interactive = true;
            params.interactive_port = std::stoi(get_next_arg(i, argc, argv, arg, params));
        }
        else if (arg == "-h" || arg == "--help") {
            gpt_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-f" || arg == "--file") {
            get_next_arg(i, argc, argv, arg, params);
            std::ifstream file(argv[i]);
            if (!file) {
                std::println(stderr, "error: failed to open file '{}'\n", argv[i]);
                break;
            }
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
        else if (arg == "-tt" || arg == "--token_test") {
            params.token_test = get_next_arg(i, argc, argv, arg, params);
        }
        else {
            std::println(stderr, "error: unknown argument: {}", arg);
            gpt_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

std::string gpt_random_prompt(std::mt19937& rng) {
    const int r = rng() % 10;
    switch (r) {
    case 0: return "So";
    case 1: return "Once upon a time";
    case 2: return "When";
    case 3: return "The";
    case 4: return "After";
    case 5: return "If";
    case 6: return "import";
    case 7: return "He";
    case 8: return "She";
    case 9: return "They";
    }

    return "The";
}

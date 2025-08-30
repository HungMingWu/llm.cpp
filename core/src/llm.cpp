module;
#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <cstring>
#include <filesystem>
#include <format>
#include <forward_list>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>
#include "unicode.h"
#include "../../ggml/inplace_vector.hpp"

#define LLAMA_LOG_WARN(...)
#define LLAMA_LOG_DEBUG(...)
#define LLAMA_LOG_INFO(...)
#define LLAMA_LOG_ERROR(...)
#define LLAMA_LOG_CONT(...)
#define LOG_ERR(...)
#define GGML_ASSERT(...)
#define GGML_ABORT(...)

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
            #include <fcntl.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

module llm;
import :ds;
import :Enumeration;
import :Util;
import ggml;

using llama_file = std::ifstream;

using llama_files = std::vector<llama_file>;

static size_t file_size(std::ifstream& ifs)
{
    auto cur_pos = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    auto file_size = ifs.tellg();
    ifs.seekg(cur_pos, std::ios::beg);
    return file_size;
}

// Hope removed from C++26
struct llama_file1 {

#if defined(_WIN32)
    // use FILE * so we don't have to re-open the file to mmap
    FILE* fp;
    HANDLE fp_win32;
    size_t size;

private:
    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = std::format("Win32 error code: {}", error_code);
        }
        else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

public:

    llama_file1(const char* fname, const char* mode) {
        fp = fopen(fname, mode);
        if (fp == NULL) {
            throw make_format_runtime_error("failed to open {}: {}", fname, strerror(errno));
        }
        fp_win32 = (HANDLE)_get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        // SetFilePointerEx returns the current position when seeking relative 0 bytes
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw make_format_runtime_error("read error: {}", GetErrorMessageWin32(GetLastError()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        // no need to convert SEEK_* to FILE_*. The enums are the same.
        // Still, keep static asserts to avoid failures in the future.
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw make_format_runtime_error("read error: {}", GetErrorMessageWin32(GetLastError()));
        }
    }

    void read_raw(void* ptr, size_t len) const {
        // On Win32 ReadFile is significant faster than fread which is again significant faster than std::fstream. Thus
        // use the Win32 API to do file io instead of the C/C++ library functions.

        // There are conditions under which ReadFile cannot read chunks >64MB.
        // Thus split the operation into smaller chunks if len exceeds this limit.
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64 * 1024 * 1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw make_format_runtime_error("read error: {}", GetErrorMessageWin32(GetLastError()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        };
    }

    uint32_t read_u32() const {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void* ptr, size_t len) const {
        // There are conditions under which WriteFile cannot write chunks >64MB.
        // Thus split the operation into smaller chunks if len exceeds this limit.
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64 * 1024 * 1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw make_format_runtime_error("write error: {}", GetErrorMessageWin32(GetLastError()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file1() {
        if (fp) {
            std::fclose(fp);
        }
    }
#else
    // use FILE * so we don't have to re-open the file to mmap
    FILE* fp;
    size_t size;

    llama_file1(const char* fname, const char* mode) {
        fp = fopen(fname, mode);
        if (fp == NULL) {
            throw make_format_runtime_error("failed to open {}: {}", fname, strerror(errno));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
#ifdef _WIN32
        __int64 ret = _ftelli64(fp);
#else
        long ret = std::ftell(fp);
#endif
        if (ret == -1) {
            throw make_format_runtime_error("ftell error: {}", strerror(errno));
        }

        return (size_t)ret;
    }

    void seek(size_t offset, int whence) const {
#ifdef _WIN32
        int ret = _fseeki64(fp, (__int64)offset, whence);
#else
        int ret = std::fseek(fp, (long)offset, whence);
#endif
        if (ret != 0) {
            throw make_format_runtime_error("seek error: {}", strerror(errno));
        }
    }

    void read_raw(void* ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, len, 1, fp);
        if (ferror(fp)) {
            throw make_format_runtime_error("read error: {}", strerror(errno));
        }
        if (ret != 1) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    uint32_t read_u32() const {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void* ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw make_format_runtime_error("write error: {}", strerror(errno));
        }
    }

    void write_u32(std::uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    ~llama_file1() {
        if (fp) {
            std::fclose(fp);
        }
    }
#endif
};

static const std::map<llm_kv, const char*> LLM_KV_NAMES = {
    { LLM_KV_GENERAL_TYPE,                  "general.type"                          },
    { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
    { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
    { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
    { LLM_KV_GENERAL_NAME,                  "general.name"                          },
    { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
    { LLM_KV_GENERAL_VERSION,               "general.version"                       },
    { LLM_KV_GENERAL_URL,                   "general.url"                           },
    { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
    { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
    { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
    { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

    { LLM_KV_VOCAB_SIZE,                        "{}.vocab_size"                        },
    { LLM_KV_CONTEXT_LENGTH,                    "{}.context_length"                    },
    { LLM_KV_EMBEDDING_LENGTH,                  "{}.embedding_length"                  },
    { LLM_KV_FEATURES_LENGTH,                   "{}.features_length"                   },
    { LLM_KV_BLOCK_COUNT,                       "{}.block_count"                       },
    { LLM_KV_LEADING_DENSE_BLOCK_COUNT,         "{}.leading_dense_block_count"         },
    { LLM_KV_FEED_FORWARD_LENGTH,               "{}.feed_forward_length"               },
    { LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        "{}.expert_feed_forward_length"        },
    { LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, "{}.expert_shared_feed_forward_length" },
    { LLM_KV_USE_PARALLEL_RESIDUAL,             "{}.use_parallel_residual"             },
    { LLM_KV_TENSOR_DATA_LAYOUT,                "{}.tensor_data_layout"                },
    { LLM_KV_EXPERT_COUNT,                      "{}.expert_count"                      },
    { LLM_KV_EXPERT_USED_COUNT,                 "{}.expert_used_count"                 },
    { LLM_KV_EXPERT_SHARED_COUNT,               "{}.expert_shared_count"               },
    { LLM_KV_EXPERT_WEIGHTS_SCALE,              "{}.expert_weights_scale"              },
    { LLM_KV_POOLING_TYPE,                      "{}.pooling_type"                      },
    { LLM_KV_LOGIT_SCALE,                       "{}.logit_scale"                       },
    { LLM_KV_DECODER_START_TOKEN_ID,            "{}.decoder_start_token_id"            },
    { LLM_KV_ATTN_LOGIT_SOFTCAPPING,            "{}.attn_logit_softcapping"            },
    { LLM_KV_FINAL_LOGIT_SOFTCAPPING,           "{}.final_logit_softcapping"           },
    { LLM_KV_SWIN_NORM,                         "{}.swin_norm"                         },
    { LLM_KV_RESCALE_EVERY_N_LAYERS,            "{}.rescale_every_n_layers"            },
    { LLM_KV_TIME_MIX_EXTRA_DIM,                "{}.time_mix_extra_dim"                },
    { LLM_KV_TIME_DECAY_EXTRA_DIM,              "{}.time_decay_extra_dim"              },
    { LLM_KV_RESIDUAL_SCALE,                    "{}.residual_scale"                    },
    { LLM_KV_EMBEDDING_SCALE,                   "{}.embedding_scale"                   },

    { LLM_KV_ATTENTION_HEAD_COUNT,             "{}.attention.head_count"             },
    { LLM_KV_ATTENTION_HEAD_COUNT_KV,          "{}.attention.head_count_kv"          },
    { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,         "{}.attention.max_alibi_bias"         },
    { LLM_KV_ATTENTION_CLAMP_KQV,              "{}.attention.clamp_kqv"              },
    { LLM_KV_ATTENTION_KEY_LENGTH,             "{}.attention.key_length"             },
    { LLM_KV_ATTENTION_VALUE_LENGTH,           "{}.attention.value_length"           },
    { LLM_KV_ATTENTION_LAYERNORM_EPS,          "{}.attention.layer_norm_epsilon"     },
    { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      "{}.attention.layer_norm_rms_epsilon" },
    { LLM_KV_ATTENTION_GROUPNORM_EPS,          "{}.attention.group_norm_epsilon"     },
    { LLM_KV_ATTENTION_GROUPNORM_GROUPS,       "{}.attention.group_norm_groups"      },
    { LLM_KV_ATTENTION_CAUSAL,                 "{}.attention.causal"                 },
    { LLM_KV_ATTENTION_Q_LORA_RANK,            "{}.attention.q_lora_rank"            },
    { LLM_KV_ATTENTION_KV_LORA_RANK,           "{}.attention.kv_lora_rank"           },
    { LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, "{}.attention.relative_buckets_count" },
    { LLM_KV_ATTENTION_SLIDING_WINDOW,         "{}.attention.sliding_window"         },
    { LLM_KV_ATTENTION_SCALE,                  "{}.attention.scale"                  },

    { LLM_KV_ROPE_DIMENSION_COUNT,             "{}.rope.dimension_count"                 },
    { LLM_KV_ROPE_DIMENSION_SECTIONS,          "{}.rope.dimension_sections"              },
    { LLM_KV_ROPE_FREQ_BASE,                   "{}.rope.freq_base"                       },
    { LLM_KV_ROPE_SCALE_LINEAR,                "{}.rope.scale_linear"                    },
    { LLM_KV_ROPE_SCALING_TYPE,                "{}.rope.scaling.type"                    },
    { LLM_KV_ROPE_SCALING_FACTOR,              "{}.rope.scaling.factor"                  },
    { LLM_KV_ROPE_SCALING_ATTN_FACTOR,         "{}.rope.scaling.attn_factor"             },
    { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,        "{}.rope.scaling.original_context_length" },
    { LLM_KV_ROPE_SCALING_FINETUNED,           "{}.rope.scaling.finetuned"               },
    { LLM_KV_ROPE_SCALING_YARN_LOG_MUL,        "{}.rope.scaling.yarn_log_multiplier"     },

    { LLM_KV_SPLIT_NO,                         "split.no"            },
    { LLM_KV_SPLIT_COUNT,                      "split.count"         },
    { LLM_KV_SPLIT_TENSORS_COUNT,              "split.tensors.count" },

    { LLM_KV_SSM_CONV_KERNEL,                  "{}.ssm.conv_kernel"    },
    { LLM_KV_SSM_INNER_SIZE,                   "{}.ssm.inner_size"     },
    { LLM_KV_SSM_STATE_SIZE,                   "{}.ssm.state_size"     },
    { LLM_KV_SSM_TIME_STEP_RANK,               "{}.ssm.time_step_rank" },
    { LLM_KV_SSM_DT_B_C_RMS,                   "{}.ssm.dt_b_c_rms"     },

    { LLM_KV_WKV_HEAD_SIZE,                    "{}.wkv.head_size" },

    { LLM_KV_POSNET_EMBEDDING_LENGTH,          "{}.posnet.embedding_length" },
    { LLM_KV_POSNET_BLOCK_COUNT,               "{}.posnet.block_count"      },

    { LLM_KV_CONVNEXT_EMBEDDING_LENGTH,        "{}.convnext.embedding_length" },
    { LLM_KV_CONVNEXT_BLOCK_COUNT,             "{}.convnext.block_count"      },

    { LLM_KV_TOKENIZER_MODEL,                  "tokenizer.ggml.model"                    },
    { LLM_KV_TOKENIZER_PRE,                    "tokenizer.ggml.pre"                      },
    { LLM_KV_TOKENIZER_LIST,                   "tokenizer.ggml.tokens"                   },
    { LLM_KV_TOKENIZER_TOKEN_TYPE,             "tokenizer.ggml.token_type"               },
    { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,       "tokenizer.ggml.token_type_count"         },
    { LLM_KV_TOKENIZER_SCORES,                 "tokenizer.ggml.scores"                   },
    { LLM_KV_TOKENIZER_MERGES,                 "tokenizer.ggml.merges"                   },
    { LLM_KV_TOKENIZER_BOS_ID,                 "tokenizer.ggml.bos_token_id"             },
    { LLM_KV_TOKENIZER_EOS_ID,                 "tokenizer.ggml.eos_token_id"             },
    { LLM_KV_TOKENIZER_EOT_ID,                 "tokenizer.ggml.eot_token_id"             },
    { LLM_KV_TOKENIZER_EOM_ID,                 "tokenizer.ggml.eom_token_id"             },
    { LLM_KV_TOKENIZER_UNK_ID,                 "tokenizer.ggml.unknown_token_id"         },
    { LLM_KV_TOKENIZER_SEP_ID,                 "tokenizer.ggml.seperator_token_id"       },
    { LLM_KV_TOKENIZER_PAD_ID,                 "tokenizer.ggml.padding_token_id"         },
    { LLM_KV_TOKENIZER_CLS_ID,                 "tokenizer.ggml.cls_token_id"             },
    { LLM_KV_TOKENIZER_MASK_ID,                "tokenizer.ggml.mask_token_id"            },
    { LLM_KV_TOKENIZER_ADD_BOS,                "tokenizer.ggml.add_bos_token"            },
    { LLM_KV_TOKENIZER_ADD_EOS,                "tokenizer.ggml.add_eos_token"            },
    { LLM_KV_TOKENIZER_ADD_PREFIX,             "tokenizer.ggml.add_space_prefix"         },
    { LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,        "tokenizer.ggml.remove_extra_whitespaces" },
    { LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP,   "tokenizer.ggml.precompiled_charsmap"     },
    { LLM_KV_TOKENIZER_HF_JSON,                "tokenizer.huggingface.json"              },
    { LLM_KV_TOKENIZER_RWKV,                   "tokenizer.rwkv.world"                    },
    { LLM_KV_TOKENIZER_FIM_PRE_ID,             "tokenizer.ggml.fim_pre_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SUF_ID,             "tokenizer.ggml.fim_suf_token_id"         },
    { LLM_KV_TOKENIZER_FIM_MID_ID,             "tokenizer.ggml.fim_mid_token_id"         },
    { LLM_KV_TOKENIZER_FIM_PAD_ID,             "tokenizer.ggml.fim_pad_token_id"         },
    { LLM_KV_TOKENIZER_FIM_REP_ID,             "tokenizer.ggml.fim_rep_token_id"         },
    { LLM_KV_TOKENIZER_FIM_SEP_ID,             "tokenizer.ggml.fim_sep_token_id"         },

    { LLM_KV_ADAPTER_TYPE,                     "adapter.type"       },
    { LLM_KV_ADAPTER_LORA_ALPHA,               "adapter.lora.alpha" },

    // deprecated
    { LLM_KV_TOKENIZER_PREFIX_ID,              "tokenizer.ggml.prefix_token_id" },
    { LLM_KV_TOKENIZER_SUFFIX_ID,              "tokenizer.ggml.suffix_token_id" },
    { LLM_KV_TOKENIZER_MIDDLE_ID,              "tokenizer.ggml.middle_token_id" },
};

enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_DECI,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GROK,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_JINA_BERT_V2,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_QWEN2MOE,
    LLM_ARCH_QWEN2VL,
    LLM_ARCH_PHI2,
    LLM_ARCH_PHI3,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_MINICPM3,
    LLM_ARCH_GEMMA,
    LLM_ARCH_GEMMA2,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_MAMBA,
    LLM_ARCH_XVERSE,
    LLM_ARCH_COMMAND_R,
    LLM_ARCH_DBRX,
    LLM_ARCH_OLMO,
    LLM_ARCH_OLMO2,
    LLM_ARCH_OLMOE,
    LLM_ARCH_OPENELM,
    LLM_ARCH_ARCTIC,
    LLM_ARCH_DEEPSEEK,
    LLM_ARCH_DEEPSEEK2,
    LLM_ARCH_CHATGLM,
    LLM_ARCH_BITNET,
    LLM_ARCH_T5,
    LLM_ARCH_T5ENCODER,
    LLM_ARCH_JAIS,
    LLM_ARCH_NEMOTRON,
    LLM_ARCH_EXAONE,
    LLM_ARCH_RWKV6,
    LLM_ARCH_GRANITE,
    LLM_ARCH_GRANITE_MOE,
    LLM_ARCH_CHAMELEON,
    LLM_ARCH_WAVTOKENIZER_DEC,
    LLM_ARCH_UNKNOWN,
};

static const std::map<llm_arch, const char*> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_FALCON,           "falcon"           },
    { LLM_ARCH_GROK,             "grok"             },
    { LLM_ARCH_GPT2,             "gpt2"             },
    { LLM_ARCH_GPTJ,             "gptj"             },
    { LLM_ARCH_GPTNEOX,          "gptneox"          },
    { LLM_ARCH_MPT,              "mpt"              },
    { LLM_ARCH_BAICHUAN,         "baichuan"         },
    { LLM_ARCH_STARCODER,        "starcoder"        },
    { LLM_ARCH_REFACT,           "refact"           },
    { LLM_ARCH_BERT,             "bert"             },
    { LLM_ARCH_NOMIC_BERT,       "nomic-bert"       },
    { LLM_ARCH_JINA_BERT_V2,     "jina-bert-v2"     },
    { LLM_ARCH_BLOOM,            "bloom"            },
    { LLM_ARCH_STABLELM,         "stablelm"         },
    { LLM_ARCH_QWEN,             "qwen"             },
    { LLM_ARCH_QWEN2,            "qwen2"            },
    { LLM_ARCH_QWEN2MOE,         "qwen2moe"         },
    { LLM_ARCH_QWEN2VL,          "qwen2vl"          },
    { LLM_ARCH_PHI2,             "phi2"             },
    { LLM_ARCH_PHI3,             "phi3"             },
    { LLM_ARCH_PLAMO,            "plamo"            },
    { LLM_ARCH_CODESHELL,        "codeshell"        },
    { LLM_ARCH_ORION,            "orion"            },
    { LLM_ARCH_INTERNLM2,        "internlm2"        },
    { LLM_ARCH_MINICPM,          "minicpm"          },
    { LLM_ARCH_MINICPM3,         "minicpm3"         },
    { LLM_ARCH_GEMMA,            "gemma"            },
    { LLM_ARCH_GEMMA2,           "gemma2"           },
    { LLM_ARCH_STARCODER2,       "starcoder2"       },
    { LLM_ARCH_MAMBA,            "mamba"            },
    { LLM_ARCH_XVERSE,           "xverse"           },
    { LLM_ARCH_COMMAND_R,        "command-r"        },
    { LLM_ARCH_DBRX,             "dbrx"             },
    { LLM_ARCH_OLMO,             "olmo"             },
    { LLM_ARCH_OLMO2,            "olmo2"            },
    { LLM_ARCH_OLMOE,            "olmoe"            },
    { LLM_ARCH_OPENELM,          "openelm"          },
    { LLM_ARCH_ARCTIC,           "arctic"           },
    { LLM_ARCH_DEEPSEEK,         "deepseek"         },
    { LLM_ARCH_DEEPSEEK2,        "deepseek2"        },
    { LLM_ARCH_CHATGLM,          "chatglm"          },
    { LLM_ARCH_BITNET,           "bitnet"           },
    { LLM_ARCH_T5,               "t5"               },
    { LLM_ARCH_T5ENCODER,        "t5encoder"        },
    { LLM_ARCH_JAIS,             "jais"             },
    { LLM_ARCH_NEMOTRON,         "nemotron"         },
    { LLM_ARCH_EXAONE,           "exaone"           },
    { LLM_ARCH_RWKV6,            "rwkv6"            },
    { LLM_ARCH_GRANITE,          "granite"          },
    { LLM_ARCH_GRANITE_MOE,      "granitemoe"       },
    { LLM_ARCH_CHAMELEON,        "chameleon"        },
    { LLM_ARCH_WAVTOKENIZER_DEC, "wavtokenizer-dec" },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

struct LLM_KV {
    LLM_KV(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    std::string operator()(llm_kv kv) const {
        return std::vformat(LLM_KV_NAMES.at(kv), std::make_format_args(LLM_ARCH_NAMES.at(arch)));
    }
};

#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

// model file types
enum llama_ftype {
    LLAMA_FTYPE_ALL_F32 = 0,
    LLAMA_FTYPE_MOSTLY_F16 = 1,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
    // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
    // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
    // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_S = 11, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_M = 12, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_L = 13, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_S = 14, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_M = 15, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_S = 16, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_M = 17, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q6_K = 18, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XS = 20, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K_S = 21, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XS = 22, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_S = 24, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_NL = 25, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_S = 26, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_M = 27, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_S = 28, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_M = 29, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_XS = 30, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_M = 31, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_BF16 = 32, // except 1d tensors
    //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
    //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
    //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
    LLAMA_FTYPE_MOSTLY_TQ1_0 = 36, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_TQ2_0 = 37, // except 1d tensors

    LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
};

struct typeNameVisitor {
    template <typename T>
    std::string operator()(const T&) const {
        return typeName<T>::value;
    }

    template <typename T>
    std::string operator()(const std::vector<T>& value) const {
        const std::string_view type_name = [] {
            if constexpr (std::is_same_v<T, bool_value>) {
                return typeName<bool>::value;
            }
            else {
                return typeName<T>::value;
            }
        }();
        const size_t size = value.size();
        return std::vformat("{}[{},{}]", std::make_format_args("arr", type_name, size));
    }
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
    void* addr = NULL;
    size_t size = 0;

    bool failed_already = false;

    llama_mlock() {}
    llama_mlock(const llama_mlock&) = delete;

    ~llama_mlock() {
        if (size) {
            raw_unlock(addr, size);
        }
    }

    void init(void* ptr) {
        GGML_ASSERT(addr == NULL && size == 0); // NOLINT
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t*)addr + size, target_size - size)) {
                size = target_size;
            }
            else {
                failed_already = true;
            }
        }
    }

#ifdef _POSIX_MEMLOCK_RANGE
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        return (size_t)sysconf(_SC_PAGESIZE);
    }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
            "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
            "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
            "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

    bool raw_lock(const void* addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);

        // Check if the resource limit is fine after all
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
            size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

#undef MLOCK_SUGGESTION

    static void raw_unlock(void* addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t)si.dwPageSize;
    }

    bool raw_lock(void* ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            // It failed but this was only the first try; increase the working
            // set size and try again.
            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            // Per MSDN: "The maximum number of pages that a process can lock
            // is equal to the number of pages in its minimum working set minus
            // a small overhead."
            // Hopefully a megabyte is enough overhead:
            size_t increment = len + 1048576;
            // The minimum must be <= the maximum, so we need to increase both:
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void* ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    static size_t lock_granularity() {
        return (size_t)65536;
    }

    bool raw_lock(const void* addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void* addr, size_t len) {}
#endif
};

using llama_mlocks = std::vector<std::unique_ptr<llama_mlock>>;

struct llama_mmap {
    void* addr;
    size_t size;

    llama_mmap(const llama_mmap&) = delete;

#ifdef _POSIX_MAPPED_FILES
    static constexpr bool SUPPORTED = true;

    // list of mapped fragments (first_offset, last_offset)
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    llama_mmap(struct llama_file1* file, size_t prefetch = (size_t)-1 /* -1 = max value */, bool numa = false) {
        size = file->size;
        int fd = fileno(file->fp);
        int flags = MAP_SHARED;
        // prefetch/readahead impairs performance on NUMA systems
        if (numa) { prefetch = 0; }
#ifdef __linux__
        // advise the kernel to read the file sequentially (increases readahead)
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) { // NOLINT
            throw make_format_runtime_error("mmap failed: {}", strerror(errno));
        }

        if (prefetch > 0) {
            // advise the kernel to preload the mapped memory
            if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                    strerror(errno));
            }
        }
        if (numa) {
            // advise the kernel not to use readahead
            // (because the next page might not belong on the same node)
            if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                    strerror(errno));
            }
        }

        // initialize list of mapped_fragments
        mapped_fragments.emplace_back(0, file->size);
    }

    static void align_range(size_t* first, size_t* last, size_t page_size) {
        // align first to the next page
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        // align last to the previous page
        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    // partially unmap the file in the range [first, last)
    void unmap_fragment(size_t first, size_t last) {
        // note: this function must not be called multiple times with overlapping ranges
        // otherwise, there is a risk of invalidating addresses that have been repurposed for other mappings
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void* next_page_start = (uint8_t*)addr + first;

        // unmap the range
        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        // update the list of mapped fragments to avoid unmapping the same range again in the destructor
        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto& frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                // the range is in the middle of the fragment, split it
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            }
            else if (frag.first < first && frag.second > first) {
                // the range starts in the middle of the fragment
                new_mapped_fragments.emplace_back(frag.first, first);
            }
            else if (frag.first < last && frag.second > last) {
                // the range ends in the middle of the fragment
                new_mapped_fragments.emplace_back(last, frag.second);
            }
            else if (frag.first >= first && frag.second <= last) {
                // the range covers the entire fragment
            }
            else {
                // the range is outside the fragment
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~llama_mmap() {
        for (const auto& frag : mapped_fragments) {
            if (munmap((char*)addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    static constexpr bool SUPPORTED = true;

    llama_mmap(struct llama_file1* file, size_t prefetch = (size_t)-1, bool numa = false) {
        size = file->size;

        HANDLE hFile = (HANDLE)_get_osfhandle(_fileno(file->fp));

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw make_format_runtime_error("CreateFileMappingA failed: {}", llama_format_win_err(error));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw make_format_runtime_error("MapViewOfFile failed: {}", llama_format_win_err(error));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            // PrefetchVirtualMemory is only present on Windows 8 and above, so we dynamically load it
            BOOL(WINAPI * pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            // may fail on pre-Windows 8 systems
            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void*)GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                // advise the kernel to preload the mapped memory
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T)std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            throw std::runtime_error("PrefetchVirtualMemory unavailable");
#endif
        }
    }

    void unmap_fragment(size_t, size_t) {
        // not supported
    }

    ~llama_mmap() {
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static constexpr bool SUPPORTED = false;

    llama_mmap(struct llama_file1* file, size_t prefetch = -1, bool numa = false) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif
};
using llama_mmaps = std::vector<std::unique_ptr<llama_mmap>>;

static llm_arch llm_arch_from_string(const std::string& name) {
    for (const auto& kv : LLM_ARCH_NAMES) { // NOLINT
        if (kv.second == name) {
            return kv.first;
        }
    }

    return LLM_ARCH_UNKNOWN;
}

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer*>;
typedef bool (*llama_progress_callback)(float progress, void* user_data);

struct llm_model_loader {
    int n_kv = 0;
    size_t n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t  n_bytes = 0;

    bool use_mmap = false;
    bool check_tensors;

    llama_files files;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string& a, const std::string& b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor& tensor;

        llama_tensor_weight(size_t file_size, uint16_t idx, const gguf_context& gguf_ctx, ggml_tensor& tensor)
            : idx(idx), tensor(tensor) {
#if 0
            const int tensor_idx = gguf_find_tensor(gguf_ctx, tensor.get_name());
            if (tensor_idx < 0) {
                throw make_format_runtime_error("tensor '{}' not found in the model", tensor.get_name());
            }

            offs = gguf_ctx.get_data_offset() + gguf_ctx.get_tensor_offset(tensor_idx);
            if (offs + tensor.nbytes() < offs || offs + tensor.nbytes() > file_size) {
                throw make_format_runtime_error("tensor '{}' data is not within the file bounds, model is corrupted or incomplete", tensor.get_name());
            }
#endif
        }
    };

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;

    std::unique_ptr<gguf_context> meta;

    std::string arch_name;
    LLM_KV      llm_kv{ LLM_ARCH_UNKNOWN };

    template<typename T>
    bool get_key(const std::string& key, T& result, const bool required = true) {
        auto v = [&]() -> std::optional<T> {
            return std::nullopt;
        }();

        const bool found = v.has_value();

        if (required && !found) {
            throw make_format_runtime_error("key not found in model: {}", key);
        }
        else {
			if (found) {
				result = v.value();
			}
        }

        return found;
    }

    template<typename T>
    bool get_key(const enum llm_kv kid, T& result, const bool required = true) {
        return get_key(llm_kv(kid), result, required);
    }

    // get span of n elements, or a single element repeated n times
    template <typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string& key, std::array<T, N_MAX>& result, uint32_t n, const bool required = true) {
        return false;
    }

    template<typename T>
    bool get_key_or_arr(const enum llm_kv kid, T& result, uint32_t n, const bool required = true) {
        return get_key_or_arr(llm_kv(kid), result, n, required);
    }

    template <typename T>
    bool get_arr_n(const std::string& key, T& result, const bool required = true)
    requires std::is_integral_v<T>
    {
        return true;
    }

    template <typename T>
    bool get_arr_n(const enum llm_kv kid, T& result, const bool required = true)
    requires std::is_integral_v<T>
    {
        return get_arr_n(llm_kv(kid), result, required);
    }

    std::string get_arch_name() const {
        return arch_name;
    }

    enum llm_arch get_arch() const {
        return llm_kv.arch;
    }

    const llama_tensor_weight* get_weight(std::string_view name) const {
        std::string tmp(name);
        auto pos = weights_map.find(tmp.c_str());
        if (pos != weights_map.end()) {
            return &pos->second;
        }

        return nullptr;
    }

    ggml_tensor* get_tensor_meta(const char* name) const {
        const auto* weight = get_weight(name);
        if (!weight) {
            return nullptr;
        }
        return &weight->tensor;
    }

    ggml_tensor* create_tensor(ggml_context* ctx, const std::string& name, const std::initializer_list<int64_t>& ne, int flags = 0) {
#if 0
        const struct ggml_tensor* cur = check_tensor_dims(name, ne, !(flags & TENSOR_NOT_REQUIRED));

        if (cur == NULL) {
            return NULL;
        }

        bool duplicated = flags & TENSOR_DUPLICATED;

        struct ggml_tensor* tensor = ggml_dup_tensor(ctx, cur);
        ggml_set_name(tensor, ggml_get_name(cur));

        if (duplicated) {
            size_data += ggml_nbytes(cur);
        }
        else {
            n_created++;
        }

        return tensor;
#else
        return nullptr;
#endif
    }

    void done_getting_tensors() const {
        if (n_created != n_tensors) {
            throw make_format_runtime_error("{}: wrong number of tensors; expected {}, got {}", __func__, n_tensors, n_created);
        }
    }

    void init_mappings(bool prefetch = true, llama_mlocks* mlock_mmaps = nullptr) {
#if 0
        if (use_mmap) {
            mappings.reserve(files.size());
            mmaps_used.reserve(files.size());
            for (const auto& file : files) {
                auto* reg = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)->get_backend_reg();
                auto* is_numa_fn = (decltype(ggml_is_numa)*)ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_is_numa");
                std::unique_ptr<llama_mmap> mapping(new llama_mmap(file.get(), prefetch ? -1 : 0, is_numa_fn()));
                mmaps_used.emplace_back(mapping->size, 0);
                if (mlock_mmaps) {
                    std::unique_ptr<llama_mlock> mlock_mmap(new llama_mlock());
                    mlock_mmap->init(mapping->addr);
                    mlock_mmaps->emplace_back(std::move(mlock_mmap));
                }
                mappings.emplace_back(std::move(mapping));
            }
        }

        // compute the total size of all tensors for progress reporting
        for (const auto& it : weights_map) {
            size_data += ggml_nbytes(it.second.tensor);
        }
#endif
    }

    void get_mapping_range(size_t* first, size_t* last, void** addr, int idx, ggml_context* ctx) const {
        GGML_ASSERT(!mappings.empty());
        const auto& mapping = mappings.at(idx);

        *first = mapping->size;
        *last = 0;
        *addr = mapping->addr;
        for (auto tensor : ctx->getTensors()) {
            const auto* weight = get_weight(tensor->get_name());
            if (!weight || weight->idx != idx) {
                continue;
            }
            *first = std::min(*first, weight->offs);
            *last = std::max(*last, weight->offs + tensor->nbytes());
        }
    }

    // Returns false if cancelled by progress_callback
    bool load_all_data(
        ggml_context* ctx,
        llama_buf_map& bufs,
        llama_mlocks* lmlocks,
        llama_progress_callback progress_callback,
        void* progress_callback_user_data) {
#if 0
        GGML_ASSERT(size_data != 0 && "call init_mappings() first");

        std::vector<no_init<uint8_t>> read_buf;
        std::vector<std::future<std::pair<ggml_tensor*, bool>>> validation_result;

        // 4 staging buffers for async uploads, each sized 1MB seems to be a good default for single NVMe drives.
        // NVMe raid configurations might require more / larger buffers.
        constexpr size_t n_buffers = 4;
        constexpr size_t buffer_size = 1 * 1024 * 1024; // 1MB

        std::vector<ggml_backend_buffer_t> host_buffers;
        std::vector<ggml_backend_event*> events;
        std::vector<void*> host_ptrs;
        size_t buffer_idx = 0; // buffer to use for async loads
        ggml_backend* upload_backend = [&](const char* func) -> ggml_backend* {
            if (use_mmap || check_tensors) {
                return nullptr;
            }
            // When not using mmaped io use async uploads from pinned memory to GPU memory.
            // First determine if the backend supports the necessary features for async uploads.
            auto* buf = bufs.count(0) ? bufs.at(0) : nullptr;
            if (!buf) {
                LLAMA_LOG_DEBUG("%s: no buffer found for async uploads\n", func);
                return nullptr;
            }

            auto* buft = ggml_backend_buffer_get_type(buf);
            auto* dev = ggml_backend_buft_get_device(buft);
            if (!dev) {
                LLAMA_LOG_DEBUG("%s: no device found for buffer type %s for async uploads\n", func,
                    ggml_backend_buft_name(buft));
                return nullptr;
            }

            if (buft != ggml_backend_dev_buffer_type(dev)) {
                LLAMA_LOG_DEBUG("%s: buffer type %s is not the default buffer type for device %s for async uploads\n", func,
                    ggml_backend_buft_name(buft), ggml_backend_dev_name(dev));
                return nullptr;
            }

            ggml_backend_dev_props props;
            dev->get_props(&props);
            if (!props.caps.async || !props.caps.host_buffer || !props.caps.events) {
                LLAMA_LOG_DEBUG("%s: device %s does not support async, host buffers or events\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            auto* host_buft = dev->get_host_buffer_type();
            if (!host_buft) {
                LLAMA_LOG_DEBUG("%s: no host buffer type found for device %s\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            // If the backend is supported, create pinned memory buffers and events for synchronisation.
            for (size_t idx = 0; idx < n_buffers; ++idx) {
                auto* buf = ggml_backend_buft_alloc_buffer(host_buft, buffer_size);
                if (!buf) {
                    LLAMA_LOG_DEBUG("%s: failed to allocate host buffer for async uploads for device %s\n", func,
                        ggml_backend_dev_name(dev));
                    return nullptr;
                }

                host_buffers.emplace_back(buf);
                host_ptrs.emplace_back(ggml_backend_buffer_get_base(buf));

                auto* event = ggml_backend_event_new(dev);
                if (!event) {
                    LLAMA_LOG_DEBUG("%s: failed to create event for async uploads for device %s\n", func,
                        ggml_backend_dev_name(dev));
                    return nullptr;
                }

                events.emplace_back(event);
            }

            ggml_backend* backend = ggml_backend_dev_init(dev, nullptr);
            if (!backend) {
                LLAMA_LOG_DEBUG("%s: failed to initialize backend for device %s for async uploads\n", func,
                    ggml_backend_dev_name(dev));
                return nullptr;
            }

            return backend;
            }(__func__);

        if (upload_backend) {
            LLAMA_LOG_DEBUG("%s: using async uploads for device %s, buffer type %s, backend %s\n", __func__,
                ggml_backend_dev_name(ggml_backend_get_device(upload_backend)),
                ggml_backend_buft_name(ggml_backend_buffer_get_type(bufs.at(0))),
                ggml_backend_name(upload_backend));
        }

        for (struct ggml_tensor* cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
            const auto* weight = get_weight(ggml_get_name(cur));
            if (weight == nullptr) {
                // this can happen with split experts models
                continue;
            }

            if (progress_callback) {
                if (!progress_callback((float)size_done / size_data, progress_callback_user_data)) {
                    return false;
                }
            }

            size_t n_size = ggml_nbytes(cur);

            if (use_mmap) {
                const auto& mapping = mappings.at(weight->idx);
                ggml_backend_buffer_t buf_mmap = nullptr;
                if (bufs.count(weight->idx)) {
                    buf_mmap = bufs.at(weight->idx);
                }
                uint8_t* data = (uint8_t*)mapping->addr + weight->offs;

                if (check_tensors) {
                    validation_result.emplace_back(std::async(std::launch::async, [cur, data, n_size] {
                        return std::make_pair(cur, ggml_validate_row_data(cur->type, data, n_size));
                        }));
                }

                GGML_ASSERT(buf_mmap || cur->data); // either we have a buffer to allocate the tensor in, or it is already allocated
                if (buf_mmap && cur->data == nullptr) {
                    ggml_backend_tensor_alloc(buf_mmap, cur, data);
                    if (lmlocks) {
                        const auto& lmlock = lmlocks->at(weight->idx);
                        lmlock->grow_to(weight->offs + n_size);
                    }

                    auto& mmap_used = mmaps_used[weight->idx];
                    mmap_used.first = std::min(mmap_used.first, weight->offs);
                    mmap_used.second = std::max(mmap_used.second, weight->offs + n_size);
                }
                else {
                    ggml_backend_tensor_set(cur, data, 0, n_size);
                }
            }
            else {
                const auto& file = files.at(weight->idx);
                if (ggml_backend_buffer_is_host(cur->buffer)) {
                    file->seek(weight->offs, SEEK_SET);
                    file->read_raw(cur->data, n_size);
                    if (check_tensors) {
                        validation_result.emplace_back(std::async(std::launch::async, [cur, n_size] {
                            return std::make_pair(cur, ggml_validate_row_data(cur->type, cur->data, n_size));
                            }));
                    }
                }
                else {
                    // If upload_backend is valid load the tensor in chunks to pinned memory and upload the buffers asynchronously to the GPU.
                    if (upload_backend) {
                        file->seek(weight->offs, SEEK_SET);

                        size_t bytes_read = 0;

                        while (bytes_read < n_size) {
                            size_t read_iteration = std::min<size_t>(buffer_size, n_size - bytes_read);

                            ggml_backend_event_synchronize(events[buffer_idx]);
                            file->read_raw(host_ptrs[buffer_idx], read_iteration);
                            ggml_backend_tensor_set_async(upload_backend, cur, host_ptrs[buffer_idx], bytes_read, read_iteration);
                            ggml_backend_event_record(events[buffer_idx], upload_backend);

                            bytes_read += read_iteration;
                            ++buffer_idx;
                            buffer_idx %= n_buffers;
                        }
                    }
                    else {
                        read_buf.resize(n_size);
                        file->seek(weight->offs, SEEK_SET);
                        file->read_raw(read_buf.data(), n_size);
                        ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);
                        if (check_tensors && !ggml_validate_row_data(cur->type, read_buf.data(), n_size)) {
                            throw std::runtime_error(format("tensor '%s' has invalid data", ggml_get_name(cur)));
                        }
                    }
                }
            }

            size_done += n_size;
        }

        // free temporary resources used for async uploads
        for (auto* event : events) {
            ggml_backend_event_synchronize(event);
            ggml_backend_event_free(event);
        }
        for (auto* buf : host_buffers) {
            ggml_backend_buffer_free(buf);
        }
        ggml_backend_free(upload_backend);

        // check validation results
        bool validation_failed = false;
        for (auto& future : validation_result) {
            auto result = future.get();
            if (!result.second) {
                LLAMA_LOG_ERROR("%s: tensor '%s' has invalid data\n", __func__, ggml_get_name(result.first));
                validation_failed = true;
            }
        }
        if (validation_failed) {
            throw std::runtime_error("found tensors with invalid data");
        }

        // check if this is the last call and do final cleanup
        if (size_done >= size_data) {
            // unmap offloaded tensors and metadata
            if (use_mmap) {
                for (uint32_t idx = 0; idx < mappings.size(); idx++) {
                    const auto& mmap_used = mmaps_used.at(idx);
                    auto& mapping = mappings.at(idx);
                    mapping->unmap_fragment(0, mmap_used.first);
                    if (mmap_used.second != 0) {
                        mapping->unmap_fragment(mmap_used.second, mapping->size);
                    }
                }
            }
            if (progress_callback) {
                // Even though the model is done loading, we still honor
                // cancellation since we need to free allocations.
                return progress_callback(1.0f, progress_callback_user_data);
            }
        }
#endif
        return true;
    }
    static const int TENSOR_NOT_REQUIRED = 1;
    static const int TENSOR_DUPLICATED = 2;

    llm_model_loader(
        const std::filesystem::path &fname,
        bool use_mmap,
        bool check_tensors,
        const llama_model_kv_override* param_overrides_p
    );
};

llm_model_loader::llm_model_loader(
    const std::filesystem::path &fname,
    bool use_mmap,
    bool check_tensors,
    const llama_model_kv_override* param_overrides_p
) {
}

struct llama_model_params {
    // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
    ggml_backend_device** devices = nullptr;
    // number of layers to store in VRAM
#ifdef GGML_USE_METAL
    // note: we usually have plenty of VRAM, so by default offload all layers to the GPU
    int32_t n_gpu_layers = 999;
#else
    int32_t n_gpu_layers = 0; 
#endif
    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split the model across multiple GPUs

    // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
    int32_t main_gpu = 0;

    // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
    const float* tensor_split = nullptr;

    // comma separated list of RPC servers to use for offloading
    const char* rpc_servers= nullptr;

    // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
    // If the provided progress_callback returns true, model loading continues.
    // If it returns false, model loading is immediately aborted.
    llama_progress_callback progress_callback = nullptr;

    // context pointer passed to the progress callback
    void* progress_callback_user_data = nullptr;

    // override key-value pairs of the model meta data
    const struct llama_model_kv_override* kv_overrides = nullptr;

    // Keep the booleans together to avoid misalignment during copy-by-value.
    bool vocab_only = false;    // only load the vocabulary, no weights
    bool use_mmap = true;      // use mmap if possible
    bool use_mlock = false;     // force system to keep model in RAM
    bool check_tensors = false; // validate model tensor data
};

struct llama_layer_posnet {
    // resnet
    ggml_tensor* norm1 = nullptr;
    ggml_tensor* norm1_b = nullptr;

    ggml_tensor* conv1 = nullptr;
    ggml_tensor* conv1_b = nullptr;

    ggml_tensor* norm2 = nullptr;
    ggml_tensor* norm2_b = nullptr;

    ggml_tensor* conv2 = nullptr;
    ggml_tensor* conv2_b = nullptr;

    // attention
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_norm_b = nullptr;

    ggml_tensor* attn_q = nullptr;
    ggml_tensor* attn_q_b = nullptr;

    ggml_tensor* attn_k = nullptr;
    ggml_tensor* attn_k_b = nullptr;

    ggml_tensor* attn_v = nullptr;
    ggml_tensor* attn_v_b = nullptr;

    ggml_tensor* attn_o = nullptr;
    ggml_tensor* attn_o_b = nullptr;

    // normalize
    ggml_tensor* norm = nullptr;
    ggml_tensor* norm_b = nullptr;
};

struct llama_layer_convnext {
    ggml_tensor* dw = nullptr;
    ggml_tensor* dw_b = nullptr;

    ggml_tensor* norm = nullptr;
    ggml_tensor* norm_b = nullptr;

    ggml_tensor* pw1 = nullptr;
    ggml_tensor* pw1_b = nullptr;

    ggml_tensor* pw2 = nullptr;
    ggml_tensor* pw2_b = nullptr;

    ggml_tensor* gamma = nullptr;
};

struct llama_layer {
    // normalization
    ggml_tensor* attn_norm = nullptr;
    ggml_tensor* attn_norm_b = nullptr;
    ggml_tensor* attn_norm_2 = nullptr;
    ggml_tensor* attn_norm_2_b = nullptr;
    ggml_tensor* attn_q_norm = nullptr;
    ggml_tensor* attn_q_norm_b = nullptr;
    ggml_tensor* attn_k_norm = nullptr;
    ggml_tensor* attn_k_norm_b = nullptr;
    ggml_tensor* attn_out_norm = nullptr;
    ggml_tensor* attn_out_norm_b = nullptr;
    ggml_tensor* attn_q_a_norm = nullptr;
    ggml_tensor* attn_kv_a_norm = nullptr;
    ggml_tensor* attn_sub_norm = nullptr;
    ggml_tensor* attn_post_norm = nullptr;
    ggml_tensor* ffn_sub_norm = nullptr;
    ggml_tensor* attn_norm_cross = nullptr;
    ggml_tensor* attn_norm_enc = nullptr;

    // attention
    ggml_tensor* wq = nullptr;
    ggml_tensor* wk = nullptr;
    ggml_tensor* wv = nullptr;
    ggml_tensor* wo = nullptr;
    ggml_tensor* wqkv = nullptr;
    ggml_tensor* wq_a = nullptr;
    ggml_tensor* wq_b = nullptr;
    ggml_tensor* wkv_a_mqa = nullptr;
    ggml_tensor* wkv_b = nullptr;
    ggml_tensor* wq_cross = nullptr;
    ggml_tensor* wk_cross = nullptr;
    ggml_tensor* wv_cross = nullptr;
    ggml_tensor* wo_cross = nullptr;
    ggml_tensor* wq_enc = nullptr;
    ggml_tensor* wk_enc = nullptr;
    ggml_tensor* wv_enc = nullptr;
    ggml_tensor* wo_enc = nullptr;

    // attention bias
    ggml_tensor* bq = nullptr;
    ggml_tensor* bk = nullptr;
    ggml_tensor* bv = nullptr;
    ggml_tensor* bo = nullptr;
    ggml_tensor* bqkv = nullptr;

    // relative position bias
    ggml_tensor* attn_rel_b = nullptr;
    ggml_tensor* attn_rel_b_enc = nullptr;
    ggml_tensor* attn_rel_b_cross = nullptr;

    // normalization
    ggml_tensor* ffn_norm = nullptr;
    ggml_tensor* ffn_norm_b = nullptr;
    ggml_tensor* ffn_post_norm = nullptr;
    ggml_tensor* layer_out_norm = nullptr;
    ggml_tensor* layer_out_norm_b = nullptr;
    ggml_tensor* ffn_norm_exps = nullptr;
    ggml_tensor* ffn_norm_enc = nullptr;

    // ff
    ggml_tensor* ffn_gate = nullptr; // w1
    ggml_tensor* ffn_down = nullptr; // w2
    ggml_tensor* ffn_up = nullptr; // w3
    ggml_tensor* ffn_gate_enc = nullptr;
    ggml_tensor* ffn_down_enc = nullptr;
    ggml_tensor* ffn_up_enc = nullptr;

    // ff MoE
    ggml_tensor* ffn_gate_inp = nullptr;
    ggml_tensor* ffn_gate_exps = nullptr;
    ggml_tensor* ffn_down_exps = nullptr;
    ggml_tensor* ffn_up_exps = nullptr;

    // ff shared expert (shexp)
    ggml_tensor* ffn_gate_inp_shexp = nullptr;
    ggml_tensor* ffn_gate_shexp = nullptr;
    ggml_tensor* ffn_down_shexp = nullptr;
    ggml_tensor* ffn_up_shexp = nullptr;

    // ff bias
    ggml_tensor* ffn_gate_b = nullptr;
    ggml_tensor* ffn_down_b = nullptr; // b2
    ggml_tensor* ffn_up_b = nullptr; // b3
    ggml_tensor* ffn_act = nullptr;

    // mamba proj
    ggml_tensor* ssm_in = nullptr;
    ggml_tensor* ssm_x = nullptr;
    ggml_tensor* ssm_dt = nullptr;
    ggml_tensor* ssm_out = nullptr;

    // mamba
    ggml_tensor* ssm_conv1d = nullptr;
    ggml_tensor* ssm_a = nullptr;
    ggml_tensor* ssm_d = nullptr;

    // mamba bias
    ggml_tensor* ssm_conv1d_b = nullptr;
    ggml_tensor* ssm_dt_b = nullptr;

    // rwkv
    ggml_tensor* time_mix_w1 = nullptr;
    ggml_tensor* time_mix_w2 = nullptr;
    ggml_tensor* time_mix_lerp_x = nullptr;
    ggml_tensor* time_mix_lerp_w = nullptr;
    ggml_tensor* time_mix_lerp_k = nullptr;
    ggml_tensor* time_mix_lerp_v = nullptr;
    ggml_tensor* time_mix_lerp_r = nullptr;
    ggml_tensor* time_mix_lerp_g = nullptr;

    ggml_tensor* time_mix_first = nullptr;
    ggml_tensor* time_mix_decay = nullptr;
    ggml_tensor* time_mix_decay_w1 = nullptr;
    ggml_tensor* time_mix_decay_w2 = nullptr;
    ggml_tensor* time_mix_key = nullptr;
    ggml_tensor* time_mix_value = nullptr;
    ggml_tensor* time_mix_receptance = nullptr;
    ggml_tensor* time_mix_gate = nullptr;

    ggml_tensor* time_mix_ln = nullptr;
    ggml_tensor* time_mix_ln_b = nullptr;
    ggml_tensor* time_mix_output = nullptr;

    ggml_tensor* channel_mix_lerp_k = nullptr;
    ggml_tensor* channel_mix_lerp_r = nullptr;

    ggml_tensor* channel_mix_key = nullptr;
    ggml_tensor* channel_mix_receptance = nullptr;
    ggml_tensor* channel_mix_value = nullptr;

    // long rope factors
    ggml_tensor* rope_long = nullptr;
    ggml_tensor* rope_short = nullptr;
    ggml_tensor* rope_freqs = nullptr;

    // bitnet scale
    ggml_tensor* wq_scale = nullptr;
    ggml_tensor* wk_scale = nullptr;
    ggml_tensor* wv_scale = nullptr;
    ggml_tensor* wo_scale = nullptr;
    ggml_tensor* ffn_gate_scale = nullptr;
    ggml_tensor* ffn_up_scale = nullptr;
    ggml_tensor* ffn_down_scale = nullptr;

    llama_layer_posnet posnet;

    llama_layer_convnext convnext;
};

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_14M,
    MODEL_17M,
    MODEL_22M,
    MODEL_33M,
    MODEL_60M,
    MODEL_70M,
    MODEL_80M,
    MODEL_109M,
    MODEL_137M,
    MODEL_160M,
    MODEL_220M,
    MODEL_250M,
    MODEL_270M,
    MODEL_335M,
    MODEL_410M,
    MODEL_450M,
    MODEL_770M,
    MODEL_780M,
    MODEL_0_5B,
    MODEL_1B,
    MODEL_1_3B,
    MODEL_1_4B,
    MODEL_1_5B,
    MODEL_1_6B,
    MODEL_2B,
    MODEL_2_8B,
    MODEL_3B,
    MODEL_4B,
    MODEL_6B,
    MODEL_6_9B,
    MODEL_7B,
    MODEL_8B,
    MODEL_9B,
    MODEL_11B,
    MODEL_12B,
    MODEL_13B,
    MODEL_14B,
    MODEL_15B,
    MODEL_16B,
    MODEL_20B,
    MODEL_30B,
    MODEL_32B,
    MODEL_34B,
    MODEL_35B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
    MODEL_236B,
    MODEL_314B,
    MODEL_SMALL,
    MODEL_MEDIUM,
    MODEL_LARGE,
    MODEL_XL,
    MODEL_A1_7B,
    MODEL_A2_7B,
    MODEL_8x7B,
    MODEL_8x22B,
    MODEL_16x12B,
    MODEL_10B_128x3_66B,
    MODEL_57B_A14B,
    MODEL_27B,
};

struct llama_hparams_posnet {
    int64_t n_embd;
    int64_t n_layer;
};

struct llama_hparams_convnext {
    int64_t n_embd;
    int64_t n_layer;
};

static constexpr size_t LLAMA_MAX_LAYERS = 512;

// TODO: use everywhere in the implementation
#define LLAMA_TOKEN_NULL -1

enum llama_rope_type {
    LLAMA_ROPE_TYPE_NONE = -1,
    LLAMA_ROPE_TYPE_NORM = 0,
    LLAMA_ROPE_TYPE_NEOX = GGML_ROPE_TYPE_NEOX,
    LLAMA_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_MROPE,
    LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
};

struct llama_hparams {
    bool vocab_only;
    bool rope_finetuned;
    bool use_par_res;
    bool swin_norm;

    uint32_t n_vocab = 0;
    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_embd_features = 0;
    uint32_t n_layer;
    uint32_t n_rot;
    uint32_t n_swa = 0; // sliding window attention (SWA)
    uint32_t n_embd_head_k; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v; // dimension of values (d_v) aka n_embd_head
    uint32_t n_expert = 0;
    uint32_t n_expert_used = 0;
    uint32_t n_vocab_type = 0; // for BERT-style token types
    uint32_t n_rel_attn_bkts = 0;

    // for WavTokenizer
    struct llama_hparams_posnet   posnet;
    struct llama_hparams_convnext convnext;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    uint32_t n_layer_dense_lead = 0;
    uint32_t n_lora_q = 0;
    uint32_t n_lora_kv = 0;
    uint32_t n_ff_exp = 0;
    uint32_t n_ff_shexp = 0;
    uint32_t n_expert_shared = 0;
    float    expert_weights_scale = 0.0;

    float f_norm_eps;
    float f_norm_rms_eps;
    float f_norm_group_eps;

    uint32_t n_norm_groups;

    float f_attn_logit_softcapping = 50.0f;
    float f_final_logit_softcapping = 30.0f;

    // for RWKV
    uint32_t rescale_every_n_layers = 0;
    uint32_t time_mix_extra_dim = 0;
    uint32_t time_decay_extra_dim = 0;
    uint32_t wkv_head_size = 0;

    float     rope_attn_factor = 1.0f;
    float     rope_freq_base_train;
    float     rope_freq_scale_train;
    uint32_t  n_ctx_orig_yarn;
    float     rope_yarn_log_mul;
    int       rope_sections[4];

    // for State Space Models
    uint32_t ssm_d_conv = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    bool ssm_dt_b_c_rms = false;

    float f_clamp_kqv = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale = 0.0f;

    // Additional scale factors (Granite/Granite MoE)
    float f_residual_scale = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;

    bool causal_attn = true;
    bool use_alibi = false;
    bool attn_soft_cap = false;

    // needed by encoder-decoder models (e.g. T5, FLAN-T5)
    // ref: https://github.com/ggerganov/llama.cpp/pull/8141
    llama_token dec_start_token_id = LLAMA_TOKEN_NULL;

    enum llama_pooling_type      pooling_type = LLAMA_POOLING_TYPE_NONE;
    enum llama_rope_type         rope_type = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    uint32_t n_head(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_head_kv(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_head_kv_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_ff(uint32_t il = 0) const {
        if (il < n_layer) {
            return n_ff_arr[il];
        }

        GGML_ABORT("fatal error");
    }

    uint32_t n_gqa(uint32_t il = 0) const {
        const uint32_t n_head = this->n_head(il);
        const uint32_t n_head_kv = this->n_head_kv(il);

        if (n_head_kv == 0) {
            return 0;
        }

        return n_head / n_head_kv;
    }

    uint32_t n_embd_k_gqa(uint32_t il = 0) const { // dimension of key embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_k * n_head_kv;
    }

    uint32_t n_embd_v_gqa(uint32_t il = 0) const { // dimension of value embeddings across all k-v heads
        const uint32_t n_head_kv = this->n_head_kv(il);

        return n_embd_head_v * n_head_kv;
    }

    uint32_t n_embd_k_s() const { // dimension of the rolling state embeddings
        // corresponds to Mamba's conv_states size or RWKV's token_shift states size
        if (wkv_head_size != 0) {
            // for RWKV models
            return 2 * n_embd;
        }

        // TODO: maybe support other convolution strides than 1
        // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
        return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * ssm_d_inner;
    }

    uint32_t n_embd_v_s() const { // dimension of the recurrent state embeddings
        if (wkv_head_size != 0) {
            // corresponds to RWKV's wkv_states size
            return n_embd * wkv_head_size;
        }

        // corresponds to Mamba's ssm_states size
        return ssm_d_state * ssm_d_inner;
    }
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");

enum llama_token_attr {
    LLAMA_TOKEN_ATTR_UNDEFINED = 0,
    LLAMA_TOKEN_ATTR_UNKNOWN = 1 << 0,
    LLAMA_TOKEN_ATTR_UNUSED = 1 << 1,
    LLAMA_TOKEN_ATTR_NORMAL = 1 << 2,
    LLAMA_TOKEN_ATTR_CONTROL = 1 << 3,  // SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
    LLAMA_TOKEN_ATTR_BYTE = 1 << 5,
    LLAMA_TOKEN_ATTR_NORMALIZED = 1 << 6,
    LLAMA_TOKEN_ATTR_LSTRIP = 1 << 7,
    LLAMA_TOKEN_ATTR_RSTRIP = 1 << 8,
    LLAMA_TOKEN_ATTR_SINGLE_WORD = 1 << 9,
};

struct llm_tokenizer {
    llm_tokenizer() {}
    virtual ~llm_tokenizer() = default;
};

enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
    LLAMA_VOCAB_TYPE_SPM = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE = 2, // GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM = 3, // BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM = 4, // T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV = 5, // RWKV tokenizer based on greedy tokenization
};

// pre-tokenization types
enum llama_vocab_pre_type {
    LLAMA_VOCAB_PRE_TYPE_DEFAULT = 0,
    LLAMA_VOCAB_PRE_TYPE_LLAMA3 = 1,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM = 2,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
    LLAMA_VOCAB_PRE_TYPE_FALCON = 4,
    LLAMA_VOCAB_PRE_TYPE_MPT = 5,
    LLAMA_VOCAB_PRE_TYPE_STARCODER = 6,
    LLAMA_VOCAB_PRE_TYPE_GPT2 = 7,
    LLAMA_VOCAB_PRE_TYPE_REFACT = 8,
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R = 9,
    LLAMA_VOCAB_PRE_TYPE_STABLELM2 = 10,
    LLAMA_VOCAB_PRE_TYPE_QWEN2 = 11,
    LLAMA_VOCAB_PRE_TYPE_OLMO = 12,
    LLAMA_VOCAB_PRE_TYPE_DBRX = 13,
    LLAMA_VOCAB_PRE_TYPE_SMAUG = 14,
    LLAMA_VOCAB_PRE_TYPE_PORO = 15,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3 = 16,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4 = 17,
    LLAMA_VOCAB_PRE_TYPE_VIKING = 18,
    LLAMA_VOCAB_PRE_TYPE_JAIS = 19,
    LLAMA_VOCAB_PRE_TYPE_TEKKEN = 20,
    LLAMA_VOCAB_PRE_TYPE_SMOLLM = 21,
    LLAMA_VOCAB_PRE_TYPE_CODESHELL = 22,
    LLAMA_VOCAB_PRE_TYPE_BLOOM = 23,
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH = 24,
    LLAMA_VOCAB_PRE_TYPE_EXAONE = 25,
    LLAMA_VOCAB_PRE_TYPE_CHAMELEON = 26,
    LLAMA_VOCAB_PRE_TYPE_MINERVA = 27,
};

struct llama_vocab {
    using id = llama_token;
    using token = std::string;
    using tattr = llama_token_attr;

    struct token_data {
        token text;
        float score;
        tattr attr;
    };

    uint32_t n_vocab = 0; // TODO: not great because has to keep in sync with hparams.n_vocab

    enum llama_vocab_type     type = LLAMA_VOCAB_TYPE_SPM;
    enum llama_vocab_pre_type type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    std::unordered_map<token, id> token_to_id;
    std::vector<token_data>       id_to_token;

    std::vector<id>    cache_special_tokens;
    std::vector<token> cache_token_to_piece; // llama_token_to_piece(special = true);

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // default LLaMA special tokens
    // TODO: should we set all of these to LLAMA_TOKEN_NULL?
    id special_bos_id = 1;
    id special_eos_id = 2;
    id special_eot_id = LLAMA_TOKEN_NULL;
    id special_eom_id = LLAMA_TOKEN_NULL;
    id special_unk_id = 0;
    id special_sep_id = LLAMA_TOKEN_NULL;
    id special_pad_id = LLAMA_TOKEN_NULL;
    id special_cls_id = LLAMA_TOKEN_NULL;
    id special_mask_id = LLAMA_TOKEN_NULL;

    id linefeed_id = 13;

    // fim tokens
    id special_fim_pre_id = LLAMA_TOKEN_NULL;
    id special_fim_suf_id = LLAMA_TOKEN_NULL;
    id special_fim_mid_id = LLAMA_TOKEN_NULL;
    id special_fim_pad_id = LLAMA_TOKEN_NULL;
    id special_fim_rep_id = LLAMA_TOKEN_NULL; // repo
    id special_fim_sep_id = LLAMA_TOKEN_NULL; // file separator

    // set of all tokens that cause "end of generation"
    std::set<id> special_eog_ids;

    // tokenizer flags
    bool tokenizer_add_space_prefix = false;
    bool tokenizer_add_bos = false;
    bool tokenizer_add_eos = false;
    bool tokenizer_ignore_merges = false;
    bool tokenizer_clean_spaces = false;  // clean_up_tokenization_spaces
    bool tokenizer_remove_extra_whitespaces = false;
    bool tokenizer_escape_whitespaces = true;
    bool tokenizer_treat_whitespace_as_suffix = false;

    std::vector<char> precompiled_charsmap;

    llm_tokenizer* tokenizer = nullptr;

    llama_vocab() = default;
    ~llama_vocab() = default;

    int find_bpe_rank(const std::string& token_left, const std::string& token_right) const;

    void init_tokenizer();
};

void llama_lora_adapter_free(struct llama_lora_adapter* adapter);

struct llama_model {
    e_model     type = MODEL_UNKNOWN;
    llm_arch    arch = LLM_ARCH_UNKNOWN;
    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

    std::string name = "n/a";

    llama_hparams hparams = {};
    llama_vocab   vocab;

    ggml_tensor* tok_embd = nullptr;
    ggml_tensor* type_embd = nullptr;
    ggml_tensor* pos_embd = nullptr;
    ggml_tensor* tok_norm = nullptr;
    ggml_tensor* tok_norm_b = nullptr;

    ggml_tensor* output_norm = nullptr;
    ggml_tensor* output_norm_b = nullptr;
    ggml_tensor* output = nullptr;
    ggml_tensor* output_b = nullptr;
    ggml_tensor* output_norm_enc = nullptr;

    // classifier
    ggml_tensor* cls = nullptr;
    ggml_tensor* cls_b = nullptr;
    ggml_tensor* cls_out = nullptr;
    ggml_tensor* cls_out_b = nullptr;

    ggml_tensor* conv1d = nullptr;
    ggml_tensor* conv1d_b = nullptr;

    std::vector<llama_layer> layers;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    llama_split_mode split_mode;
    int main_gpu;
    int n_gpu_layers;

    std::vector<std::string> rpc_servers;

    // list of devices used in this model
    std::vector<ggml_backend_device*> devices;


    // lists of buffer types used for each layer
    using buft_list_t = std::vector<std::pair<ggml_backend_device*, ggml_backend_buffer_type*>>;
    buft_list_t cpu_buft_list;
    std::map<ggml_backend_device*, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_device* dev;
        buft_list_t* buft_list;
    };
    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;

    // contexts where the model tensors metadata is stored
    std::vector<std::unique_ptr<ggml_context>> ctxs;

    // the model memory buffers for the tensor data
    std::vector<std::unique_ptr<ggml_backend_buffer>> bufs;

    // model memory mapped files
    llama_mmaps mappings;

    // objects representing data potentially being locked in memory
    llama_mlocks mlock_bufs;
    llama_mlocks mlock_mmaps;

    // for quantize-stats only
    std::vector<std::pair<std::string, ggml_tensor*>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    // total number of parameters in the model
    uint64_t n_elements = 0;

    // total size of all the tensors in the model in bytes
    size_t  n_bytes = 0;

    // keep track of loaded lora adapters
    std::set<struct llama_lora_adapter*> lora_adapters;

    ~llama_model() {
        while (!lora_adapters.empty()) {
            llama_lora_adapter_free(*lora_adapters.begin());
        }
    }
};

struct llama_lora_weight {
    ggml_tensor* a = nullptr;
    ggml_tensor* b = nullptr;
    llama_lora_weight() = default;
    llama_lora_weight(ggml_tensor* a, ggml_tensor* b) : a(a), b(b) {}
};

struct llama_lora_adapter {
    llama_model* base_model;
    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_lora_weight> ab_map;
    std::vector<std::unique_ptr<ggml_context>> ctxs;
    std::vector<std::unique_ptr<ggml_backend_buffer>> bufs;

    float alpha;

    llama_lora_adapter(llama_model* base_model) : base_model(base_model) {
        base_model->lora_adapters.insert(this);
    }

    llama_lora_weight* get_weight(ggml_tensor* w) {
        std::string name(w->get_name());
        auto pos = ab_map.find(name);
        if (ab_map.find(name) != ab_map.end()) {
            return &pos->second;
        }
        return nullptr;
    }

    ~llama_lora_adapter() {
        auto pos = base_model->lora_adapters.find(this);
        if (pos != base_model->lora_adapters.end()) {
            base_model->lora_adapters.erase(pos);
        }
    }
};

void llama_lora_adapter_free(struct llama_lora_adapter* adapter) {
    delete adapter;
}

static void llm_load_arch(llm_model_loader& ml, llama_model& model) {
    model.arch = ml.get_arch();
    if (model.arch == LLM_ARCH_UNKNOWN) {
        throw std::runtime_error("unknown model architecture: '" + ml.get_arch_name() + "'");
    }
}

struct ArrayVisitor {
    template <typename T>
    bool operator()(const T&) {
        return false;
    }
    template <typename T>
    bool operator()(const std::vector<T>&) {
		return true;
    }
};

bool is_array(const gguf_value& value)
{
	return std::visit(ArrayVisitor{}, value);
}

static const std::map<llama_rope_scaling_type, const char*> LLAMA_ROPE_SCALING_TYPES = {
    { LLAMA_ROPE_SCALING_TYPE_NONE,       "none"       },
    { LLAMA_ROPE_SCALING_TYPE_LINEAR,     "linear"     },
    { LLAMA_ROPE_SCALING_TYPE_YARN,       "yarn"       },
    { LLAMA_ROPE_SCALING_TYPE_LONGROPE,   "longrope"   },
};

static llama_rope_scaling_type llama_rope_scaling_type_from_string(const std::string& name) {
    for (const auto& kv : LLAMA_ROPE_SCALING_TYPES) {
        if (kv.second == name) {
            return (llama_rope_scaling_type)kv.first;
        }
    }

    return LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
}

enum llama_rope_type llama_rope_type(const struct llama_model* model) {
    switch (model->arch) {
        // these models do not use RoPE
    case LLM_ARCH_GPT2:
    case LLM_ARCH_GPTJ:
    case LLM_ARCH_MPT:
    case LLM_ARCH_REFACT:
    case LLM_ARCH_BLOOM:
    case LLM_ARCH_MAMBA:
    case LLM_ARCH_JINA_BERT_V2:
    case LLM_ARCH_T5:
    case LLM_ARCH_T5ENCODER:
    case LLM_ARCH_JAIS:
    case LLM_ARCH_RWKV6:
    case LLM_ARCH_WAVTOKENIZER_DEC:
        return LLAMA_ROPE_TYPE_NONE;

        // use what we call a normal RoPE, operating on pairs of consecutive head values
    case LLM_ARCH_LLAMA:
    case LLM_ARCH_DECI:
    case LLM_ARCH_BAICHUAN:
    case LLM_ARCH_STARCODER:
    case LLM_ARCH_PLAMO:
    case LLM_ARCH_ORION:
    case LLM_ARCH_INTERNLM2:
    case LLM_ARCH_MINICPM:
    case LLM_ARCH_XVERSE:
    case LLM_ARCH_COMMAND_R:
    case LLM_ARCH_OLMO:
    case LLM_ARCH_ARCTIC:
    case LLM_ARCH_DEEPSEEK:
    case LLM_ARCH_DEEPSEEK2:
    case LLM_ARCH_CHATGLM:
    case LLM_ARCH_GRANITE:
    case LLM_ARCH_GRANITE_MOE:
    case LLM_ARCH_CHAMELEON:
        return LLAMA_ROPE_TYPE_NORM;

        // the pairs of head values are offset by n_rot/2
    case LLM_ARCH_FALCON:
    case LLM_ARCH_GROK:
    case LLM_ARCH_DBRX:
    case LLM_ARCH_BERT:
    case LLM_ARCH_NOMIC_BERT:
    case LLM_ARCH_STABLELM:
    case LLM_ARCH_BITNET:
    case LLM_ARCH_QWEN:
    case LLM_ARCH_QWEN2:
    case LLM_ARCH_QWEN2MOE:
    case LLM_ARCH_OLMO2:
    case LLM_ARCH_OLMOE:
    case LLM_ARCH_PHI2:
    case LLM_ARCH_PHI3:
    case LLM_ARCH_GEMMA:
    case LLM_ARCH_GEMMA2:
    case LLM_ARCH_STARCODER2:
    case LLM_ARCH_OPENELM:
    case LLM_ARCH_GPTNEOX:
    case LLM_ARCH_CODESHELL:
    case LLM_ARCH_NEMOTRON:
    case LLM_ARCH_EXAONE:
    case LLM_ARCH_MINICPM3:
        return LLAMA_ROPE_TYPE_NEOX;

    case LLM_ARCH_QWEN2VL:
        return LLAMA_ROPE_TYPE_MROPE;

        // all model arches should be listed explicitly here
    case LLM_ARCH_UNKNOWN:
        GGML_ABORT("unknown architecture");
    }

    return LLAMA_ROPE_TYPE_NONE;
}

static void llm_load_hparams(
    llm_model_loader& ml,
    llama_model& model) {
    auto& hparams = model.hparams;
    const gguf_context* ctx = ml.meta.get();

    // get metadata as string
#if 0
    for (const auto &[key, value] : ctx->kv) {
        if (is_array(value)) {
            continue;
        }
        const std::string strValue = toString(value);
        model.gguf_kv.emplace(key, strValue);
    }
#endif

    // get general kv
    ml.get_key(LLM_KV_GENERAL_NAME, model.name, false);

    // get hparams kv
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, hparams.n_vocab, false);

    // everything past this point is not vocab-related
    if (hparams.vocab_only) {
        return;
    }

    ml.get_key(LLM_KV_CONTEXT_LENGTH, hparams.n_ctx_train);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH, hparams.n_embd);
    ml.get_key(LLM_KV_BLOCK_COUNT, hparams.n_layer);
    ml.get_key(LLM_KV_EXPERT_COUNT, hparams.n_expert, false);
    ml.get_key(LLM_KV_EXPERT_USED_COUNT, hparams.n_expert_used, false);

    if (model.arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        ml.get_key(LLM_KV_FEATURES_LENGTH, hparams.n_embd_features);

        ml.get_key(LLM_KV_POSNET_EMBEDDING_LENGTH, hparams.posnet.n_embd);
        ml.get_key(LLM_KV_POSNET_BLOCK_COUNT, hparams.posnet.n_layer);

        ml.get_key(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, hparams.convnext.n_embd);
        ml.get_key(LLM_KV_CONVNEXT_BLOCK_COUNT, hparams.convnext.n_layer);
    }

    GGML_ASSERT(hparams.n_expert <= LLAMA_MAX_EXPERTS);
    GGML_ASSERT(hparams.n_expert_used <= hparams.n_expert);
    if (hparams.n_expert > 0) {
        GGML_ASSERT(hparams.n_expert_used > 0);
    }
    else {
        GGML_ASSERT(hparams.n_expert_used == 0);
    }

    // zero-out the array hparams
    std::fill(hparams.n_head_arr.begin(), hparams.n_head_arr.end(), 0);
    std::fill(hparams.n_head_kv_arr.begin(), hparams.n_head_kv_arr.end(), 0);
    std::fill(hparams.n_ff_arr.begin(), hparams.n_ff_arr.end(), 0);

    ml.get_key_or_arr(LLM_KV_FEED_FORWARD_LENGTH, hparams.n_ff_arr, hparams.n_layer, false);
    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT, hparams.n_head_arr, hparams.n_layer, false);

    // n_head_kv is optional, default to n_head
    hparams.n_head_kv_arr = hparams.n_head_arr;

    ml.get_key_or_arr(LLM_KV_ATTENTION_HEAD_COUNT_KV, hparams.n_head_kv_arr, hparams.n_layer, false);

    bool rope_finetuned = false;
    ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
    hparams.rope_finetuned = rope_finetuned;

    hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
    ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);

    // rope_freq_base (optional)
    hparams.rope_freq_base_train = 10000.0f;
    ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);

    std::string rope_scaling("linear");
    ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
    hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
    GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);

    // rope_freq_scale (inverse of the kv) is optional
    float ropescale = 0.0f;
    if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
        // try the old key name
        ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
    }
    hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f / ropescale;

    ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);

    // non-transformer models do not have attention heads
    if (hparams.n_head() > 0) {
        // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
        // gpt-j n_rot = rotary_dim

        hparams.n_embd_head_k = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k, false);

        hparams.n_embd_head_v = hparams.n_embd / hparams.n_head();
        ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v, false);

        // sanity check for n_rot (optional)
        hparams.n_rot = hparams.n_embd_head_k;

        ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot, false);

        if (model.arch == LLM_ARCH_LLAMA || model.arch == LLM_ARCH_DECI || model.arch == LLM_ARCH_FALCON) {
            if (hparams.n_rot != hparams.n_embd_head_k) {
                throw make_format_runtime_error("invalid n_rot: {}, expected {}", hparams.n_rot, hparams.n_embd_head_k);
            }
        }
    }
    else {
        hparams.n_rot = 0;
        hparams.n_embd_head_k = 0;
        hparams.n_embd_head_v = 0;
    }

    // arch-specific KVs
    switch (model.arch) {
    case LLM_ARCH_LLAMA:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        if (hparams.n_expert == 8) {
            switch (hparams.n_layer) {
            case 32: model.type = e_model::MODEL_8x7B; break;
            case 56: model.type = e_model::MODEL_8x22B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            }
        }
        else {
            switch (hparams.n_layer) {
            case 16: model.type = e_model::MODEL_1B; break; // Llama 3.2 1B
            case 22: model.type = e_model::MODEL_1B; break;
            case 26: model.type = e_model::MODEL_3B; break;
            case 28: model.type = e_model::MODEL_3B; break; // Llama 3.2 3B
                // granite uses a vocab with len 49152
            case 32: model.type = hparams.n_vocab == 49152 ? e_model::MODEL_3B : (hparams.n_vocab < 40000 ? e_model::MODEL_7B : e_model::MODEL_8B); break;
            case 36: model.type = e_model::MODEL_8B; break; // granite
            case 40: model.type = e_model::MODEL_13B; break;
            case 48: model.type = e_model::MODEL_34B; break;
            case 60: model.type = e_model::MODEL_30B; break;
            case 80: model.type = hparams.n_head() == hparams.n_head_kv() ? e_model::MODEL_65B : e_model::MODEL_70B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            }
        }
    } break;
#if 0
    case LLM_ARCH_DECI:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 80: model.type = e_model::MODEL_70B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_MINICPM:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
        ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
        ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);

        switch (hparams.n_layer) {
        case 52: model.type = e_model::MODEL_1B; break;
        case 40: model.type = e_model::MODEL_2B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_MINICPM3:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
        ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);

        switch (hparams.n_layer) {
        case 62: model.type = e_model::MODEL_4B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_GROK:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 64: model.type = e_model::MODEL_314B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_FALCON:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 60: model.type = e_model::MODEL_40B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_BAICHUAN:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 40: model.type = e_model::MODEL_13B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }

        if (model.type == e_model::MODEL_13B) {
            // TODO: become GGUF KV parameter
            hparams.f_max_alibi_bias = 8.0f;
        }
    } break;
    case LLM_ARCH_STARCODER:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1B; break;
        case 36: model.type = e_model::MODEL_3B; break;
        case 42: model.type = e_model::MODEL_7B; break;
        case 40: model.type = e_model::MODEL_15B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_REFACT:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_1B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }

        // TODO: become GGUF KV parameter
        hparams.f_max_alibi_bias = 8.0f;
    } break;
    case LLM_ARCH_BERT:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_CAUSAL, hparams.causal_attn);
        ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
        ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);

        switch (hparams.n_layer) {
        case 3:
            model.type = e_model::MODEL_17M; break; // bge-micro
        case 6:
            model.type = e_model::MODEL_22M; break; // MiniLM-L6
        case 12:
            switch (hparams.n_embd) {
            case 384: model.type = e_model::MODEL_33M; break; // MiniLM-L12, bge-small
            case 768: model.type = e_model::MODEL_109M; break; // bge-base
            } break;
        case 24:
            model.type = e_model::MODEL_335M; break; // bge-large
        }
    } break;
    case LLM_ARCH_JINA_BERT_V2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_CAUSAL, hparams.causal_attn);
        ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
        ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type, false);
        hparams.f_max_alibi_bias = 8.0f;

        switch (hparams.n_layer) {
        case 4:  model.type = e_model::MODEL_33M;  break; // jina-embeddings-small
        case 12: model.type = e_model::MODEL_137M; break; // jina-embeddings-base
        }
    } break;
    case LLM_ARCH_NOMIC_BERT:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_CAUSAL, hparams.causal_attn);
        ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, hparams.n_vocab_type);
        ml.get_key(LLM_KV_POOLING_TYPE, hparams.pooling_type);

        if (hparams.n_layer == 12 && hparams.n_embd == 768) {
            model.type = e_model::MODEL_137M;
        }
    } break;
    case LLM_ARCH_BLOOM:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1B; break;
        case 30:
            switch (hparams.n_embd) {
            case 2560: model.type = e_model::MODEL_3B; break;
            case 4096: model.type = e_model::MODEL_7B; break;
            } break;
        }

        // TODO: become GGUF KV parameter
        hparams.f_max_alibi_bias = 8.0f;
    } break;
    case LLM_ARCH_MPT:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV, hparams.f_clamp_kqv, false);
        ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 48: model.type = e_model::MODEL_30B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_STABLELM:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1B; break;
        case 32: model.type = e_model::MODEL_3B; break;
        case 40: model.type = e_model::MODEL_12B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_QWEN:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 40: model.type = e_model::MODEL_13B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_QWEN2VL:
    {
        std::array<int, 4> section_dims;
        ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, section_dims, 4, true);
        std::copy(section_dims.begin(), section_dims.begin() + 4, std::begin(hparams.rope_sections));
    }
    // fall through
    case LLM_ARCH_QWEN2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 24: model.type = hparams.n_embd == 1024 ? e_model::MODEL_0_5B : e_model::MODEL_1B; break;
        case 28: model.type = hparams.n_embd == 1536 ? e_model::MODEL_1_5B : e_model::MODEL_7B; break;
        case 32: model.type = e_model::MODEL_7B; break;
        case 36: model.type = e_model::MODEL_3B; break;
        case 40: model.type = hparams.n_head() == 20 ? e_model::MODEL_4B : e_model::MODEL_13B; break;
        case 48: model.type = e_model::MODEL_14B; break;
        case 64: model.type = e_model::MODEL_32B; break;
        case 80: model.type = e_model::MODEL_70B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_QWEN2MOE:
    {
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp, false);
        ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);

        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_A2_7B; break;
        case 28: model.type = e_model::MODEL_57B_A14B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_PHI2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1B; break;
        case 32: model.type = e_model::MODEL_3B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_PHI3:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1B; break;
        case 32: model.type = e_model::MODEL_3B; break;
        case 40: model.type = e_model::MODEL_14B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }

        // for backward compatibility ; see: https://github.com/ggerganov/llama.cpp/pull/8931
        if ((hparams.n_layer == 32 || hparams.n_layer == 40) && hparams.n_ctx_train == 4096) {
            // default value for Phi-3-mini-4k-instruct and Phi-3-medium-4k-instruct
            hparams.n_swa = 2047;
        }
        else if (hparams.n_layer == 32 && hparams.n_head_kv(0) == 32 && hparams.n_ctx_train == 131072) {
            // default value for Phi-3-mini-128k-instruct
            hparams.n_swa = 262144;
        }
        else if (hparams.n_layer == 40 && hparams.n_ctx_train == 131072) {
            // default value for Phi-3-medium-128k-instruct
            hparams.n_swa = 131072;
        }
        bool found_swa = ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
        if (!found_swa && hparams.n_swa == 0) {
            throw std::runtime_error("invalid value for sliding_window");
        }
    } break;
    case LLM_ARCH_PLAMO:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 40: model.type = e_model::MODEL_13B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_GPT2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        switch (hparams.n_layer) {
        case 12: model.type = e_model::MODEL_SMALL; break;
        case 24: model.type = e_model::MODEL_MEDIUM; break;
        case 36: model.type = e_model::MODEL_LARGE; break;
        case 48: model.type = e_model::MODEL_XL; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_CODESHELL:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        switch (hparams.n_layer) {
        case 42: model.type = e_model::MODEL_7B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_ORION:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

        switch (hparams.n_layer) {
        case 40: model.type = e_model::MODEL_14B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_INTERNLM2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 48: model.type = e_model::MODEL_20B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_GEMMA:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 18: model.type = e_model::MODEL_2B; break;
        case 28: model.type = e_model::MODEL_7B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_GEMMA2:
    {
        hparams.n_swa = 4096; // default value of gemma 2
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_ATTN_LOGIT_SOFTCAPPING, hparams.f_attn_logit_softcapping, false);
        ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);
        hparams.attn_soft_cap = true;

        switch (hparams.n_layer) {
        case 26: model.type = e_model::MODEL_2B; break;
        case 42: model.type = e_model::MODEL_9B; break;
        case 46: model.type = e_model::MODEL_27B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_STARCODER2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        switch (hparams.n_layer) {
        case 30: model.type = e_model::MODEL_3B; break;
        case 32: model.type = e_model::MODEL_7B; break;
        case 40: model.type = e_model::MODEL_15B; break;
        case 52: model.type = e_model::MODEL_20B; break; // granite
        case 88: model.type = e_model::MODEL_34B; break; // granite
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_MAMBA:
    {
        ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);
        ml.get_key(LLM_KV_SSM_INNER_SIZE, hparams.ssm_d_inner);
        ml.get_key(LLM_KV_SSM_STATE_SIZE, hparams.ssm_d_state);
        ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
        ml.get_key(LLM_KV_SSM_DT_B_C_RMS, hparams.ssm_dt_b_c_rms, false);

        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 24:
            switch (hparams.n_embd) {
            case 768: model.type = e_model::MODEL_SMALL; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 48:
            switch (hparams.n_embd) {
            case 1024: model.type = e_model::MODEL_MEDIUM; break;
            case 1536: model.type = e_model::MODEL_LARGE; break;
            case 2048: model.type = e_model::MODEL_XL; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 64:
            switch (hparams.n_embd) {
            case 2560: model.type = e_model::MODEL_3B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_XVERSE:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 40: model.type = e_model::MODEL_13B; break;
        case 80: model.type = e_model::MODEL_65B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_COMMAND_R:
    {
        ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        switch (hparams.n_layer) {
        case 40: model.type = e_model::MODEL_35B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_DBRX:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV, hparams.f_clamp_kqv);

        switch (hparams.n_layer) {
        case 40: model.type = e_model::MODEL_16x12B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_OLMO:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_CLAMP_KQV, hparams.f_clamp_kqv, false);

        switch (hparams.n_layer) {
        case 22: model.type = e_model::MODEL_1B; break;
        case 32: model.type = e_model::MODEL_7B; break;
        case 80: model.type = e_model::MODEL_70B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_OLMO2:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 16: model.type = e_model::MODEL_1B; break;
        case 32: model.type = e_model::MODEL_7B; break;
        case 40: model.type = e_model::MODEL_13B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_OLMOE:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 16: model.type = e_model::MODEL_A1_7B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_OPENELM:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 16: model.type = e_model::MODEL_270M; break;
        case 20: model.type = e_model::MODEL_450M; break;
        case 28: model.type = e_model::MODEL_1B; break;
        case 36: model.type = e_model::MODEL_3B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_GPTNEOX:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_USE_PARALLEL_RESIDUAL, hparams.use_par_res);
        switch (hparams.n_layer) {
        case 6:
            switch (hparams.n_ff()) {
            case 512: model.type = e_model::MODEL_14M; break;
            case 2048: model.type = e_model::MODEL_70M; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 12:
            switch (hparams.n_ff()) {
            case 3072: model.type = e_model::MODEL_160M; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 16:
            switch (hparams.n_ff()) {
            case 8192: model.type = e_model::MODEL_1B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 24:
            switch (hparams.n_ff()) {
            case 4096: model.type = e_model::MODEL_410M; break;
            case 8192: model.type = e_model::MODEL_1_4B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 32:
            switch (hparams.n_ff()) {
            case 10240: model.type = e_model::MODEL_2_8B; break;
            case 16384: model.type = e_model::MODEL_6_9B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 36:
            switch (hparams.n_ff()) {
            case 20480: model.type = e_model::MODEL_12B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 44:
            switch (hparams.n_ff()) {
            case 24576: model.type = e_model::MODEL_20B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_ARCTIC:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        if (hparams.n_expert == 128) {
            switch (hparams.n_layer) {
            case 35: model.type = e_model::MODEL_10B_128x3_66B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            }
        }
        else {
            model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_DEEPSEEK:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);

        switch (hparams.n_layer) {
        case 28: model.type = e_model::MODEL_20B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_DEEPSEEK2:
    {
        bool is_lite = (hparams.n_layer == 27);
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT, hparams.n_layer_dense_lead);
        if (!is_lite) {
            ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK, hparams.n_lora_q);
        }
        ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK, hparams.n_lora_kv);
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT, hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
        ml.get_key(LLM_KV_ROPE_SCALING_YARN_LOG_MUL, hparams.rope_yarn_log_mul);

        switch (hparams.n_layer) {
        case 27: model.type = e_model::MODEL_16B; break;
        case 60: model.type = e_model::MODEL_236B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_CHATGLM:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        switch (hparams.n_layer) {
        case 28: model.type = e_model::MODEL_6B; break;
        case 40: model.type = e_model::MODEL_9B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_BITNET:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 26: model.type = e_model::MODEL_3B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_T5:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);

        uint32_t dec_start_token_id;
        if (ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, dec_start_token_id, false)) {
            hparams.dec_start_token_id = dec_start_token_id;
        }

        switch (hparams.n_layer) {
        case 6:  model.type = e_model::MODEL_60M;  break; // t5-small
        case 8:  model.type = e_model::MODEL_80M;  break; // flan-t5-small
        case 12:
            switch (hparams.n_ff()) {
            case 3072: model.type = e_model::MODEL_220M; break; // t5-base
            case 2048: model.type = e_model::MODEL_250M; break; // flan-t5-base
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 24:
            switch (hparams.n_ff()) {
            case 4096:  model.type = e_model::MODEL_770M; break; // t5-large
            case 2816:  model.type = e_model::MODEL_780M; break; // flan-t5-large
            case 16384: model.type = e_model::MODEL_3B;   break; // t5-3b
            case 5120:  model.type = e_model::MODEL_3B;   break; // flan-t5-xl
            case 65536: model.type = e_model::MODEL_11B;  break; // t5-11b
            case 10240: model.type = e_model::MODEL_11B;  break; // flan-t5-xxl
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_T5ENCODER:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_attn_bkts);
        model.type = e_model::MODEL_UNKNOWN;
    } break;
    case LLM_ARCH_JAIS:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, hparams.f_max_alibi_bias);

        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1_3B; break;
        case 40: model.type = e_model::MODEL_13B; break;
            /* TODO: add variants */
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_NEMOTRON:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_4B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_EXAONE:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_8B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_RWKV6:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_WKV_HEAD_SIZE, hparams.wkv_head_size);
        ml.get_key(LLM_KV_TIME_MIX_EXTRA_DIM, hparams.time_mix_extra_dim);
        ml.get_key(LLM_KV_TIME_DECAY_EXTRA_DIM, hparams.time_decay_extra_dim);
        ml.get_key(LLM_KV_RESCALE_EVERY_N_LAYERS, hparams.rescale_every_n_layers, false);

        switch (hparams.n_layer) {
        case 24: model.type = e_model::MODEL_1_6B; break;
        case 32:
            switch (hparams.n_embd) {
            case 2560: model.type = e_model::MODEL_3B; break;
            case 4096: model.type = e_model::MODEL_7B; break;
            default: model.type = e_model::MODEL_UNKNOWN;
            } break;
        case 61: model.type = e_model::MODEL_14B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_GRANITE:
    case LLM_ARCH_GRANITE_MOE:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        ml.get_key(LLM_KV_LOGIT_SCALE, hparams.f_logit_scale);
        ml.get_key(LLM_KV_RESIDUAL_SCALE, hparams.f_residual_scale);
        ml.get_key(LLM_KV_EMBEDDING_SCALE, hparams.f_embedding_scale);
        ml.get_key(LLM_KV_ATTENTION_SCALE, hparams.f_attention_scale);

        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_3B; break;
        case 40: model.type = e_model::MODEL_3B; break;
            // Add additional layer/vocab/etc checks here for other model sizes
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_CHAMELEON:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
        hparams.f_norm_eps = 1e-5;  // eps for qk-norm, torch default
        ml.get_key(LLM_KV_SWIN_NORM, hparams.swin_norm);

        switch (hparams.n_layer) {
        case 32: model.type = e_model::MODEL_7B; break;
        case 48: model.type = e_model::MODEL_34B; break;
        default: model.type = e_model::MODEL_UNKNOWN;
        }
    } break;
    case LLM_ARCH_WAVTOKENIZER_DEC:
    {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);
        ml.get_key(LLM_KV_ATTENTION_GROUPNORM_EPS, hparams.f_norm_group_eps);
        ml.get_key(LLM_KV_ATTENTION_GROUPNORM_GROUPS, hparams.n_norm_groups);
        ml.get_key(LLM_KV_ATTENTION_CAUSAL, hparams.causal_attn);
    } break;
#endif
    default: (void)0;
    }

    model.ftype = ml.ftype;

    if (hparams.f_max_alibi_bias > 0.0f) {
        hparams.use_alibi = true;
    }

    hparams.rope_type = llama_rope_type(&model);
}

enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
    LLAMA_TOKEN_TYPE_UNDEFINED = 0,
    LLAMA_TOKEN_TYPE_NORMAL = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN = 2,
    LLAMA_TOKEN_TYPE_CONTROL = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED = 5,
    LLAMA_TOKEN_TYPE_BYTE = 6,
};

static enum llama_vocab_type llama_vocab_get_type(const llama_vocab& vocab) {
    return vocab.type;
}

llama_token llama_byte_to_token_impl(const llama_vocab& vocab, uint8_t ch) {
    GGML_ASSERT(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    static const char* hex = "0123456789ABCDEF";
    switch (llama_vocab_get_type(vocab)) {
    case LLAMA_VOCAB_TYPE_SPM:
    case LLAMA_VOCAB_TYPE_UGM: {
        const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
        auto token = vocab.token_to_id.find(buf);
        if (token != vocab.token_to_id.end()) {
            return (*token).second;
        }
        // Try to fall back to just the byte as a string
        const char buf2[2] = { (char)ch, 0 };
        return vocab.token_to_id.at(buf2);
    }
    case LLAMA_VOCAB_TYPE_WPM:
    case LLAMA_VOCAB_TYPE_BPE: {
        return vocab.token_to_id.at(unicode_byte_to_utf8(ch));
    }
    default:
        GGML_ABORT("fatal error");
    }
}

//
// (de-) tokenize
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_vocab::id _token)
        :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {
    }

    fragment_buffer_variant(const std::string& _raw_text, int64_t _offset, int64_t _length)
        :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_vocab::id)-1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length) {
        GGML_ASSERT(_offset >= 0);
        GGML_ASSERT(_length >= 1);
        GGML_ASSERT(offset + length <= raw_text.length());
    }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_vocab::id token;
    const std::string _dummy;
    const std::string& raw_text;
    const uint64_t offset;
    const uint64_t length;
};

// #define PRETOKENIZERDEBUG

static void tokenizer_st_partition(const llama_vocab& vocab, std::forward_list<fragment_buffer_variant>& buffer, bool parse_special) {
    // for each special token
    for (const llama_vocab::id special_id : vocab.cache_special_tokens) {
        const auto& data = vocab.id_to_token[special_id];
        const auto& special_token = data.text;

        if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto& fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto& raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    auto match = raw_text.find(special_token, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

                    // check if match is within bounds of offset <-> length
                    if (match + special_token.length() > raw_text_base_offset + raw_text_base_length) break;

#ifdef PRETOKENIZERDEBUG
                    LLAMA_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + special_token.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + special_token.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + special_token.length());

                        if (data.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        }
                        else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    }
                    else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        }
                        else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

static void llama_escape_whitespace(std::string& text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char* text;
    size_t n;
};

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm& l, llm_bigram_spm& r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm_session {
    llm_tokenizer_spm_session(const llama_vocab& vocab) : vocab(vocab) {}

    void tokenize(const std::string& text, std::vector<llama_vocab::id>& output) {

        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (int i = 1; i < (int)symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto& left_sym = symbols[bigram.left];
            auto& right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto& symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol& symbol, std::vector<llama_vocab::id>& output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.token_to_id.find(text);

        // Do we need to support is_unused?
        if (token != vocab.token_to_id.end()) {
            output.push_back((*token).second);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_vocab::id token_id = llama_byte_to_token_impl(vocab, symbol.text[j]);
                output.push_back(token_id);
            }
            return;
        }

        resegment(symbols[p->second.first], output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab.id_to_token.size()) {
            return;
        }

        const auto& tok_data = vocab.id_to_token[(*token).second];

        llm_bigram_spm bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab& vocab;
    // currently unused
    // const llm_tokenizer_spm * spm_tokenizer;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
    std::map<std::string, std::pair<int, int>> rev_merge;
};

struct llm_tokenizer_bpe : llm_tokenizer {
    llm_tokenizer_bpe(const llama_vocab& vocab) : llm_tokenizer() {
        GGML_ASSERT(vocab.type == LLAMA_VOCAB_TYPE_BPE);
        switch (vocab.type_pre) {
        case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
            regex_exprs = {
                // original regex from tokenizer.json
                //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_DBRX:
        case LLAMA_VOCAB_PRE_TYPE_SMAUG:
            regex_exprs = {
                // same as llama3
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM:
            regex_exprs = {
                "[\r\n]",
                "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                "\\s+$",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER:
            regex_exprs = {
                "[\r\n]",
                "\\s?\\p{L}+",
                "\\s?\\p{P}+",
                "[一-龥ࠀ-一가-퟿]+",
                "\\p{N}",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_FALCON:
            regex_exprs = {
                "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "[0-9][0-9][0-9]",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_STARCODER:
        case LLAMA_VOCAB_PRE_TYPE_REFACT:
        case LLAMA_VOCAB_PRE_TYPE_COMMAND_R:
        case LLAMA_VOCAB_PRE_TYPE_SMOLLM:
        case LLAMA_VOCAB_PRE_TYPE_CODESHELL:
        case LLAMA_VOCAB_PRE_TYPE_EXAONE:
        case LLAMA_VOCAB_PRE_TYPE_MINERVA:
            regex_exprs = {
                "\\p{N}",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_GPT2:
        case LLAMA_VOCAB_PRE_TYPE_MPT:
        case LLAMA_VOCAB_PRE_TYPE_OLMO:
        case LLAMA_VOCAB_PRE_TYPE_JAIS:
            regex_exprs = {
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
        case LLAMA_VOCAB_PRE_TYPE_QWEN2:
            regex_exprs = {
                // original regex from tokenizer.json
                // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_PORO:
        case LLAMA_VOCAB_PRE_TYPE_BLOOM:
        case LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH:
            regex_exprs = {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
            regex_exprs = {
                "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_VIKING:
            regex_exprs = {
                " ?[^(\\s|.,!?…。，、।۔،)]+",
                "\\p{N}",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_TEKKEN:
            // original regex from tokenizer.json
            // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
            regex_exprs = {
                "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
            };
            break;
        case LLAMA_VOCAB_PRE_TYPE_CHAMELEON:
            // Note: in theory, the special token (sentinel and image token) regex_exprs below
            // are unnecessary, as they are split in `tokenizer_st_partition` anyway.
            // However, since the upstream pre-tokenizer uses them, they are also
            // included here (see https://huggingface.co/facebook/chameleon-7b).
            regex_exprs = {
                "<sentinel:[0-9]+>",  // Sentinel tokens
                "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",  // Image tokens
                "([\\t\\n]|    |  )",  // directly from tokenizer.json
                "\\p{N}", // Individual digits
                "[\\p{P}!-/:-@\\[-`{-~]",  // Punctuation, Isolated
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
            break;
        default:
            // default regex for BPE tokenization pre-processing
            regex_exprs = {
                "[\\p{P}\\$\\+<=>\\^~\\|]+",
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                "\\p{N}+",
                "[0-9][0-9][0-9]",
            };
            break;
        }
    }

    std::vector<std::string> regex_exprs;
};

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() = delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe& l, const llm_bigram_bpe& r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const llama_vocab& vocab) : vocab(vocab),
        bpe_tokenizer(static_cast<const llm_tokenizer_bpe*>(vocab.tokenizer)) {
    }

    static void append(const llama_vocab::id token_id, std::vector<llama_vocab::id>& output) {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<llama_vocab::id>& output) const {
        if (vocab.tokenizer_add_bos) {
            GGML_ASSERT(vocab.special_bos_id != -1);
            output.push_back(vocab.special_bos_id);
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<llama_vocab::id>& output) const {
        if (vocab.tokenizer_add_eos) {
            GGML_ASSERT(vocab.special_eos_id != -1);
            output.push_back(vocab.special_eos_id);
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<llama_vocab::id>& output) const {
        if (vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
            LLAMA_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.tokenizer_add_eos && output.size() >= 2 && *(output.end() - 2) == vocab.special_eos_id) {
            LLAMA_LOG_WARN(
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    void tokenize(const std::string& text, std::vector<llama_vocab::id>& output) {
        int final_prev_index = -1;
        const auto word_collection = unicode_regex_split(text, bpe_tokenizer->regex_exprs);

        symbols_final.clear();

        for (const auto& word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
                symbols.emplace_back(llm_symbol{ -1, -1, word.c_str(), word.size() });
                offset = word.size();
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t)unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (int i = 1; i < (int)symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto& left_symbol = symbols[bigram.left];
                auto& right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto& sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto& symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.token_to_id.find(str);

                if (token == vocab.token_to_id.end()) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.token_to_id.find(byte_str);
                        if (token_multibyte != vocab.token_to_id.end()) {
                            output.push_back(token_multibyte->second);
                        }
                    }
                }
                else {
                    output.push_back((*token).second);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token = std::string(symbols[left].text, symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left = left;
        bigram.right = right;
        bigram.text = left_token + right_token;
        bigram.size = left_token.size() + right_token.size();
        bigram.rank = rank_found;

        work_queue.push(bigram);
    }

    const llama_vocab& vocab;
    const llm_tokenizer_bpe* bpe_tokenizer;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};

struct llm_tokenizer_wpm_session {
    llm_tokenizer_wpm_session(const llama_vocab& vocab) : vocab(vocab) {}

    void tokenize(const std::string& text, std::vector<llama_vocab::id>& output) {
        const auto& token_map = vocab.token_to_id;
        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);
        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string& word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len + 1); j > i; j--) {
                    auto it = token_map.find(word1.substr(i, j - i));
                    if (it != token_map.end()) {
                        output.push_back(it->second);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.special_unk_id);
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    static std::vector<std::string> preprocess(const std::string& text) {
        const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts_nfd) {
            const auto flags = unicode_cpt_flags_from_cpt(cpt);

            if (flags.is_whitespace) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            assert(!flags.is_separator);
            if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (flags.is_punctuation || (cpt < 0x7F && flags.is_symbol) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            }
            else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }

    static bool is_chinese_char(uint32_t cpt) {
        return
            (cpt >= 0x04E00 && cpt <= 0x09FFF) ||
            (cpt >= 0x03400 && cpt <= 0x04DBF) ||
            (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
            (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
            (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
            (cpt >= 0x2B920 && cpt <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (cpt >= 0x0F900 && cpt <= 0x0FAFF) ||
            (cpt >= 0x2F800 && cpt <= 0x2FA1F);
        //(cpt >= 0x3000  && cpt <= 0x303F)  ||
        //(cpt >= 0xFF00  && cpt <= 0xFFEF);
    }

private:
    const llama_vocab& vocab;
    // currently unused
    // const llm_tokenizer_wpm * wpm_tokenizer;
};

struct naive_trie {
    naive_trie() : token(std::nullopt) {
    }
    void insert(const char* key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->token = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        }
        else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }
    std::pair<const char*, size_t> get_longest_prefix(const char* key, size_t len, size_t offset = 0) const {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        }

        return std::make_pair(key, offset);
    }
    const naive_trie* traverse(const char c) const {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        }

        return nullptr;
    }
    bool has_value() const {
		return token.has_value();
    }
    llama_token value() const {
        return token.value();
	}
    std::map<char, naive_trie> children;
    std::optional<llama_token> token;
};

static bool llama_is_normal_token(const llama_vocab& vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_NORMAL;
}

static bool llama_is_user_defined_token(const llama_vocab& vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_USER_DEFINED;
}

static bool llama_is_unused_token(const llama_vocab& vocab, llama_token id) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNUSED;
}

//
// UGM tokenizer
//

struct llm_tokenizer_ugm : llm_tokenizer {
    llm_tokenizer_ugm(const llama_vocab& vocab) : llm_tokenizer() {
        if (vocab.precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t*)&vocab.precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);
            if (xcda_blob_size + charsmap_offset >= vocab.precompiled_charsmap.size()) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }

            // Next xcda_blob_size bytes contain entries of XOR-compressed compact
            // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
            xcda_array = (const uint32_t*)&vocab.precompiled_charsmap[charsmap_offset];
            xcda_array_size = xcda_blob_size / sizeof(uint32_t);
            charsmap_offset += xcda_blob_size;

            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &vocab.precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = vocab.precompiled_charsmap.size() - charsmap_offset;
        }

        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto& token_data = vocab.id_to_token[id];

            if (llama_is_normal_token(vocab, id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (llama_is_normal_token(vocab, id) ||
                llama_is_user_defined_token(vocab, id) ||
                llama_is_unused_token(vocab, id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (llama_is_user_defined_token(vocab, id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }

    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";

    const char* prefix_replacements = NULL;
    size_t prefix_replacements_size = 0;

    const uint32_t* xcda_array = NULL;
    size_t xcda_array_size = 0;

    struct naive_trie user_defined_token_matcher;

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;

    float unknown_token_score_penalty = 10.0;
    float unknown_token_score;

    struct naive_trie token_matcher;
};

struct llm_tokenizer_ugm_session {
    llm_tokenizer_ugm_session(const llama_vocab& vocab) : vocab(vocab),
        ugm_tokenizer(static_cast<const llm_tokenizer_ugm*>(vocab.tokenizer)) {
    }

    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
    void tokenize(const std::string& text, std::vector<llama_vocab::id>& output) {
        // get current size of output (for reversal later)
        size_t output_size = output.size();

        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }

        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, { vocab.special_unk_id, 0, -FLT_MAX });
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.special_unk_id, 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization& current_best = tokenization_results[input_offset];
            const struct naive_trie* node = ugm_tokenizer->token_matcher.traverse(normalized[prefix_offset++]);

            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value()) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    llama_token token_id = node->value();
                    const auto& token_data = vocab.id_to_token[token_id];

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = llama_is_user_defined_token(vocab, token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization& current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, (float)challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if (!single_codepoint_token_found) {
                const double challenger_score = current_best.score_sum + ugm_tokenizer->unknown_token_score;
                prefix_offset = input_offset + n_utf8_code_units;
                struct best_tokenization& current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    struct best_tokenization challenger = { vocab.special_unk_id, input_offset, (float)challenger_score };
                    current_champ = challenger;
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        bool is_prev_unknown = false;
        for (struct best_tokenization& tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
            bool is_unknown = tokenization.token_id == vocab.special_unk_id;
            if (!(is_prev_unknown && is_unknown)) {
                output.push_back(tokenization.token_id);
            }
            if (tokenization.input_offset == 0) {
                break;
            }
            is_prev_unknown = is_unknown;
        }

        // reverse the output since we added tokens starting from the end of the input
        std::reverse(output.begin() + output_size, output.end());
    }

private:

    // helper structure for returning normalization results
    struct normalization_result {
        const char* normalized;
        size_t normalized_len;
        size_t consumed_input;
    };

    void normalize(const std::string& input, std::string* normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.tokenizer_escape_whitespaces ? ugm_tokenizer->escaped_space : " ";

        bool shall_prepend_space = !vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
        bool shall_append_space = vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
        bool shall_merge_spaces = vocab.tokenizer_remove_extra_whitespaces;

        bool is_space_prepended = false;
        bool processing_non_ws = false;

        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                }
                else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }

            input_offset += norm_res.consumed_input;
        }

        if (shall_append_space) {
            normalized->append(space);
        }
    }

    /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Each bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
    struct xcda_array_view {
    public:
        xcda_array_view(const uint32_t* xcda_array, size_t xcda_array_size) : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {
        }
        uint32_t get_base(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
        }
        uint32_t get_lcheck(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) | 0xff);
        }
        bool get_leaf(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 8) & 1;
        }
        uint32_t get_value(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) - 1);
        }
    private:
        uint32_t get_node(size_t index) {
            if (index > xcda_array_size) {
                throw std::runtime_error("Index out of array bounds in XCDA array!");
            }
            return xcda_array[index];
        }
        const uint32_t* xcda_array;
        size_t xcda_array_size;
    };

    // this structure stores the best tokenization so far at input_offset
    struct best_tokenization {
        llama_token token_id;
        size_t input_offset;
        float score_sum;
    };

    struct normalization_result normalize_prefix(const std::string& input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match =
            ugm_tokenizer->user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }

        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (ugm_tokenizer->xcda_array_size > 0) {
            struct xcda_array_view xcda_view(ugm_tokenizer->xcda_array, ugm_tokenizer->xcda_array_size);

            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;
            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);
            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
        }

        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= ugm_tokenizer->prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char* prefix_replacement = &(ugm_tokenizer->prefix_replacements)[longest_prefix_offset];
            return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
        }

        // check if the input prefix contains a valid sequence of UTF-8 code units
        try {
            // if yes, return this sequence unmodified
            size_t prefix_offset = input_offset;
            unicode_cpt_from_utf8(input, prefix_offset);
            return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
        }
        catch (std::invalid_argument& /*ex*/) {
            // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
            return { "\xEF\xBF\xBD", 3, 1 };
        }
    }

    const llama_vocab& vocab;
    const llm_tokenizer_ugm* ugm_tokenizer;
};

//
// RWKV tokenizer
//

static std::vector<uint8_t> llama_unescape_rwkv_token(const std::string& escaped) {
    std::vector<uint8_t> output;
    output.reserve(escaped.size());

    // Parser state
    bool escaping = false;
    uint8_t hex_remaining = 0;
    uint8_t hex_acc = 0;

    // Step through characters, performing parsing
    for (const char& c : escaped) {
        // If we're parsing a hex code, interpret the next character
        if (hex_remaining != 0) {
            uint8_t value = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            hex_acc = (hex_acc << 4) + value;

            hex_remaining -= 1;
            if (hex_remaining == 0) {
                output.push_back(hex_acc);
                hex_acc = 0;
            }

            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            if (c == 't') {
                output.push_back('\t');
            }
            else if (c == 'n') {
                output.push_back('\n');
            }
            else if (c == 'r') {
                output.push_back('\r');
            }
            else if (c == 'x') {
                hex_remaining = 2;
            }
            else {
                output.push_back(c);
            }

            escaping = false;
            continue;
        }

        if (c == '\\') {
            escaping = true;
            continue;
        }

        output.push_back(c);
    }

    return output;
}

struct llm_tokenizer_rwkv : llm_tokenizer {
    llm_tokenizer_rwkv(const llama_vocab& vocab) : llm_tokenizer() {
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.

        // build trie
        for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
            const auto& token = vocab.id_to_token[id];
            const auto data = llama_unescape_rwkv_token(token.text);
            token_matcher.insert((const char*)data.data(), data.size(), id);
        }
    }

    struct naive_trie token_matcher;
};

struct llm_tokenizer_rwkv_session {
    llm_tokenizer_rwkv_session(const llama_vocab& vocab) : vocab(vocab),
        rwkv_tokenizer(static_cast<const llm_tokenizer_rwkv&>(*vocab.tokenizer)) {
    }

    void tokenize(const std::string& text, std::vector<llama_vocab::id>& output) {
        uint32_t position = 0;
        while (position < text.size()) {
            const struct naive_trie* node = rwkv_tokenizer.token_matcher.traverse(text[position]);
            if (node == NULL) {
                // no matching token found, add unknown token
                output.push_back(vocab.special_unk_id);
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            uint32_t token_id = 0;
            uint32_t token_length = 0;
            while (node != NULL) {
                if (node->has_value()) {
                    token_id = node->value();
                    token_length = position + 1;
                }
                node = node->traverse(text[++position]);
            }

            // add the longest matching token
            output.push_back(token_id);
            position = token_length;
        }
    }

private:
    const llama_vocab& vocab;
    const llm_tokenizer_rwkv& rwkv_tokenizer;
};

std::vector<llama_vocab::id> llama_tokenize_internal(
    const llama_vocab& vocab,
    std::string raw_text,
    bool add_special,
    bool parse_special = false);

std::vector<llama_vocab::id> llama_tokenize_internal(
    const llama_vocab& vocab,
    std::string raw_text,
    bool add_special,
    bool parse_special) {
    GGML_ASSERT(vocab.tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    std::vector<llama_vocab::id> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(vocab, fragment_buffer, parse_special);
    }

    switch (vocab.type) {
    case LLAMA_VOCAB_TYPE_SPM:
    {
        // OG tokenizer behavior:
        //
        // tokenizer.encode('', add_special_tokens=True)  returns [1]
        // tokenizer.encode('', add_special_tokens=False) returns []

        bool is_prev_special = true;  // prefix with space if first token

        if (add_special && vocab.tokenizer_add_bos) {
            GGML_ASSERT(vocab.special_bos_id != -1);
            output.push_back(vocab.special_bos_id);
            is_prev_special = true;
        }

        for (const auto& fragment : fragment_buffer) {
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

                // prefix with space if previous is special
                if (vocab.tokenizer_add_space_prefix && is_prev_special) {
                    raw_text = " " + raw_text;
                }

#ifdef PRETOKENIZERDEBUG
                LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                llama_escape_whitespace(raw_text);
                llm_tokenizer_spm_session session(vocab);
                session.tokenize(raw_text, output);
                is_prev_special = false;
            }
            else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                output.push_back(fragment.token);
                is_prev_special = true;
            }
        }

        if (add_special && vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
            LLAMA_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }

        if (add_special && vocab.tokenizer_add_eos) {
            GGML_ASSERT(vocab.special_eos_id != -1);
            output.push_back(vocab.special_eos_id);
        }
    } break;
    case LLAMA_VOCAB_TYPE_BPE:
    {
        llm_tokenizer_bpe_session session(vocab);
        // it calls some other methods that are not exist in llm_tokenizer,
        // here just cast it to bpe tokenizer object
        if (add_special) {
            session.append_bos(output);
        }
        for (const auto& fragment : fragment_buffer) {
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                session.tokenize(raw_text, output);
            }
            else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                session.append(fragment.token, output);
            }
        }

        if (add_special) {
            session.append_eos(output);
            session.check_double_bos_eos(output);
        }
    } break;
    case LLAMA_VOCAB_TYPE_WPM:
    {
        if (add_special) {
            GGML_ASSERT(vocab.special_cls_id != -1);
            output.push_back(vocab.special_cls_id);
        }

        llm_tokenizer_wpm_session session(vocab);

        for (const auto& fragment : fragment_buffer) {
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                session.tokenize(raw_text, output);
            }
            else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                output.push_back(fragment.token);
            }
        }

        if (add_special) {
            GGML_ASSERT(vocab.special_sep_id != -1);
            output.push_back(vocab.special_sep_id);
        }
    } break;
    case LLAMA_VOCAB_TYPE_UGM:
    {
        if (add_special && vocab.tokenizer_add_bos) {
            GGML_ASSERT(vocab.special_bos_id != -1);
            output.push_back(vocab.special_bos_id);
        }
        llm_tokenizer_ugm_session session(vocab);

        for (const auto& fragment : fragment_buffer) {
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);
#ifdef PRETOKENIZERDEBUG
                LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif
                session.tokenize(raw_text, output);
            }
            else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                output.push_back(fragment.token);
            }
        }

        if (add_special && vocab.tokenizer_add_bos && output.size() >= 2 && output[1] == vocab.special_bos_id) {
            LLAMA_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }

        if (add_special && vocab.tokenizer_add_eos) {
            GGML_ASSERT(vocab.special_eos_id != -1);
            output.push_back(vocab.special_eos_id);
        }
    } break;
    case LLAMA_VOCAB_TYPE_RWKV:
    {
        llm_tokenizer_rwkv_session session(vocab);
        for (const auto& fragment : fragment_buffer) {
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                auto raw_text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", raw_text.length(), fragment.offset, fragment.length, raw_text.c_str());
#endif

                session.tokenize(raw_text, output);
            }
            else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                output.push_back(fragment.token);
            }
        }
    } break;
    case LLAMA_VOCAB_TYPE_NONE:
        GGML_ABORT("fatal error");
    }

    return output;
}

llama_token_attr llama_token_get_attr_impl(const struct llama_vocab& vocab, llama_token token) {
    GGML_ASSERT(vocab.type != LLAMA_VOCAB_TYPE_NONE);
    return vocab.id_to_token[token].attr;
}

static uint8_t llama_token_to_byte(const llama_vocab& vocab, llama_token id) {
    GGML_ASSERT(llama_vocab_get_type(vocab) != LLAMA_VOCAB_TYPE_NONE);
    GGML_ASSERT(llama_is_byte_token(vocab, id));
    const auto& token_data = vocab.id_to_token.at(id);
    switch (llama_vocab_get_type(vocab)) {
    case LLAMA_VOCAB_TYPE_SPM:
    case LLAMA_VOCAB_TYPE_UGM: {
        auto buf = token_data.text.substr(3, 2);
        return strtol(buf.c_str(), NULL, 16);
    }
    case LLAMA_VOCAB_TYPE_BPE: {
        GGML_ABORT("fatal error");
        //return unicode_utf8_to_byte(token_data.text); // TODO: why is this here after GGML_ASSERT?
    }
    case LLAMA_VOCAB_TYPE_WPM: {
        GGML_ABORT("fatal error");
    }
    default:
        GGML_ABORT("fatal error");
    }
}

static void llama_unescape_whitespace(std::string& word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

static std::string llama_decode_text(const std::string& text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        }
        catch (const std::out_of_range& /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += std::format("{:02x}", (uint8_t)c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

// does not write null-terminator to buf
int32_t llama_token_to_piece_impl(const struct llama_vocab& vocab, llama_token token, char* buf, int32_t length, int32_t lstrip, bool special) {
    // ref: https://github.com/ggerganov/llama.cpp/pull/7587#discussion_r1620983843
    static const int attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
    const llama_token_attr attr = llama_token_get_attr_impl(vocab, token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=](const char* token, size_t size) -> int32_t {
        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t)size;
        }
        memcpy(buf, token, size);
        return (int32_t)size;
        };

    // if we have a cache - use it
    {
        const auto& cache = vocab.cache_token_to_piece;

        if (!cache.empty()) {
            const auto& result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t)vocab.id_to_token.size()) {
        const std::string& token_text = vocab.id_to_token[token].text;
        switch (llama_vocab_get_type(vocab)) {
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            // NOTE: we accept all unsupported token types,
            // suppressing them like CONTROL tokens.
            if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                return _try_copy(token_text.data(), token_text.size());
            }
            if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                std::string result = token_text;
                llama_unescape_whitespace(result);
                return _try_copy(result.data(), result.size());
            }
            if (attr & LLAMA_TOKEN_ATTR_BYTE) {
                char byte = (char)llama_token_to_byte(vocab, token);
                return _try_copy((char*)&byte, 1);
            }
            break;
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            // NOTE: we accept all unsupported token types,
            // suppressing them like CONTROL tokens.
            if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                return _try_copy(token_text.data(), token_text.size());
            }
            if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                std::string result = llama_decode_text(token_text);
                return _try_copy(result.data(), result.size());
            }
            break;
        }
        case LLAMA_VOCAB_TYPE_RWKV: {
            std::vector<uint8_t> result = llama_unescape_rwkv_token(token_text);

            // If we don't have enough space, return an error
            if (result.size() > (size_t)length) {
                return -(int)result.size();
            }

            memcpy(buf, result.data(), result.size());
            return (int)result.size();
        }
        default:
            GGML_ABORT("fatal error");
        }
    }

    return 0;
}

int llama_vocab::find_bpe_rank(const std::string& token_left, const std::string& token_right) const {
    GGML_ASSERT(token_left.find(' ') == std::string::npos);
    GGML_ASSERT(token_left.find('\n') == std::string::npos);
    GGML_ASSERT(token_right.find(' ') == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

struct llm_tokenizer_spm : llm_tokenizer {
    llm_tokenizer_spm(const llama_vocab& /*vocab*/) : llm_tokenizer() {}
};

struct llm_tokenizer_wpm : llm_tokenizer {
    llm_tokenizer_wpm(const llama_vocab& /*vocab*/) : llm_tokenizer() {}
};

void llama_vocab::init_tokenizer() {
    switch (type) {
    case LLAMA_VOCAB_TYPE_SPM:
        tokenizer = new llm_tokenizer_spm(*this);
        break;
    case LLAMA_VOCAB_TYPE_BPE:
        tokenizer = new llm_tokenizer_bpe(*this);
        break;
    case LLAMA_VOCAB_TYPE_WPM:
        tokenizer = new llm_tokenizer_wpm(*this);
        break;
    case LLAMA_VOCAB_TYPE_UGM:
        tokenizer = new llm_tokenizer_ugm(*this);
        break;
    case LLAMA_VOCAB_TYPE_RWKV:
        tokenizer = new llm_tokenizer_rwkv(*this);
        break;
    default:
        GGML_ABORT("unsupported vocab type");
    }
}

int32_t llama_token_to_piece(
    const struct llama_model* model,
    llama_token   token,
    char* buf,
    int32_t   length,
    int32_t   lstrip,
    bool   special) {
    return llama_token_to_piece_impl(model->vocab, token, buf, length, lstrip, special);
}

// NOTE: avoid ever using this except for building the token_to_piece caches
static std::string llama_token_to_piece(const struct llama_model* model, llama_token token, bool special) {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = llama_token_to_piece(model, token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = llama_token_to_piece(model, token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

static void llm_load_vocab(
    llm_model_loader& ml,
    llama_model& model) {
    auto& vocab = model.vocab;

    struct gguf_context* ctx = ml.meta.get();

    const auto kv = LLM_KV(model.arch);

    // determine vocab type
    {
        std::string tokenizer_model;
        std::string tokenizer_pre;

        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);
        ml.get_key(LLM_KV_TOKENIZER_PRE, tokenizer_pre, false);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
            vocab.type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            vocab.special_bos_id = LLAMA_TOKEN_NULL;
            vocab.special_eos_id = LLAMA_TOKEN_NULL;
            vocab.special_unk_id = LLAMA_TOKEN_NULL;
            vocab.special_sep_id = LLAMA_TOKEN_NULL;
            vocab.special_pad_id = LLAMA_TOKEN_NULL;
            vocab.special_cls_id = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;
            vocab.linefeed_id = LLAMA_TOKEN_NULL;

            // read vocab size from metadata
            if (!ml.get_key(LLM_KV_VOCAB_SIZE, vocab.n_vocab, false)) {
                vocab.n_vocab = 0;
                LLAMA_LOG_WARN("%s: there is no vocab_size in metadata, vocab.n_vocab will be set to %u\n", __func__, vocab.n_vocab);
            }
            return;
        }

        if (tokenizer_model == "llama") {
            vocab.type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            vocab.special_bos_id = 1;
            vocab.special_eos_id = 2;
            vocab.special_unk_id = 0;
            vocab.special_sep_id = LLAMA_TOKEN_NULL;
            vocab.special_pad_id = LLAMA_TOKEN_NULL;
            vocab.special_cls_id = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;
        }
        else if (tokenizer_model == "bert") {
            vocab.type = LLAMA_VOCAB_TYPE_WPM;

            // default special tokens
            vocab.special_bos_id = LLAMA_TOKEN_NULL;
            vocab.special_eos_id = LLAMA_TOKEN_NULL;
            vocab.special_unk_id = 100;
            vocab.special_sep_id = 102;
            vocab.special_pad_id = 0;
            vocab.special_cls_id = 101;
            vocab.special_mask_id = 103;
        }
        else if (tokenizer_model == "gpt2") {
#if 0
            vocab.type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = ctx->find_key(kv(LLM_KV_TOKENIZER_MERGES);
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                vocab.bpe_ranks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            vocab.special_bos_id = 11;
            vocab.special_eos_id = 11;
            vocab.special_unk_id = LLAMA_TOKEN_NULL;
            vocab.special_sep_id = LLAMA_TOKEN_NULL;
            vocab.special_pad_id = LLAMA_TOKEN_NULL;
            vocab.special_cls_id = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;
#endif
        }
        else if (tokenizer_model == "t5") {
#if 0
            vocab.type = LLAMA_VOCAB_TYPE_UGM;

            // default special tokens
            vocab.special_bos_id = LLAMA_TOKEN_NULL;
            vocab.special_eos_id = 1;
            vocab.special_unk_id = 2;
            vocab.special_sep_id = LLAMA_TOKEN_NULL;
            vocab.special_pad_id = 0;
            vocab.special_cls_id = LLAMA_TOKEN_NULL;
            vocab.special_mask_id = LLAMA_TOKEN_NULL;

            const int precompiled_charsmap_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP).c_str());
            if (precompiled_charsmap_keyidx != -1) {
                size_t n_precompiled_charsmap = gguf_get_arr_n(ctx, precompiled_charsmap_keyidx);
                const char* precompiled_charsmap = (const char*)gguf_get_arr_data(ctx, precompiled_charsmap_keyidx);
                vocab.precompiled_charsmap.assign(precompiled_charsmap, precompiled_charsmap + n_precompiled_charsmap);
#ifdef IS_BIG_ENDIAN
                // correct endiannes of data in precompiled_charsmap binary blob
                uint32_t* xcda_blob_size = (uint32_t*)&vocab.precompiled_charsmap[0];
                *xcda_blob_size = __builtin_bswap32(*xcda_blob_size);
                assert(*xcda_blob_size + sizeof(uint32_t) < n_precompiled_charsmap);
                size_t xcda_array_size = *xcda_blob_size / sizeof(uint32_t);
                uint32_t* xcda_array = (uint32_t*)&vocab.precompiled_charsmap[sizeof(uint32_t)];
                for (size_t i = 0; i < xcda_array_size; ++i) {
                    xcda_array[i] = __builtin_bswap32(xcda_array[i]);
                }
#endif
            }
        }
        else if (tokenizer_model == "rwkv") {
            vocab.type = LLAMA_VOCAB_TYPE_RWKV;

            // default special tokens
            vocab.special_bos_id = LLAMA_TOKEN_NULL;
            vocab.special_eos_id = LLAMA_TOKEN_NULL;
            vocab.special_unk_id = LLAMA_TOKEN_NULL;
            vocab.special_sep_id = LLAMA_TOKEN_NULL;
            vocab.special_pad_id = LLAMA_TOKEN_NULL;
        }
        else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (vocab.type == LLAMA_VOCAB_TYPE_BPE) {
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            if (tokenizer_pre.empty()) {
                LLAMA_LOG_WARN("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                LLAMA_LOG_WARN("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            }
            else if (tokenizer_pre == "default") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            }
            else if (
                tokenizer_pre == "llama3" ||
                tokenizer_pre == "llama-v3" ||
                tokenizer_pre == "llama-bpe" ||
                tokenizer_pre == "falcon3") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            }
            else if (
                tokenizer_pre == "deepseek-llm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "deepseek-coder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "falcon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_FALCON;
            }
            else if (
                tokenizer_pre == "mpt") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MPT;
            }
            else if (
                tokenizer_pre == "starcoder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STARCODER;
            }
            else if (
                tokenizer_pre == "gpt-2" ||
                tokenizer_pre == "phi-2" ||
                tokenizer_pre == "jina-es" ||
                tokenizer_pre == "jina-de" ||
                tokenizer_pre == "gigachat" ||
                tokenizer_pre == "jina-v1-en" ||
                tokenizer_pre == "jina-v2-es" ||
                tokenizer_pre == "jina-v2-de" ||
                tokenizer_pre == "jina-v2-code" ||
                tokenizer_pre == "roberta-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT2;
            }
            else if (
                tokenizer_pre == "refact") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_REFACT;
            }
            else if (
                tokenizer_pre == "command-r") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_COMMAND_R;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "qwen2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "stablelm2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STABLELM2;
            }
            else if (
                tokenizer_pre == "olmo") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_OLMO;
            }
            else if (
                tokenizer_pre == "dbrx") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DBRX;
            }
            else if (
                tokenizer_pre == "smaug-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMAUG;
            }
            else if (
                tokenizer_pre == "poro-chat") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_PORO;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "chatglm-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHATGLM4;
                vocab.special_bos_id = LLAMA_TOKEN_NULL;
            }
            else if (
                tokenizer_pre == "viking") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_VIKING;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "jais") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_JAIS;
            }
            else if (
                tokenizer_pre == "tekken") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_TEKKEN;
                vocab.tokenizer_clean_spaces = false;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            }
            else if (
                tokenizer_pre == "smollm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMOLLM;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "codeshell") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CODESHELL;
            }
            else if (
                tokenizer_pre == "bloom") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_BLOOM;
            }
            else if (
                tokenizer_pre == "gpt3-finnish") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH;
            }
            else if (
                tokenizer_pre == "exaone") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_EXAONE;
            }
            else if (
                tokenizer_pre == "chameleon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHAMELEON;
                vocab.tokenizer_add_bos = true;
                vocab.tokenizer_clean_spaces = false;
            }
            else if (
                tokenizer_pre == "minerva-7b") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MINERVA;
            }
            else if (
                tokenizer_pre == "megrez") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
            }
            else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
#endif
        }
        else if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = true;
            vocab.tokenizer_clean_spaces = false;
            vocab.tokenizer_add_bos = true;
            vocab.tokenizer_add_eos = false;
        }
        else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            vocab.tokenizer_add_bos = true;
            vocab.tokenizer_add_eos = false;
        }
        else if (vocab.type == LLAMA_VOCAB_TYPE_UGM) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_bos = false;
            vocab.tokenizer_add_eos = true;
        }
        else if (vocab.type == LLAMA_VOCAB_TYPE_RWKV) {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = false;
            vocab.tokenizer_add_bos = false;
            vocab.tokenizer_add_eos = false;
        }
        else {
            vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        }

        ml.get_key(LLM_KV_TOKENIZER_ADD_PREFIX, vocab.tokenizer_add_space_prefix, false);
        ml.get_key(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS, vocab.tokenizer_remove_extra_whitespaces, false);
    }

    std::optional<size_t> token_idx = ctx->find_key(kv(LLM_KV_TOKENIZER_LIST));
    if (!token_idx.has_value()) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float* scores = nullptr;
    std::optional<size_t> score_idx = ctx->find_key(kv(LLM_KV_TOKENIZER_SCORES));
    if (!score_idx.has_value()) {
#if 0
        scores = (const float*)gguf_get_arr_data(ctx, score_idx);
#endif
    }

    const int* toktypes = nullptr;
    std::optional<size_t> toktype_idx = ctx->find_key(kv(LLM_KV_TOKENIZER_TOKEN_TYPE));
    if (!toktype_idx.has_value()) {
#if 0
        toktypes = (const int*)gguf_get_arr_data(ctx, toktype_idx);
#endif
    }

    const auto& vocabs = ctx->kv[token_idx.value()].data_string;
    const uint32_t n_vocab = vocabs.size();

    vocab.n_vocab = n_vocab;
    vocab.id_to_token.resize(n_vocab);

    for (size_t i = 0; i < vocabs.size(); i++) {
        auto word = vocabs[i];
        //GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);
        if (word.empty()) {
            LLAMA_LOG_WARN("%s: empty token at index %u\n", __func__, i);
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        vocab.token_to_id[word] = i;
        vocab.max_token_len = std::max(vocab.max_token_len, (int)word.size());

        auto& token_data = vocab.id_to_token[i];
        token_data.text = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch (toktypes[i]) {
            case LLAMA_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
            case LLAMA_TOKEN_TYPE_UNUSED:       token_data.attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
            case LLAMA_TOKEN_TYPE_NORMAL:       token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
            case LLAMA_TOKEN_TYPE_CONTROL:      token_data.attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
            case LLAMA_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
            case LLAMA_TOKEN_TYPE_BYTE:         token_data.attr = LLAMA_TOKEN_ATTR_BYTE;         break;
            case LLAMA_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            default:                            token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    GGML_ASSERT(vocab.id_to_token.size() == vocab.token_to_id.size());

    vocab.init_tokenizer();

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        try {
            vocab.linefeed_id = llama_byte_to_token_impl(vocab, '\n');
        }
        catch (const std::exception& e) {
            LLAMA_LOG_WARN("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            vocab.linefeed_id = vocab.special_pad_id;
        }
    }
    else if (vocab.type == LLAMA_VOCAB_TYPE_WPM) {
        vocab.linefeed_id = vocab.special_pad_id;
    }
    else if (vocab.type == LLAMA_VOCAB_TYPE_RWKV) {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\n", false);
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        vocab.linefeed_id = ids[0];
    }
    else {
        const std::vector<int> ids = llama_tokenize_internal(vocab, "\xC4\x8A", false); // U+010A

        //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        if (ids.empty()) {
            LLAMA_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            vocab.linefeed_id = vocab.special_pad_id;
        }
        else {
            vocab.linefeed_id = ids[0];
        }
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t&>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     vocab.special_bos_id     },
            { LLM_KV_TOKENIZER_EOS_ID,     vocab.special_eos_id     },
            { LLM_KV_TOKENIZER_EOT_ID,     vocab.special_eot_id     },
            { LLM_KV_TOKENIZER_EOM_ID,     vocab.special_eom_id     },
            { LLM_KV_TOKENIZER_UNK_ID,     vocab.special_unk_id     },
            { LLM_KV_TOKENIZER_SEP_ID,     vocab.special_sep_id     },
            { LLM_KV_TOKENIZER_PAD_ID,     vocab.special_pad_id     },
            { LLM_KV_TOKENIZER_CLS_ID,     vocab.special_cls_id     },
            { LLM_KV_TOKENIZER_MASK_ID,    vocab.special_mask_id    },
            { LLM_KV_TOKENIZER_FIM_PRE_ID, vocab.special_fim_pre_id },
            { LLM_KV_TOKENIZER_FIM_SUF_ID, vocab.special_fim_suf_id },
            { LLM_KV_TOKENIZER_FIM_MID_ID, vocab.special_fim_mid_id },
            { LLM_KV_TOKENIZER_FIM_PAD_ID, vocab.special_fim_pad_id },
            { LLM_KV_TOKENIZER_FIM_REP_ID, vocab.special_fim_rep_id },
            { LLM_KV_TOKENIZER_FIM_SEP_ID, vocab.special_fim_sep_id },

            // deprecated
            { LLM_KV_TOKENIZER_PREFIX_ID, vocab.special_fim_pre_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, vocab.special_fim_suf_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, vocab.special_fim_mid_id },
        };

        for (const auto &[item, id] : special_token_types) {
            const std::string& key = kv(item);

            uint32_t new_id;
            if (!ml.get_key(item, new_id, false)) {
                continue;
            }
            if (new_id >= vocab.id_to_token.size()) {
                LLAMA_LOG_WARN("%s: bad special token: '%s' = %ud, using default id %d\n",
                    __func__, key.c_str(), new_id, id);
            }
            else {
                id = new_id;
            }
        }

        // Handle add_bos_token and add_eos_token
        {
            bool temp = true;

            if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
                vocab.tokenizer_add_bos = temp;
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {
                vocab.tokenizer_add_eos = temp;
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text

        for (const auto& t : vocab.token_to_id) {
            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if (vocab.special_eot_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|eot_id|>"
                    || t.first == "<|im_end|>"
                    || t.first == "<|end|>"
                    || t.first == "<end_of_turn>"
                    || t.first == "<|endoftext|>"
                    || t.first == "<EOT>"
                    || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                    ) {
                    vocab.special_eot_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find EOM token: "<|eom_id|>"
            if (vocab.special_eom_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|eom_id|>"
                    ) {
                    vocab.special_eom_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if (vocab.special_fim_pre_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|fim_prefix|>"  // Qwen
                    || t.first == "<fim-prefix>"
                    || t.first == "<｜fim▁begin｜>" // DeepSeek
                    || t.first == "<PRE>"
                    ) {
                    vocab.special_fim_pre_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if (vocab.special_fim_suf_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|fim_suffix|>" // Qwen
                    || t.first == "<fim-suffix>"
                    || t.first == "<｜fim▁hole｜>" // DeepSeek
                    || t.first == "<SUF>"
                    ) {
                    vocab.special_fim_suf_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if (vocab.special_fim_mid_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|fim_middle|>" // Qwen
                    || t.first == "<fim-middle>"
                    || t.first == "<｜fim▁end｜>"  // DeepSeek
                    || t.first == "<MID>"
                    ) {
                    vocab.special_fim_mid_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if (vocab.special_fim_pad_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|fim_pad|>" // Qwen
                    || t.first == "<fim-pad>"
                    || t.first == "<PAD>"
                    ) {
                    vocab.special_fim_pad_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if (vocab.special_fim_rep_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|fim_repo|>"  // Qwen
                    || t.first == "<|repo_name|>"
                    || t.first == "<fim-repo>"
                    || t.first == "<REPO>"
                    ) {
                    vocab.special_fim_rep_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if (vocab.special_fim_sep_id == LLAMA_TOKEN_NULL) {
                if (false
                    || t.first == "<|file_sep|>" // Qwen
                    ) {
                    vocab.special_fim_sep_id = t.second;
                    if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                        vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggerganov/llama.cpp/issues/9606
        vocab.special_eog_ids.clear();

        if (vocab.special_fim_pad_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_fim_pad_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_fim_pad_id);
        }

        if (vocab.special_fim_rep_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_fim_rep_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_fim_rep_id);
        }

        if (vocab.special_fim_sep_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_fim_sep_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_fim_sep_id);
        }

        for (const auto& t : vocab.token_to_id) {
            if (false
                || t.first == "<|eot_id|>"
                || t.first == "<|im_end|>"
                || t.first == "<|end|>"
                || t.first == "<end_of_turn>"
                || t.first == "<|endoftext|>"
                || t.first == "<|eom_id|>"
                || t.first == "<EOT>"
                ) {
                vocab.special_eog_ids.insert(t.second);
                if ((vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                    LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                        __func__, t.second, t.first.c_str());
                    vocab.id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                }
            }
            else {
                // token is control, but not marked as EOG -> print a debug log
                if (vocab.id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL && vocab.special_eog_ids.count(t.second) == 0) {
                    LLAMA_LOG_DEBUG("%s: control token: %6d '%s' is not marked as EOG\n",
                        __func__, t.second, t.first.c_str());
                }
            }
        }

        // sanity checks
        if (vocab.special_eos_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_eos_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_eos_id);
            LLAMA_LOG_WARN("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (vocab.special_eot_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_eot_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_eot_id);
            LLAMA_LOG_WARN("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (vocab.special_eom_id != LLAMA_TOKEN_NULL && vocab.special_eog_ids.count(vocab.special_eom_id) == 0) {
            vocab.special_eog_ids.insert(vocab.special_eom_id);
            LLAMA_LOG_WARN("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }
    }

    // build special tokens cache
    {
        for (llama_vocab::id id = 0; id < (llama_vocab::id)n_vocab; ++id) {
            if (vocab.id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                vocab.cache_special_tokens.push_back(id);
            }
        }

        std::sort(vocab.cache_special_tokens.begin(), vocab.cache_special_tokens.end(),
            [&](const llama_vocab::id a, const llama_vocab::id b) {
                return vocab.id_to_token[a].text.size() > vocab.id_to_token[b].text.size();
            }
        );

        LLAMA_LOG_INFO("%s: special tokens cache size = %u\n", __func__, (uint32_t)vocab.cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<llama_vocab::token> cache_token_to_piece(n_vocab);

        for (uint32_t id = 0; id < n_vocab; ++id) {
            cache_token_to_piece[id] = llama_token_to_piece(&model, id, true);

            size_cache += cache_token_to_piece[id].size();
        }

        std::swap(vocab.cache_token_to_piece, cache_token_to_piece);

        LLAMA_LOG_INFO("%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the GGUF file.
    //TODO: Extract attributes from GGUF file.
    {
        auto _contains_any = [](const std::string& str, const std::vector<std::string>& substrs) -> bool {
            for (auto substr : substrs) {
                if (str.find(substr) < std::string::npos) {
                    return true;
                }
            }
            return false;
            };

        auto _set_tokenid_attr = [&](const llama_vocab::id id, llama_token_attr attr, bool value) {
            uint32_t current = vocab.id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            vocab.id_to_token[id].attr = (llama_token_attr)current;
            };

        auto _set_token_attr = [&](const std::string& token, llama_token_attr attr, bool value) {
            _set_tokenid_attr(vocab.token_to_id.at(token), attr, value);
            };

        std::string model_name;
        std::string tokenizer_pre;

        ml.get_key(LLM_KV_GENERAL_NAME, model_name, false);
        ml.get_key(LLM_KV_TOKENIZER_PRE, tokenizer_pre, false);

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [](const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer name
        if (_contains_any(tokenizer_pre, { "jina-v2-de", "jina-v2-es", "jina-v2-code" })) {
            _set_token_attr("<mask>", LLAMA_TOKEN_ATTR_LSTRIP, true);
        }
        else if (_contains_any(model_name, { "phi-3", "phi3" })) {
            for (auto id : vocab.cache_special_tokens) {
                _set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (auto token : { "</s>" }) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (auto token : { "<unk>", "<s>", "<|endoftext|>" }) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false);
            }
        }
    }
}

static void llm_load_stats(llm_model_loader& ml, llama_model& model) {
    model.n_elements = ml.n_elements;
    model.n_bytes = ml.n_bytes;
}

static const size_t kiB = 1024;
static const size_t MiB = 1024 * kiB;
static const size_t GiB = 1024 * MiB;

static void llm_load_print_meta(llm_model_loader& ml, llama_model& model) {
    const auto& hparams = model.hparams;
    const auto& vocab = model.vocab;

    const char* rope_scaling_type = LLAMA_ROPE_SCALING_TYPES.at(hparams.rope_scaling_type_train);

    auto print_f = [](const std::function<uint32_t(uint32_t)>& f, uint32_t n) {
        bool is_var = false;

        std::vector<uint32_t> v;
        for (uint32_t i = 0; i < n; ++i) {
            v.push_back(f(i));
            if (v[i] != v[0]) {
                is_var = true;
            }
        }

        std::stringstream ss;

        if (is_var) {
            ss << "[";
            for (uint32_t i = 0; i < n; ++i) {
                ss << v[i];
                if (i < n - 1) {
                    ss << ", ";
                }
            }
            ss << "]";
        }
        else {
            ss << v[0];
        }

        return ss.str();
    };

    // hparams
    LLAMA_LOG_INFO("%s: format           = %s\n", __func__, llama_file_version_name(ml.fver));
    LLAMA_LOG_INFO("%s: arch             = %s\n", __func__, LLM_ARCH_NAMES.at(model.arch));
    LLAMA_LOG_INFO("%s: vocab type       = %s\n", __func__, llama_model_vocab_type_name(vocab.type));
    LLAMA_LOG_INFO("%s: n_vocab          = %u\n", __func__, hparams.n_vocab);
    LLAMA_LOG_INFO("%s: n_merges         = %u\n", __func__, (int)vocab.bpe_ranks.size());
    LLAMA_LOG_INFO("%s: vocab_only       = %d\n", __func__, hparams.vocab_only);

    if (!hparams.vocab_only) {
        LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n", __func__, hparams.n_ctx_train);
        LLAMA_LOG_INFO("%s: n_embd           = %u\n", __func__, hparams.n_embd);
        LLAMA_LOG_INFO("%s: n_layer          = %u\n", __func__, hparams.n_layer);
        LLAMA_LOG_INFO("%s: n_head           = %s\n", __func__, print_f([&](uint32_t il) { return hparams.n_head(il);    }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_head_kv        = %s\n", __func__, print_f([&](uint32_t il) { return hparams.n_head_kv(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_rot            = %u\n", __func__, hparams.n_rot);
        LLAMA_LOG_INFO("%s: n_swa            = %u\n", __func__, hparams.n_swa);
        LLAMA_LOG_INFO("%s: n_embd_head_k    = %u\n", __func__, hparams.n_embd_head_k);
        LLAMA_LOG_INFO("%s: n_embd_head_v    = %u\n", __func__, hparams.n_embd_head_v);
        LLAMA_LOG_INFO("%s: n_gqa            = %s\n", __func__, print_f([&](uint32_t il) { return hparams.n_gqa(il);        }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_k_gqa     = %s\n", __func__, print_f([&](uint32_t il) { return hparams.n_embd_k_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_embd_v_gqa     = %s\n", __func__, print_f([&](uint32_t il) { return hparams.n_embd_v_gqa(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n", __func__, hparams.f_norm_eps);
        LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n", __func__, hparams.f_norm_rms_eps);
        LLAMA_LOG_INFO("%s: f_clamp_kqv      = %.1e\n", __func__, hparams.f_clamp_kqv);
        LLAMA_LOG_INFO("%s: f_max_alibi_bias = %.1e\n", __func__, hparams.f_max_alibi_bias);
        LLAMA_LOG_INFO("%s: f_logit_scale    = %.1e\n", __func__, hparams.f_logit_scale);
        LLAMA_LOG_INFO("%s: n_ff             = %s\n", __func__, print_f([&](uint32_t il) { return hparams.n_ff(il); }, hparams.n_layer).c_str());
        LLAMA_LOG_INFO("%s: n_expert         = %u\n", __func__, hparams.n_expert);
        LLAMA_LOG_INFO("%s: n_expert_used    = %u\n", __func__, hparams.n_expert_used);
        LLAMA_LOG_INFO("%s: causal attn      = %d\n", __func__, hparams.causal_attn);
        LLAMA_LOG_INFO("%s: pooling type     = %d\n", __func__, hparams.pooling_type);
        LLAMA_LOG_INFO("%s: rope type        = %d\n", __func__, hparams.rope_type);
        LLAMA_LOG_INFO("%s: rope scaling     = %s\n", __func__, rope_scaling_type);
        LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n", __func__, hparams.rope_freq_base_train);
        LLAMA_LOG_INFO("%s: freq_scale_train = %g\n", __func__, hparams.rope_freq_scale_train);
        LLAMA_LOG_INFO("%s: n_ctx_orig_yarn  = %u\n", __func__, hparams.n_ctx_orig_yarn);
        LLAMA_LOG_INFO("%s: rope_finetuned   = %s\n", __func__, hparams.rope_finetuned ? "yes" : "unknown");
        LLAMA_LOG_INFO("%s: ssm_d_conv       = %u\n", __func__, hparams.ssm_d_conv);
        LLAMA_LOG_INFO("%s: ssm_d_inner      = %u\n", __func__, hparams.ssm_d_inner);
        LLAMA_LOG_INFO("%s: ssm_d_state      = %u\n", __func__, hparams.ssm_d_state);
        LLAMA_LOG_INFO("%s: ssm_dt_rank      = %u\n", __func__, hparams.ssm_dt_rank);
        LLAMA_LOG_INFO("%s: ssm_dt_b_c_rms   = %d\n", __func__, hparams.ssm_dt_b_c_rms);
    }

    LLAMA_LOG_INFO("%s: model type       = %s\n", __func__, llama_model_type_name(model.type));
    LLAMA_LOG_INFO("%s: model ftype      = %s\n", __func__, llama_model_ftype_name(model.ftype).c_str());
    if (ml.n_elements >= 1e12) {
        LLAMA_LOG_INFO("%s: model params     = %.2f T\n", __func__, ml.n_elements * 1e-12);
    }
    else if (ml.n_elements >= 1e9) {
        LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, ml.n_elements * 1e-9);
    }
    else if (ml.n_elements >= 1e6) {
        LLAMA_LOG_INFO("%s: model params     = %.2f M\n", __func__, ml.n_elements * 1e-6);
    }
    else {
        LLAMA_LOG_INFO("%s: model params     = %.2f K\n", __func__, ml.n_elements * 1e-3);
    }
    if (ml.n_bytes < GiB) {
        LLAMA_LOG_INFO("%s: model size       = %.2f MiB (%.2f BPW) \n", __func__, ml.n_bytes / 1024.0 / 1024.0, ml.n_bytes * 8.0 / ml.n_elements);
    }
    else {
        LLAMA_LOG_INFO("%s: model size       = %.2f GiB (%.2f BPW) \n", __func__, ml.n_bytes / 1024.0 / 1024.0 / 1024.0, ml.n_bytes * 8.0 / ml.n_elements);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n", __func__, model.name.c_str());

    // special tokens
    if (vocab.special_bos_id != -1) { LLAMA_LOG_INFO("%s: BOS token        = %d '%s'\n", __func__, vocab.special_bos_id, vocab.id_to_token[vocab.special_bos_id].text.c_str()); }
    if (vocab.special_eos_id != -1) { LLAMA_LOG_INFO("%s: EOS token        = %d '%s'\n", __func__, vocab.special_eos_id, vocab.id_to_token[vocab.special_eos_id].text.c_str()); }
    if (vocab.special_eot_id != -1) { LLAMA_LOG_INFO("%s: EOT token        = %d '%s'\n", __func__, vocab.special_eot_id, vocab.id_to_token[vocab.special_eot_id].text.c_str()); }
    if (vocab.special_eom_id != -1) { LLAMA_LOG_INFO("%s: EOM token        = %d '%s'\n", __func__, vocab.special_eom_id, vocab.id_to_token[vocab.special_eom_id].text.c_str()); }
    if (vocab.special_unk_id != -1) { LLAMA_LOG_INFO("%s: UNK token        = %d '%s'\n", __func__, vocab.special_unk_id, vocab.id_to_token[vocab.special_unk_id].text.c_str()); }
    if (vocab.special_sep_id != -1) { LLAMA_LOG_INFO("%s: SEP token        = %d '%s'\n", __func__, vocab.special_sep_id, vocab.id_to_token[vocab.special_sep_id].text.c_str()); }
    if (vocab.special_pad_id != -1) { LLAMA_LOG_INFO("%s: PAD token        = %d '%s'\n", __func__, vocab.special_pad_id, vocab.id_to_token[vocab.special_pad_id].text.c_str()); }
    if (vocab.special_cls_id != -1) { LLAMA_LOG_INFO("%s: CLS token        = %d '%s'\n", __func__, vocab.special_cls_id, vocab.id_to_token[vocab.special_cls_id].text.c_str()); }
    if (vocab.special_mask_id != -1) { LLAMA_LOG_INFO("%s: MASK token       = %d '%s'\n", __func__, vocab.special_mask_id, vocab.id_to_token[vocab.special_mask_id].text.c_str()); }

    if (vocab.linefeed_id != -1) { LLAMA_LOG_INFO("%s: LF token         = %d '%s'\n", __func__, vocab.linefeed_id, vocab.id_to_token[vocab.linefeed_id].text.c_str()); }

    if (vocab.special_fim_pre_id != -1) { LLAMA_LOG_INFO("%s: FIM PRE token    = %d '%s'\n", __func__, vocab.special_fim_pre_id, vocab.id_to_token[vocab.special_fim_pre_id].text.c_str()); }
    if (vocab.special_fim_suf_id != -1) { LLAMA_LOG_INFO("%s: FIM SUF token    = %d '%s'\n", __func__, vocab.special_fim_suf_id, vocab.id_to_token[vocab.special_fim_suf_id].text.c_str()); }
    if (vocab.special_fim_mid_id != -1) { LLAMA_LOG_INFO("%s: FIM MID token    = %d '%s'\n", __func__, vocab.special_fim_mid_id, vocab.id_to_token[vocab.special_fim_mid_id].text.c_str()); }
    if (vocab.special_fim_pad_id != -1) { LLAMA_LOG_INFO("%s: FIM PAD token    = %d '%s'\n", __func__, vocab.special_fim_pad_id, vocab.id_to_token[vocab.special_fim_pad_id].text.c_str()); }
    if (vocab.special_fim_rep_id != -1) { LLAMA_LOG_INFO("%s: FIM REP token    = %d '%s'\n", __func__, vocab.special_fim_rep_id, vocab.id_to_token[vocab.special_fim_rep_id].text.c_str()); }
    if (vocab.special_fim_sep_id != -1) { LLAMA_LOG_INFO("%s: FIM SEP token    = %d '%s'\n", __func__, vocab.special_fim_sep_id, vocab.id_to_token[vocab.special_fim_sep_id].text.c_str()); }

    for (const auto& id : vocab.special_eog_ids) {
        LLAMA_LOG_INFO("%s: EOG token        = %d '%s'\n", __func__, id, vocab.id_to_token[id].text.c_str());
    }

    LLAMA_LOG_INFO("%s: max token length = %d\n", __func__, vocab.max_token_len);

    if (model.arch == LLM_ARCH_DEEPSEEK) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n", __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n", __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n", __func__, hparams.expert_weights_scale);
    }

    if (model.arch == LLM_ARCH_DEEPSEEK2) {
        LLAMA_LOG_INFO("%s: n_layer_dense_lead   = %d\n", __func__, hparams.n_layer_dense_lead);
        LLAMA_LOG_INFO("%s: n_lora_q             = %d\n", __func__, hparams.n_lora_q);
        LLAMA_LOG_INFO("%s: n_lora_kv            = %d\n", __func__, hparams.n_lora_kv);
        LLAMA_LOG_INFO("%s: n_ff_exp             = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_expert_shared      = %d\n", __func__, hparams.n_expert_shared);
        LLAMA_LOG_INFO("%s: expert_weights_scale = %.1f\n", __func__, hparams.expert_weights_scale);
        LLAMA_LOG_INFO("%s: rope_yarn_log_mul    = %.4f\n", __func__, hparams.rope_yarn_log_mul);
    }

    if (model.arch == LLM_ARCH_QWEN2MOE) {
        LLAMA_LOG_INFO("%s: n_ff_exp         = %d\n", __func__, hparams.n_ff_exp);
        LLAMA_LOG_INFO("%s: n_ff_shexp       = %d\n", __func__, hparams.n_ff_shexp);
    }

    if (model.arch == LLM_ARCH_MINICPM || model.arch == LLM_ARCH_GRANITE || model.arch == LLM_ARCH_GRANITE_MOE) {
        LLAMA_LOG_INFO("%s: f_embedding_scale = %f\n", __func__, hparams.f_embedding_scale);
        LLAMA_LOG_INFO("%s: f_residual_scale  = %f\n", __func__, hparams.f_residual_scale);
        LLAMA_LOG_INFO("%s: f_attention_scale = %f\n", __func__, hparams.f_attention_scale);
    }
}

// CPU: ACCEL -> CPU extra -> GPU host -> CPU
static llama_model::buft_list_t make_cpu_buft_list(llama_model& model) {
    llama_model::buft_list_t buft_list;

    // add ACCEL buffer types
    for (auto dev : backend_devs()) {
        if (dev->get_type() == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            auto* buft = dev->get_buffer_type();
            // skip
            if (buft != ggml_backend_cpu_buffer_type()) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add extra buffer types
    ggml_backend_device* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

    for (auto & extra_bufts : cpu_dev->get_extra_bufts())
        buft_list.emplace_back(cpu_dev, extra_bufts);

    // add a host buffer type
    // storing the tensors in a host buffer is useful when the processing of large batches
    // is offloaded to a GPU device, since it reduces the time spent on data transfers
    // generally, this will be done using the first device in the list
    // a better approach would be to handle this on a weight-by-weight basis using the offload_op
    // function of the device to determine if it would benefit from being stored in a host buffer
    for (auto* dev : model.devices) {
        ggml_backend_buffer_type* buft = dev->get_host_buffer_type();
        if (buft) {
            buft_list.emplace_back(dev, buft);
            break;
        }
    }

    // add the CPU buffer type
    for (auto dev : backend_devs()) {
        if (dev->get_type() == GGML_BACKEND_DEVICE_TYPE_CPU) {
            buft_list.emplace_back(dev, dev->get_buffer_type());
        }
    }

    return buft_list;
}

// GPU: split if LLAMA_SPLIT_MODE_ROW -> GPU
static llama_model::buft_list_t make_gpu_buft_list(ggml_backend_device* dev, enum llama_split_mode split_mode, const float* tensor_split) {
    llama_model::buft_list_t buft_list;

#if 0
    // add the device split buffer type if requested and available
    if (split_mode == LLAMA_SPLIT_MODE_ROW) {
        ggml_backend_reg_t reg = dev->get_backend_reg();
        auto ggml_backend_split_buffer_type_fn = (ggml_backend_split_buffer_type_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_split_buffer_type");
        if (ggml_backend_split_buffer_type_fn) {
            size_t dev_index = [&]() {
                auto* reg = dev->get_backend_reg();
                for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                    if (ggml_backend_reg_dev_get(reg, i) == dev) {
                        return i;
                    }
                }
                throw std::runtime_error(format("device %s not found in its backend reg", ggml_backend_dev_name(dev)));
                }();
            auto* buft = ggml_backend_split_buffer_type_fn(dev_index, tensor_split);
            if (buft != nullptr) {
                buft_list.emplace_back(dev, buft);
            }
        }
    }

    // add the device default buffer type
    buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev));
#endif
    return buft_list;
}

static int llama_get_device_count(const llama_model& model) {
    return (int)model.devices.size();
}

enum llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_TOKEN_EMBD_NORM,
    LLM_TENSOR_TOKEN_TYPES,
    LLM_TENSOR_POS_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ROPE_FREQS,
    LLM_TENSOR_ROPE_FACTORS_LONG,
    LLM_TENSOR_ROPE_FACTORS_SHORT,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_QKV,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_NORM_2,
    LLM_TENSOR_ATTN_OUT_NORM,
    LLM_TENSOR_ATTN_POST_NORM,
    LLM_TENSOR_ATTN_ROT_EMBD,
    LLM_TENSOR_FFN_GATE_INP,
    LLM_TENSOR_FFN_GATE_INP_SHEXP,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_FFN_POST_NORM,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_FFN_ACT,
    LLM_TENSOR_FFN_DOWN_EXP,  // split experts for backward compatibility
    LLM_TENSOR_FFN_GATE_EXP,
    LLM_TENSOR_FFN_UP_EXP,
    LLM_TENSOR_FFN_NORM_EXPS,
    LLM_TENSOR_FFN_DOWN_EXPS, // merged experts
    LLM_TENSOR_FFN_GATE_EXPS,
    LLM_TENSOR_FFN_UP_EXPS,
    LLM_TENSOR_FFN_DOWN_SHEXP,
    LLM_TENSOR_FFN_GATE_SHEXP,
    LLM_TENSOR_FFN_UP_SHEXP,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
    LLM_TENSOR_LAYER_OUT_NORM,
    LLM_TENSOR_SSM_IN,
    LLM_TENSOR_SSM_CONV1D,
    LLM_TENSOR_SSM_X,
    LLM_TENSOR_SSM_DT,
    LLM_TENSOR_SSM_A,
    LLM_TENSOR_SSM_D,
    LLM_TENSOR_SSM_OUT,
    LLM_TENSOR_TIME_MIX_W1,
    LLM_TENSOR_TIME_MIX_W2,
    LLM_TENSOR_TIME_MIX_LERP_X,
    LLM_TENSOR_TIME_MIX_LERP_W,
    LLM_TENSOR_TIME_MIX_LERP_K,
    LLM_TENSOR_TIME_MIX_LERP_V,
    LLM_TENSOR_TIME_MIX_LERP_R,
    LLM_TENSOR_TIME_MIX_LERP_G,
    LLM_TENSOR_TIME_MIX_FIRST,
    LLM_TENSOR_TIME_MIX_DECAY,
    LLM_TENSOR_TIME_MIX_DECAY_W1,
    LLM_TENSOR_TIME_MIX_DECAY_W2,
    LLM_TENSOR_TIME_MIX_KEY,
    LLM_TENSOR_TIME_MIX_VALUE,
    LLM_TENSOR_TIME_MIX_RECEPTANCE,
    LLM_TENSOR_TIME_MIX_GATE,
    LLM_TENSOR_TIME_MIX_LN,
    LLM_TENSOR_TIME_MIX_OUTPUT,
    LLM_TENSOR_CHANNEL_MIX_LERP_K,
    LLM_TENSOR_CHANNEL_MIX_LERP_R,
    LLM_TENSOR_CHANNEL_MIX_KEY,
    LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,
    LLM_TENSOR_CHANNEL_MIX_VALUE,
    LLM_TENSOR_ATTN_Q_A,
    LLM_TENSOR_ATTN_Q_B,
    LLM_TENSOR_ATTN_KV_A_MQA,
    LLM_TENSOR_ATTN_KV_B,
    LLM_TENSOR_ATTN_Q_A_NORM,
    LLM_TENSOR_ATTN_KV_A_NORM,
    LLM_TENSOR_ATTN_SUB_NORM,
    LLM_TENSOR_FFN_SUB_NORM,
    LLM_TENSOR_DEC_ATTN_NORM,
    LLM_TENSOR_DEC_ATTN_Q,
    LLM_TENSOR_DEC_ATTN_K,
    LLM_TENSOR_DEC_ATTN_V,
    LLM_TENSOR_DEC_ATTN_OUT,
    LLM_TENSOR_DEC_ATTN_REL_B,
    LLM_TENSOR_DEC_CROSS_ATTN_NORM,
    LLM_TENSOR_DEC_CROSS_ATTN_Q,
    LLM_TENSOR_DEC_CROSS_ATTN_K,
    LLM_TENSOR_DEC_CROSS_ATTN_V,
    LLM_TENSOR_DEC_CROSS_ATTN_OUT,
    LLM_TENSOR_DEC_CROSS_ATTN_REL_B,
    LLM_TENSOR_DEC_FFN_NORM,
    LLM_TENSOR_DEC_FFN_GATE,
    LLM_TENSOR_DEC_FFN_DOWN,
    LLM_TENSOR_DEC_FFN_UP,
    LLM_TENSOR_DEC_OUTPUT_NORM,
    LLM_TENSOR_ENC_ATTN_NORM,
    LLM_TENSOR_ENC_ATTN_Q,
    LLM_TENSOR_ENC_ATTN_K,
    LLM_TENSOR_ENC_ATTN_V,
    LLM_TENSOR_ENC_ATTN_OUT,
    LLM_TENSOR_ENC_ATTN_REL_B,
    LLM_TENSOR_ENC_FFN_NORM,
    LLM_TENSOR_ENC_FFN_GATE,
    LLM_TENSOR_ENC_FFN_DOWN,
    LLM_TENSOR_ENC_FFN_UP,
    LLM_TENSOR_ENC_OUTPUT_NORM,
    LLM_TENSOR_CLS,
    LLM_TENSOR_CLS_OUT,
    LLM_TENSOR_CONV1D,
    LLM_TENSOR_CONVNEXT_DW,
    LLM_TENSOR_CONVNEXT_NORM,
    LLM_TENSOR_CONVNEXT_PW1,
    LLM_TENSOR_CONVNEXT_PW2,
    LLM_TENSOR_CONVNEXT_GAMMA,
    LLM_TENSOR_POS_NET_CONV1,
    LLM_TENSOR_POS_NET_CONV2,
    LLM_TENSOR_POS_NET_NORM,
    LLM_TENSOR_POS_NET_NORM1,
    LLM_TENSOR_POS_NET_NORM2,
    LLM_TENSOR_POS_NET_ATTN_NORM,
    LLM_TENSOR_POS_NET_ATTN_Q,
    LLM_TENSOR_POS_NET_ATTN_K,
    LLM_TENSOR_POS_NET_ATTN_V,
    LLM_TENSOR_POS_NET_ATTN_OUT,
};

enum llm_tensor_layer {
    LLM_TENSOR_LAYER_INPUT,
    LLM_TENSOR_LAYER_REPEATING,
    LLM_TENSOR_LAYER_OUTPUT,
};

struct llm_tensor_info {
    llm_tensor_layer layer;
    ggml_op op;
};

static const std::map<llm_tensor, llm_tensor_info> llm_tensor_info_mapping = {
    {LLM_TENSOR_TOKEN_EMBD,                 {LLM_TENSOR_LAYER_INPUT, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_POS_EMBD,                   {LLM_TENSOR_LAYER_INPUT, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_TOKEN_EMBD_NORM,            {LLM_TENSOR_LAYER_INPUT, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_TOKEN_TYPES,                {LLM_TENSOR_LAYER_INPUT, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_OUTPUT,                     {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CLS,                        {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CLS_OUT,                    {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_OUTPUT_NORM,                {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL}},
    {LLM_TENSOR_DEC_OUTPUT_NORM,            {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL}},
    {LLM_TENSOR_ENC_OUTPUT_NORM,            {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL}},
    {LLM_TENSOR_ROPE_FREQS,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ROPE_FACTORS_LONG,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ROPE_FACTORS_SHORT,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ROPE}},
    {LLM_TENSOR_ATTN_Q,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_K,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_QKV,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_OUT,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN_SHEXP,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE_SHEXP,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP_SHEXP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_A,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_B,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_KV_A_MQA,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_KV_B,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_ATTN_Q,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_ATTN_K,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_K,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_V,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_QKV,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_OUT,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_DOWN_SHEXP,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE_SHEXP,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_UP_SHEXP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_A,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_Q_B,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_KV_A_MQA,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ATTN_KV_B,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_ATTN_Q,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_ATTN_K,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_ATTN_V,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_ATTN_OUT,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_CROSS_ATTN_Q,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_CROSS_ATTN_K,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_CROSS_ATTN_V,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_CROSS_ATTN_OUT,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_FFN_GATE,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_FFN_DOWN,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_DEC_FFN_UP,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_ATTN_Q,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_ATTN_K,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_ATTN_V,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_ATTN_OUT,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_FFN_GATE,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_FFN_DOWN,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_ENC_FFN_UP,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE_INP_SHEXP,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_GATE_INP,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_IN,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_X,                      {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_DT,                     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_SSM_OUT,                    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_W1,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_W2,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_DECAY_W1,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_DECAY_W2,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_KEY,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_VALUE,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_RECEPTANCE,        {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_GATE,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_TIME_MIX_OUTPUT,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CHANNEL_MIX_KEY,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,     {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CHANNEL_MIX_VALUE,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_FFN_ACT,                    {LLM_TENSOR_LAYER_REPEATING, GGML_OP_DIV}},
    {LLM_TENSOR_SSM_CONV1D,                 {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_CONV}},
    {LLM_TENSOR_SSM_A,                      {LLM_TENSOR_LAYER_REPEATING, GGML_OP_SSM_SCAN}},
    {LLM_TENSOR_SSM_D,                      {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_TIME_MIX_LERP_X,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_TIME_MIX_LN,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_CHANNEL_MIX_LERP_K,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_CHANNEL_MIX_LERP_R,         {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_TIME_MIX_LERP_W,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_TIME_MIX_LERP_K,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_TIME_MIX_LERP_V,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_TIME_MIX_LERP_R,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_TIME_MIX_LERP_G,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_TIME_MIX_DECAY,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_ADD}},
    {LLM_TENSOR_TIME_MIX_FIRST,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_RWKV_WKV6}},
    {LLM_TENSOR_ATTN_NORM,                  {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_NORM_2,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_OUT_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_POST_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_NORM,                   {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_POST_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_NORM_EXPS,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_Q_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_K_NORM,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_LAYER_OUT_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_Q_A_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_KV_A_NORM,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ATTN_SUB_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_FFN_SUB_NORM,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_DEC_ATTN_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_DEC_CROSS_ATTN_NORM,        {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_DEC_FFN_NORM,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ENC_ATTN_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_ENC_FFN_NORM,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_DEC_ATTN_REL_B,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_ENC_ATTN_REL_B,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_GET_ROWS}},
    {LLM_TENSOR_FFN_DOWN_EXPS,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    {LLM_TENSOR_FFN_GATE_EXPS,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    {LLM_TENSOR_FFN_UP_EXPS,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT_ID}},
    // this tensor is loaded for T5, but never used
    {LLM_TENSOR_DEC_CROSS_ATTN_REL_B,       {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
    {LLM_TENSOR_CONV1D,                     {LLM_TENSOR_LAYER_INPUT,     GGML_OP_IM2COL}},
    {LLM_TENSOR_POS_NET_NORM,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_POS_NET_NORM1,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_POS_NET_NORM2,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_POS_NET_CONV1,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_IM2COL}},
    {LLM_TENSOR_POS_NET_CONV2,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_IM2COL}},
    {LLM_TENSOR_POS_NET_ATTN_NORM,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_POS_NET_ATTN_Q,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_POS_NET_ATTN_K,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_POS_NET_ATTN_V,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_POS_NET_ATTN_OUT,           {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CONVNEXT_DW,                {LLM_TENSOR_LAYER_REPEATING, GGML_OP_IM2COL}},
    {LLM_TENSOR_CONVNEXT_NORM,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    {LLM_TENSOR_CONVNEXT_PW1,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CONVNEXT_PW2,               {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL_MAT}},
    {LLM_TENSOR_CONVNEXT_GAMMA,             {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
};

// find the first buffer type in the list that can use the tensor
static ggml_backend_buffer_type* select_weight_buft(const llama_model& model, ggml_tensor &tensor, ggml_op op, const llama_model::buft_list_t& buft_list) {
    GGML_ASSERT(!buft_list.empty());
    for (const auto& [cur_dev, cur_buft] : buft_list) {
        // TODO
#if 0
        if (weight_buft_supported(model.hparams, tensor, op, cur_buft, cur_dev)) {
            return cur_buft;
        }
#endif
    }
    return nullptr;
}

static const std::map<llm_arch, std::map<llm_tensor, const char*>> LLM_TENSOR_NAMES = {
    {
        LLM_ARCH_LLAMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.{}.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.{}.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.{}.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.{}.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_DECI,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.{}.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.{}.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.{}.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.{}.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_BAICHUAN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_FALCON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.{}.attn_norm_2" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_GROK,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.{}.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.{}.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.{}.ffn_up.%d" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.{}.ffn_up_exps" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.{}.layer_output_norm" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.{}.attn_output_norm" },
        },
    },
    {
        LLM_ARCH_GPT2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
        },
    },
    {
        LLM_ARCH_GPTJ,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
    {
        LLM_ARCH_GPTNEOX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_MPT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output"},
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_ACT,         "blk.{}.ffn.act" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm"},
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm"},
        },
    },
    {
        LLM_ARCH_STARCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
        },
    },
    {
        LLM_ARCH_REFACT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_POS_EMBD,        "position_embd" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.{}.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.{}.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_CLS,             "cls" },
            { LLM_TENSOR_CLS_OUT,         "cls.output" },
        },
    },
    {
        LLM_ARCH_NOMIC_BERT,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.{}.attn_output_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.{}.layer_output_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_JINA_BERT_V2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_TOKEN_TYPES,     "token_types" },
            { LLM_TENSOR_ATTN_NORM_2,     "blk.{}.attn_norm_2" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.{}.attn_output_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_LAYER_OUT_NORM,  "blk.{}.layer_output_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_CLS,             "cls" },
        },
    },
    {
        LLM_ARCH_BLOOM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM, "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
        },
    },
    {
        LLM_ARCH_STABLELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_QWEN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2VL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_QWEN2MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.{}.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.{}.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.{}.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.{}.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.{}.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_PHI2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_PHI3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,           "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,           "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_PLAMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_CODESHELL,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_ORION,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_INTERNLM2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_MINICPM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXP,    "blk.{}.ffn_gate.%d" },
            { LLM_TENSOR_FFN_DOWN_EXP,    "blk.{}.ffn_down.%d" },
            { LLM_TENSOR_FFN_UP_EXP,      "blk.{}.ffn_up.%d" },
        },
    },
    {
        LLM_ARCH_MINICPM3,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ROPE_FACTORS_LONG,  "rope_factors_long" },
            { LLM_TENSOR_ROPE_FACTORS_SHORT, "rope_factors_short" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.{}.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.{}.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,           "blk.{}.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,           "blk.{}.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.{}.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,          "blk.{}.attn_kv_b" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.{}.ffn_down" },
        },
    },
    {
        LLM_ARCH_GEMMA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_GEMMA2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.{}.post_attention_norm" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.{}.post_ffw_norm" },
        },
    },
    {
        LLM_ARCH_STARCODER2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_MAMBA,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_SSM_IN,          "blk.{}.ssm_in" },
            { LLM_TENSOR_SSM_CONV1D,      "blk.{}.ssm_conv1d" },
            { LLM_TENSOR_SSM_X,           "blk.{}.ssm_x" },
            { LLM_TENSOR_SSM_DT,          "blk.{}.ssm_dt" },
            { LLM_TENSOR_SSM_A,           "blk.{}.ssm_a" },
            { LLM_TENSOR_SSM_D,           "blk.{}.ssm_d" },
            { LLM_TENSOR_SSM_OUT,         "blk.{}.ssm_out" },
        },
    },
    {
        LLM_ARCH_XVERSE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_COMMAND_R,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_DBRX,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_OUT_NORM,   "blk.{}.attn_output_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.{}.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_OLMO,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_OLMO2,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_POST_NORM,  "blk.{}.post_attention_norm" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm" },
            { LLM_TENSOR_FFN_POST_NORM,   "blk.{}.post_ffw_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_OLMOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_Q_NORM,        "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,        "blk.{}.attn_k_norm" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.{}.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_OPENELM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_ARCTIC,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_NORM_EXPS,   "blk.{}.ffn_norm_exps" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.{}.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_DEEPSEEK,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ROPE_FREQS,         "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,      "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.{}.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.{}.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.{}.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.{}.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.{}.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_DEEPSEEK2,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_OUTPUT,             "output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q_A_NORM,      "blk.{}.attn_q_a_norm" },
            { LLM_TENSOR_ATTN_KV_A_NORM,     "blk.{}.attn_kv_a_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_Q_A,           "blk.{}.attn_q_a" },
            { LLM_TENSOR_ATTN_Q_B,           "blk.{}.attn_q_b" },
            { LLM_TENSOR_ATTN_KV_A_MQA,      "blk.{}.attn_kv_a_mqa" },
            { LLM_TENSOR_ATTN_KV_B,          "blk.{}.attn_kv_b" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_UP,             "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,           "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_GATE_INP,       "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,      "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,        "blk.{}.ffn_up_exps" },
            { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.{}.ffn_gate_inp_shexp" },
            { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.{}.ffn_gate_shexp" },
            { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.{}.ffn_down_shexp" },
            { LLM_TENSOR_FFN_UP_SHEXP,       "blk.{}.ffn_up_shexp" },
        },
    },
    {
        LLM_ARCH_CHATGLM,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
        },
    },
    {
        LLM_ARCH_BITNET,
        {
            { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
            { LLM_TENSOR_ATTN_Q,             "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,             "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,             "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,           "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_NORM,          "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_SUB_NORM,      "blk.{}.attn_sub_norm" },
            { LLM_TENSOR_FFN_GATE,           "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,           "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,             "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_NORM,           "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_SUB_NORM,       "blk.{}.ffn_sub_norm" },
        },
    },
    {
        LLM_ARCH_T5,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_DEC_OUTPUT_NORM,      "dec.output_norm" },
            { LLM_TENSOR_DEC_ATTN_NORM,        "dec.blk.{}.attn_norm" },
            { LLM_TENSOR_DEC_ATTN_Q,           "dec.blk.{}.attn_q" },
            { LLM_TENSOR_DEC_ATTN_K,           "dec.blk.{}.attn_k" },
            { LLM_TENSOR_DEC_ATTN_V,           "dec.blk.{}.attn_v" },
            { LLM_TENSOR_DEC_ATTN_OUT,         "dec.blk.{}.attn_o" },
            { LLM_TENSOR_DEC_ATTN_REL_B,       "dec.blk.{}.attn_rel_b" },
            { LLM_TENSOR_DEC_CROSS_ATTN_NORM,  "dec.blk.{}.cross_attn_norm" },
            { LLM_TENSOR_DEC_CROSS_ATTN_Q,     "dec.blk.{}.cross_attn_q" },
            { LLM_TENSOR_DEC_CROSS_ATTN_K,     "dec.blk.{}.cross_attn_k" },
            { LLM_TENSOR_DEC_CROSS_ATTN_V,     "dec.blk.{}.cross_attn_v" },
            { LLM_TENSOR_DEC_CROSS_ATTN_OUT,   "dec.blk.{}.cross_attn_o" },
            { LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "dec.blk.{}.cross_attn_rel_b" },
            { LLM_TENSOR_DEC_FFN_NORM,         "dec.blk.{}.ffn_norm" },
            { LLM_TENSOR_DEC_FFN_GATE,         "dec.blk.{}.ffn_gate" },
            { LLM_TENSOR_DEC_FFN_DOWN,         "dec.blk.{}.ffn_down" },
            { LLM_TENSOR_DEC_FFN_UP,           "dec.blk.{}.ffn_up" },
            { LLM_TENSOR_ENC_OUTPUT_NORM,      "enc.output_norm" },
            { LLM_TENSOR_ENC_ATTN_NORM,        "enc.blk.{}.attn_norm" },
            { LLM_TENSOR_ENC_ATTN_Q,           "enc.blk.{}.attn_q" },
            { LLM_TENSOR_ENC_ATTN_K,           "enc.blk.{}.attn_k" },
            { LLM_TENSOR_ENC_ATTN_V,           "enc.blk.{}.attn_v" },
            { LLM_TENSOR_ENC_ATTN_OUT,         "enc.blk.{}.attn_o" },
            { LLM_TENSOR_ENC_ATTN_REL_B,       "enc.blk.{}.attn_rel_b" },
            { LLM_TENSOR_ENC_FFN_NORM,         "enc.blk.{}.ffn_norm" },
            { LLM_TENSOR_ENC_FFN_GATE,         "enc.blk.{}.ffn_gate" },
            { LLM_TENSOR_ENC_FFN_DOWN,         "enc.blk.{}.ffn_down" },
            { LLM_TENSOR_ENC_FFN_UP,           "enc.blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_T5ENCODER,
        {
            { LLM_TENSOR_TOKEN_EMBD,           "token_embd" },
            { LLM_TENSOR_OUTPUT,               "output" },
            { LLM_TENSOR_ENC_OUTPUT_NORM,      "enc.output_norm" },
            { LLM_TENSOR_ENC_ATTN_NORM,        "enc.blk.{}.attn_norm" },
            { LLM_TENSOR_ENC_ATTN_Q,           "enc.blk.{}.attn_q" },
            { LLM_TENSOR_ENC_ATTN_K,           "enc.blk.{}.attn_k" },
            { LLM_TENSOR_ENC_ATTN_V,           "enc.blk.{}.attn_v" },
            { LLM_TENSOR_ENC_ATTN_OUT,         "enc.blk.{}.attn_o" },
            { LLM_TENSOR_ENC_ATTN_REL_B,       "enc.blk.{}.attn_rel_b" },
            { LLM_TENSOR_ENC_FFN_NORM,         "enc.blk.{}.ffn_norm" },
            { LLM_TENSOR_ENC_FFN_GATE,         "enc.blk.{}.ffn_gate" },
            { LLM_TENSOR_ENC_FFN_DOWN,         "enc.blk.{}.ffn_down" },
            { LLM_TENSOR_ENC_FFN_UP,           "enc.blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_JAIS,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_QKV,        "blk.{}.attn_qkv" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
        },
    },
    {
        LLM_ARCH_NEMOTRON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_EXAONE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.{}.attn_rot_embd" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_RWKV6,
        {
            { LLM_TENSOR_TOKEN_EMBD,                "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM,           "token_embd_norm" },
            { LLM_TENSOR_OUTPUT_NORM,               "output_norm" },
            { LLM_TENSOR_OUTPUT,                    "output" },
            { LLM_TENSOR_ATTN_NORM,                 "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_NORM_2,               "blk.{}.attn_norm_2" },
            { LLM_TENSOR_TIME_MIX_W1,               "blk.{}.time_mix_w1" },
            { LLM_TENSOR_TIME_MIX_W2,               "blk.{}.time_mix_w2" },
            { LLM_TENSOR_TIME_MIX_LERP_X,           "blk.{}.time_mix_lerp_x" },
            { LLM_TENSOR_TIME_MIX_LERP_W,           "blk.{}.time_mix_lerp_w" },
            { LLM_TENSOR_TIME_MIX_LERP_K,           "blk.{}.time_mix_lerp_k" },
            { LLM_TENSOR_TIME_MIX_LERP_V,           "blk.{}.time_mix_lerp_v" },
            { LLM_TENSOR_TIME_MIX_LERP_R,           "blk.{}.time_mix_lerp_r" },
            { LLM_TENSOR_TIME_MIX_LERP_G,           "blk.{}.time_mix_lerp_g" },
            { LLM_TENSOR_TIME_MIX_FIRST,            "blk.{}.time_mix_first" },
            { LLM_TENSOR_TIME_MIX_DECAY,            "blk.{}.time_mix_decay" },
            { LLM_TENSOR_TIME_MIX_DECAY_W1,         "blk.{}.time_mix_decay_w1" },
            { LLM_TENSOR_TIME_MIX_DECAY_W2,         "blk.{}.time_mix_decay_w2" },
            { LLM_TENSOR_TIME_MIX_KEY,              "blk.{}.time_mix_key" },
            { LLM_TENSOR_TIME_MIX_VALUE,            "blk.{}.time_mix_value" },
            { LLM_TENSOR_TIME_MIX_RECEPTANCE,       "blk.{}.time_mix_receptance" },
            { LLM_TENSOR_TIME_MIX_GATE,             "blk.{}.time_mix_gate" },
            { LLM_TENSOR_TIME_MIX_LN,               "blk.{}.time_mix_ln" },
            { LLM_TENSOR_TIME_MIX_OUTPUT,           "blk.{}.time_mix_output" },
            { LLM_TENSOR_CHANNEL_MIX_LERP_K,        "blk.{}.channel_mix_lerp_k" },
            { LLM_TENSOR_CHANNEL_MIX_LERP_R,        "blk.{}.channel_mix_lerp_r" },
            { LLM_TENSOR_CHANNEL_MIX_KEY,           "blk.{}.channel_mix_key" },
            { LLM_TENSOR_CHANNEL_MIX_VALUE,         "blk.{}.channel_mix_value" },
            { LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,    "blk.{}.channel_mix_receptance" },
        },
    },
    {
        LLM_ARCH_GRANITE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
        },
    },
    {
        LLM_ARCH_GRANITE_MOE,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE_INP,    "blk.{}.ffn_gate_inp" },
            { LLM_TENSOR_FFN_GATE_EXPS,   "blk.{}.ffn_gate_exps" },
            { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.{}.ffn_down_exps" },
            { LLM_TENSOR_FFN_UP_EXPS,     "blk.{}.ffn_up_exps" },
        },
    },
    {
        LLM_ARCH_CHAMELEON,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
            { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
            { LLM_TENSOR_OUTPUT,          "output" },
            { LLM_TENSOR_ATTN_NORM,       "blk.{}.attn_norm" },
            { LLM_TENSOR_ATTN_Q,          "blk.{}.attn_q" },
            { LLM_TENSOR_ATTN_K,          "blk.{}.attn_k" },
            { LLM_TENSOR_ATTN_V,          "blk.{}.attn_v" },
            { LLM_TENSOR_ATTN_OUT,        "blk.{}.attn_output" },
            { LLM_TENSOR_FFN_NORM,        "blk.{}.ffn_norm" },
            { LLM_TENSOR_FFN_GATE,        "blk.{}.ffn_gate" },
            { LLM_TENSOR_FFN_DOWN,        "blk.{}.ffn_down" },
            { LLM_TENSOR_FFN_UP,          "blk.{}.ffn_up" },
            { LLM_TENSOR_ATTN_Q_NORM,     "blk.{}.attn_q_norm" },
            { LLM_TENSOR_ATTN_K_NORM,     "blk.{}.attn_k_norm" },
        },
    },
    {
        LLM_ARCH_WAVTOKENIZER_DEC,
        {
            { LLM_TENSOR_TOKEN_EMBD,        "token_embd" },
            { LLM_TENSOR_TOKEN_EMBD_NORM,   "token_embd_norm" },
            { LLM_TENSOR_CONV1D,            "conv1d" },
            { LLM_TENSOR_CONVNEXT_DW,       "convnext.{}.dw" },
            { LLM_TENSOR_CONVNEXT_NORM,     "convnext.{}.norm" },
            { LLM_TENSOR_CONVNEXT_PW1,      "convnext.{}.pw1" },
            { LLM_TENSOR_CONVNEXT_PW2,      "convnext.{}.pw2" },
            { LLM_TENSOR_CONVNEXT_GAMMA,    "convnext.{}.gamma" },
            { LLM_TENSOR_OUTPUT_NORM,       "output_norm" },
            { LLM_TENSOR_OUTPUT,            "output" },
            { LLM_TENSOR_POS_NET_CONV1,     "posnet.{}.conv1" },
            { LLM_TENSOR_POS_NET_CONV2,     "posnet.{}.conv2" },
            { LLM_TENSOR_POS_NET_NORM,      "posnet.{}.norm" },
            { LLM_TENSOR_POS_NET_NORM1,     "posnet.{}.norm1" },
            { LLM_TENSOR_POS_NET_NORM2,     "posnet.{}.norm2" },
            { LLM_TENSOR_POS_NET_ATTN_NORM, "posnet.{}.attn_norm" },
            { LLM_TENSOR_POS_NET_ATTN_Q,    "posnet.{}.attn_q" },
            { LLM_TENSOR_POS_NET_ATTN_K,    "posnet.{}.attn_k" },
            { LLM_TENSOR_POS_NET_ATTN_V,    "posnet.{}.attn_v" },
            { LLM_TENSOR_POS_NET_ATTN_OUT,  "posnet.{}.attn_output" },
        },
    },
    {
        LLM_ARCH_UNKNOWN,
        {
            { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        },
    },
};

// helper to handle gguf constants
// usage:
//
//   const auto tn = LLM_TN(LLM_ARCH_LLAMA);
//
//   std::string name = tn(LLM_TENSOR_OUTPUT);                     -> "output"
//   std::string name = tn(LLM_TENSOR_TOKEN_EMBD, "bias");         -> "token_embd.bias"
//   std::string name = tn(LLM_TENSOR_ATTN_NORM, "weight", 3);     -> "blk.3.attn_norm.weight"
//
struct LLM_TN_IMPL {
    const llm_arch arch;
    const llm_tensor tensor;
    const char* const suffix;
    const int bid;
    const int xid;

    std::string str() const {
        if (LLM_TENSOR_NAMES.at(arch).find(tensor) == LLM_TENSOR_NAMES.at(arch).end()) {
            return "__missing__";
        }

        std::string name = std::vformat(LLM_TENSOR_NAMES.at(arch).at(tensor), std::make_format_args(bid, xid));

        if (suffix != nullptr) {
            name += ".";
            name += suffix;
        }

        return name;
    }

    operator std::string() const {
        return str();
    }

    friend bool operator==(const std::string& str, const LLM_TN_IMPL& tn) {
        return str == tn.str();
    }

    friend bool operator!=(const std::string& str, const LLM_TN_IMPL& tn) {
        return str != tn.str();
    }
};

struct LLM_TN {
    LLM_TN(llm_arch arch) : arch(arch) {}

    llm_arch arch;

    LLM_TN_IMPL operator()(llm_tensor tensor, const char* suffix, int bid = -1, int xid = -1) const {
        return { arch, tensor, suffix, bid, xid };
    }

    LLM_TN_IMPL operator()(llm_tensor tensor, int bid = -1, int xid = -1) const {
        return { arch, tensor, nullptr, bid, xid };
    }
};

bool llama_supports_gpu_offload(void) {
#if 0
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
        llama_supports_rpc();
#else
    return false;
#endif
}

// Returns false if cancelled by progress_callback
static bool llm_load_tensors(
    llm_model_loader& ml,
    llama_model& model,
    int n_gpu_layers,
    enum llama_split_mode split_mode,
    int main_gpu,
    const float* tensor_split,
    bool use_mlock,
    llama_progress_callback progress_callback,
    void* progress_callback_user_data) {
    auto& hparams = model.hparams;

    model.split_mode = split_mode;
    model.main_gpu = main_gpu;
    model.n_gpu_layers = n_gpu_layers;

    const int n_layer = hparams.n_layer;

    bool use_mmap_buffer = true;

    // build a list of buffer types for the CPU and GPU devices
    model.cpu_buft_list = make_cpu_buft_list(model);
    for (auto* dev : model.devices) {
        llama_model::buft_list_t buft_list = make_gpu_buft_list(dev, split_mode, tensor_split);
        // add CPU buffer types as a fallback
        buft_list.insert(buft_list.end(), model.cpu_buft_list.begin(), model.cpu_buft_list.end());
        model.gpu_buft_list.emplace(dev, std::move(buft_list));
    }

    // calculate the split points
    int device_count = llama_get_device_count(model);
    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + device_count, [](float x) { return x == 0.0f; });
    std::vector<float> splits(device_count);
    if (all_zero) {
        // default split, by free memory
        for (int i = 0; i < device_count; ++i) {
            ggml_backend_device* dev = model.devices[i];
            size_t total;
            size_t free;
            dev->get_memory(&free, &total);
            splits[i] = free;
        }
    }
    else {
        std::copy(tensor_split, tensor_split + device_count, splits.begin());
    }

    // sum and normalize the splits to get the split points
    float split_sum = 0.0f;
    for (int i = 0; i < device_count; ++i) {
        split_sum += splits[i];
        splits[i] = split_sum;
    }
    for (int i = 0; i < device_count; ++i) {
        splits[i] /= split_sum;
    }

    ggml_backend_device* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const int i_gpu_start = std::max((int)hparams.n_layer - n_gpu_layers, (int)0);
    const int act_gpu_layers = model.devices.empty() ? 0 : std::min(n_gpu_layers, (int)n_layer + 1);
    auto get_layer_buft_list = [&](int il) -> llama_model::layer_dev {
        if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
            return { cpu_dev, &model.cpu_buft_list };
        }
        int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + device_count, float(il - i_gpu_start) / act_gpu_layers) - splits.begin();
        auto* dev = model.devices.at(layer_gpu);
        return { dev, &model.gpu_buft_list.at(dev) };
        };

    // assign the input layer
    // there is very little benefit to offloading the input layer, so always keep it on the CPU
    model.dev_input = { cpu_dev, &model.cpu_buft_list };

    // assign the repeating layers to the devices according to the splits
    model.dev_layer.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        model.dev_layer[il] = get_layer_buft_list(il);
    }
    // assign the output layer
    model.dev_output = get_layer_buft_list(n_layer);

    std::map<ggml_backend_buffer_type*, ggml_context*> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type* buft) -> ggml_context* {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            auto ctx = std::make_unique<ggml_context>();
            if (!ctx) {
                throw std::runtime_error("failed to create ggml context");
            }
            ctx_map[buft] = ctx.get();
            model.ctxs.emplace_back(ctx.get());
            return ctx.release();
        }
        return it->second;
        };

    // create tensors for the weights
    {
        // note: cast to int64_t since we will use these for the tensor dimensions
        const int64_t n_head = hparams.n_head();
        const int64_t n_head_kv = hparams.n_head_kv();
        const int64_t n_embd = hparams.n_embd;
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa();
        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa();
        const int64_t n_embd_head_k = hparams.n_embd_head_k;
        const int64_t n_embd_head_v = hparams.n_embd_head_v;
        const int64_t n_ff = hparams.n_ff();
        const int64_t n_embd_gqa = n_embd_v_gqa;
        const int64_t n_vocab = hparams.n_vocab;
        const int64_t n_vocab_type = hparams.n_vocab_type;
        const int64_t n_rot = hparams.n_rot;
        const int64_t n_expert = hparams.n_expert;
        const int64_t n_expert_used = hparams.n_expert_used;
        const int64_t n_ctx_train = hparams.n_ctx_train;

        if (n_expert > 0 && hparams.n_expert_used == 0) {
            throw std::runtime_error("model has expert layers but no expert layers are used");
        }

        int n_moved_tensors = 0;
        ggml_tensor* first_moved_tensor = nullptr;
        ggml_backend_buffer_type* first_moved_from_buft = nullptr;
        ggml_backend_buffer_type* first_moved_to_buft = nullptr;

        auto create_tensor = [&](const LLM_TN_IMPL& tn, const std::initializer_list<int64_t>& ne, int flags) -> ggml_tensor* {
            ggml_tensor* t_meta = ml.get_tensor_meta(tn.str().c_str());

            if (!t_meta) {
                if (flags & llm_model_loader::TENSOR_NOT_REQUIRED) {
                    return nullptr;
                }
                throw make_format_runtime_error("missing tensor '{}'", tn.str());
            }

            // some models use the token embedding tensor as the output, but since these are used in different layers and with different ops
            // the tensor is duplicated
            // to handle this, we check if the tensor is duplicated, and if so, we assume that it is being loaded as the output tensor
            llm_tensor tn_tensor = tn.tensor;
            if (tn.tensor == LLM_TENSOR_TOKEN_EMBD && flags & llm_model_loader::TENSOR_DUPLICATED) {
                tn_tensor = LLM_TENSOR_OUTPUT;
            }

            auto it = llm_tensor_info_mapping.find(tn_tensor);
            if (it == llm_tensor_info_mapping.end()) {
                throw make_format_runtime_error("missing tensor info mapping for {}", tn.str());
            }
            const auto& info = it->second;

            // tensors with "bias" suffix are always used with GGML_OP_ADD
            ggml_op op;
            bool bias = tn.suffix != nullptr && strcmp(tn.suffix, "bias") == 0;
            if (bias) {
                op = GGML_OP_ADD;
            }
            else {
                op = info.op;
            }

            // sanity checks
            if (info.layer == LLM_TENSOR_LAYER_INPUT || info.layer == LLM_TENSOR_LAYER_OUTPUT) {
                if (tn.bid != -1) {
                    GGML_ABORT("input/output layer tensor %s used with a layer number", tn.str().c_str());
                }
            }
            else {
                if (tn.bid == -1) {
                    GGML_ABORT("repeating layer tensor %s used without a layer number", tn.str().c_str());
                }
            }

            // select the buffer type for this tensor
            llama_model::buft_list_t* buft_list;
            switch (info.layer) {
            case LLM_TENSOR_LAYER_INPUT:
                buft_list = model.dev_input.buft_list;
                break;
            case LLM_TENSOR_LAYER_OUTPUT:
                buft_list = model.dev_output.buft_list;
                break;
            case LLM_TENSOR_LAYER_REPEATING:
                buft_list = model.dev_layer.at(tn.bid).buft_list;
                break;
            default:
                GGML_ABORT("invalid layer %d for tensor %s", info.layer, tn.str().c_str());
            }

            ggml_backend_buffer_type* buft = select_weight_buft(model, *t_meta, op, *buft_list);
            if (!buft) {
                throw make_format_runtime_error("failed to find a compatible buffer type for tensor {}", tn.str());
            }

            // avoid using a host buffer when using mmap
            auto* buft_dev = ggml_backend_buft_get_device(buft);
            if (ml.use_mmap && buft_dev && buft == buft_dev->get_host_buffer_type()) {
                auto* cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                buft = cpu_dev->get_buffer_type();
            }

            if (buft != buft_list->front().second) {
                n_moved_tensors++;
                if (!first_moved_tensor) {
                    first_moved_tensor = t_meta;
                    first_moved_from_buft = buft_list->front().second;
                    first_moved_to_buft = buft;
                }
            }

            ggml_context* ctx = ctx_for_buft(buft);

            // if duplicated, check if the original tensor was allocated in the same buffer type context and avoid creating a new one
            if (flags & llm_model_loader::TENSOR_DUPLICATED) {
                ggml_tensor* t = ctx->find(tn.str());
                if (t) {
                    return t;
                }
            }
            return ml.create_tensor(ctx, tn, ne, flags);
            };

        model.layers.resize(n_layer);

        // TODO: move to a separate function
        const auto tn = LLM_TN(model.arch);
        switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);

            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                // optional bias tensors
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                    layer.rope_long = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                    layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                }
                else {
                    layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                }

                if (n_expert == 0) {
                    layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                    layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                    layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);

                    // optional MLP bias
                    layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
                    layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                    layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
                }
                else {
                    layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
                    layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd,   n_ff, n_expert }, llm_model_loader::TENSOR_NOT_REQUIRED);
                    layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff, n_embd, n_expert }, 0);
                    layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd,   n_ff, n_expert }, 0);
                }
            }
        } break;
        case LLM_ARCH_DECI:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);

            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];
                const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(i);
                const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(i);
                const int64_t n_embd_gqa = hparams.n_embd_v_gqa(i);
                const int64_t n_ff = hparams.n_ff(i);
                const int64_t n_head = hparams.n_head(i);
                const int64_t n_head_kv = hparams.n_head_kv(i);

                if (n_head_kv == 0 && n_head > 0) {
                    // linear attention for DeciLMCausalModel
                    layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                }
                else if (n_head_kv > 0) {
                    layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                    layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                    layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                    layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);
                }

                // optional bias tensors
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
                    layer.rope_long = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                    layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                }
                else {
                    layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                }

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);

                // optional MLP bias
                layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
            }
        } break;
        case LLM_ARCH_MINICPM3:
        {
            const int64_t n_embd_head_qk_rope = hparams.n_rot;
            const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

            const int64_t q_lora_rank = hparams.n_lora_q;
            const int64_t kv_lora_rank = hparams.n_lora_kv;
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);

            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), { q_lora_rank }, 0);

                layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), { kv_lora_rank }, 0);

                layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), { n_embd, q_lora_rank }, 0);
                layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), { q_lora_rank, n_head * n_embd_head_k }, 0);

                layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), { n_embd, kv_lora_rank + (n_embd_head_qk_rope) }, 0);
                layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i), { kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v) }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_head * (n_embd_head_v), n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);

                layer.rope_long = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG, "weight", i), { n_embd_head_qk_rope / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head_qk_rope / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
            }
        } break;
        case LLM_ARCH_GROK:
        {
            if (n_expert == 0) {
                throw std::runtime_error("Grok model cannot have zero experts");
            }

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);

            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff, n_expert }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff, n_embd, n_expert }, 0);
                layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd,   n_ff, n_expert }, 0);

                layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), { n_embd }, 0);
            }
        } break;
        case LLM_ARCH_DBRX:
        {
            if (n_expert == 0) {
                throw std::runtime_error("DBRX model cannot have zero experts");
            }

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff,   n_expert }, 0);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff,   n_embd, n_expert }, 0);
                layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff,   n_expert }, 0);
            }
        } break;
        case LLM_ARCH_BAICHUAN:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
            {
                model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_FALCON:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            {
                model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);

                model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
                if (!model.output) {
                    model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED); // needs to be on GPU
                }
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_STARCODER:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
            model.pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), { n_embd, n_ctx_train }, 0);

            // output
            {
                model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
                model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
                if (!model.output) {
                    // needs to be on GPU
                    model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
                }

            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_BERT:
        case LLM_ARCH_NOMIC_BERT:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
            model.type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), { n_embd, n_vocab_type }, 0);

            if (model.arch == LLM_ARCH_BERT) {
                model.pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), { n_embd, n_ctx_train }, 0);

                model.cls = create_tensor(tn(LLM_TENSOR_CLS, "weight"), { n_embd, n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                model.cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                model.cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
                model.cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
            }

            model.tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), { n_embd }, 0);
            model.tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), { n_embd }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                if (model.arch == LLM_ARCH_BERT) {
                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                    layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, 0);

                    layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                    layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, 0);

                    layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                    layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, 0);
                }
                else {
                    layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                }

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), { n_embd }, 0);
                layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);

                if (model.arch == LLM_ARCH_BERT) {
                    layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);
                    layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
                    layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);
                }
                else {
                    layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                }

                layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), { n_embd }, 0);
                layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i), { n_embd }, 0);
            }
        } break;
        case LLM_ARCH_JINA_BERT_V2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0); // word_embeddings
            model.type_embd = create_tensor(tn(LLM_TENSOR_TOKEN_TYPES, "weight"), { n_embd, n_vocab_type }, 0); // token_type_embeddings

            model.tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), { n_embd }, 0); // LayerNorm
            model.tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), { n_embd }, 0); //LayerNorm bias

            model.cls = create_tensor(tn(LLM_TENSOR_CLS, "weight"), { n_embd, 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
            model.cls_b = create_tensor(tn(LLM_TENSOR_CLS, "bias"), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i]; // JinaBertLayer

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, 0);

                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, 0);

                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0); //output_dens
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0); //output_dens

                layer.attn_out_norm = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), { n_embd }, 0); //output_norm
                layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias", i), { n_embd }, 0);

                layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.layer_out_norm = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), { n_embd }, 0);
                layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias", i), { n_embd }, 0);
            }
        } break;
        case LLM_ARCH_BLOOM:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
            model.tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), { n_embd }, 0);
            model.tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), { n_embd }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_MPT:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
            model.pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), { n_embd, n_ctx_train }, llm_model_loader::TENSOR_NOT_REQUIRED);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            if (!model.output) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED); // needs to be on GPU
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                // AWQ ScaleActivation layer
                layer.ffn_act = create_tensor(tn(LLM_TENSOR_FFN_ACT, "scales", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
            }
        } break;
        case LLM_ARCH_STABLELM:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                // optional bias tensors, present in Stable LM 2 1.6B
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);

                // optional q and k layernorms, present in StableLM 2 12B
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k, n_head }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k, n_head_kv }, llm_model_loader::TENSOR_NOT_REQUIRED);

                // optional FFN norm, not present in StableLM 2 12B which uses parallel residual
                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_QWEN:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd * 3 }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd * 3 }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff / 2 }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff / 2, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff / 2 }, 0);
            }
        } break;
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN2VL:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                // optional bias tensors
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, 0);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, 0);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_QWEN2MOE:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                // optional bias tensors
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, 0);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, 0);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);

                if (n_expert == 0) {
                    throw std::runtime_error("n_expert must be > 0 for QWEN2MOE");
                }
                if (n_expert_used == 0) {
                    throw std::runtime_error("n_expert_used must be > 0 for QWEN2MOE");
                }

                // MoE branch
                const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp,   n_embd, n_expert }, 0);
                layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);

                // Shared expert branch
                const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

                layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", i), { n_embd }, 0);
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_shexp }, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_shexp,     n_embd }, 0);
                layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_shexp }, 0);
            }
        } break;
        case LLM_ARCH_PHI2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);
            model.output_b = create_tensor(tn(LLM_TENSOR_OUTPUT, "bias"), { n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);

                if (layer.wqkv == nullptr) {
                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                    layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, 0);

                    layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                    layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, 0);

                    layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                    layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, 0);
                }

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_PHI3:
        {
            const int64_t n_embd_head = n_embd / n_head;

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, 2 * n_ff }, 0);

                layer.rope_long = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG, "weight", i), { n_embd_head / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), { n_embd_head / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
            }
        } break;
        case LLM_ARCH_PLAMO:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_GPT2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
            model.pos_embd = create_tensor(tn(LLM_TENSOR_POS_EMBD, "weight"), { n_embd, n_ctx_train }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_CODESHELL:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_ORION:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_INTERNLM2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                // layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), {n_embd, n_embd + 2*n_embd_gqa}, 0);
                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_GEMMA:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
            }
        } break;
        case LLM_ARCH_GEMMA2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED); // same as tok_embd, duplicated to allow offloading

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);
                layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
            }
        } break;
        case LLM_ARCH_STARCODER2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);

            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                // optional bias tensors
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, 0);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, 0);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);

                // optional bias tensors
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_MAMBA:
        {
            const int64_t d_conv = hparams.ssm_d_conv;
            const int64_t d_inner = hparams.ssm_d_inner;
            const int64_t d_state = hparams.ssm_d_state;
            const int64_t dt_rank = hparams.ssm_dt_rank;

            // only an expansion factor of 2 is supported for now
            if (2 * n_embd != d_inner) {
                throw std::runtime_error("only an expansion factor of 2 is supported for now");
            }

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);

            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed, duplicated to allow offloading
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                // norm
                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.ssm_in = create_tensor(tn(LLM_TENSOR_SSM_IN, "weight", i), { n_embd, 2 * d_inner }, 0);

                layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), { d_conv, d_inner }, 0);
                layer.ssm_conv1d_b = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "bias", i), { d_inner }, 0);

                layer.ssm_x = create_tensor(tn(LLM_TENSOR_SSM_X, "weight", i), { d_inner, dt_rank + 2 * d_state }, 0);

                layer.ssm_dt = create_tensor(tn(LLM_TENSOR_SSM_DT, "weight", i), { dt_rank, d_inner }, 0);
                layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), { d_inner }, 0);

                // no "weight" suffix for these
                layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), { d_state, d_inner }, 0);
                layer.ssm_d = create_tensor(tn(LLM_TENSOR_SSM_D, i), { d_inner }, 0);

                // out_proj
                layer.ssm_out = create_tensor(tn(LLM_TENSOR_SSM_OUT, "weight", i), { d_inner, n_embd }, 0);
            }
        } break;
        case LLM_ARCH_XVERSE:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_COMMAND_R:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            // init output from the input tok embed
            model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                if (n_layer >= 64) {
                    layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k, n_head }, 0);
                    layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k, n_head_kv }, 0);
                }

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_OLMO:  // adapted from LLM_ARCH_LLAMA with norm params removed
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_OLMO2:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd }, 0);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd }, 0);
                layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), { n_embd }, 0);
            }
        } break;
        case LLM_ARCH_OLMOE:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd }, 0);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);

                if (n_expert == 0) {
                    throw std::runtime_error("n_expert must be > 0");
                }
                if (n_expert_used == 0) {
                    throw std::runtime_error("n_expert_used must be > 0");
                }

                // MoE branch
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff,   n_expert }, 0);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff,   n_embd, n_expert }, 0);
                layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff,   n_expert }, 0);
            }
        } break;
        case LLM_ARCH_OPENELM:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            // init output from the input tok embed
            model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);

            for (int i = 0; i < n_layer; ++i) {
                const int64_t n_head = hparams.n_head(i);
                const int64_t n_head_qkv = 2 * hparams.n_head_kv(i) + n_head;
                const int64_t n_ff = hparams.n_ff(i);

                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_head_qkv * n_embd_head_k }, 0);
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_head * n_embd_head_k, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
            }
        } break;
        case LLM_ARCH_GPTNEOX:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_ARCTIC:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);

            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_embd }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_embd, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);
                layer.ffn_norm_exps = create_tensor(tn(LLM_TENSOR_FFN_NORM_EXPS, "weight", i), { n_embd }, 0);
                layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd,   n_ff, n_expert }, false);
                layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff, n_embd, n_expert }, 0);
                layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd,   n_ff, n_expert }, 0);
            }
        } break;
        case LLM_ARCH_DEEPSEEK:
        {

            const int64_t n_ff_exp = hparams.n_ff_exp;
            const int64_t n_expert_shared = hparams.n_expert_shared;

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                if (i < (int)hparams.n_layer_dense_lead) {
                    layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                    layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                    layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
                }
                else {
                    layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);

                    if (n_expert == 0) {
                        throw std::runtime_error("n_expert must be > 0");
                    }
                    if (n_expert_used == 0) {
                        throw std::runtime_error("n_expert_used must be > 0");
                    }

                    // MoE branch
                    layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
                    layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp,   n_embd, n_expert }, 0);
                    layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);

                    // Shared expert branch
                    layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_exp * n_expert_shared }, 0);
                    layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_exp * n_expert_shared, n_embd }, 0);
                    layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_exp * n_expert_shared }, 0);
                }
            }
        } break;
        case LLM_ARCH_DEEPSEEK2:
        {
            const bool is_lite = (hparams.n_layer == 27);

            const int64_t n_embd_head_qk_rope = hparams.n_rot;
            const int64_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;

            const int64_t q_lora_rank = hparams.n_lora_q;
            const int64_t kv_lora_rank = hparams.n_lora_kv;

            const int64_t n_ff_exp = hparams.n_ff_exp;
            const int64_t n_expert_shared = hparams.n_expert_shared;

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                if (!is_lite) {
                    layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), { q_lora_rank }, 0);
                }

                layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), { kv_lora_rank }, 0);

                if (!is_lite) {
                    layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), { n_embd, q_lora_rank }, 0);
                    layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), { q_lora_rank, n_head * n_embd_head_k }, 0);
                }
                else {
                    layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                }

                layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), { n_embd, kv_lora_rank + (n_embd_head_qk_rope) }, 0);
                layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i), { kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v) }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_head * (n_embd_head_v), n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                if (i < (int)hparams.n_layer_dense_lead) {
                    layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                    layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                    layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
                }
                else {
                    layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), { n_embd, n_expert }, 0);

                    if (n_expert == 0) {
                        throw std::runtime_error("n_expert must be > 0");
                    }
                    if (n_expert_used == 0) {
                        throw std::runtime_error("n_expert_used must be > 0");
                    }

                    // MoE branch
                    layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);
                    layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), { n_ff_exp,   n_embd, n_expert }, 0);
                    layer.ffn_up_exps = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS, "weight", i), { n_embd, n_ff_exp, n_expert }, 0);

                    // Shared expert branch
                    layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), { n_embd, n_ff_exp * n_expert_shared }, 0);
                    layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), { n_ff_exp * n_expert_shared, n_embd }, 0);
                    layer.ffn_up_shexp = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP, "weight", i), { n_embd, n_ff_exp * n_expert_shared }, 0);
                }
            }
        } break;
        case LLM_ARCH_BITNET:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_sub_norm = create_tensor(tn(LLM_TENSOR_ATTN_SUB_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wq_scale = create_tensor(tn(LLM_TENSOR_ATTN_Q, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wk_scale = create_tensor(tn(LLM_TENSOR_ATTN_K, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv_scale = create_tensor(tn(LLM_TENSOR_ATTN_V, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.wo_scale = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_sub_norm = create_tensor(tn(LLM_TENSOR_FFN_SUB_NORM, "weight", i), { n_ff }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_gate_scale = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_scale = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_scale = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), { 1 }, llm_model_loader::TENSOR_NOT_REQUIRED);
            }
        } break;
        case LLM_ARCH_T5:
        {
            const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), { n_embd }, 0);

            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), { n_head, n_rel_attn_bkts }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), { n_embd_v_gqa, n_embd }, 0);

                layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), { n_embd,   n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_rel_b = create_tensor(tn(LLM_TENSOR_DEC_ATTN_REL_B, "weight", i), { n_head, n_rel_attn_bkts }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), { n_embd_v_gqa, n_embd }, 0);

                layer.attn_norm_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM, "weight", i), { n_embd }, 0);
                // this tensor seems to be unused in HF transformers implementation
                layer.attn_rel_b_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_REL_B, "weight", i), { n_head, n_rel_attn_bkts }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), { n_embd_v_gqa, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE, "weight", i), { n_embd,   n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_T5ENCODER:
        {
            const auto n_rel_attn_bkts = hparams.n_rel_attn_bkts;

            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_rel_b_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_B, "weight", i), { n_head, n_rel_attn_bkts }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wq_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wk_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo_enc = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), { n_embd_v_gqa, n_embd }, 0);

                layer.ffn_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_gate_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE, "weight", i), { n_embd,   n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_down_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up_enc = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_JAIS:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), { n_ff }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, 0);
            }
        } break;
        case LLM_ARCH_CHATGLM:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "weight", i), { n_embd, n_embd + 2 * n_embd_gqa }, 0);
                layer.bqkv = create_tensor(tn(LLM_TENSOR_ATTN_QKV, "bias", i), { n_embd + 2 * n_embd_gqa }, 0);

                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff * 2 }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
            }
        } break;
        case LLM_ARCH_NEMOTRON:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                // optional bias tensors
                layer.bq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bk = create_tensor(tn(LLM_TENSOR_ATTN_K, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bv = create_tensor(tn(LLM_TENSOR_ATTN_V, "bias", i), { n_embd_gqa }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.bo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.ffn_norm_b = create_tensor(tn(LLM_TENSOR_FFN_NORM, "bias", i), { n_embd }, 0);

                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);

                // optional MLP bias
                layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), { n_embd }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP, "bias", i), { n_ff }, llm_model_loader::TENSOR_NOT_REQUIRED);
            }
        } break;
        case LLM_ARCH_EXAONE:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_k_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_v_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
                layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), { n_rot / 2 }, llm_model_loader::TENSOR_NOT_REQUIRED | (i != 0 ? llm_model_loader::TENSOR_DUPLICATED : 0));
                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_RWKV6:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // Block 0, LN0
            model.tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), { n_embd }, 0);
            model.tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), { n_embd }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, 0);

            const int time_mix_extra_dim = hparams.time_mix_extra_dim;
            const int time_decay_extra_dim = hparams.time_decay_extra_dim;
            const int head_size = hparams.wkv_head_size;
            const int attn_hidden_size = n_embd;
            const int ffn_size = hparams.n_ff_arr[0];

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "bias", i), { n_embd }, 0);

                layer.attn_norm_2 = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "weight", i), { n_embd }, 0);
                layer.attn_norm_2_b = create_tensor(tn(LLM_TENSOR_ATTN_NORM_2, "bias", i), { n_embd }, 0);

                layer.time_mix_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W1, "weight", i), { n_embd, time_mix_extra_dim * 5 }, 0);
                layer.time_mix_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_W2, "weight", i), { time_mix_extra_dim, n_embd, 5 }, 0);

                layer.time_mix_lerp_x = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_X, "weight", i), { n_embd, 1, 1 }, 0);
                layer.time_mix_lerp_w = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_W, "weight", i), { n_embd, 1, 1 }, 0);
                layer.time_mix_lerp_k = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_K, "weight", i), { n_embd, 1, 1 }, 0);
                layer.time_mix_lerp_v = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_V, "weight", i), { n_embd, 1, 1 }, 0);
                layer.time_mix_lerp_r = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_R, "weight", i), { n_embd, 1, 1 }, 0);
                layer.time_mix_lerp_g = create_tensor(tn(LLM_TENSOR_TIME_MIX_LERP_G, "weight", i), { n_embd, 1, 1 }, 0);

                layer.time_mix_first = create_tensor(tn(LLM_TENSOR_TIME_MIX_FIRST, "weight", i), { head_size, n_embd / head_size }, 0);
                layer.time_mix_decay = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY, "weight", i), { n_embd }, 0);
                layer.time_mix_decay_w1 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W1, "weight", i), { n_embd, time_decay_extra_dim }, 0);
                layer.time_mix_decay_w2 = create_tensor(tn(LLM_TENSOR_TIME_MIX_DECAY_W2, "weight", i), { time_decay_extra_dim, attn_hidden_size }, 0);
                layer.time_mix_key = create_tensor(tn(LLM_TENSOR_TIME_MIX_KEY, "weight", i), { attn_hidden_size, n_embd }, 0);
                layer.time_mix_value = create_tensor(tn(LLM_TENSOR_TIME_MIX_VALUE, "weight", i), { attn_hidden_size, n_embd }, 0);
                layer.time_mix_receptance = create_tensor(tn(LLM_TENSOR_TIME_MIX_RECEPTANCE, "weight", i), { attn_hidden_size, n_embd }, 0);
                layer.time_mix_gate = create_tensor(tn(LLM_TENSOR_TIME_MIX_GATE, "weight", i), { attn_hidden_size, n_embd }, 0);

                layer.time_mix_ln = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "weight", i), { n_embd }, 0);
                layer.time_mix_ln_b = create_tensor(tn(LLM_TENSOR_TIME_MIX_LN, "bias", i), { n_embd }, 0);
                layer.time_mix_output = create_tensor(tn(LLM_TENSOR_TIME_MIX_OUTPUT, "weight", i), { n_embd, attn_hidden_size }, 0);

                layer.channel_mix_lerp_k = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_K, "weight", i), { n_embd, 1, 1 }, 0);
                layer.channel_mix_lerp_r = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_LERP_R, "weight", i), { n_embd, 1, 1 }, 0);

                layer.channel_mix_key = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_KEY, "weight", i), { n_embd, ffn_size }, 0);
                layer.channel_mix_value = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_VALUE, "weight", i), { ffn_size, n_embd }, 0);
                layer.channel_mix_receptance = create_tensor(tn(LLM_TENSOR_CHANNEL_MIX_RECEPTANCE, "weight", i), { n_embd, n_embd }, 0);
            }

        } break;
        case LLM_ARCH_CHAMELEON:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

            // output
            model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_NOT_REQUIRED);
            // if output is NULL, init from the input tok embed
            if (model.output == NULL) {
                model.output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, llm_model_loader::TENSOR_DUPLICATED);
            }

            for (int i = 0; i < n_layer; ++i) {
                auto& layer = model.layers[i];

                layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
                layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k, n_head }, 0);
                layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k, n_head_kv }, 0);
                layer.attn_q_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "bias", i), { n_embd_head_k, n_head }, llm_model_loader::TENSOR_NOT_REQUIRED);
                layer.attn_k_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "bias", i), { n_embd_head_k, n_head_kv }, llm_model_loader::TENSOR_NOT_REQUIRED);

                layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), { n_embd, n_embd }, 0);
                layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), { n_embd, n_embd_gqa }, 0);
                layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd, n_embd }, 0);

                layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);

                layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd,   n_ff }, 0);
                layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
                layer.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd,   n_ff }, 0);
            }
        } break;
        case LLM_ARCH_WAVTOKENIZER_DEC:
        {
            model.tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { hparams.n_embd_features, n_vocab }, 0);

            model.conv1d = create_tensor(tn(LLM_TENSOR_CONV1D, "weight"), { 7, hparams.n_embd_features, hparams.posnet.n_embd }, 0);
            model.conv1d_b = create_tensor(tn(LLM_TENSOR_CONV1D, "bias"), { 1, hparams.posnet.n_embd }, 0);

            // posnet
            {
                const int64_t n_embd = hparams.posnet.n_embd;

                for (uint32_t i = 0; i < hparams.posnet.n_layer; ++i) {
                    auto& layer = model.layers[i].posnet;

                    // posnet:
                    //
                    //  - resnet
                    //  - resnet
                    //  - attn
                    //  - resnet
                    //  - resnet
                    //  - norm
                    //
                    switch (i) {
                    case 0:
                    case 1:
                    case 3:
                    case 4:
                    {
                        layer.norm1 = create_tensor(tn(LLM_TENSOR_POS_NET_NORM1, "weight", i), { 1, n_embd }, 0);
                        layer.norm1_b = create_tensor(tn(LLM_TENSOR_POS_NET_NORM1, "bias", i), { 1, n_embd }, 0);

                        layer.conv1 = create_tensor(tn(LLM_TENSOR_POS_NET_CONV1, "weight", i), { 3, n_embd, n_embd }, 0);
                        layer.conv1_b = create_tensor(tn(LLM_TENSOR_POS_NET_CONV1, "bias", i), { 1, n_embd }, 0);

                        layer.norm2 = create_tensor(tn(LLM_TENSOR_POS_NET_NORM2, "weight", i), { 1, n_embd }, 0);
                        layer.norm2_b = create_tensor(tn(LLM_TENSOR_POS_NET_NORM2, "bias", i), { 1, n_embd }, 0);

                        layer.conv2 = create_tensor(tn(LLM_TENSOR_POS_NET_CONV2, "weight", i), { 3, n_embd, n_embd }, 0);
                        layer.conv2_b = create_tensor(tn(LLM_TENSOR_POS_NET_CONV2, "bias", i), { 1, n_embd }, 0);
                    } break;
                    case 2:
                    {
                        layer.attn_norm = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "weight", i), { 1, n_embd }, 0);
                        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "bias", i), { 1, n_embd }, 0);

                        layer.attn_q = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_Q, "weight", i), { 1, n_embd, n_embd }, 0);
                        layer.attn_q_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_Q, "bias", i), { 1, n_embd }, 0);

                        layer.attn_k = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_K, "weight", i), { 1, n_embd, n_embd }, 0);
                        layer.attn_k_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_K, "bias", i), { 1, n_embd }, 0);

                        layer.attn_v = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_V, "weight", i), { 1, n_embd, n_embd }, 0);
                        layer.attn_v_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_V, "bias", i), { 1, n_embd }, 0);

                        layer.attn_o = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_OUT, "weight", i), { 1, n_embd, n_embd }, 0);
                        layer.attn_o_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_OUT, "bias", i), { 1, n_embd }, 0);
                    } break;
                    case 5:
                    {
                        layer.norm = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "weight", i), { 1, n_embd }, 0);
                        layer.norm_b = create_tensor(tn(LLM_TENSOR_POS_NET_ATTN_NORM, "bias", i), { 1, n_embd }, 0);
                    } break;
                    default: GGML_ABORT("unknown posnet layer");
                    };
                }
            }

            GGML_ASSERT(hparams.posnet.n_embd == hparams.convnext.n_embd);

            model.tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight"), { hparams.posnet.n_embd }, 0);
            model.tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias"), { hparams.posnet.n_embd }, 0);

            // convnext
            {
                const int64_t n_embd = hparams.convnext.n_embd;

                for (uint32_t i = 0; i < hparams.convnext.n_layer; ++i) {
                    auto& layer = model.layers[i].convnext;

                    layer.dw = create_tensor(tn(LLM_TENSOR_CONVNEXT_DW, "weight", i), { 7, 1, n_embd }, 0);
                    layer.dw_b = create_tensor(tn(LLM_TENSOR_CONVNEXT_DW, "bias", i), { 1, n_embd }, 0);

                    layer.norm = create_tensor(tn(LLM_TENSOR_CONVNEXT_NORM, "weight", i), { n_embd }, 0);
                    layer.norm_b = create_tensor(tn(LLM_TENSOR_CONVNEXT_NORM, "bias", i), { n_embd }, 0);

                    layer.pw1 = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW1, "weight", i), { n_embd, n_ff }, 0);
                    layer.pw1_b = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW1, "bias", i), { n_ff }, 0);

                    layer.pw2 = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW2, "weight", i), { n_ff, n_embd }, 0);
                    layer.pw2_b = create_tensor(tn(LLM_TENSOR_CONVNEXT_PW2, "bias", i), { n_embd }, 0);

                    layer.gamma = create_tensor(tn(LLM_TENSOR_CONVNEXT_GAMMA, "weight", i), { n_embd }, 0);
                }

                // output
                model.output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
                model.output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"), { n_embd }, 0);
            }

            model.output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { hparams.convnext.n_embd, n_embd }, 0);
            model.output_b = create_tensor(tn(LLM_TENSOR_OUTPUT, "bias"), { n_embd }, 0);
        } break;
        default:
            throw std::runtime_error("unknown architecture");
        }

        if (n_moved_tensors > 0) {
            LLAMA_LOG_DEBUG("%s: tensor '%s' (%s) (and %d others) cannot be used with preferred buffer type %s, using %s instead\n",
                __func__, first_moved_tensor->name, ggml_type_name(first_moved_tensor->type), n_moved_tensors - 1,
                ggml_backend_buft_name(first_moved_from_buft), ggml_backend_buft_name(first_moved_to_buft));
        }
    }

    ml.done_getting_tensors();

    ml.init_mappings(true, use_mlock ? &model.mlock_mmaps : nullptr);
    model.mappings.reserve(ml.mappings.size());

    // create the backend buffers
    std::vector<std::pair<ggml_context*, llama_buf_map>> ctx_bufs;
    ctx_bufs.reserve(ctx_map.size());

    // Ensure we have enough capacity for the maximum backend buffer we will potentially create
    const size_t n_max_backend_buffer = ctx_map.size() * ml.files.size();
    model.bufs.reserve(n_max_backend_buffer);

    for (auto& it : ctx_map) {
        ggml_backend_buffer_type* buft = it.first;
        ggml_context* ctx = it.second;

        // skip contexts without tensors
        if (ctx->getTensors().empty()) {
            continue;
        }

        llama_buf_map bufs;
        bufs.reserve(n_max_backend_buffer);

        // check if it is possible to use buffer_from_host_ptr with this buffer type
        ggml_backend_device* dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            // FIXME: workaround for CPU backend buft having a NULL device
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
        ggml_backend_dev_props props;
        dev->get_props(&props);
        bool buffer_from_host_ptr_supported = props.caps.buffer_from_host_ptr;
        bool is_default_buft = buft == dev->get_buffer_type();

        if (ml.use_mmap && use_mmap_buffer && buffer_from_host_ptr_supported && is_default_buft) {
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                // only the mmap region containing the tensors in the model is mapped to the backend buffer
                // this is important for metal with apple silicon: if the entire model could be mapped to a metal buffer, then we could just use metal for all layers
                // this allows using partial offloading when the model size exceeds the metal buffer size, but not the RAM size
                void* addr = nullptr;
                size_t first, last; // NOLINT
                ml.get_mapping_range(&first, &last, &addr, idx, ctx);
                if (first >= last) {
                    continue;
                }
                const size_t max_size = ggml_get_max_tensor_size(ctx);
                ggml_backend_buffer* buf = dev->buffer_from_host_ptr((char*)addr + first, last - first, max_size);
                if (buf == nullptr) {
                    throw make_format_runtime_error("unable to allocate {} buffer", buft->get_name());
                }
                model.bufs.emplace_back(buf);
                bufs.emplace(idx, buf);
            }
        }
        else {
            std::unique_ptr<ggml_backend_buffer> buf = buft->alloc_tensors(ctx);
            if (buf == nullptr) {
                throw make_format_runtime_error("unable to allocate {} buffer", buft->get_name());
            }
            model.bufs.emplace_back(buf.get());
            if (use_mlock && buf->is_host()) {
                model.mlock_bufs.emplace_back(new llama_mlock);
                auto& mlock_buf = model.mlock_bufs.back();
                mlock_buf->init(buf->get_base());
                mlock_buf->grow_to(buf->get_size());
            }
            for (uint32_t idx = 0; idx < ml.files.size(); idx++) {
                bufs.emplace(idx, buf.get());
            }
        }

        if (bufs.empty()) {
            throw std::runtime_error("failed to allocate buffer");
        }

        for (auto& buf : bufs) {
            // indicate that this buffer contains weights
            // this is used by ggml_backend_sched to improve op scheduling: ops that use a weight are preferably scheduled to the backend that contains the weight
            buf.second->setUsage(GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        ctx_bufs.emplace_back(ctx, bufs);
    }

    if (llama_supports_gpu_offload()) {
        const int n_gpu = std::min(n_gpu_layers, int(hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int)hparams.n_layer) {
            LLAMA_LOG_INFO("%s: offloading output layer to GPU\n", __func__);
        }

        const int max_backend_supported_layers = hparams.n_layer + 1;
        const int max_offloadable_layers = hparams.n_layer + 1;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers), max_backend_supported_layers);
    }

    // print memory requirements per buffer type
    for (auto& buf : model.bufs) {
        LLAMA_LOG_INFO("%s: %12s model buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get()) / 1024.0 / 1024.0);
    }

    // populate tensors_by_name
    for (auto& ctx : model.ctxs) {
        for (auto cur : ctx->getTensors()) {
            model.tensors_by_name.emplace_back(cur->get_name(), cur);
        }
    }

    // load tensor data
    for (auto& it : ctx_bufs) {
        ggml_context* ctx = it.first;
        auto& bufs = it.second;
        if (!ml.load_all_data(ctx, bufs, use_mlock ? &model.mlock_mmaps : NULL, progress_callback, progress_callback_user_data)) {
            return false;
        }
    }

    if (use_mmap_buffer) {
        for (auto& mapping : ml.mappings) {
            model.mappings.emplace_back(std::move(mapping));
        }
    }

    return true;
}

int llama_model_load(const std::filesystem::path& path, const llama_model_params &params, llama_model &model)
{
    // TODO
    // model.t_start_us = ggml_time_us();

    try {
        llm_model_loader ml(path, params.use_mmap, params.check_tensors, params.kv_overrides);
        model.hparams.vocab_only = params.vocab_only;

        try {
            llm_load_arch(ml, model);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            llm_load_hparams(ml, model);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            llm_load_vocab(ml, model);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        llm_load_stats(ml, model);
        llm_load_print_meta(ml, model);

        if (model.vocab.type != LLAMA_VOCAB_TYPE_NONE &&
            model.hparams.n_vocab != model.vocab.id_to_token.size()) {
            throw std::runtime_error("vocab size mismatch");
        }

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!llm_load_tensors(
            ml, model, params.n_gpu_layers, params.split_mode, params.main_gpu, params.tensor_split, params.use_mlock,
            params.progress_callback, params.progress_callback_user_data
        )) {
            return -2;
        }
    }
    catch (const std::exception& err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

#if 0
    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model->t_load_us = ggml_time_us() - model.t_start_us;
#endif
    return 0;
}

llama_model_params common_model_params_to_llama(common_params& params) {
    llama_model_params mparams{};

    if (!params.devices.empty()) {
        mparams.devices = params.devices.data();
    }
    if (params.n_gpu_layers != -1) {
        mparams.n_gpu_layers = params.n_gpu_layers;
    }
    mparams.rpc_servers = params.rpc_servers.c_str();
    mparams.main_gpu = params.main_gpu;
    mparams.split_mode = params.split_mode;
    mparams.tensor_split = params.tensor_split;
    mparams.use_mmap = params.use_mmap;
    mparams.use_mlock = params.use_mlock;
    mparams.check_tensors = params.check_tensors;
    if (params.kv_overrides.empty()) {
        mparams.kv_overrides = NULL;
    }
    else {
        GGML_ASSERT(params.kv_overrides.back().key[0] == 0 && "KV overrides not terminated with empty key");
        mparams.kv_overrides = params.kv_overrides.data();
    }

    return mparams;
}

llama_model* llama_load_model_from_file(
    std::filesystem::path path_model,
    struct llama_model_params   params) {

    auto model = std::make_unique<llama_model>();

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void* ctx) {
            unsigned* cur_percentage_p = (unsigned*)ctx;
            unsigned percentage = (unsigned)(100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    if (params.rpc_servers != nullptr && params.rpc_servers[0] != '\0') {
        // split the servers set them into model->rpc_servers
        std::string servers(params.rpc_servers);
        size_t pos = 0;
        while ((pos = servers.find(',')) != std::string::npos) {
            std::string server = servers.substr(0, pos);
            model->rpc_servers.push_back(server);
            servers.erase(0, pos + 1);
        }
        model->rpc_servers.push_back(servers);
    }

    // TODO
#if 0
    // add RPC devices
    if (!model->rpc_servers.empty()) {
        ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
        if (!rpc_reg) {
            LLAMA_LOG_ERROR("%s: failed to find RPC backend\n", __func__);
            llama_free_model(model);
            return nullptr;
        }

        typedef ggml_backend_device*(*ggml_backend_rpc_add_device_t)(const char* endpoint);
        ggml_backend_rpc_add_device_t ggml_backend_rpc_add_device_fn = (ggml_backend_rpc_add_device_t)ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_device");
        if (!ggml_backend_rpc_add_device_fn) {
            LLAMA_LOG_ERROR("%s: failed to find RPC device add function\n", __func__);
            llama_free_model(model);
            return nullptr;
        }

        for (const std::string& server : model->rpc_servers) {
            ggml_backend_device* dev = ggml_backend_rpc_add_device_fn(server.c_str());
            if (dev) {
                model->devices.push_back(dev);
            }
            else {
                LLAMA_LOG_ERROR("%s: failed to add RPC device for server '%s'\n", __func__, server.c_str());
                llama_free_model(model);
                return nullptr;
            }
        }
    }
#endif

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_device** dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    }
    else {
        // use all available devices
        for (auto dev : backend_devs()) {
            switch (dev->get_type()) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
            case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                // skip CPU backends since they are handled separately
                break;

            case GGML_BACKEND_DEVICE_TYPE_GPU:
                model->devices.push_back(dev);
                break;
            }
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0 || params.main_gpu >= (int)model->devices.size()) {
            LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %d)\n", __func__, params.main_gpu, (int)model->devices.size());
            return nullptr;
        }
        ggml_backend_device* main_gpu = model->devices[params.main_gpu];
        model->devices.clear();
        model->devices.push_back(main_gpu);
    }

    for (auto* dev : model->devices) {
        size_t free, total; // NOLINT
        dev->get_memory(&free, &total);
        LLAMA_LOG_INFO("%s: using device %s (%s) - %zu MiB free\n", __func__, ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), free / 1024 / 1024);
    }

    int status = llama_model_load(path_model, params, *model);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        }
        else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }
        return nullptr;
    }

    return model.release();
}


// NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
//       https://github.com/ggerganov/llama.cpp/pull/7544
struct llama_context_params {
    uint32_t n_ctx = 512;             // text context, 0 = from model
    uint32_t n_batch = 2048;           // logical maximum batch size that can be submitted to llama_decode
    uint32_t n_ubatch = 512;          // physical maximum batch size
    uint32_t n_seq_max = 1;         // max number of sequences (i.e. distinct states for recurrent models)
    int32_t  n_threads = GGML_DEFAULT_N_THREADS;         // number of threads to use for generation
    int32_t  n_threads_batch = GGML_DEFAULT_N_THREADS;   // number of threads to use for batch processing

    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED; // RoPE scaling type, from `enum llama_rope_scaling_type`
    enum llama_pooling_type      pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;      // whether to pool (sum) embedding results by sequence id
    enum llama_attention_type    attention_type = LLAMA_ATTENTION_TYPE_UNSPECIFIED;    // attention type to use for embeddings

    // ref: https://github.com/ggerganov/llama.cpp/pull/2054
    float    rope_freq_base = 0.0f;   // RoPE base frequency, 0 = from model
    float    rope_freq_scale = 0.0f;  // RoPE frequency scaling factor, 0 = from model
    float    yarn_ext_factor = -1.0f;  // YaRN extrapolation mix factor, negative = from model
    float    yarn_attn_factor = 1.0f; // YaRN magnitude scaling factor
    float    yarn_beta_fast = 32.0f;   // YaRN low correction dim
    float    yarn_beta_slow = 1.0f;   // YaRN high correction dim
    uint32_t yarn_orig_ctx = 0;    // YaRN original context size
    float    defrag_thold = -1.0f;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

    ggml_backend_sched_eval_callback cb_eval = nullptr;

    enum ggml_type type_k = GGML_TYPE_F16; // data type for K cache [EXPERIMENTAL]
    enum ggml_type type_v = GGML_TYPE_F16; // data type for V cache [EXPERIMENTAL]

    // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
    // TODO: move at the end of the struct
    bool logits_all = false;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
    bool embeddings = false;  // if true, extract embeddings (together with logits)
    bool offload_kqv = true; // whether to offload the KQV ops (including the KV cache) to GPU
    bool flash_attn = false;  // whether to use flash attention [EXPERIMENTAL]
    bool no_perf = true;     // whether to measure performance timings

    // Abort callback
    // if it returns true, execution of llama_decode() will be aborted
    // currently works only with CPU execution
    ggml_abort_callback abort_callback = nullptr;
};

llama_context_params common_context_params_to_llama(const common_params& params) {
    llama_context_params cparams{};

    cparams.n_ctx = params.n_ctx;
    cparams.n_seq_max = params.n_parallel;
    cparams.n_batch = params.n_batch;
    cparams.n_ubatch = params.n_ubatch;
    cparams.n_threads = params.cpuparams.n_threads;
    cparams.n_threads_batch = params.cpuparams_batch.n_threads == -1 ?
        params.cpuparams.n_threads : params.cpuparams_batch.n_threads;
    cparams.logits_all = params.logits_all;
    cparams.embeddings = params.embedding;
    cparams.rope_scaling_type = params.rope_scaling_type;
    cparams.rope_freq_base = params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale;
    cparams.yarn_ext_factor = params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor;
    cparams.yarn_beta_fast = params.yarn_beta_fast;
    cparams.yarn_beta_slow = params.yarn_beta_slow;
    cparams.yarn_orig_ctx = params.yarn_orig_ctx;
    cparams.pooling_type = params.pooling_type;
    cparams.attention_type = params.attention_type;
    cparams.defrag_thold = params.defrag_thold;
    cparams.cb_eval = params.cb_eval;
    cparams.offload_kqv = !params.no_kv_offload;
    cparams.flash_attn = params.flash_attn;
    cparams.no_perf = params.no_perf;

    if (params.reranking) {
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }

    cparams.type_k = params.cache_type_k;
    cparams.type_v = params.cache_type_v;

    return cparams;
}

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int      n_threads;       // number of threads to use for generation
    int      n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool no_perf;

    enum llama_pooling_type pooling_type;

    ggml_backend_sched_eval_callback cb_eval;
};

using llama_seq_id = int32_t;
using llama_pos = int32_t;

struct llama_sbatch_seq {
    int32_t n_seq_id;
    llama_seq_id* seq_id;
    size_t offset;
    size_t length;
};

// Input data for llama_decode
// A llama_batch object can contain input about one or many sequences
// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
//
// - token  : the token ids of the input (used when embd is NULL)
// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
// - pos    : the positions of the respective token in the sequence
//            (if set to NULL, the token position will be tracked automatically by llama_decode)
// - seq_id : the sequence to which the respective token belongs
//            (if set to NULL, the sequence ID will be assumed to be 0)
// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
//            (if set to NULL, only the logits for last token will be returned)
//
struct llama_batch {
    int32_t n_tokens;

    llama_token* token;
    float* embd;
    llama_pos* pos;
    int32_t* n_seq_id;
    llama_seq_id** seq_id;
    int8_t* logits; // TODO: rename this to "output"
};

// very similar to llama_batch,
// but has more metadata about sequences
struct llama_ubatch {
    bool equal_seqs;
    // TODO: whole_seqs for embeddings?

    uint32_t n_tokens; // total tokens (n_seq_tokens * n_seqs)
    uint32_t n_seq_tokens; // tokens per sequence
    uint32_t n_seqs;

    llama_token* token;    // [n_tokens]
    float* embd;     // [n_embd, n_tokens]
    llama_pos* pos;      // [n_tokens]
    int32_t* n_seq_id; // [n_seqs]
    llama_seq_id** seq_id;   // [n_seqs]
    int8_t* output;   // [n_tokens]
};

// sequence-length-aware batch splitting
struct llama_sbatch {
    // tokens left in this batch
    size_t n_tokens;

    size_t n_embd;

    bool logits_all; // TODO: remove once lctx.logits_all is removed too

    // sorted indices into the batch
    std::vector<size_t> ids;
    // batch indices of the output
    std::vector<size_t> out_ids;
    std::vector<llama_sbatch_seq> seq;

    const llama_batch* batch = nullptr;

    // buffers for the ubatch
    std::vector<llama_token>    ubatch_token;
    std::vector<float>          ubatch_embd;
    std::vector<llama_pos>      ubatch_pos;
    std::vector<int32_t>        ubatch_n_seq_id;
    std::vector<llama_seq_id*> ubatch_seq_id;
    std::vector<int8_t>         ubatch_output;

    llama_ubatch reserve_ubatch(size_t n_ubatch, bool has_embd = false) {
        // clear empty sequences
        // the previous ubatch is assumed to be gone,
        // so nothing should refer to values in these sequences anymore.
        for (size_t i = seq.size(); i-- > 0;) {
            if (seq[i].length == 0) {
                seq.pop_back();
            }
            else {
                break;
            }
        }
        ubatch_token.resize(!has_embd ? n_ubatch : 0);
        ubatch_embd.resize(has_embd ? n_embd * n_ubatch : 0);
        ubatch_pos.resize(n_ubatch);
        ubatch_n_seq_id.resize(n_ubatch);
        ubatch_seq_id.resize(n_ubatch);
        ubatch_output.resize(n_ubatch);
        llama_ubatch ubatch = {
            /*equal_seqs   =*/ true,
            /*n_tokens     =*/ 0,
            /*n_seq_tokens =*/ 0,
            /*n_seqs       =*/ 0,
            /*token        =*/ !has_embd ? ubatch_token.data() : nullptr,
            /*embd         =*/ has_embd ? ubatch_embd.data() : nullptr,
            /*pos          =*/ ubatch_pos.data(),
            /*n_seq_id     =*/ ubatch_n_seq_id.data(),
            /*seq_id       =*/ ubatch_seq_id.data(),
            /*output       =*/ ubatch_output.data(),
        };
        return ubatch;
    }

    void add_seq_to_ubatch(llama_ubatch& ubatch, llama_sbatch_seq& seq, size_t length) {
        GGML_ASSERT(batch != nullptr);
        GGML_ASSERT(length <= seq.length);
        // Can only add sequences of equal lengths to a batch,
        // otherwise it isn't clear to which sequence a token belongs
        GGML_ASSERT(seq.n_seq_id == 0 || ubatch.n_seqs == 0 || length == (size_t)ubatch.n_tokens / ubatch.n_seqs);
        GGML_ASSERT((seq.n_seq_id != 0) == ubatch.equal_seqs);
        // NOTE: loops are separated for cache-friendliness
        if (batch->token) {
            if (ubatch.equal_seqs) {
                for (size_t i = 0; i < length; ++i) {
                    ubatch.token[ubatch.n_tokens + i] = batch->token[ids[seq.offset + i]];
                }
            }
            else {
                // simple split
                ubatch.token = batch->token + seq.offset;
            }
        }
        else {
            ubatch.token = nullptr;
        }
        if (batch->embd) {
            if (ubatch.equal_seqs) {
                for (size_t i = 0; i < length; ++i) {
                    memcpy(
                        ubatch.embd + n_embd * (ubatch.n_tokens + i),
                        batch->embd + n_embd * ids[seq.offset + i],
                        n_embd * sizeof(float)
                    );
                }
            }
            else {
                // simple split
                ubatch.embd = batch->embd + (n_embd * seq.offset);
            }
        }
        else {
            ubatch.embd = nullptr;
        }
        if (ubatch.equal_seqs) {
            for (size_t i = 0; i < length; ++i) {
                ubatch.pos[ubatch.n_tokens + i] = batch->pos[ids[seq.offset + i]];
            }
        }
        else {
            // simple split
            ubatch.pos = batch->pos + seq.offset;
        }
        if (ubatch.equal_seqs) {
            ubatch.n_seq_id[ubatch.n_seqs] = seq.n_seq_id;
            if (seq.seq_id) {
                ubatch.seq_id[ubatch.n_seqs] = seq.seq_id;
            }
        }
        else {
            // simple split
            if (batch->n_seq_id) {
                ubatch.n_seq_id = batch->n_seq_id + seq.offset;
            }
            else {
                for (size_t i = 0; i < length; ++i) {
                    ubatch.n_seq_id[ubatch.n_seqs + i] = 1;
                }
            }
            if (batch->seq_id) {
                ubatch.seq_id = batch->seq_id + seq.offset;
            }
        }
        if (logits_all) {
            for (size_t i = 0; i < length; ++i) {
                ubatch.output[ubatch.n_tokens + i] = 1;
                out_ids.push_back(ids[seq.offset + i]);
            }
        }
        else if (batch->logits) {
            if (ubatch.equal_seqs) {
                for (size_t i = 0; i < length; ++i) {
                    size_t id = ids[seq.offset + i];
                    int8_t is_output = batch->logits[id];
                    ubatch.output[ubatch.n_tokens + i] = is_output;
                    if (is_output) { out_ids.push_back(id); }
                }
            }
            else {
                // simple split
                ubatch.output = batch->logits + seq.offset;
                for (size_t i = 0; i < length; ++i) {
                    if (ubatch.output[i] != 0) { out_ids.push_back(seq.offset + i); }
                }
            }
        }
        else {
            // only get last output
            for (size_t i = 0; i < length; ++i) {
                size_t id = ids[seq.offset + i];
                int8_t is_last = id == ids.size() - 1;
                ubatch.output[ubatch.n_tokens + i] = is_last;
                if (is_last) { out_ids.push_back(id); }
            }
        }
        if (ubatch.n_tokens == 0 && ubatch.n_seqs == 0) {
            ubatch.n_seq_tokens = ubatch.equal_seqs ? length : 1;
        }
        ubatch.n_tokens += length;
        ubatch.n_seqs += ubatch.equal_seqs ? 1 : length; // virtual sequences for simple splits
        seq.offset += length;
        seq.length -= length;
        n_tokens -= length;
        GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);
    }

    // simple split, unknown number of sequences of unequal lengths
    llama_ubatch split_simple(size_t n_ubatch) {
        n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
        llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
        ubatch.equal_seqs = false;
        if (!seq.empty()) {
            llama_sbatch_seq& s = seq[0];
            size_t length = s.length < n_ubatch ? s.length : n_ubatch;
            GGML_ASSERT(seq.size() == 1 && s.n_seq_id == 0); // don't mix with other splits
            add_seq_to_ubatch(ubatch, s, length);
        }
        return ubatch;
    }

    // make batches of equal-length sequences
    llama_ubatch split_equal(size_t n_ubatch) {
        n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
        llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
        if (!seq.empty()) {
            size_t length = 0;
            size_t n_tokens_in_ubatch = 0;
            GGML_ASSERT(seq[0].n_seq_id > 0); // should not be mixed with simple splits
            // smallest first, because it's easier to split this way;
            // starting from the end to pop in constant time.
            for (size_t i = seq.size(); i-- > 0;) {
                llama_sbatch_seq& s = seq[i];
                GGML_ASSERT(s.length > 0);
                if (length == 0) {
                    length = s.length < n_ubatch ? s.length : n_ubatch;
                }
                add_seq_to_ubatch(ubatch, s, length);
                n_tokens_in_ubatch += length;
                // shared prompts can't be mixed with any of their sequences,
                // so it's safer to compute them in their own ubatch
                if (s.n_seq_id > 1) { break; }
                // stop when there isn't enough space for another sequence
                if (length + n_tokens_in_ubatch > n_ubatch) { break; }
            }
        }
        return ubatch;
    }

    // sequence-wise split
    llama_ubatch split_seq(size_t n_ubatch) {
        n_ubatch = n_tokens < n_ubatch ? n_tokens : n_ubatch;
        llama_ubatch ubatch = reserve_ubatch(n_ubatch, /* has_embd */ batch->embd != nullptr);
        if (!seq.empty()) {
            llama_sbatch_seq& s = seq[seq.size() - 1];
            size_t length = s.length < n_ubatch ? s.length : n_ubatch;
            GGML_ASSERT(s.n_seq_id > 0); // should not be mixed with simple splits
            add_seq_to_ubatch(ubatch, s, length);
        }
        return ubatch;
    }

    void from_batch(const llama_batch& batch, const size_t n_embd, const bool simple_split = false, const bool logits_all = false) {
        GGML_ASSERT(batch.n_tokens >= 0);
        this->batch = &batch;
        this->n_embd = n_embd;
        this->logits_all = logits_all;

        n_tokens = batch.n_tokens;
        ids.resize(n_tokens);
        out_ids.clear();
        // TODO: reserve out_ids and seq

        for (size_t i = 0; i < n_tokens; ++i) {
            ids[i] = i;
        }
        if (simple_split) {
            seq.resize(1);
            llama_sbatch_seq& s = seq[0];
            s.n_seq_id = 0;
            s.seq_id = nullptr;
            s.offset = 0;
            s.length = n_tokens;
            return;
        }
        std::sort(ids.begin(), ids.end(),
            [&batch](size_t a, size_t b) {
                int32_t n_seq_a = batch.n_seq_id ? batch.n_seq_id[a] : 1;
                int32_t n_seq_b = batch.n_seq_id ? batch.n_seq_id[b] : 1;
                // sort by seq_id, then by pos
                if (n_seq_a == n_seq_b) {
                    if (batch.seq_id) {
                        for (int32_t i = 0; i < n_seq_a; ++i) {
                            llama_seq_id seq_id_a = batch.seq_id[a][i];
                            llama_seq_id seq_id_b = batch.seq_id[b][i];
                            // smaller seq_ids go first
                            if (seq_id_a != seq_id_b) {
                                return seq_id_a < seq_id_b;
                            }
                        }
                    }
                    // when all else is equal, sort by pos
                    if (batch.pos) {
                        return batch.pos[a] < batch.pos[b];
                    }
                    // no pos, sort by id
                    return a < b;
                }
                // shared prompts go first
                return n_seq_a > n_seq_b;
            }
        );
        // init seq
        llama_sbatch_seq* last_seq = nullptr;

        for (size_t i = 0; i < n_tokens; ++i) {
            const size_t bi = ids[i];
            const int32_t n_seqs = batch.n_seq_id[bi];
            llama_seq_id* seq_ids = batch.seq_id[bi];
            if (last_seq != nullptr) {
                bool same = n_seqs == last_seq->n_seq_id;
                for (int32_t j = 0; same && j < n_seqs; ++j) {
                    if (seq_ids[j] != last_seq->seq_id[j]) {
                        same = false;
                    }
                }
                if (same) {
                    last_seq->length += 1;
                    continue;
                }
            }
            llama_sbatch_seq new_seq = { n_seqs, seq_ids, i, 1 };
            seq.push_back(new_seq);
            last_seq = &seq.back();
        }
        // keep shared prompts first at the end, then sort by length descending.
        std::sort(seq.begin(), seq.end(),
            [](llama_sbatch_seq& a, llama_sbatch_seq& b) {
                if (a.n_seq_id == b.n_seq_id) {
                    return a.length > b.length;
                }
                return a.n_seq_id < b.n_seq_id;
            }
        );
    }
};

struct llama_kv_cell {
    llama_pos pos = -1;
    llama_pos delta = 0;
    int32_t   src = -1; // used by recurrent state models to copy states
    int32_t   tail = -1;

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id& id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell& other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool v_trans = true;  // the value tensor is transposed

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<ggml_tensor*> k_l; // per layer
    std::vector<ggml_tensor*> v_l;

    std::vector<std::unique_ptr<ggml_context>> ctxs;
    std::vector<std::unique_ptr<ggml_backend_buffer>> bufs;

    size_t total_size() {
        size_t size = 0;
        for (auto& buf : bufs) {
            size += buf->get_size();
        }
        return size;
    }
};

struct llama_control_vector {
    std::vector<ggml_tensor*> tensors; // per layer
    std::vector<std::unique_ptr<ggml_context>> ctxs;
    std::vector<std::unique_ptr<ggml_backend_buffer>> bufs;

    int32_t layer_start = -1;
    int32_t layer_end = -1;

    ggml_tensor* tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end || (size_t)il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    ggml_tensor* apply_to(ggml_context* ctx, ggml_tensor* cur, int  il) const {
        // TODO
#if 0
        ggml_tensor* layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
#else
        return nullptr;
#endif
    }
};

struct llama_context {
    llama_context(const llama_model& model)
        : model(model)
        , t_start_us(model.t_start_us)
        , t_load_us(model.t_load_us) {
    }

    const struct llama_model& model;

    struct llama_cparams        cparams;
    struct llama_sbatch         sbatch;
    struct llama_kv_cache       kv_self;
    struct llama_control_vector cvec;

    std::unordered_map<struct llama_lora_adapter*, float> lora_adapters;

    std::vector<std::unique_ptr<ggml_backend>> backends;

    ggml_backend* backend_cpu = nullptr;

    ggml_threadpool_t threadpool = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    bool has_evaluated_once = false;

    mutable int64_t t_start_us;
    mutable int64_t t_load_us;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval = 0; // number of eval calls

    // host buffer for the model output (logits and embeddings)
    std::unique_ptr<ggml_backend_buffer> buf_output;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float* logits = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float* embd = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // TODO: find a better way to accommodate mutli-dimension position encoding methods
    // number of position id each token get, 1 for each token in most cases.
    // when using m-rope, it will be 3 position ids per token to representing 3 dimension coordinate.
    int n_pos_per_token = 1;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    std::unique_ptr<ggml_backend_sched> sched;

    ggml_abort_callback abort_callback = nullptr;

    // input tensors
    ggml_tensor* inp_tokens;      // I32 [n_batch]
    ggml_tensor* inp_embd;        // F32 [n_embd, n_batch]
    ggml_tensor* inp_pos;         // I32 [n_batch]
    ggml_tensor* inp_out_ids;     // I32 [n_outputs]
    ggml_tensor* inp_KQ_mask;     // F32 [kv_size, n_batch]
    ggml_tensor* inp_KQ_mask_swa; // F32 [kv_size, n_batch]
    ggml_tensor* inp_K_shift;     // I32 [kv_size]
    ggml_tensor* inp_mean;        // F32 [n_batch, n_batch]
    ggml_tensor* inp_cls;         // I32 [n_batch]
    ggml_tensor* inp_s_copy;      // I32 [kv_size]
    ggml_tensor* inp_s_mask;      // F32 [1, n_kv]
    ggml_tensor* inp_s_seq;       // I32 [n_kv, n_batch]
    ggml_tensor* inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    ggml_tensor* inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    ggml_tensor* inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]
};

static uint32_t llama_kv_cache_get_padding(const llama_cparams& cparams) {
    // the FA kernels require padding to avoid extra runtime boundary checks
    return cparams.flash_attn ? 256u : 32u;
}

bool llama_model_has_encoder(const llama_model* model) {
    switch (model->arch) {
    case LLM_ARCH_T5:        return true;
    case LLM_ARCH_T5ENCODER: return true;
    default:                 return false;
    }
}

bool llama_model_is_recurrent(const llama_model* model) {
    switch (model->arch) {
    case LLM_ARCH_MAMBA:  return true;
    case LLM_ARCH_RWKV6:  return true;
    default:              return false;
    }
}

void llama_set_abort_callback(struct llama_context* ctx, std::function<bool()> abort_callback) {
    ctx->abort_callback = abort_callback;

    for (auto& backend : ctx->backends) {
        auto* reg = backend->get_device()->get_backend_reg();
        auto* set_abort_callback_fn = (ggml_backend_set_abort_callback_t)reg->get_proc_address("ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), ctx->abort_callback);
        }
    }
}

//
// kv cache helpers
//

static bool llama_kv_cache_init(
    struct llama_kv_cache& cache,
    const llama_context* ctx,
    ggml_type   type_k,
    ggml_type   type_v,
    uint32_t   kv_size,
    bool   offload) {
    const llama_model& model = ctx->model;
    const llama_cparams& cparams = ctx->cparams;

    const struct llama_hparams& hparams = model.hparams;

    const int32_t n_layer = hparams.n_layer;

    LLAMA_LOG_INFO("%s: kv_size = %d, offload = %d, type_k = '%s', type_v = '%s', n_layer = %d\n", __func__, kv_size, offload, ggml_type_name(type_k), ggml_type_name(type_v), n_layer);

    cache.has_shift = false;

    cache.recurrent = llama_model_is_recurrent(&model);
    cache.v_trans = !cache.recurrent && !cparams.flash_attn;

    cache.head = 0;
    cache.size = kv_size;
    cache.used = 0;

    cache.type_k = type_k;
    cache.type_v = type_v;

    cache.cells.clear();
    cache.cells.resize(kv_size);

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type*, ggml_context*> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type* buft) -> ggml_context* {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            auto ctx = std::make_unique<ggml_context>();
            if (!ctx) {
                return nullptr;
            }
            ctx_map[buft] = ctx.get();
            cache.ctxs.emplace_back(ctx.get());
            return ctx.release();
        }
        return it->second;
        };

    cache.k_l.reserve(n_layer);
    cache.v_l.reserve(n_layer);

    for (int i = 0; i < n_layer; i++) {
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(i) + hparams.n_embd_k_s();
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(i) + hparams.n_embd_v_s();

        LLAMA_LOG_DEBUG("%s: layer %d: n_embd_k_gqa = %d, n_embd_v_gqa = %d\n", __func__, i, n_embd_k_gqa, n_embd_v_gqa);

        ggml_backend_buffer_type* buft;
        if (offload) {
            auto* dev = model.dev_layer.at(i).dev;
            buft = dev->get_buffer_type();
        }
        else {
            buft = ggml_backend_cpu_buffer_type();
        }
        ggml_context* ctx = ctx_for_buft(buft);

        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to create ggml context for kv cache\n", __func__);
            return false;
        }

        ggml_tensor* k = ctx->create(type_k, { n_embd_k_gqa * kv_size });
        ggml_tensor* v = ctx->create(type_v, { n_embd_v_gqa * kv_size });
        k->set_name("cache_k_l{}", i);
        v->set_name("cache_v_l{}", i);
        cache.k_l.push_back(k);
        cache.v_l.push_back(v);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto it : ctx_map) {
        auto* buft = it.first;
        auto* ctx = it.second;

        std::unique_ptr<ggml_backend_buffer> buf = buft->alloc_tensors(ctx);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for kv cache\n", __func__);
            return false;
        }
        buf->clear(0);
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0);
        cache.bufs.emplace_back(buf.release());
    }

    return true;
}

// Make sure enough space is available for outputs.
// Returns max number of outputs for which space was reserved.
static size_t llama_output_reserve(llama_context& lctx, size_t n_outputs) {
    const auto& cparams = lctx.cparams;
    const auto& hparams = lctx.model.hparams;

    const size_t n_outputs_max = std::max(n_outputs, (size_t)cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = hparams.n_vocab;
    const auto n_embd = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool has_logits = !cparams.embeddings;
    const bool has_embd = cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    const size_t logits_size = has_logits ? n_vocab * n_outputs_max : 0;
    const size_t embd_size = has_embd ? n_embd * n_outputs_max : 0;

    if (lctx.output_ids.empty()) {
        // init, never resized afterwards
        lctx.output_ids.resize(n_batch);
    }

    const size_t prev_size = lctx.buf_output ? lctx.buf_output->get_size() : 0;
    const size_t new_size = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!lctx.buf_output || prev_size < new_size) {
        if (lctx.buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            lctx.buf_output = nullptr;
            lctx.logits = nullptr;
            lctx.embd = nullptr;
        }

        auto* buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto* output_dev = lctx.model.dev_output.dev;
        auto* output_dev_host_buft = output_dev ? output_dev->get_host_buffer_type() : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        lctx.buf_output.reset(buft->alloc_buffer(new_size).release());
        if (lctx.buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float* output_base = (float*)lctx.buf_output->get_base();

    lctx.logits = has_logits ? output_base : nullptr;
    lctx.embd = has_embd ? output_base + logits_size : nullptr;

    lctx.output_size = n_outputs_max;
    lctx.logits_size = logits_size;
    lctx.embd_size = embd_size;

    // set all ids as invalid (negative)
    std::fill(lctx.output_ids.begin(), lctx.output_ids.end(), -1);

    lctx.buf_output->clear(0);

    lctx.n_outputs = 0;

    return n_outputs_max;
}

llama_token llama_token_bos_impl(const struct llama_vocab& vocab) {
    return vocab.type != LLAMA_VOCAB_TYPE_WPM ? vocab.special_bos_id : vocab.special_cls_id;
}

llama_token llama_token_bos(const llama_model* model) {
    return llama_token_bos_impl(model->vocab);
}

using llm_build_cb = std::function<void(ggml_tensor* cur, const char* name, int nl)>;

static ggml_tensor* llm_build_inp_embd(
    ggml_context* ctx,
    struct llama_context& lctx,
    const llama_hparams& hparams,
    const llama_ubatch& batch,
    ggml_tensor* tok_embd,
    const llm_build_cb& cb) {
    const int64_t n_embd = hparams.n_embd;

    struct ggml_tensor* inpL;

    if (batch.token) {
        lctx.inp_tokens = ctx->create(GGML_TYPE_I32, { batch.n_tokens });
        cb(lctx.inp_tokens, "inp_tokens", -1);
        lctx.inp_tokens->set_flag(GGML_TENSOR_FLAG_INPUT);

        inpL = ggml_get_rows(ctx, tok_embd, lctx.inp_tokens);
    }
    else {
        lctx.inp_embd = ctx->create(GGML_TYPE_F32, { n_embd, batch.n_tokens });
        inpL = lctx.inp_embd;
        lctx.inp_embd->set_flag(GGML_TENSOR_FLAG_INPUT);
    }

    // For Granite architecture
    if (hparams.f_embedding_scale != 0.0f) {
        inpL = ggml_scale(ctx, inpL, hparams.f_embedding_scale);
    }

    cb(inpL, "inp_embd", -1);

    return inpL;
}

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

// do mat_mul, while optionally apply lora
static ggml_tensor* llm_build_lora_mm(
    struct llama_context& lctx,
    ggml_context* ctx0,
    ggml_tensor* w,
    ggml_tensor* cur) {
    struct ggml_tensor* res = ggml_mul_mat(ctx0, w, cur);
    for (auto& it : lctx.lora_adapters) {
        struct llama_lora_weight* lora = it.first->get_weight(w);
        if (lora == nullptr) {
            continue;
        }
        const float alpha = it.first->alpha;
        const float rank = (float)lora->b->ne[0];
        const float scale = alpha ? it.second * alpha / rank : it.second;
        struct ggml_tensor* ab_cur = ggml_mul_mat(
            ctx0, lora->b,
            ggml_mul_mat(ctx0, lora->a, cur)
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}

static void llm_build_kv_store(
    ggml_context* ctx,
    const llama_hparams& hparams,
    const llama_cparams& cparams,
    const llama_kv_cache& kv,
    ggml_cgraph* graph,
    ggml_tensor* k_cur,
    ggml_tensor* v_cur,
    int32_t   n_tokens,
    int32_t   kv_head,
    const llm_build_cb& cb,
    int64_t   il) {
    const int64_t n_ctx = cparams.n_ctx;

    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    GGML_ASSERT(kv.size == n_ctx);

    ggml_tensor* k_cache_view = ggml_view(ctx, kv.k_l[il], { n_tokens * n_embd_k_gqa }, {}, ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa) * kv_head);
    cb(k_cache_view, "k_cache_view", il);

    // note: storing RoPE-ed version of K in the KV cache
    graph->build_forward_expand(ggml_cpy(ctx, k_cur, k_cache_view));

    assert(v_cur->ne[0] == n_embd_v_gqa && v_cur->ne[1] == n_tokens);

    struct ggml_tensor* v_cache_view = nullptr;

    if (cparams.flash_attn) {
        v_cache_view = ggml_view(ctx, kv.v_l[il], { n_tokens * n_embd_v_gqa }, {}, ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa) * kv_head);
    }
    else {
        // note: the V cache is transposed when not using flash attention
        v_cache_view = ggml_view(ctx, kv.v_l[il], { n_tokens, n_embd_v_gqa },
            { (n_ctx)*ggml_element_size(kv.v_l[il]) },
            (kv_head)*ggml_element_size(kv.v_l[il]));

        v_cur = ggml_transpose(ctx, v_cur);
    }
    cb(v_cache_view, "v_cache_view", il);

    graph->build_forward_expand(ggml_cpy(ctx, v_cur, v_cache_view));
}

static ggml_tensor* llm_build_kqv(
    ggml_context* ctx,
    struct llama_context& lctx,
    const llama_kv_cache& kv,
    ggml_cgraph* graph,
    ggml_tensor* wo,
    ggml_tensor* wo_b,
    ggml_tensor* q_cur,
    ggml_tensor* kq_mask,
    int32_t   n_tokens,
    int32_t   n_kv,
    float     kq_scale,
    const llm_build_cb& cb,
    int       il) {
    const llama_model& model = lctx.model;
    const llama_hparams& hparams = lctx.model.hparams;
    const llama_cparams& cparams = lctx.cparams;

    const int64_t n_ctx = cparams.n_ctx;
    const int64_t n_head = hparams.n_head(il);
    const int64_t n_head_kv = hparams.n_head_kv(il);
    const int64_t n_embd_head_k = hparams.n_embd_head_k;
    const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const int64_t n_embd_head_v = hparams.n_embd_head_v;
    const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

    struct ggml_tensor* q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);
    cb(q, "q", il);

    struct ggml_tensor* k =
        ggml_view(ctx, kv.k_l[il],
            { n_embd_head_k, n_kv, n_head_kv },
            { ggml_row_size(kv.k_l[il]->type, n_embd_k_gqa),
            ggml_row_size(kv.k_l[il]->type, n_embd_head_k) },
            0);
    cb(k, "k", il);

    struct ggml_tensor* cur;

    if (cparams.flash_attn) {
        // TODO
#if 0
        GGML_UNUSED(model);
        GGML_UNUSED(n_ctx);
#endif
        // split cached v into n_head heads (not transposed)
        struct ggml_tensor* v =
            ggml_view(ctx, kv.v_l[il],
                { n_embd_head_v, n_kv, n_head_kv },
                { ggml_row_size(kv.v_l[il]->type, n_embd_v_gqa),
                ggml_row_size(kv.v_l[il]->type, n_embd_head_v) },
                0);
        cb(v, "v", il);

        cur = ggml_flash_attn_ext(ctx, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
            hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);

        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape(ctx, cur, { n_embd_head_v * n_head, n_tokens });
    }
    else {
        struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q);
        cb(kq, "kq", il);

        // note: this op tends to require high floating point range
        //       while for some models F16 is enough, for others it is not, so we default to F32 here
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        if (model.arch == LLM_ARCH_GROK) {
            // need to do the following:
            // multiply by attn_output_multiplyer of 0.08838834764831845
            // and then :
            // kq = 30 * tanh(kq / 30)
            // before the softmax below

            kq = ggml_tanh(ctx, ggml_scale(ctx, kq, 0.08838834764831845f / 30.0f));
            kq = ggml_scale(ctx, kq, 30);
        }

        if (hparams.attn_soft_cap) {
            kq = ggml_scale(ctx, kq, 1.0f / hparams.f_attn_logit_softcapping);
            kq = ggml_tanh(ctx, kq);
            kq = ggml_scale(ctx, kq, hparams.f_attn_logit_softcapping);
        }

        kq = ggml_soft_max(ctx, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        cb(kq, "kq_soft_max", il);

        GGML_ASSERT(kv.size == n_ctx);

        // split cached v into n_head heads
        ggml_tensor* v =
            ggml_view(ctx, kv.v_l[il],
                { n_kv, n_embd_head_v, n_head_kv },
                { ggml_element_size(kv.v_l[il]) * n_ctx,
                ggml_element_size(kv.v_l[il]) * n_ctx * n_embd_head_v },
                0);
        cb(v, "v", il);

        struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq);
        cb(kqv, "kqv", il);

        struct ggml_tensor* kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);
        cb(kqv_merged, "kqv_merged", il);

        cur = ggml_cont(ctx, kqv_merged, { n_embd_head_v * n_head, n_tokens });
        cb(cur, "kqv_merged_cont", il);
    }

    graph->build_forward_expand(cur);

    if (wo) {
        cur = llm_build_lora_mm(lctx, ctx, wo, cur);
    }

    if (wo_b) {
        cb(cur, "kqv_wo", il);
    }

    if (wo_b) {
        cur = ggml_add(ctx, cur, wo_b);
    }

    return cur;
}

static ggml_tensor* llm_build_kv(
    ggml_context* ctx,
    struct llama_context& lctx,
    const llama_kv_cache& kv,
    ggml_cgraph* graph,
    ggml_tensor* wo,
    ggml_tensor* wo_b,
    ggml_tensor* k_cur,
    ggml_tensor* v_cur,
    ggml_tensor* q_cur,
    ggml_tensor* kq_mask,
    int32_t   n_tokens,
    int32_t   kv_head,
    int32_t   n_kv,
    float     kq_scale,
    const llm_build_cb& cb,
    int       il) {
    const llama_hparams& hparams = lctx.model.hparams;
    const llama_cparams& cparams = lctx.cparams;

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    graph->build_forward_expand(q_cur);
    graph->build_forward_expand(k_cur);
    graph->build_forward_expand(v_cur);

    llm_build_kv_store(ctx, hparams, cparams, kv, graph, k_cur, v_cur, n_tokens, kv_head, cb, il);

    struct ggml_tensor* cur;

    cur = llm_build_kqv(ctx, lctx, kv, graph, wo, wo_b, q_cur, kq_mask, n_tokens, n_kv, kq_scale, cb, il);
    cb(cur, "kqv_out", il);

    return cur;
}

static ggml_tensor* llm_build_ffn(
    ggml_context* ctx,
    struct llama_context& lctx,
    ggml_tensor* cur,
    ggml_tensor* up,
    ggml_tensor* up_b,
    ggml_tensor* up_s,
    ggml_tensor* gate,
    ggml_tensor* gate_b,
    ggml_tensor* gate_s,
    ggml_tensor* down,
    ggml_tensor* down_b,
    ggml_tensor* down_s,
    ggml_tensor* act_scales,
    llm_ffn_op_type   type_op,
    llm_ffn_gate_type   type_gate,
    const llm_build_cb& cb,
    int   il) {
    struct ggml_tensor* tmp = up ? llm_build_lora_mm(lctx, ctx, up, cur) : cur;
    cb(tmp, "ffn_up", il);

    if (up_b) {
        tmp = ggml_add(ctx, tmp, up_b);
        cb(tmp, "ffn_up_b", il);
    }

    if (up_s) {
        tmp = ggml_mul(ctx, tmp, up_s);
        cb(tmp, "ffn_up_s", il);
    }

    if (gate) {
        switch (type_gate) {
        case LLM_FFN_SEQ:
        {
            cur = llm_build_lora_mm(lctx, ctx, gate, tmp);
            cb(cur, "ffn_gate", il);
        } break;
        case LLM_FFN_PAR:
        {
            cur = llm_build_lora_mm(lctx, ctx, gate, cur);
            cb(cur, "ffn_gate", il);
        } break;
        }

        if (gate_b) {
            cur = ggml_add(ctx, cur, gate_b);
            cb(cur, "ffn_gate_b", il);
        }

        if (gate_s) {
            cur = ggml_mul(ctx, cur, gate_s);
            cb(cur, "ffn_gate_s", il);
        }

    }
    else {
        cur = tmp;
    }

    switch (type_op) {
    case LLM_FFN_SILU:
    {
        cur = ggml_silu(ctx, cur);
        cb(cur, "ffn_silu", il);
    } break;
    case LLM_FFN_GELU:
    {
        cur = ggml_gelu(ctx, cur);
        cb(cur, "ffn_gelu", il);
        if (act_scales != NULL) {
            cur = ggml_div(ctx, cur, act_scales);
            cb(cur, "ffn_act", il);
        }
    } break;
    case LLM_FFN_RELU:
    {
        cur = ggml_relu(ctx, cur);
        cb(cur, "ffn_relu", il);
    } break;
    case LLM_FFN_RELU_SQR:
    {
        cur = ggml_relu(ctx, cur);
        cb(cur, "ffn_relu", il);

        cur = ggml_sqr(ctx, cur);
        cb(cur, "ffn_sqr(relu)", il);
    } break;
    case LLM_FFN_SWIGLU:
    {
        // Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        int64_t split_point = cur->ne[0] / 2;
        struct ggml_tensor* x0 = ggml_cont(ctx, ggml_view(ctx, cur, { split_point, cur->ne[1] }, { cur->nb[1] }, 0));
        struct ggml_tensor* x1 = ggml_cont(ctx, ggml_view(ctx, cur, { split_point, cur->ne[1] }, { cur->nb[1] }, split_point * ggml_element_size(cur)));

        x0 = ggml_silu(ctx, x0);
        cb(cur, "ffn_silu", il);

        cur = ggml_mul(ctx, x0, x1);
        cb(cur, "ffn_mul", il);
    } break;
    }

    if (type_gate == LLM_FFN_PAR) {
        cur = ggml_mul(ctx, cur, tmp);
        cb(cur, "ffn_gate_par", il);
    }

    if (down) {
        cur = llm_build_lora_mm(lctx, ctx, down, cur);
    }

    if (down_b) {
        cb(cur, "ffn_down", il);
    }

    if (down_b) {
        cur = ggml_add(ctx, cur, down_b);
    }

    if (down_s) {
        cur = ggml_mul(ctx, cur, down_s);
        cb(cur, "ffn_down_s", il);
    }

    return cur;
}

// do mat_mul_id, while optionally apply lora
static ggml_tensor* llm_build_lora_mm_id(
    struct llama_context& lctx,
    ggml_context* ctx0,
    ggml_tensor* w,   // struct ggml_tensor * as
    ggml_tensor* cur, // struct ggml_tensor * b
    ggml_tensor* ids) {
    struct ggml_tensor* res = ggml_mul_mat_id(ctx0, w, cur, ids);
    for (auto& it : lctx.lora_adapters) {
        struct llama_lora_weight* lora = it.first->get_weight(w);
        if (lora == nullptr) {
            continue;
        }
        const float alpha = it.first->alpha;
        const float rank = (float)lora->b->ne[0];
        const float scale = alpha ? it.second * alpha / rank : it.second;
        struct ggml_tensor* ab_cur = ggml_mul_mat_id(
            ctx0, lora->b,
            ggml_mul_mat_id(ctx0, lora->a, cur, ids),
            ids
        );
        ab_cur = ggml_scale(ctx0, ab_cur, scale);
        res = ggml_add(ctx0, res, ab_cur);
    }
    return res;
}

static ggml_tensor* llm_build_moe_ffn(
    ggml_context* ctx,
    struct llama_context& lctx,
    ggml_tensor* cur,
    ggml_tensor* gate_inp,
    ggml_tensor* up_exps,
    ggml_tensor* gate_exps,
    ggml_tensor* down_exps,
    int64_t   n_expert,
    int64_t   n_expert_used,
    llm_ffn_op_type   type_op,
    bool   norm_w,
    bool   scale_w,
    float   w_scale,
    const llm_build_cb& cb,
    int   il) {
    int64_t n_embd = cur->ne[0];
    int64_t n_tokens = cur->ne[1];

    ggml_tensor* logits = llm_build_lora_mm(lctx, ctx, gate_inp, cur); // [n_expert, n_tokens]
    cb(logits, "ffn_moe_logits", il);

    ggml_tensor* probs = ggml_soft_max(ctx, logits); // [n_expert, n_tokens]
    cb(probs, "ffn_moe_probs", il);

    // select experts
    ggml_tensor* selected_experts = ggml_top_k(ctx, probs, n_expert_used); // [n_expert_used, n_tokens]
    // TODO
#if 0
    cb(selected_experts->src[0], "ffn_moe_argsort", il);
#endif
    cb(selected_experts, "ffn_moe_topk", il);

    ggml_tensor* weights = ggml_get_rows(ctx,
        ggml_reshape(ctx, probs, { 1, n_expert, n_tokens }), selected_experts); // [1, n_expert_used, n_tokens]
    cb(weights, "ffn_moe_weights", il);

    if (norm_w) {
        weights = ggml_reshape(ctx, weights, { n_expert_used, n_tokens });

        ggml_tensor* weights_sum = ggml_sum_rows(ctx, weights); // [1, n_tokens]
        cb(weights_sum, "ffn_moe_weights_sum", il);

        weights = ggml_div(ctx, weights, weights_sum); // [n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights_norm", il);

        weights = ggml_reshape(ctx, weights, { 1, n_expert_used, n_tokens });
    }
    if (scale_w) {
        weights = ggml_scale(ctx, weights, w_scale);
        cb(weights, "ffn_moe_weights_scaled", il);
    }

    cur = ggml_reshape(ctx, cur, { n_embd, 1, n_tokens });
    ggml_tensor* up = llm_build_lora_mm_id(lctx, ctx, up_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(up, "ffn_moe_up", il);

    ggml_tensor* gate = llm_build_lora_mm_id(lctx, ctx, gate_exps, cur, selected_experts); // [n_ff, n_expert_used, n_tokens]
    cb(gate, "ffn_moe_gate", il);

    switch (type_op) {
    case LLM_FFN_SILU:
    {
        gate = ggml_silu(ctx, gate);
        cb(gate, "ffn_moe_silu", il);
    } break;
    case LLM_FFN_GELU:
    {
        gate = ggml_gelu(ctx, gate);
        cb(gate, "ffn_moe_gelu", il);
    } break;
    default:
        GGML_ABORT("fatal error");
    }

    ggml_tensor* par = ggml_mul(ctx, up, gate); // [n_ff, n_expert_used, n_tokens]
    cb(par, "ffn_moe_gate_par", il);

    ggml_tensor* experts = llm_build_lora_mm_id(lctx, ctx, down_exps, par, selected_experts); // [n_embd, n_expert_used, n_tokens]
    cb(experts, "ffn_moe_down", il);

    experts = ggml_mul(ctx, experts, weights);

    // aggregate experts
    ggml_tensor* moe_out = nullptr;
    for (int i = 0; i < n_expert_used; ++i) {
        ggml_tensor* cur_expert = ggml_view(ctx, experts, { n_embd, n_tokens },
            { experts->nb[2] }, i* experts->nb[1]);

        if (i == 0) {
            moe_out = cur_expert;
        }
        else {
            moe_out = ggml_add(ctx, moe_out, cur_expert);
        }
    }

    if (n_expert_used == 1) {
        // avoid returning a non-contiguous tensor
        moe_out = ggml_cont(ctx, moe_out);
    }

    return moe_out;
}

static ggml_tensor* llm_build_copy_mask_state(
    ggml_context* ctx,
    ggml_cgraph* graph,
    ggml_tensor* s,
    ggml_tensor* state_copy,
    ggml_tensor* state_mask,
    int32_t   n_state,
    int32_t   kv_size,
    int32_t   kv_head,
    int32_t   n_kv,
    int32_t   n_seqs) {
    struct ggml_tensor* states = ggml_reshape(ctx, s, { n_state, kv_size });

    // copy states
    // NOTE: assuming the copy destinations are ALL contained between kv_head and kv_head + n_kv
    // this shrinks the tensors's ne[1] to n_kv
    states = ggml_get_rows(ctx, states, state_copy);

    // clear states of sequences which are starting at the beginning of this batch
    // FIXME: zero-out NANs?
    states = ggml_mul(ctx, states, state_mask);

    // copy states which won't be changed further (between n_seqs and n_kv)
    graph->build_forward_expand(
        ggml_cpy(ctx,
            ggml_view(ctx, states, { n_state * (n_kv - n_seqs) }, {}, n_seqs * n_state * ggml_element_size(states)),
            ggml_view(ctx, s, { n_state * (n_kv - n_seqs) }, {}, (kv_head + n_seqs) * n_state * ggml_element_size(s))));

    // the part of the states that will be used and modified
    return ggml_view(ctx, states, { n_state, n_seqs }, { states->nb[1] }, 0);
}

// TODO: split
static ggml_tensor* llm_build_mamba(
    ggml_context* ctx,
    struct llama_context& lctx,
    const llama_ubatch& batch,
    ggml_cgraph* graph,
    ggml_tensor* cur,
    ggml_tensor* state_copy,
    ggml_tensor* state_mask,
    int32_t   kv_head,
    int32_t   n_kv,
    const llm_build_cb& cb,
    int       il) {
    const llama_model& model = lctx.model;
    const llama_hparams& hparams = model.hparams;
    const llama_kv_cache& kv = lctx.kv_self;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t d_state = hparams.ssm_d_state;
    const int64_t dt_rank = hparams.ssm_dt_rank;
    const int64_t n_seqs = batch.n_seqs;
    // Some variants of Mamba arch (e.g. FalconMamba do apply layer norm on B and Dt layers)
    const bool ssm_dt_b_c_rms = hparams.ssm_dt_b_c_rms;
    // Use the same RMS norm as the final layer norm
    const float norm_rms_eps = hparams.f_norm_rms_eps;

    const int64_t n_seq_tokens = batch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(batch.equal_seqs);
    GGML_ASSERT(batch.n_tokens == n_seq_tokens * n_seqs);

    struct ggml_tensor* conv_states_all = kv.k_l[il];
    struct ggml_tensor* ssm_states_all = kv.v_l[il];

    // (ab)using the KV cache to store the states
    struct ggml_tensor* conv = llm_build_copy_mask_state(ctx,
        graph, conv_states_all, state_copy, state_mask,
        hparams.n_embd_k_s(), kv.size, kv_head, n_kv, n_seqs);
    conv = ggml_reshape(ctx, conv, { d_conv - 1, d_inner, n_seqs });
    struct ggml_tensor* ssm = llm_build_copy_mask_state(ctx,
        graph, ssm_states_all, state_copy, state_mask,
        hparams.n_embd_v_s(), kv.size, kv_head, n_kv, n_seqs);
    ssm = ggml_reshape(ctx, ssm, { d_state, d_inner, n_seqs });

    // {n_embd, n_tokens} => {n_embd, n_seq_tokens, n_seqs}
    cur = ggml_reshape(ctx, cur, { cur->ne[0], n_seq_tokens, n_seqs });

    // {n_embd, 2*d_inner} @ {n_embd, n_seq_tokens, n_seqs} => {2*d_inner, n_seq_tokens, n_seqs}
    struct ggml_tensor* xz = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_in, cur);
    // split the above in two
    // => {d_inner, n_seq_tokens, n_seqs}
    ggml_tensor* x = ggml_view(ctx, xz, { d_inner, xz->ne[1], xz->ne[2] }, { xz->nb[1], xz->nb[2] }, 0);
    ggml_tensor* z = ggml_view(ctx, xz, { d_inner, xz->ne[1], xz->ne[2] }, { xz->nb[1], xz->nb[2] }, d_inner * ggml_element_size(xz));

    // conv
    {
        // => {d_conv - 1 + n_seq_tokens, d_inner, n_seqs}
        struct ggml_tensor* conv_x = ggml_concat(ctx, conv, ggml_transpose(ctx, x), 0);

        // copy last (d_conv - 1) columns back into the state cache
        struct ggml_tensor* last_conv = ggml_view(ctx, conv_x, { d_conv - 1, d_inner, n_seqs }, { conv_x->nb[1], conv_x->nb[2] }, n_seq_tokens* (conv_x->nb[0]));

        graph->build_forward_expand(
            ggml_cpy(ctx, last_conv,
                ggml_view(ctx, conv_states_all,
                    { (d_conv - 1) * (d_inner) * (n_seqs) }, {},
                    kv_head * (d_conv - 1) * (d_inner)*ggml_element_size(conv_states_all))));

        // 1D convolution
        // The equivalent is to make a self-overlapping view of conv_x
        // over d_conv columns at each stride in the 3rd dimension,
        // then element-wise multiply that with the conv1d weight,
        // then sum the elements of each row,
        // (the last two steps are a dot product over rows (also doable with mul_mat))
        // then permute away the ne[0] dimension,
        // and then you're left with the resulting x tensor.
        // For simultaneous sequences, all sequences need to have the same length.
        x = ggml_ssm_conv(ctx, conv_x, model.layers[il].ssm_conv1d);

        // bias
        x = ggml_add(ctx, x, model.layers[il].ssm_conv1d_b);

        x = ggml_silu(ctx, x);
    }

    // ssm
    {
        // {d_inner, dt_rank + 2*d_state} @ {d_inner, n_seq_tokens, n_seqs} => {dt_rank + 2*d_state, n_seq_tokens, n_seqs}
        struct ggml_tensor* x_db = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_x, x);
        // split
        ggml_tensor* dt = ggml_view(ctx, x_db, { dt_rank, n_seq_tokens, n_seqs }, { x_db->nb[1], x_db->nb[2] }, 0);
        ggml_tensor* B = ggml_view(ctx, x_db, { d_state, n_seq_tokens, n_seqs }, { x_db->nb[1], x_db->nb[2] }, ggml_element_size(x_db) * dt_rank);
        ggml_tensor* C = ggml_view(ctx, x_db, { d_state, n_seq_tokens, n_seqs }, { x_db->nb[1], x_db->nb[2] }, ggml_element_size(x_db) * (dt_rank + d_state));

        // Some Mamba variants (e.g. FalconMamba) apply RMS norm in B, C & Dt layers
        if (ssm_dt_b_c_rms) {
            dt = ggml_rms_norm(ctx, dt, norm_rms_eps);
            B = ggml_rms_norm(ctx, B, norm_rms_eps);
            C = ggml_rms_norm(ctx, C, norm_rms_eps);
        }

        // {dt_rank, d_inner} @ {dt_rank, n_seq_tokens, n_seqs} => {d_inner, n_seq_tokens, n_seqs}
        dt = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_dt, dt);
        dt = ggml_add(ctx, dt, model.layers[il].ssm_dt_b);

        // Custom operator to optimize the parallel associative scan
        // as described in the Annex D of the Mamba paper.
        // => {d_inner, n_seq_tokens, n_seqs} and {d_state, d_inner, n_seqs}
        // The answer is wwrong, fix it
        struct ggml_tensor* y_ssm = ggml_ssm_scan(ctx, ssm, x, dt, model.layers[il].ssm_a, B, C, nullptr);

        // store last states
        graph->build_forward_expand(
            ggml_cpy(ctx,
                ggml_view(ctx, y_ssm, { d_state * d_inner * n_seqs }, {}, x->nb[3]),
                ggml_view(ctx, ssm_states_all, { d_state * d_inner * n_seqs }, {}, kv_head * d_state * d_inner * ggml_element_size(ssm_states_all))));

        ggml_tensor* y = ggml_view(ctx, y_ssm, { d_inner, n_seq_tokens, n_seqs }, { x->nb[1], x->nb[2] }, 0);

        // TODO: skip computing output earlier for unused tokens

        // {d_inner, n_seq_tokens, n_seqs} * {d_inner} => {d_inner, n_seq_tokens, n_seqs}
        y = ggml_add(ctx, y, ggml_mul(ctx, x, model.layers[il].ssm_d));
        y = ggml_mul(ctx, y, ggml_silu(ctx, ggml_cont(ctx, z)));

        // {d_inner, n_embd} @ {d_inner, n_seq_tokens, n_seqs} => {n_embd, n_seq_tokens, n_seqs}
        cur = llm_build_lora_mm(lctx, ctx, model.layers[il].ssm_out, y);
    }

    // {n_embd, n_seq_tokens, n_seqs} => {n_embd, n_tokens}
    cur = ggml_reshape(ctx, cur, { cur->ne[0], n_seq_tokens * n_seqs });
    cb(cur, "mamba_out", il);

    return cur;
}

static ggml_tensor* llm_build_rwkv6_time_mix(
    struct llama_context& lctx,
    ggml_context* ctx,
    const struct llama_layer* layer,
    ggml_tensor* cur,
    ggml_tensor* x_prev,
    ggml_tensor** wkv_state) {
    size_t n_embd = cur->ne[0];
    size_t n_seq_tokens = cur->ne[1];
    size_t n_seqs = cur->ne[2];

    size_t head_size = layer->time_mix_first->ne[0];
    size_t head_count = layer->time_mix_first->ne[1];

    size_t n_tokens = n_seqs * n_seq_tokens;

    struct ggml_tensor* sx = ggml_sub(ctx, x_prev, cur);

    sx = ggml_reshape(ctx, sx, { (int64_t)n_embd, (int64_t)n_tokens });
    cur = ggml_reshape(ctx, cur, { (int64_t)n_embd, (int64_t)n_tokens });

    struct ggml_tensor* xxx = ggml_add(ctx, ggml_mul(ctx, sx, layer->time_mix_lerp_x), cur);

    xxx = ggml_reshape(
        ctx,
        ggml_tanh(
            ctx,
            ggml_mul_mat(ctx, layer->time_mix_w1, xxx)
        ),
        { layer->time_mix_w1->ne[1] / 5, 1, 5, (int64_t)n_tokens }
    );

    xxx = ggml_cont(ctx, ggml_permute(ctx, xxx, 0, 1, 3, 2));

    xxx = ggml_mul_mat(
        ctx,
        ggml_reshape(
            ctx,
            layer->time_mix_w2,
            { layer->time_mix_w2->ne[0], layer->time_mix_w2->ne[1], 1, 5 }
        ),
        xxx
    );

    ggml_tensor* mw = ggml_view(ctx, xxx, { (int64_t)n_embd, (int64_t)n_tokens }, { xxx->nb[1]}, 0);
    ggml_tensor* mk = ggml_view(ctx, xxx, { (int64_t)n_embd, (int64_t)n_tokens }, { xxx->nb[1]}, n_embd * n_tokens * sizeof(float));
    ggml_tensor* mv = ggml_view(ctx, xxx, { (int64_t)n_embd, (int64_t)n_tokens }, { xxx->nb[1]}, n_embd * n_tokens * 2 * sizeof(float));
    ggml_tensor* mr = ggml_view(ctx, xxx, { (int64_t)n_embd, (int64_t)n_tokens }, { xxx->nb[1]}, n_embd * n_tokens * 3 * sizeof(float));
    ggml_tensor* mg = ggml_view(ctx, xxx, { (int64_t)n_embd, (int64_t)n_tokens }, { xxx->nb[1]}, n_embd * n_tokens * 4 * sizeof(float));

    ggml_tensor* xw = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_add(ctx, mw, layer->time_mix_lerp_w),
            sx
        ),
        cur
    );

    struct ggml_tensor* xk = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_add(ctx, mk, layer->time_mix_lerp_k),
            sx
        ),
        cur
    );

    struct ggml_tensor* xv = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_add(ctx, mv, layer->time_mix_lerp_v),
            sx
        ),
        cur
    );

    struct ggml_tensor* xr = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_add(ctx, mr, layer->time_mix_lerp_r),
            sx
        ),
        cur
    );

    struct ggml_tensor* xg = ggml_add(
        ctx,
        ggml_mul(
            ctx,
            ggml_add(ctx, mg, layer->time_mix_lerp_g),
            sx
        ),
        cur
    );

    struct ggml_tensor* r = ggml_reshape(ctx, llm_build_lora_mm(lctx, ctx, layer->time_mix_receptance, xr), { (int64_t)head_size, 1, (int64_t)head_count, (int64_t)n_tokens });
    struct ggml_tensor* k = ggml_reshape(ctx, llm_build_lora_mm(lctx, ctx, layer->time_mix_key, xk), { 1, (int64_t)head_size, (int64_t)head_count, (int64_t)n_tokens });
    struct ggml_tensor* v = ggml_reshape(ctx, llm_build_lora_mm(lctx, ctx, layer->time_mix_value, xv), { (int64_t)head_size, 1, (int64_t)head_count, (int64_t)n_tokens });
    struct ggml_tensor* g = ggml_silu(
        ctx,
        llm_build_lora_mm(lctx, ctx, layer->time_mix_gate, xg)
    );

    struct ggml_tensor* w = ggml_mul_mat(
        ctx,
        layer->time_mix_decay_w2,
        ggml_tanh(
            ctx,
            ggml_mul_mat(ctx, layer->time_mix_decay_w1, xw)
        )
    );

    w = ggml_add(ctx, w, ggml_reshape(ctx, layer->time_mix_decay, { (int64_t)n_embd }));
    w = ggml_exp(ctx, ggml_neg(ctx, ggml_exp(ctx, w)));
    w = ggml_reshape(ctx, w, { 1, (int64_t)head_size, (int64_t)head_count, (int64_t)n_tokens });

    k = ggml_transpose(ctx, k);
    v = ggml_transpose(ctx, v);
    r = ggml_transpose(ctx, r);

    struct ggml_tensor* wkv_output = ggml_rwkv_wkv6(ctx, k, v, r, layer->time_mix_first, w, *wkv_state);
    cur = ggml_view(ctx, wkv_output, { (int64_t)(n_embd * n_tokens) }, {}, 0);
    *wkv_state = ggml_view(ctx, wkv_output, { (int64_t)(n_embd * head_size * n_seqs) }, {}, n_embd * n_tokens * sizeof(float));

    // group norm with head_count groups
    cur = ggml_reshape(ctx, cur, { (int64_t)(n_embd / head_count), (int64_t)head_count, (int64_t)n_tokens });
    cur = ggml_norm(ctx, cur, 64e-5f);

    // Convert back to regular vectors.
    cur = ggml_reshape(ctx, cur, { (int64_t)n_embd, (int64_t)n_tokens });
    cur = ggml_add(ctx, ggml_mul(ctx, cur, layer->time_mix_ln), layer->time_mix_ln_b);

    cur = ggml_mul(ctx, cur, g);
    cur = llm_build_lora_mm(lctx, ctx, layer->time_mix_output, cur);

    return ggml_reshape(ctx, cur, { (int64_t)n_embd, (int64_t)n_seq_tokens, (int64_t)n_seqs });
}

static ggml_tensor* llm_build_rwkv6_channel_mix(
    struct llama_context& lctx,
    ggml_context* ctx,
    const struct llama_layer* layer,
    ggml_tensor* cur,
    ggml_tensor* x_prev) {
    struct ggml_tensor* sx = ggml_sub(ctx, x_prev, cur);
    struct ggml_tensor* xk = ggml_add(ctx, ggml_mul(ctx, sx, layer->channel_mix_lerp_k), cur);
    struct ggml_tensor* xr = ggml_add(ctx, ggml_mul(ctx, sx, layer->channel_mix_lerp_r), cur);

    struct ggml_tensor* r = ggml_sigmoid(ctx, llm_build_lora_mm(lctx, ctx, layer->channel_mix_receptance, xr));
    struct ggml_tensor* k = ggml_sqr(
        ctx,
        ggml_relu(
            ctx,
            llm_build_lora_mm(lctx, ctx, layer->channel_mix_key, xk)
        )
    );

    return ggml_mul(ctx, r, llm_build_lora_mm(lctx, ctx, layer->channel_mix_value, k));
}

static ggml_tensor* llm_build_norm(
    ggml_context* ctx,
    ggml_tensor* cur,
    const llama_hparams& hparams,
    ggml_tensor* mw,
    ggml_tensor* mb,
    llm_norm_type   type,
    const llm_build_cb& cb,
    int   il) {
    switch (type) {
    case LLM_NORM:       cur = ggml_norm(ctx, cur, hparams.f_norm_eps);     break;
    case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps); break;
    case LLM_NORM_GROUP:
    {
        cur = ggml_reshape(ctx, cur, { cur->ne[0], 1, cur->ne[1] });
        cur = ggml_group_norm(ctx, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
        cur = ggml_reshape(ctx, cur, { cur->ne[0], cur->ne[2] });
    } break;
    }

    if (mw || mb) {
        cb(cur, "norm", il);
    }

    if (mw) {
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cb(cur, "norm_w", il);
        }
    }

    if (mb) {
        cur = ggml_add(ctx, cur, mb);
    }

    return cur;
}

struct llm_build_context {
    const llama_model& model;
    llama_context& lctx;
    const llama_hparams& hparams;
    const llama_cparams& cparams;
    const llama_ubatch& ubatch;
    const llama_kv_cache& kv_self;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int32_t n_tokens;
    const int32_t n_kv;     // size of KV cache to consider (n_kv <= kv_self.size)
    const int32_t n_outputs;
    const int32_t n_outputs_enc;
    const int32_t kv_head;  // index of where we store new KV data in the cache
    const int32_t n_ctx_orig;

    const bool flash_attn;

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    const llm_build_cb& cb;

    ggml_context* ctx0 = nullptr;

    // TODO: consider making the entire interface noexcept
    llm_build_context(
        llama_context& lctx,
        const llama_ubatch& ubatch,
        const llm_build_cb& cb,
        bool   worst_case) :
        model(lctx.model),
        lctx(lctx),
        hparams(model.hparams),
        cparams(lctx.cparams),
        ubatch(ubatch),
        kv_self(lctx.kv_self),
        n_embd(hparams.n_embd),
        n_layer(hparams.n_layer),
        n_rot(hparams.n_rot),
        n_ctx(cparams.n_ctx),
        n_head(hparams.n_head()),
        n_head_kv(hparams.n_head_kv()),
        n_embd_head_k(hparams.n_embd_head_k),
        n_embd_k_gqa(hparams.n_embd_k_gqa()),
        n_embd_head_v(hparams.n_embd_head_v),
        n_embd_v_gqa(hparams.n_embd_v_gqa()),
        n_expert(hparams.n_expert),
        n_expert_used(hparams.n_expert_used),
        freq_base(cparams.rope_freq_base),
        freq_scale(cparams.rope_freq_scale),
        ext_factor(cparams.yarn_ext_factor),
        attn_factor(cparams.yarn_attn_factor),
        beta_fast(cparams.yarn_beta_fast),
        beta_slow(cparams.yarn_beta_slow),
        norm_eps(hparams.f_norm_eps),
        norm_rms_eps(hparams.f_norm_rms_eps),
        n_tokens(ubatch.n_tokens),
        n_kv(worst_case ? kv_self.size : kv_self.n),
        n_outputs(worst_case ? n_tokens : lctx.n_outputs),
        n_outputs_enc(worst_case ? n_tokens : lctx.embd_enc.size() / hparams.n_embd),
        kv_head(worst_case ? (kv_self.recurrent ? 0 : kv_self.size - n_tokens) : kv_self.head),
        n_ctx_orig(cparams.n_ctx_orig_yarn),
        flash_attn(cparams.flash_attn),
        pooling_type(cparams.pooling_type),
        rope_type(hparams.rope_type),
        cb(cb) {
        // all initializations should be done in init()
    }

    void init() {
        ctx0 = std::make_unique<ggml_context>().release();

        lctx.inp_tokens = nullptr;
        lctx.inp_embd = nullptr;
        lctx.inp_pos = nullptr;
        lctx.inp_out_ids = nullptr;
        lctx.inp_KQ_mask = nullptr;
        lctx.inp_KQ_mask_swa = nullptr;
        lctx.inp_K_shift = nullptr;
        lctx.inp_mean = nullptr;
        lctx.inp_cls = nullptr;
        lctx.inp_s_copy = nullptr;
        lctx.inp_s_mask = nullptr;
        lctx.inp_s_seq = nullptr;
        lctx.inp_pos_bucket = nullptr;
        lctx.inp_embd_enc = nullptr;
        lctx.inp_KQ_mask_cross = nullptr;
    }

    void free() {
        delete ctx0;
        ctx0 = nullptr;
    }

    ggml_cgraph* build_k_shift() {
        ggml_cgraph* gf = new ggml_cgraph;

        GGML_ASSERT(kv_self.size == n_ctx);

        lctx.inp_K_shift = ctx0->create(GGML_TYPE_I32, { n_ctx });
        cb(lctx.inp_K_shift, "K_shift", -1);
        lctx.inp_K_shift->set_flag(GGML_TENSOR_FLAG_INPUT);

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
            struct ggml_tensor* rope_factors = build_rope_factors(il);
            struct ggml_tensor* k =
                ggml_view(ctx0, kv_self.k_l[il],
                    { n_embd_head_k, n_head_kv, n_ctx },
                    { ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa) },
                    0);

            struct ggml_tensor* tmp;
            if (ggml_is_quantized(k->type)) {
                // dequantize to f32 -> RoPE -> quantize back
                tmp = ggml_cast(ctx0, k, GGML_TYPE_F32);
                cb(tmp, "K_f32", il);
                for (auto& backend : lctx.backends) {
                    // Figure out which backend KV cache belongs to
                    if (backend->supports_buft(kv_self.k_l[il]->buffer->get_type())) {
                        lctx.sched->set_tensor_backend(tmp, backend.get());
                        break;
                    }
                }
                tmp = ggml_rope_ext_inplace(ctx0, tmp,
                    lctx.inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(tmp, "K_shifted_f32", il);
                tmp = ggml_cpy(ctx0, tmp, k);
            }
            else {
                // we rotate only the first n_rot dimensions
                tmp = ggml_rope_ext_inplace(ctx0, k,
                    lctx.inp_K_shift, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            }
            cb(tmp, "K_shifted", il);
            gf->build_forward_expand(tmp);
        }

        return gf;
    }

    ggml_cgraph* build_defrag(const std::vector<uint32_t>& ids) {
        ggml_cgraph* gf = new ggml_cgraph;

        for (uint32_t i = 0; i < ids.size(); ++i) {
            const uint32_t id = ids[i];

            if (i == id || id == ids.size()) {
                continue;
            }

            uint32_t nm = 1;

            while (i + nm < ids.size() && ids[i + nm] == id + nm) {
                nm++;
            }

            for (int il = 0; il < n_layer; ++il) {
                const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
                const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

                ggml_tensor* view_k_src = ggml_view(ctx0, kv_self.k_l[il],
                    { n_embd_k_gqa, nm },
                    { ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa) },
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa * i));

                ggml_tensor* view_k_dst = ggml_view(ctx0, kv_self.k_l[il],
                    { n_embd_k_gqa, nm },
                    { ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa) },
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa * id));

                ggml_tensor* view_v_src;
                ggml_tensor* view_v_dst;

                if (flash_attn) {
                    // NOTE: the V cache is not transposed when using flash attention
                    view_v_src = ggml_view(ctx0, kv_self.v_l[il],
                        { n_embd_v_gqa, nm },
                        { ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa) },
                        ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa * i));

                    view_v_dst = ggml_view(ctx0, kv_self.v_l[il],
                        { n_embd_v_gqa, nm },
                        { ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa) },
                        ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa * id));
                }
                else {
                    view_v_src = ggml_view(ctx0, kv_self.v_l[il],
                        { nm, n_embd_v_gqa },
                        { ggml_row_size(kv_self.v_l[il]->type, kv_self.size) },
                        ggml_row_size(kv_self.v_l[il]->type, i));

                    view_v_dst = ggml_view(ctx0, kv_self.v_l[il],
                        { nm, n_embd_v_gqa },
                        { ggml_row_size(kv_self.v_l[il]->type, kv_self.size) },
                        ggml_row_size(kv_self.v_l[il]->type, id));
                }

                gf->build_forward_expand(ggml_cpy(ctx0, view_k_src, view_k_dst));
                gf->build_forward_expand(ggml_cpy(ctx0, view_v_src, view_v_dst));
            }

            i += nm - 1;
        }

        //LLAMA_LOG_INFO("gf->n_nodes = %d\n", gf->n_nodes);

        return gf;
    }

    ggml_tensor* build_inp_pos() {
        lctx.inp_pos = ctx0->create(GGML_TYPE_I32, { n_tokens });
        cb(lctx.inp_pos, "inp_pos", -1);
        lctx.inp_pos->set_flag(GGML_TENSOR_FLAG_INPUT);
        return lctx.inp_pos;
    }

    ggml_tensor* build_rope_factors(int il) {
        // choose long/short freq factors based on the context size
        const auto n_ctx_pre_seq = cparams.n_ctx / cparams.n_seq_max;

        if (model.layers[il].rope_freqs != nullptr) {
            return model.layers[il].rope_freqs;
        }

        if (n_ctx_pre_seq > hparams.n_ctx_orig_yarn) {
            return model.layers[il].rope_long;
        }

        return model.layers[il].rope_short;
    }

    ggml_tensor* build_inp_out_ids() {
        lctx.inp_out_ids = ctx0->create(GGML_TYPE_I32, { n_outputs });
        cb(lctx.inp_out_ids, "inp_out_ids", -1);
        lctx.inp_out_ids->set_flag(GGML_TENSOR_FLAG_INPUT);
        return lctx.inp_out_ids;
    }

    ggml_tensor* build_inp_KQ_mask(bool causal = true) {
        lctx.inp_KQ_mask = causal
            ? ctx0->create(GGML_TYPE_F32, { n_kv, (int64_t)GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) })
            : ctx0->create(GGML_TYPE_F32, { n_tokens, (int64_t)GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) });
        cb(lctx.inp_KQ_mask, "KQ_mask", -1);
        lctx.inp_KQ_mask->set_flag(GGML_TENSOR_FLAG_INPUT);

        return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask, GGML_TYPE_F16) : lctx.inp_KQ_mask;
    }

    ggml_tensor* build_inp_KQ_mask_swa(bool causal = true) {
        GGML_ASSERT(hparams.n_swa > 0);

        lctx.inp_KQ_mask_swa = causal
            ? ctx0->create(GGML_TYPE_F32, { n_kv, (int64_t)GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) })
            : ctx0->create(GGML_TYPE_F32, { n_tokens, (int64_t)GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) });
        cb(lctx.inp_KQ_mask_swa, "KQ_mask_swa", -1);
        lctx.inp_KQ_mask_swa->set_flag(GGML_TENSOR_FLAG_INPUT);

        return flash_attn ? ggml_cast(ctx0, lctx.inp_KQ_mask_swa, GGML_TYPE_F16) : lctx.inp_KQ_mask_swa;
    }

    ggml_tensor* build_inp_mean() {
        lctx.inp_mean = ctx0->create(GGML_TYPE_F32, { n_tokens, n_tokens });
        cb(lctx.inp_mean, "inp_mean", -1);
        lctx.inp_mean->set_flag(GGML_TENSOR_FLAG_INPUT);
        return lctx.inp_mean;
    }

    ggml_tensor* build_inp_cls() {
        lctx.inp_cls = ctx0->create(GGML_TYPE_I32, { n_tokens });
        cb(lctx.inp_cls, "inp_cls", -1);
        lctx.inp_cls->set_flag(GGML_TENSOR_FLAG_INPUT);
        return lctx.inp_cls;
    }

    ggml_tensor* build_inp_s_copy() {
        lctx.inp_s_copy = ctx0->create(GGML_TYPE_I32, { n_kv });
        cb(lctx.inp_s_copy, "inp_s_copy", -1);
        lctx.inp_s_copy->set_flag(GGML_TENSOR_FLAG_INPUT);
        return lctx.inp_s_copy;
    }

    ggml_tensor* build_inp_s_mask() {
        lctx.inp_s_mask = ctx0->create(GGML_TYPE_F32, { 1, n_kv });
        cb(lctx.inp_s_mask, "inp_s_mask", -1);
        lctx.inp_s_mask->set_flag(GGML_TENSOR_FLAG_INPUT);
        return lctx.inp_s_mask;
    }

    ggml_cgraph* append_pooling(ggml_cgraph* gf) {
        // find result_norm tensor for input
        struct ggml_tensor* inp = nullptr;
        for (int i = gf->nodes.size() - 1; i >= 0; --i) {
            inp = gf->nodes[i];
            if (inp->get_name() == "result_norm" || inp->get_name() == "result_embd") {
                break;
            }
            else {
                inp = nullptr;
            }
        }
        GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd tensor");

        struct ggml_tensor* cur;

        switch (pooling_type) {
        case LLAMA_POOLING_TYPE_NONE:
        {
            cur = inp;
        } break;
        case LLAMA_POOLING_TYPE_MEAN:
        {
            struct ggml_tensor* inp_mean = build_inp_mean();
            cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
        } break;
        case LLAMA_POOLING_TYPE_CLS:
        case LLAMA_POOLING_TYPE_LAST:
        {
            struct ggml_tensor* inp_cls = build_inp_cls();
            cur = ggml_get_rows(ctx0, inp, inp_cls);
        } break;
        case LLAMA_POOLING_TYPE_RANK:
        {
            struct ggml_tensor* inp_cls = build_inp_cls();
            inp = ggml_get_rows(ctx0, inp, inp_cls);

            // classification head
            // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
            GGML_ASSERT(model.cls != nullptr);
            GGML_ASSERT(model.cls_b != nullptr);

            cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.cls, inp), model.cls_b);
            cur = ggml_tanh(ctx0, cur);

            // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
            // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
            if (model.cls_out) {
                GGML_ASSERT(model.cls_out_b != nullptr);

                cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.cls_out, cur), model.cls_out_b);
            }
        } break;
        default:
        {
            GGML_ABORT("unknown pooling type");
        }
        }

        cb(cur, "result_embd_pooled", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_tensor* llm_build_pos_bucket(bool causal) {
        if (causal) {
            lctx.inp_pos_bucket = ctx0->create(GGML_TYPE_I32, { n_kv, n_tokens });
        }
        else {
            lctx.inp_pos_bucket = ctx0->create(GGML_TYPE_I32, { n_tokens, n_tokens });
        }

        lctx.inp_pos_bucket->set_flag(GGML_TENSOR_FLAG_INPUT);
        cb(lctx.inp_pos_bucket, "pos_bucket", -1);

        return lctx.inp_pos_bucket;
    }

    ggml_tensor* llm_build_pos_bias(ggml_tensor* pos_bucket, ggml_tensor* attn_rel_b) {
        ggml_tensor* pos_bucket_1d = ggml_view(ctx0, pos_bucket, { pos_bucket->ne[0] * pos_bucket->ne[1] }, {}, 0);
        cb(pos_bucket_1d, "pos_bucket_1d", -1);

        ggml_tensor* pos_bias = ggml_get_rows(ctx0, attn_rel_b, pos_bucket_1d);
        cb(pos_bias, "pos_bias", -1);

        pos_bias = ggml_view(ctx0, pos_bias, { pos_bias->ne[0], lctx.inp_pos_bucket->ne[0], lctx.inp_pos_bucket->ne[1] }, { ggml_element_size(pos_bias) * pos_bias->ne[0], ggml_element_size(pos_bias) * pos_bias->ne[0] * lctx.inp_pos_bucket->ne[0] }, 0);
        cb(pos_bias, "pos_bias", -1);

        pos_bias = ggml_permute(ctx0, pos_bias, 2, 0, 1, 3);
        cb(pos_bias, "pos_bias", -1);

        pos_bias = ggml_cont(ctx0, pos_bias);
        cb(pos_bias, "pos_bias", -1);

        return pos_bias;
    }

    ggml_tensor* llm_build_inp_embd_enc() {
        const int64_t n_embd = hparams.n_embd;
        lctx.inp_embd_enc = ctx0->create(GGML_TYPE_F32, { n_embd, n_outputs_enc });
        lctx.inp_embd_enc->set_flag(GGML_TENSOR_FLAG_INPUT);
        cb(lctx.inp_embd_enc, "embd_enc", -1);
        return lctx.inp_embd_enc;
    }

    ggml_tensor* llm_build_inp_KQ_mask_cross() {
        lctx.inp_KQ_mask_cross = ctx0->create(GGML_TYPE_F32, { n_outputs_enc, (int64_t)GGML_PAD(n_tokens, GGML_KQ_MASK_PAD) });
        lctx.inp_KQ_mask_cross->set_flag(GGML_TENSOR_FLAG_INPUT);
        cb(lctx.inp_KQ_mask_cross, "KQ_mask_cross", -1);
        return lctx.inp_KQ_mask_cross;
    }

    ggml_cgraph* build_llama() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor* rope_factors = build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }
            else {
                // MoE branch
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    false, 0.0,
                    cb, il);
                cb(cur, "ffn_moe_out", il);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        // For Granite architecture
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_deci() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head = hparams.n_head(il);

            if (n_head == 0) {
                // attention-free layer of Llama-3_1-Nemotron-51B
                cur = inpL;
            }
            else {
                // norm
                cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "attn_norm", il);
            }

            if (n_head > 0 && n_head_kv == 0) {
                // "linear attention" of Llama-3_1-Nemotron-51B
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
                cb(cur, "wo", il);
            }
            else if (n_head > 0) {
                // self-attention
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor* rope_factors = build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            // modified to support attention-free layer of Llama-3_1-Nemotron-51B
            struct ggml_tensor* ffn_inp = cur;
            if (n_head > 0) {
                ffn_inp = ggml_add(ctx0, cur, inpSA);
                cb(ffn_inp, "ffn_inp", il);
            }

            // feed-forward network
            if (model.layers[il].ffn_gate_inp == nullptr) {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // For Granite architecture
            if (hparams.f_residual_scale) {
                cur = ggml_scale(ctx0, cur, hparams.f_residual_scale);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        // For Granite architecture
        if (hparams.f_logit_scale) {
            cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale);
        }

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_baichuan() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = model.type == MODEL_7B ? build_inp_pos() : nullptr;

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                switch (model.type) {
                case MODEL_7B:
                    Qcur = ggml_rope_ext(
                        ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                    );
                    Kcur = ggml_rope_ext(
                        ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                        n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                        ext_factor, attn_factor, beta_fast, beta_slow
                    );
                    break;
                case MODEL_13B:
                    Qcur = ggml_reshape(ctx0, Qcur, { n_embd / n_head, n_head, n_tokens });
                    Kcur = ggml_reshape(ctx0, Kcur, { n_embd / n_head, n_head, n_tokens });
                    break;
                default:
                    GGML_ABORT("fatal error");
                }
                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_xverse() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, NULL, LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_falcon() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* attn_norm;

            attn_norm = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                if (model.layers[il].attn_norm_2) {
                    // Falcon-40B
                    cur = llm_build_norm(ctx0, inpL, hparams,
                        model.layers[il].attn_norm_2,
                        model.layers[il].attn_norm_2_b,
                        LLM_NORM, cb, il);
                    cb(cur, "attn_norm_2", il);
                }
                else {
                    cur = attn_norm;
                }

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
                attn_norm = ggml_get_rows(ctx0, attn_norm, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = cur;

            // feed forward
            {
                cur = llm_build_ffn(ctx0, lctx, attn_norm, // !! use the attn norm, not the result
                    model.layers[il].ffn_up, NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = ggml_add(ctx0, cur, inpL);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_grok() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // multiply by embedding_multiplier_scale of 78.38367176906169
        inpL = ggml_scale(ctx0, inpL, 78.38367176906169f);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);


            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Grok
            // if attn_out_norm is present then apply it before adding the input
            if (model.layers[il].attn_out_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                    model.layers[il].attn_out_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "attn_out_norm", il);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                n_expert, n_expert_used,
                LLM_FFN_GELU, true,
                false, 0.0,
                cb, il);
            cb(cur, "ffn_moe_out", il);

            // Grok
            // if layer_out_norm is present then apply it before adding the input
            // Idea: maybe ffn_out_norm is a better name
            if (model.layers[il].layer_out_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                    model.layers[il].layer_out_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "layer_out_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        // Grok
        // multiply logits by output_multiplier_scale of 0.5773502691896257

        cur = ggml_scale(ctx0, cur, 0.5773502691896257f);

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_dbrx() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = nullptr;
                struct ggml_tensor* Kcur = nullptr;
                struct ggml_tensor* Vcur = nullptr;

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                cb(cur, "wqkv_clamped", il);

                Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            // MoE branch
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].attn_out_norm, NULL,
                LLM_NORM, cb, il);
            cb(cur, "attn_out_norm", il);

            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                false, 0.0,
                cb, il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_starcoder() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        struct ggml_tensor* pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_refact() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });
                cb(Kcur, "Kcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                cb(Qcur, "Qcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_bert() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();

        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;
        struct ggml_tensor* inp_pos = nullptr;

        if (model.arch != LLM_ARCH_JINA_BERT_V2) {
            inp_pos = build_inp_pos();
        }

        // construct input embeddings (token, type, position)
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // token types are hardcoded to zero ("Sentence A")
        ggml_tensor* type_row0 = ggml_view(ctx0, model.type_embd, { n_embd }, {}, 0);
        inpL = ggml_add(ctx0, inpL, type_row0);
        if (model.arch == LLM_ARCH_BERT) {
            inpL = ggml_add(ctx0, ggml_get_rows(ctx0, model.pos_embd, inp_pos), inpL);
        }
        cb(inpL, "inp_embd", -1);

        // embed layer norm
        inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);
        cb(inpL, "inp_norm", -1);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask(false);

        // iterate layers
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* cur = inpL;

            struct ggml_tensor* Qcur;
            struct ggml_tensor* Kcur;
            struct ggml_tensor* Vcur;

            // self-attention
            if (model.arch == LLM_ARCH_BERT || model.arch == LLM_ARCH_JINA_BERT_V2) {
                Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur), model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = llm_build_norm(ctx0, Qcur, hparams,
                        model.layers[il].attn_q_norm,
                        model.layers[il].attn_q_norm_b,
                        LLM_NORM, cb, il);
                }

                Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur), model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                if (model.layers[il].attn_k_norm) {
                    Kcur = llm_build_norm(ctx0, Kcur, hparams,
                        model.layers[il].attn_k_norm,
                        model.layers[il].attn_k_norm_b,
                        LLM_NORM, cb, il);
                }
                Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur), model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });
            }
            else {
                // compute Q and K and RoPE them
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);
            }

            struct ggml_tensor* q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            struct ggml_tensor* k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

            struct ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
            cb(kq, "kq", il);

            kq = ggml_soft_max(ctx0, kq, KQ_mask, 1.0f / sqrtf(float(n_embd_head)), hparams.f_max_alibi_bias);
            cb(kq, "kq_soft_max", il);

            struct ggml_tensor* v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape(ctx0, Vcur, { n_embd_gqa, n_tokens })));
            cb(v, "v", il);

            struct ggml_tensor* kqv = ggml_mul_mat(ctx0, ggml_reshape(ctx0, v, { n_tokens, n_embd_head, n_head_kv }), kq);
            cb(kqv, "kqv", il);

            struct ggml_tensor* kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
            cb(kqv_merged, "kqv_merged", il);

            cur = ggml_cont(ctx0, kqv_merged, { n_embd_gqa, n_tokens });
            cb(cur, "kqv_merged_cont", il);

            gf->build_forward_expand(cur);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
            if (model.layers[il].bo) {
                cb(cur, "kqv_wo", il);
            }

            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
            cb(cur, "kqv_out", il);

            if (il == n_layer - 1 && pooling_type == LLAMA_POOLING_TYPE_NONE) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // re-add the layer input
            cur = ggml_add(ctx0, cur, inpL);

            // attention layer norm
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b, LLM_NORM, cb, il);

            if (model.layers[il].attn_norm_2 != nullptr) {
                cur = ggml_add(ctx0, cur, inpL); // re-add the layer input
                cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].attn_norm_2, model.layers[il].attn_norm_2_b, LLM_NORM, cb, il);
            }

            struct ggml_tensor* ffn_inp = cur;
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (model.arch == LLM_ARCH_BERT) {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            }
            else if (model.arch == LLM_ARCH_JINA_BERT_V2) {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
            }
            else {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            }
            cb(cur, "ffn_out", il);

            // attentions bypass the intermediate layer
            cur = ggml_add(ctx0, cur, ffn_inp);

            // output layer norm
            cur = llm_build_norm(ctx0, cur, hparams, model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b, LLM_NORM, cb, il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cb(cur, "result_embd", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_bloom() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        inpL = llm_build_norm(ctx0, inpL, hparams,
            model.tok_norm,
            model.tok_norm_b,
            LLM_NORM, cb, -1);
        cb(inpL, "inp_norm", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_mpt() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* pos;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        if (model.pos_embd) {
            // inp_pos - contains the positions
            struct ggml_tensor* inp_pos = build_inp_pos();
            pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
            cb(pos, "pos_embd", -1);

            inpL = ggml_add(ctx0, inpL, pos);
            cb(inpL, "inpL", -1);
        }

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* attn_norm;

            attn_norm = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(attn_norm, "attn_norm", il);

            // self-attention
            {
                cur = attn_norm;

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                if (model.layers[il].bqkv) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);
                }

                if (hparams.f_clamp_kqv > 0.0f) {
                    cur = ggml_clamp(ctx0, cur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(cur, "wqkv_clamped", il);
                }

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                // Q/K Layernorm
                if (model.layers[il].attn_q_norm) {
                    Qcur = llm_build_norm(ctx0, Qcur, hparams,
                        model.layers[il].attn_q_norm,
                        model.layers[il].attn_q_norm_b,
                        LLM_NORM, cb, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = llm_build_norm(ctx0, Kcur, hparams,
                        model.layers[il].attn_k_norm,
                        model.layers[il].attn_k_norm_b,
                        LLM_NORM, cb, il);
                    cb(Kcur, "Kcur", il);

                    Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                    Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                    cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
                }
                else {
                    Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });

                    cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                        model.layers[il].wo, model.layers[il].bo,
                        Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
                }
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed forward
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    model.layers[il].ffn_act,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_stablelm() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {


            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor* inpSA = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                cb(Qcur, "Qcur", il);
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });
                cb(Kcur, "Kcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = llm_build_norm(ctx0, Qcur, hparams,
                        model.layers[il].attn_q_norm,
                        NULL,
                        LLM_NORM, cb, il);
                    cb(Qcur, "Qcur", il);
                }
                if (model.layers[il].attn_k_norm) {
                    Kcur = llm_build_norm(ctx0, Kcur, hparams,
                        model.layers[il].attn_k_norm,
                        NULL,
                        LLM_NORM, cb, il);
                    cb(Kcur, "Kcur", il);
                }


                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                if (model.layers[il].ffn_norm) {
                    cur = llm_build_norm(ctx0, ffn_inp, hparams,
                        model.layers[il].ffn_norm,
                        model.layers[il].ffn_norm_b,
                        LLM_NORM, cb, il);
                    cb(cur, "ffn_norm", il);
                }
                else {
                    // parallel residual
                    cur = inpSA;
                }
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_qwen() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 2 * sizeof(float) * (n_embd)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_qwen2() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_qwen2vl() {
        ggml_cgraph* gf = new ggml_cgraph;
        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        lctx.inp_pos = ctx0->create(GGML_TYPE_I32, { n_tokens * 4 });
        cb(lctx.inp_pos, "inp_pos", -1);
        lctx.inp_pos->set_flag(GGML_TENSOR_FLAG_INPUT);
        struct ggml_tensor* inp_pos = lctx.inp_pos;

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();
        int sections[4];
        std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_multi(
                    ctx0,
                    ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_multi(
                    ctx0,
                    ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_qwen2moe() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            ggml_tensor* moe_out =
                llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, false,
                    false, 0.0,
                    cb, il);
            cb(cur, "ffn_moe_out", il);

            // FFN shared expert
            {
                ggml_tensor* cur_gate_inp = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_gate_inp_shexp, cur);
                cb(cur_gate_inp, "ffn_shexp_gate_inp", il);

                // sigmoid
                ggml_tensor* cur_gate = ggml_div(ctx0, ggml_silu(ctx0, cur_gate_inp), cur_gate_inp);
                cb(cur_gate, "ffn_shexp_gate", il);

                ggml_tensor* cur_ffn = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up_shexp, NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur_ffn, "ffn_shexp", il);

                ggml_tensor* ffn_shexp_out = ggml_mul(ctx0, cur_ffn, cur_gate);
                cb(ffn_shexp_out, "ffn_shexp_out", il);

                moe_out = ggml_add(ctx0, moe_out, ffn_shexp_out);
                cb(moe_out, "ffn_out", il);

                cur = moe_out;
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_phi2() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* attn_norm_output;
        struct ggml_tensor* ffn_output;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            attn_norm_output = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(attn_norm_output, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = nullptr;
                struct ggml_tensor* Kcur = nullptr;
                struct ggml_tensor* Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                    cb(cur, "bqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));
                }
                else {
                    Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                // with phi2, we scale the Q to avoid precision issues
                // ref: https://github.com/ml-explore/mlx-examples/blob/08e862336ade809bc37d1035f94b359e7d1a5152/phi2/phi2.py#L64-L66
                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
                attn_norm_output = ggml_get_rows(ctx0, attn_norm_output, inp_out_ids);
            }

            // FF
            {
                ffn_output = llm_build_ffn(ctx0, lctx, attn_norm_output,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(ffn_output, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_output);
            cur = ggml_add(ctx0, cur, inpL);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output_no_bias", -1);

        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_output", -1);
        gf->build_forward_expand(cur);
        return gf;
    }

    ggml_cgraph* build_phi3() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = nullptr;
        if (hparams.n_swa == 0) {
            // Phi-4 doesn't use sliding window attention
            KQ_mask = build_inp_KQ_mask();
        }
        else {
            KQ_mask = build_inp_KQ_mask_swa();
        }

        for (int il = 0; il < n_layer; ++il) {
            auto residual = inpL;

            // self-attention
            {
                // rope freq factors for 128k context
                struct ggml_tensor* rope_factors = build_rope_factors(il);

                struct ggml_tensor* attn_norm_output = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm,
                    NULL,
                    LLM_NORM_RMS, cb, il);
                cb(attn_norm_output, "attn_norm", il);

                struct ggml_tensor* Qcur = nullptr;
                struct ggml_tensor* Kcur = nullptr;
                struct ggml_tensor* Vcur = nullptr;

                if (model.layers[il].wqkv) {
                    cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, attn_norm_output);
                    cb(cur, "wqkv", il);

                    Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                    Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                    Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));
                }
                else {
                    Qcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, attn_norm_output), model.layers[il].bq);
                    Kcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, attn_norm_output), model.layers[il].bk);
                    Vcur = ggml_add(ctx0, llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, attn_norm_output), model.layers[il].bv);
                }

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head)));
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
            }

            cur = ggml_add(ctx0, cur, residual);
            residual = cur;

            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            // FF
            // special-case: the up and gate tensors are merged into a single tensor
            // TOOD: support into llm_build_ffn
            {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, residual, cur);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }


    ggml_cgraph* build_plamo() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            struct ggml_tensor* attention_norm = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_rot, n_head, n_tokens }), inp_pos, nullptr,
                    n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_rot, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_embd_head, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }
            struct ggml_tensor* sa_out = cur;

            cur = attention_norm;

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                sa_out = ggml_get_rows(ctx0, sa_out, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // feed-forward network
            {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cur = ggml_add(ctx0, cur, inpL);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_gpt2() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* pos;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        pos = ggml_get_rows(ctx0, model.pos_embd, inp_pos);
        cb(pos, "pos_embd", -1);

        inpL = ggml_add(ctx0, inpL, pos);
        cb(inpL, "inpL", -1);

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_codeshell() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* tmpq = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* tmpk = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(tmpq, "tmpq", il);
                cb(tmpk, "tmpk", il);
                cb(Vcur, "Vcur", il);

                struct ggml_tensor* Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, tmpq, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, tmpk, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_orion() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                // if (model.layers[il].bq) {
                //     Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                //     cb(Qcur, "Qcur", il);
                // }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                // if (model.layers[il].bk) {
                //     Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                //     cb(Kcur, "Kcur", il);
                // }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                // if (model.layers[il].bv) {
                //     Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                //     cb(Vcur, "Vcur", il);
                // }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_internlm2() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_minicpm3() {
        ggml_cgraph* gf = new ggml_cgraph;

        //TODO: if the model varies, these parameters need to be read from the model
        const int64_t n_embd_base = 256;
        const float scale_embd = 12.0f;
        const float scale_depth = 1.4f;
        const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // scale the input embeddings
        inpL = ggml_scale(ctx0, inpL, scale_embd);
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            struct ggml_tensor* rope_factors = build_rope_factors(il);
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                struct ggml_tensor* q = NULL;
                // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                cb(q, "q", il);

                q = llm_build_norm(ctx0, q, hparams,
                    model.layers[il].attn_q_a_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(q, "q", il);

                // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                cb(q, "q", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor* q_nope = ggml_view(ctx0, q, { n_embd_head_qk_nope, n_head, n_tokens },
                    { ggml_row_size(q->type, hparams.n_embd_head_k),
                    ggml_row_size(q->type, hparams.n_embd_head_k * n_head) },
                    0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                ggml_tensor* q_pe = ggml_view(ctx0, q, { n_embd_head_qk_rope, n_head, n_tokens },
                    { ggml_row_size(q->type, hparams.n_embd_head_k),
                    ggml_row_size(q->type, hparams.n_embd_head_k * n_head) },
                    ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor* kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                struct ggml_tensor* kv_compressed = ggml_view(ctx0, kv_pe_compresseed, { (int64_t)kv_lora_rank, (int64_t)n_tokens },
                    { kv_pe_compresseed->nb[1] },
                    0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                ggml_tensor* k_pe = ggml_view(ctx0, kv_pe_compresseed, { n_embd_head_qk_rope, 1, n_tokens },
                    { kv_pe_compresseed->nb[1],
                    kv_pe_compresseed->nb[1] },
                    ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                kv_compressed = ggml_cont(ctx0, kv_compressed); // TODO: the CUDA backend does not support non-contiguous norm
                kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams,
                    model.layers[il].attn_kv_a_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor* kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor* k_nope = ggml_view(ctx0, kv, { n_embd_head_qk_nope, n_head, n_tokens },
                    { ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                    ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)) },
                    0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                ggml_tensor* v_states = ggml_view(ctx0, kv, { hparams.n_embd_head_v, n_head, n_tokens },
                    { ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                    ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v) * n_head) },
                    ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view(ctx0, v_states, { hparams.n_embd_head_v * n_head, n_tokens },
                    { ggml_row_size(kv->type, hparams.n_embd_head_v * n_head) },
                    0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(ctx0, q_pe); // TODO: the CUDA backend does not support non-contiguous RoPE
                q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(ctx0, k_pe); // TODO: the CUDA backend does not support non-contiguous RoPE
                k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                struct ggml_tensor* q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor* k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    k_states, v_states, q_states, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // scale_res - scale the hidden states for residual connection
            const float scale_res = scale_depth / sqrtf(float(n_layer));
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled", il);

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // scale the hidden states for residual connection
            cur = ggml_scale(ctx0, cur, scale_res);
            cb(cur, "hidden_scaled_ffn", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head scaling
        const float scale_lmhead = float(n_embd_base) / float(n_embd);
        cur = ggml_scale(ctx0, cur, scale_lmhead);
        cb(cur, "lmhead_scaling", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_gemma() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head_k, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head_k, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            struct ggml_tensor* sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = llm_build_norm(ctx0, sa_out, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, sa_out);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_gemma2() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head_k = hparams.n_embd_head_k;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        inpL = ggml_scale(ctx0, inpL, sqrtf(n_embd));
        cb(inpL, "inp_scaled", -1);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        // gemma 2 requires different mask for layers using sliding window (SWA)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask(true);
        struct ggml_tensor* KQ_mask_swa = build_inp_KQ_mask_swa(true);

        for (int il = 0; il < n_layer; ++il) {
            // (il % 2) layers use SWA
            struct ggml_tensor* KQ_mask_l = (il % 2 == 0) ? KQ_mask_swa : KQ_mask;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head_k, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Qcur, "Qcur", il);

                // ref: https://github.com/google/gemma_pytorch/commit/03e657582d17cb5a8617ebf333c1c16f3694670e
                switch (model.type) {
                case e_model::MODEL_2B:
                case e_model::MODEL_9B:  Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd_head_k)));   break;
                case e_model::MODEL_27B: Qcur = ggml_scale(ctx0, Qcur, 1.0f / sqrtf(float(n_embd / n_head))); break;
                default: GGML_ABORT("fatal error");
                };
                cb(Qcur, "Qcur_scaled", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head_k, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask_l, n_tokens, kv_head, n_kv, 1.0f, cb, il);
            }

            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].attn_post_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_post_norm", il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            struct ggml_tensor* sa_out = ggml_add(ctx0, cur, inpL);
            cb(sa_out, "sa_out", il);

            cur = llm_build_norm(ctx0, sa_out, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            // feed-forward network
            {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, cb, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, sa_out);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        // final logit soft-capping
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }


    ggml_cgraph* build_starcoder2() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network

            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, model.layers[il].ffn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_mamba() {
        ggml_cgraph* gf = new ggml_cgraph;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        // {n_embd, n_tokens}
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        struct ggml_tensor* state_copy = build_inp_s_copy();
        struct ggml_tensor* state_mask = build_inp_s_mask();

        for (int il = 0; il < n_layer; ++il) {
            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            cur = llm_build_mamba(ctx0, lctx, ubatch, gf, cur,
                state_copy, state_mask,
                kv_head, n_kv, cb, il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // residual
            cur = ggml_add(ctx0, cur, inpL);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        // final rmsnorm
        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_command_r() {

        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        const float f_logit_scale = hparams.f_logit_scale;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);
            struct ggml_tensor* ffn_inp = cur;

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view(ctx0, Qcur, { n_embd_head, n_head, n_tokens },
                        { ggml_element_size(Qcur) * n_embd_head,
                        ggml_element_size(Qcur) * n_embd_head * n_head },
                        0);
                    cb(Qcur, "Qcur", il);
                    Kcur = ggml_view(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens },
                        { ggml_element_size(Kcur) * n_embd_head,
                        ggml_element_size(Kcur) * n_embd_head * n_head_kv },
                        0);
                    cb(Kcur, "Kcur", il);

                    Qcur = llm_build_norm(ctx0, Qcur, hparams,
                        model.layers[il].attn_q_norm,
                        NULL,
                        LLM_NORM, cb, il);
                    cb(Qcur, "Qcur", il);

                    Kcur = llm_build_norm(ctx0, Kcur, hparams,
                        model.layers[il].attn_k_norm,
                        NULL,
                        LLM_NORM, cb, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
                ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
            }

            struct ggml_tensor* attn_out = cur;

            // feed-forward network
            {
                cur = llm_build_ffn(ctx0, lctx, ffn_inp,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            // add together residual + FFN + self-attention
            cur = ggml_add(ctx0, cur, inpL);
            cur = ggml_add(ctx0, cur, attn_out);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        if (f_logit_scale) {
            cur = ggml_scale(ctx0, cur, f_logit_scale);
        }

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;

    }

    // ref: https://allenai.org/olmo
    // based on the original build_llama() function, changes:
    //   * non-parametric layer norm
    //   * clamp qkv
    //   * removed bias
    //   * removed MoE
    ggml_cgraph* build_olmo() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                NULL, NULL,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Qcur = ggml_clamp(ctx0, Qcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Kcur = ggml_clamp(ctx0, Kcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (hparams.f_clamp_kqv > 0.0f) {
                    Vcur = ggml_clamp(ctx0, Vcur, -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                NULL, NULL,
                LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            NULL, NULL,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_olmo2() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = inpL;

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].attn_post_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_post_norm", il);

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_ffn(ctx0, lctx, ffn_inp,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].ffn_post_norm, NULL,
                LLM_NORM_RMS, cb, -1);
            cb(cur, "ffn_post_norm", -1);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    // based on the build_qwen2moe() function, changes:
    //   * removed shared experts
    //   * removed bias
    //   * added q, k norm
    ggml_cgraph* build_olmoe() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = llm_build_norm(ctx0, Qcur, hparams, model.layers[il].attn_q_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(Qcur, "Qcur_normed", il);

                Kcur = llm_build_norm(ctx0, Kcur, hparams, model.layers[il].attn_k_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(Kcur, "Kcur_normed", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // MoE branch
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                n_expert, n_expert_used,
                LLM_FFN_SILU, false,
                false, 0.0,
                cb, il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_openelm() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            const int64_t n_head = hparams.n_head(il);
            const int64_t n_head_kv = hparams.n_head_kv(il);
            const int64_t n_head_qkv = 2 * n_head_kv + n_head;

            cur = inpL;
            struct ggml_tensor* residual = cur;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_reshape(ctx0, cur, { n_embd_head_k, n_head_qkv, n_tokens });

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_head, n_head, n_tokens }, { cur->nb[1], cur->nb[2] }, 0));
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_head, n_head_kv, n_tokens }, { cur->nb[1], cur->nb[2] }, cur->nb[1] * n_head));
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_head, n_head_kv, n_tokens }, { cur->nb[1], cur->nb[2] }, cur->nb[1] * (n_head + n_head_kv)));
                cb(Vcur, "Vcur", il);

                Qcur = llm_build_norm(ctx0, Qcur, hparams,
                    model.layers[il].attn_q_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(Qcur, "Qcur", il);

                Kcur = llm_build_norm(ctx0, Kcur, hparams,
                    model.layers[il].attn_k_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(Kcur, "Kcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, NULL, n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                Vcur = ggml_reshape(ctx0, Vcur, { n_embd_head * n_head_kv, n_tokens });
                cb(Qcur, "Vcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                residual = ggml_get_rows(ctx0, residual, inp_out_ids);
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, residual, cur);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            inpL = cur;
        }

        cur = inpL;

        // norm
        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_gptneox() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // ffn
            if (hparams.use_par_res) {
                // attention and ffn are computed in parallel
                // x = x + attn(ln1(x)) + ffn(ln2(x))

                struct ggml_tensor* attn_out = cur;

                cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, inpL);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, attn_out);
                cur = lctx.cvec.apply_to(ctx0, cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            }
            else {
                // attention and ffn are computed sequentially
                // x = x + attn(ln1(x))
                // x = x + ffn(ln2(x))

                struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
                cb(ffn_inp, "ffn_inp", il);

                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);

                cur = ggml_add(ctx0, cur, ffn_inp);
                cur = lctx.cvec.apply_to(ctx0, cur, il);
                cb(cur, "l_out", il);

                // input for next layer
                inpL = cur;
            }
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_arctic() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            struct ggml_tensor* ffn_out = ggml_add(ctx0, cur, ffn_inp);
            cb(ffn_out, "ffn_out", il);

            // MoE
            cur = llm_build_norm(ctx0, inpSA, hparams,
                model.layers[il].ffn_norm_exps, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm_exps", il);

            cur = llm_build_moe_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_gate_inp,
                model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps,
                model.layers[il].ffn_down_exps,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true,
                false, 0.0,
                cb, il);
            cb(cur, "ffn_moe_out", il);

            cur = ggml_add(ctx0, cur, ffn_out);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_deepseek() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();
        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor* rope_factors = build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }


            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t)il < hparams.n_layer_dense_lead) {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }
            else {
                // MoE branch
                ggml_tensor* moe_out =
                    llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        false, hparams.expert_weights_scale,
                        cb, il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor* ffn_shexp = llm_build_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_deepseek2() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        bool is_lite = (hparams.n_layer == 27);

        // We have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly.
        // See https://github.com/ggerganov/llama.cpp/discussions/7416 for detailed explanation.
        const float mscale = attn_factor * (1.0f + hparams.rope_yarn_log_mul * logf(1.0f / freq_scale));
        const float kq_scale = 1.0f * mscale * mscale / sqrtf(float(hparams.n_embd_head_k));
        const float attn_factor_scaled = 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));

        const uint32_t n_embd_head_qk_rope = hparams.n_rot;
        const uint32_t n_embd_head_qk_nope = hparams.n_embd_head_k - hparams.n_rot;
        const uint32_t kv_lora_rank = hparams.n_lora_kv;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        // {n_embd, n_tokens}
        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self_attention
            {
                struct ggml_tensor* q = NULL;
                if (!is_lite) {
                    // {n_embd, q_lora_rank} * {n_embd, n_tokens} -> {q_lora_rank, n_tokens}
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_a, cur);
                    cb(q, "q", il);

                    q = llm_build_norm(ctx0, q, hparams,
                        model.layers[il].attn_q_a_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                    cb(q, "q", il);

                    // {q_lora_rank, n_head * hparams.n_embd_head_k} * {q_lora_rank, n_tokens} -> {n_head * hparams.n_embd_head_k, n_tokens}
                    q = ggml_mul_mat(ctx0, model.layers[il].wq_b, q);
                    cb(q, "q", il);
                }
                else {
                    q = ggml_mul_mat(ctx0, model.layers[il].wq, cur);
                    cb(q, "q", il);
                }

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor* q_nope = ggml_view(ctx0, q, { n_embd_head_qk_nope, n_head, n_tokens },
                    { ggml_row_size(q->type, hparams.n_embd_head_k),
                    ggml_row_size(q->type, hparams.n_embd_head_k * n_head) },
                    0);
                cb(q_nope, "q_nope", il);

                // and {n_head * n_embd_head_qk_rope, n_tokens}
                ggml_tensor* q_pe = ggml_view(ctx0, q, { n_embd_head_qk_rope, n_head, n_tokens },
                    { ggml_row_size(q->type, hparams.n_embd_head_k),
                    ggml_row_size(q->type, hparams.n_embd_head_k * n_head) },
                    ggml_row_size(q->type, n_embd_head_qk_nope));
                cb(q_pe, "q_pe", il);

                // {n_embd, kv_lora_rank + n_embd_head_qk_rope} * {n_embd, n_tokens} -> {kv_lora_rank + n_embd_head_qk_rope, n_tokens}
                struct ggml_tensor* kv_pe_compresseed = ggml_mul_mat(ctx0, model.layers[il].wkv_a_mqa, cur);
                cb(kv_pe_compresseed, "kv_pe_compresseed", il);

                // split into {kv_lora_rank, n_tokens}
                struct ggml_tensor* kv_compressed = ggml_view(ctx0, kv_pe_compresseed, { kv_lora_rank, n_tokens },
                    { kv_pe_compresseed->nb[1] },
                    0);
                cb(kv_compressed, "kv_compressed", il);

                // and {n_embd_head_qk_rope, n_tokens}
                ggml_tensor* k_pe = ggml_view(ctx0, kv_pe_compresseed, { n_embd_head_qk_rope, 1, n_tokens },
                    { kv_pe_compresseed->nb[1],
                    kv_pe_compresseed->nb[1] },
                    ggml_row_size(kv_pe_compresseed->type, kv_lora_rank));
                cb(k_pe, "k_pe", il);

                kv_compressed = ggml_cont(ctx0, kv_compressed); // TODO: the CUDA backend does not support non-contiguous norm
                kv_compressed = llm_build_norm(ctx0, kv_compressed, hparams,
                    model.layers[il].attn_kv_a_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(kv_compressed, "kv_compressed", il);

                // {kv_lora_rank, n_head * (n_embd_head_qk_nope + n_embd_head_v)} * {kv_lora_rank, n_tokens} -> {n_head * (n_embd_head_qk_nope + n_embd_head_v), n_tokens}
                struct ggml_tensor* kv = ggml_mul_mat(ctx0, model.layers[il].wkv_b, kv_compressed);
                cb(kv, "kv", il);

                // split into {n_head * n_embd_head_qk_nope, n_tokens}
                ggml_tensor* k_nope = ggml_view(ctx0, kv, { n_embd_head_qk_nope, n_head, n_tokens },
                    { ggml_row_size(kv->type, n_embd_head_qk_nope + hparams.n_embd_head_v),
                    ggml_row_size(kv->type, n_head * (n_embd_head_qk_nope + hparams.n_embd_head_v)) },
                    0);
                cb(k_nope, "k_nope", il);

                // and {n_head * n_embd_head_v, n_tokens}
                ggml_tensor* v_states = ggml_view(ctx0, kv, { hparams.n_embd_head_v, n_head, n_tokens },
                    { ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v)),
                    ggml_row_size(kv->type, (n_embd_head_qk_nope + hparams.n_embd_head_v) * n_head) },
                    ggml_row_size(kv->type, (n_embd_head_qk_nope)));
                cb(v_states, "v_states", il);

                v_states = ggml_cont(ctx0, v_states);
                cb(v_states, "v_states", il);

                v_states = ggml_view(ctx0, v_states, { hparams.n_embd_head_v * n_head, n_tokens },
                    { ggml_row_size(kv->type, hparams.n_embd_head_v * n_head) },
                    0);
                cb(v_states, "v_states", il);

                q_pe = ggml_cont(ctx0, q_pe); // TODO: the CUDA backend does not support non-contiguous RoPE
                q_pe = ggml_rope_ext(
                    ctx0, q_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow
                );
                cb(q_pe, "q_pe", il);

                // shared RoPE key
                k_pe = ggml_cont(ctx0, k_pe); // TODO: the CUDA backend does not support non-contiguous RoPE
                k_pe = ggml_rope_ext(
                    ctx0, k_pe, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor_scaled, beta_fast, beta_slow
                );
                cb(k_pe, "k_pe", il);

                struct ggml_tensor* q_states = ggml_concat(ctx0, q_nope, q_pe, 0);
                cb(q_states, "q_states", il);

                struct ggml_tensor* k_states = ggml_concat(ctx0, k_nope, ggml_repeat(ctx0, k_pe, q_pe), 0);
                cb(k_states, "k_states", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    k_states, v_states, q_states, KQ_mask, n_tokens, kv_head, n_kv, kq_scale, cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            if ((uint32_t)il < hparams.n_layer_dense_lead) {
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }
            else {
                // MoE branch
                ggml_tensor* moe_out =
                    llm_build_moe_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_gate_inp,
                        model.layers[il].ffn_up_exps,
                        model.layers[il].ffn_gate_exps,
                        model.layers[il].ffn_down_exps,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, false,
                        true, hparams.expert_weights_scale,
                        cb, il);
                cb(moe_out, "ffn_moe_out", il);

                // FFN shared expert
                {
                    ggml_tensor* ffn_shexp = llm_build_ffn(ctx0, lctx, cur,
                        model.layers[il].ffn_up_shexp, NULL, NULL,
                        model.layers[il].ffn_gate_shexp, NULL, NULL,
                        model.layers[il].ffn_down_shexp, NULL, NULL,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                    cb(ffn_shexp, "ffn_shexp", il);

                    cur = ggml_add(ctx0, moe_out, ffn_shexp);
                    cb(cur, "ffn_out", il);
                }
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = ggml_mul_mat(ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_bitnet() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                if (model.layers[il].wq_scale) {
                    Qcur = ggml_mul(ctx0, Qcur, model.layers[il].wq_scale);
                }
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                // B1.K
                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                if (model.layers[il].wk_scale) {
                    Kcur = ggml_mul(ctx0, Kcur, model.layers[il].wk_scale);
                }
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                // B1.V
                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                if (model.layers[il].wv_scale) {
                    Vcur = ggml_mul(ctx0, Vcur, model.layers[il].wv_scale);
                }
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    NULL, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);

                cur = llm_build_norm(ctx0, cur, hparams,
                    model.layers[il].attn_sub_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "attn_sub_norm", il);

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
                if (model.layers[il].wo_scale) {
                    cur = ggml_mul(ctx0, cur, model.layers[il].wo_scale);
                }
                if (model.layers[il].bo) {
                    cur = ggml_add(ctx0, cur, model.layers[il].bo);
                }
                cb(cur, "attn_o_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward forward
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_scale,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_scale,
                NULL, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_sub_out", il);

            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].ffn_sub_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_sub_norm", il);

            cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].ffn_down, cur);
            if (model.layers[il].ffn_down_scale) {
                cur = ggml_mul(ctx0, cur, model.layers[il].ffn_down_scale);
            }
            cb(cur, "ffn_down", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        // FIXME: do not use model.tok_embd directly, duplicate as model.output
        cur = llm_build_lora_mm(lctx, ctx0, model.tok_embd, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);
        return gf;
    }

    ggml_cgraph* build_t5_enc() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        GGML_ASSERT(lctx.is_encoding);
        struct ggml_tensor* pos_bucket_enc = llm_build_pos_bucket(false);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask_enc = build_inp_KQ_mask(false);

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm_enc, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_enc, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk_enc, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv_enc, cur);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens });

                struct ggml_tensor* q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                struct ggml_tensor* k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

                struct ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
                cb(kq, "kq", il);

                struct ggml_tensor* attn_rel_b = model.layers[il].attn_rel_b_enc ? model.layers[il].attn_rel_b_enc : model.layers[0].attn_rel_b_enc;
                struct ggml_tensor* pos_bias = llm_build_pos_bias(pos_bucket_enc, attn_rel_b);
                struct ggml_tensor* kq_b = ggml_add(ctx0, kq, pos_bias);
                cb(kq_b, "kq_b", il);

                kq = ggml_soft_max(ctx0, kq_b, KQ_mask_enc, 1.0f, hparams.f_max_alibi_bias);
                cb(kq, "kq_soft_max_ext", il);

                struct ggml_tensor* v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape(ctx0, Vcur, { n_embd_gqa, n_tokens })));
                cb(v, "v", il);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx0, ggml_reshape(ctx0, v, { n_tokens, n_embd_head, n_head_kv }), kq);
                cb(kqv, "kqv", il);

                struct ggml_tensor* kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                cb(kqv_merged, "kqv_merged", il);

                cur = ggml_cont(ctx0, kqv_merged, { n_embd_gqa, n_tokens });
                cb(cur, "kqv_merged_cont", il);

                gf->build_forward_expand(cur);

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_enc, cur);
                cb(cur, "kqv_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm_enc, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                // T5 uses relu, flan-T5 uses gelu-gated
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up_enc, NULL, NULL,
                    model.layers[il].ffn_gate_enc, NULL, NULL,
                    model.layers[il].ffn_down_enc, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
                    cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            ggml_tensor* layer_dir = lctx.cvec.tensor_for(il);
            if (layer_dir != nullptr) {
                cur = ggml_add(ctx0, cur, layer_dir);
            }
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cb(cur, "result_embd", -1);

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm_enc, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_t5_dec() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        GGML_ASSERT(!lctx.is_encoding);
        GGML_ASSERT(n_outputs_enc > 0 && "call llama_encode() first");

        struct ggml_tensor* embd_enc = llm_build_inp_embd_enc();
        struct ggml_tensor* pos_bucket_dec = llm_build_pos_bucket(true);

        struct ggml_tensor* KQ_mask_dec = build_inp_KQ_mask();
        struct ggml_tensor* KQ_mask_cross = llm_build_inp_KQ_mask_cross();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                llm_build_kv_store(ctx0, hparams, cparams, kv_self, gf, Kcur, Vcur, n_tokens, kv_head, cb, il);

                struct ggml_tensor* k =
                    ggml_view(ctx0, kv_self.k_l[il],
                        { n_embd_head_k, n_kv, n_head_kv },
                        { ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa),
                        ggml_row_size(kv_self.k_l[il]->type, n_embd_head_k) },
                        0);
                cb(k, "k", il);

                struct ggml_tensor* v =
                    ggml_view(ctx0, kv_self.v_l[il],
                        { n_kv, n_embd_head_v, n_head_kv },
                        { ggml_element_size(kv_self.v_l[il]) * n_ctx,
                        ggml_element_size(kv_self.v_l[il]) * n_ctx * n_embd_head_v },
                        0);
                cb(v, "v", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });

                struct ggml_tensor* q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);

                struct ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
                cb(kq, "kq", il);

                struct ggml_tensor* attn_rel_b = model.layers[il].attn_rel_b ? model.layers[il].attn_rel_b : model.layers[0].attn_rel_b;
                struct ggml_tensor* pos_bias = llm_build_pos_bias(pos_bucket_dec, attn_rel_b);
                struct ggml_tensor* kq_b = ggml_add(ctx0, kq, pos_bias);
                cb(kq_b, "kq_b", il);

                kq = ggml_soft_max(ctx0, kq_b, KQ_mask_dec, 1.0f, hparams.f_max_alibi_bias);
                cb(kq, "kq_soft_max", il);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx0, v, kq);
                cb(kqv, "kqv", il);

                struct ggml_tensor* kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                cb(kqv_merged, "kqv_merged", il);

                cur = ggml_cont(ctx0, kqv_merged, { n_embd_gqa, n_tokens });
                cb(cur, "kqv_merged_cont", il);

                gf->build_forward_expand(cur);

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo, cur);
                cb(cur, "kqv_out", il);
            }

            cur = ggml_add(ctx0, cur, inpSA);
            cb(cur, "cross_inp", il);

            struct ggml_tensor* inpCA = cur;

            // norm
            cur = llm_build_norm(ctx0, cur, hparams,
                model.layers[il].attn_norm_cross, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm_cross", il);

            // cross-attention
            {
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq_cross, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk_cross, embd_enc);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv_cross, embd_enc);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });
                Kcur = ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_outputs_enc });

                struct ggml_tensor* q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
                struct ggml_tensor* k = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3));

                struct ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);
                cb(kq, "kq", il);

                kq = ggml_soft_max(ctx0, kq, KQ_mask_cross, 1.0f, hparams.f_max_alibi_bias);
                cb(kq, "kq_soft_max", il);

                struct ggml_tensor* v = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape(ctx0, Vcur, { n_embd_gqa, n_outputs_enc })));
                cb(v, "v", il);

                struct ggml_tensor* kqv = ggml_mul_mat(ctx0, ggml_reshape(ctx0, v, { n_outputs_enc, n_embd_head, n_head_kv }), kq);
                cb(kqv, "kqv", il);

                struct ggml_tensor* kqv_merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                cb(kqv_merged, "kqv_merged", il);

                cur = ggml_cont(ctx0, kqv_merged, { n_embd_gqa, n_tokens });
                cb(cur, "kqv_merged_cont", il);

                gf->build_forward_expand(cur);

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wo_cross, cur);
                cb(cur, "kqv_out", il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
                inpCA = ggml_get_rows(ctx0, inpCA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpCA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                // T5 uses relu, flan-T5 uses gelu-gated
                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_GELU : LLM_FFN_RELU,
                    model.layers[il].ffn_gate_enc ? LLM_FFN_PAR : LLM_FFN_SEQ,
                    cb, il);
                cb(cur, "ffn_out", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            ggml_tensor* layer_dir = lctx.cvec.tensor_for(il);
            if (layer_dir != nullptr) {
                cur = ggml_add(ctx0, cur, layer_dir);
            }
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        cb(cur, "result_embd", -1);

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_jais() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                struct ggml_tensor* Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { (int64_t)n_embd, (int64_t)n_tokens }, { cur->nb[1] }, 0 * cur->nb[0] * (n_embd)));
                struct ggml_tensor* Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { (int64_t)n_embd_gqa, (int64_t)n_tokens }, { cur->nb[1] }, 1 * cur->nb[0] * (n_embd)));
                struct ggml_tensor* Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { (int64_t)n_embd_gqa, (int64_t)n_tokens }, { cur->nb[1] }, 1 * cur->nb[0] * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);

                Qcur = ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens });

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / float(n_embd_head), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
            }

            // add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpL);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    model.layers[il].ffn_norm_b,
                    LLM_NORM, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                    model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, NULL,
                    model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
                cb(cur, "ffn_out", il);
            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_chatglm() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        const int64_t n_embd_gqa = hparams.n_embd_v_gqa();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                struct ggml_tensor* Qcur = nullptr;
                struct ggml_tensor* Kcur = nullptr;
                struct ggml_tensor* Vcur = nullptr;

                cur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wqkv, cur);
                cb(cur, "wqkv", il);

                cur = ggml_add(ctx0, cur, model.layers[il].bqkv);
                cb(cur, "bqkv", il);

                Qcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd, n_tokens }, { cur->nb[1] }, 0 * sizeof(float) * (n_embd)));
                Kcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd)));
                Vcur = ggml_cont(ctx0, ggml_view(ctx0, cur, { n_embd_gqa, n_tokens }, { cur->nb[1] }, 1 * sizeof(float) * (n_embd + n_embd_gqa)));

                cb(Qcur, "Qcur", il);
                cb(Kcur, "Kcur", il);
                cb(Vcur, "Vcur", il);
                //printf("freq_base: %f freq_scale: %f ext_factor: %f attn_factor: %f\n", freq_base, freq_scale, ext_factor, attn_factor);
                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur_rope", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur_rope", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, NULL,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);

            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            // Add the input
            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // FF
            {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm,
                    NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);

                cur = llm_build_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_up, NULL, NULL,
                    NULL, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU, LLM_FFN_SEQ, cb, il);
                cb(cur, "ffn_out", il);

            }

            inpL = ggml_add(ctx0, cur, ffn_inp);
            cb(inpL, "l_out", il);
        }

        cur = llm_build_norm(ctx0, inpL, hparams,
            model.output_norm,
            NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_nemotron() {
        ggml_cgraph* gf = new ggml_cgraph;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        //GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm,
                model.layers[il].attn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm,
                model.layers[il].ffn_norm_b,
                LLM_NORM, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL,
                NULL,
                LLM_FFN_RELU_SQR, LLM_FFN_SEQ, cb, il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, model.output_norm_b,
            LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_exaone() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            cur = llm_build_norm(ctx0, inpL, hparams,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "attn_norm", il);

            // self-attention
            {
                // rope freq factors for llama3; may return nullptr for llama2 and other models
                struct ggml_tensor* rope_factors = build_rope_factors(il);

                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);
                if (model.layers[il].bq) {
                    Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
                    cb(Qcur, "Qcur", il);
                }

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);
                if (model.layers[il].bk) {
                    Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
                    cb(Kcur, "Kcur", il);
                }

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);
                if (model.layers[il].bv) {
                    Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
                    cb(Vcur, "Vcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, model.layers[il].bo,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            cur = llm_build_norm(ctx0, ffn_inp, hparams,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, cb, il);
            cb(cur, "ffn_norm", il);

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_rwkv6() {
        ggml_cgraph* gf = new ggml_cgraph;

        // Token shift state dimensions should be 2 * n_emb
        GGML_ASSERT(n_embd == hparams.n_embd_k_s() / 2);

        const int64_t n_seqs = ubatch.n_seqs;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_tokens = ubatch.n_tokens;
        GGML_ASSERT(n_seqs != 0);
        GGML_ASSERT(ubatch.equal_seqs);
        GGML_ASSERT(n_tokens == n_seq_tokens * n_seqs);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;
        struct ggml_tensor* state_copy = build_inp_s_copy();
        struct ggml_tensor* state_mask = build_inp_s_mask();

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);
        inpL = llm_build_norm(ctx0, inpL, hparams, model.tok_norm, model.tok_norm_b, LLM_NORM, cb, -1);

        for (int il = 0; il < n_layer; ++il) {
            const llama_layer* layer = &model.layers[il];

            // (ab)using the KV cache to store the states
            struct ggml_tensor* token_shift = llm_build_copy_mask_state(ctx0,
                gf, kv_self.k_l[il], state_copy, state_mask,
                hparams.n_embd_k_s(), kv_self.size, kv_head, n_kv, n_seqs);
            struct ggml_tensor* wkv_states = llm_build_copy_mask_state(ctx0,
                gf, kv_self.v_l[il], state_copy, state_mask,
                hparams.n_embd_v_s(), kv_self.size, kv_head, n_kv, n_seqs);

            cur = ggml_reshape(ctx0, inpL, { n_embd, n_seq_tokens, n_seqs });
            token_shift = ggml_reshape(ctx0, token_shift, { n_embd, 2, n_seqs });

            ggml_tensor* att_shift = ggml_view(ctx0, token_shift, { n_embd, 1, n_seqs }, { token_shift->nb[1], token_shift->nb[2] }, 0);
            ggml_tensor* ffn_shift = ggml_view(ctx0, token_shift, { n_embd, 1, n_seqs }, { token_shift->nb[1], token_shift->nb[2] }, n_embd * ggml_element_size(token_shift));

            struct ggml_tensor* x_norm_att = llm_build_norm(ctx0, cur, hparams, layer->attn_norm, layer->attn_norm_b, LLM_NORM, cb, il);
            struct ggml_tensor* x_prev = ggml_concat(
                ctx0,
                att_shift,
                ggml_view(ctx0, x_norm_att, { n_embd, n_seq_tokens - 1, n_seqs }, { x_norm_att->nb[1], x_norm_att->nb[2] }, 0),
                1
            );

            cur = ggml_add(ctx0, cur, llm_build_rwkv6_time_mix(lctx, ctx0, layer, x_norm_att, x_prev, &wkv_states));
            gf->build_forward_expand(cur);
            gf->build_forward_expand(
                ggml_cpy(
                    ctx0,
                    wkv_states,
                    ggml_view(
                        ctx0,
                        kv_self.v_l[il],
                        { hparams.n_embd_v_s() * n_seqs }, {},
                        hparams.n_embd_v_s() * kv_head * ggml_element_size(kv_self.v_l[il])
                    )
                )
            );

            struct ggml_tensor* x_norm_ffn = llm_build_norm(ctx0, cur, hparams, layer->attn_norm_2, layer->attn_norm_2_b, LLM_NORM, cb, il);
            x_prev = ggml_concat(
                ctx0,
                ffn_shift,
                ggml_view(ctx0, x_norm_ffn, { n_embd, n_seq_tokens - 1, n_seqs }, { x_norm_ffn->nb[1], x_norm_ffn->nb[2] }, 0),
                1
            );
            cur = ggml_add(ctx0, cur, llm_build_rwkv6_channel_mix(lctx, ctx0, layer, x_norm_ffn, x_prev));
            gf->build_forward_expand(cur);

            ggml_tensor* last_norm_att = ggml_view(ctx0, x_norm_att, { n_embd, 1, n_seqs }, { x_norm_att->nb[1], x_norm_att->nb[2] }, (n_seq_tokens - 1) * n_embd * ggml_element_size(x_norm_att));
            ggml_tensor* last_norm_ffn = ggml_view(ctx0, x_norm_ffn, { n_embd, 1, n_seqs }, { x_norm_ffn->nb[1], x_norm_ffn->nb[2] }, (n_seq_tokens - 1) * n_embd * ggml_element_size(x_norm_ffn));

            token_shift = ggml_concat(ctx0, last_norm_att, last_norm_ffn, 1);

            gf->build_forward_expand(
                ggml_cpy(
                    ctx0,
                    ggml_view(ctx0, token_shift, { n_embd * n_seqs * 2 }, {}, 0),
                    ggml_view(ctx0, kv_self.k_l[il], { hparams.n_embd_k_s() * n_seqs }, {}, hparams.n_embd_k_s() * kv_head * ggml_element_size(kv_self.k_l[il]))
                )
            );

            if (hparams.rescale_every_n_layers != 0 && (il + 1) % hparams.rescale_every_n_layers == 0) {
                cur = ggml_scale(ctx0, cur, 0.5F);
            }

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;
        struct ggml_tensor* inp_out_ids = build_inp_out_ids();
        cur = ggml_reshape(ctx0, cur, { n_embd, n_tokens });
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);

        cur = llm_build_norm(ctx0, cur, hparams, model.output_norm, model.output_norm_b, LLM_NORM, cb, -1);
        cb(cur, "result_norm", -1);

        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    // ref: https://github.com/facebookresearch/chameleon
    // based on the original build_llama() function, changes:
    //   * qk-norm
    //   * swin-norm
    //   * removed bias
    //   * removed MoE
    ggml_cgraph* build_chameleon() {
        ggml_cgraph* gf = new ggml_cgraph;

        // mutable variable, needed during the last layer of the computation to skip unused tokens
        int32_t n_tokens = this->n_tokens;

        const int64_t n_embd_head = hparams.n_embd_head_v;
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
        GGML_ASSERT(n_embd_head == hparams.n_rot);

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        // inp_pos - contains the positions
        struct ggml_tensor* inp_pos = build_inp_pos();

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor* KQ_mask = build_inp_KQ_mask();

        for (int il = 0; il < n_layer; ++il) {
            struct ggml_tensor* inpSA = inpL;

            // norm
            if (hparams.swin_norm) {
                cur = inpL;
            }
            else {
                cur = llm_build_norm(ctx0, inpL, hparams,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "attn_norm", il);
            }

            // self-attention
            {
                // compute Q and K and RoPE them
                struct ggml_tensor* Qcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wq, cur);
                cb(Qcur, "Qcur", il);

                struct ggml_tensor* Kcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wk, cur);
                cb(Kcur, "Kcur", il);

                struct ggml_tensor* Vcur = llm_build_lora_mm(lctx, ctx0, model.layers[il].wv, cur);
                cb(Vcur, "Vcur", il);

                if (model.layers[il].attn_q_norm) {
                    Qcur = ggml_view(ctx0, Qcur, { n_embd_head, n_head, n_tokens },
                        { ggml_element_size(Qcur) * n_embd_head,
                        ggml_element_size(Qcur) * n_embd_head * n_head },
                        0);
                    cb(Qcur, "Qcur", il);

                    Qcur = llm_build_norm(ctx0, Qcur, hparams,
                        model.layers[il].attn_q_norm,
                        model.layers[il].attn_q_norm_b,
                        LLM_NORM, cb, il);
                    cb(Qcur, "Qcur", il);
                }

                if (model.layers[il].attn_k_norm) {
                    Kcur = ggml_view(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens },
                        { ggml_element_size(Kcur) * n_embd_head,
                        ggml_element_size(Kcur) * n_embd_head * n_head_kv },
                        0);
                    cb(Kcur, "Kcur", il);

                    Kcur = llm_build_norm(ctx0, Kcur, hparams,
                        model.layers[il].attn_k_norm,
                        model.layers[il].attn_k_norm_b,
                        LLM_NORM, cb, il);
                    cb(Kcur, "Kcur", il);
                }

                Qcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Qcur, { n_embd_head, n_head, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Qcur, "Qcur", il);

                Kcur = ggml_rope_ext(
                    ctx0, ggml_reshape(ctx0, Kcur, { n_embd_head, n_head_kv, n_tokens }), inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );
                cb(Kcur, "Kcur", il);

                cur = llm_build_kv(ctx0, lctx, kv_self, gf,
                    model.layers[il].wo, nullptr,
                    Kcur, Vcur, Qcur, KQ_mask, n_tokens, kv_head, n_kv, 1.0f / sqrtf(float(n_embd_head)), cb, il);

                if (hparams.swin_norm) {
                    cur = llm_build_norm(ctx0, cur, hparams,
                        model.layers[il].attn_norm, NULL,
                        LLM_NORM_RMS, cb, il);
                }
            }

            if (il == n_layer - 1) {
                // skip computing output for unused tokens
                struct ggml_tensor* inp_out_ids = build_inp_out_ids();
                n_tokens = n_outputs;
                cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            }

            struct ggml_tensor* ffn_inp = ggml_add(ctx0, cur, inpSA);
            cb(ffn_inp, "ffn_inp", il);

            // feed-forward network
            if (!hparams.swin_norm) {
                cur = llm_build_norm(ctx0, ffn_inp, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);
            }

            cur = llm_build_ffn(ctx0, lctx, cur,
                model.layers[il].ffn_up, NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, cb, il);
            cb(cur, "ffn_out", il);

            if (hparams.swin_norm) {
                cur = llm_build_norm(ctx0, cur, hparams,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, cb, il);
                cb(cur, "ffn_norm", il);
            }

            cur = ggml_add(ctx0, cur, ffn_inp);
            cb(cur, "ffn_out", il);

            cur = lctx.cvec.apply_to(ctx0, cur, il);
            cb(cur, "l_out", il);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm, NULL,
            LLM_NORM_RMS, cb, -1);
        cb(cur, "result_norm", -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);
        cb(cur, "result_output_with_img_logits", -1);

        // TODO: this suppresses the output of image tokens, which is required to enable text-only outputs.
        // Needs to be removed once image outputs are supported.
        int img_token_end_idx = 8196;
        int img_token_start_idx = 4;
        int num_img_tokens = img_token_end_idx - img_token_start_idx;
        // creates 1d tensor of size num_img_tokens and values -FLT_MAX,
        // which ensures that text token values are always at least larger than image token values
        ggml_tensor* img_logits = ctx0->create(GGML_TYPE_F32, { num_img_tokens });
        img_logits = ggml_clamp(ctx0, img_logits, -FLT_MAX, -FLT_MAX);
        cb(img_logits, "img_logits", -1);
        cur = ggml_set_1d(ctx0, cur, img_logits, ggml_element_size(cur) * img_token_start_idx);
        cb(cur, "result_output", -1);

        gf->build_forward_expand(cur);

        return gf;
    }

    ggml_cgraph* build_wavtokenizer_dec() {
        ggml_cgraph* gf = new ggml_cgraph;

        struct ggml_tensor* cur;
        struct ggml_tensor* inpL;

        inpL = llm_build_inp_embd(ctx0, lctx, hparams, ubatch, model.tok_embd, cb);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

        cur = ggml_conv_1d_ph(ctx0, model.conv1d, cur, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_b);

        // posnet
        for (uint32_t il = 0; il < hparams.posnet.n_layer; ++il) {
            const auto& layer = model.layers[il].posnet;

            inpL = cur;

            switch (il) {
            case 0:
            case 1:
            case 3:
            case 4:
            {
                cur = llm_build_norm(ctx0, cur, hparams,
                    layer.norm1,
                    layer.norm1_b,
                    LLM_NORM_GROUP, cb, 0);

                cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                cur = ggml_conv_1d_ph(ctx0, layer.conv1, cur, 1, 1);
                cur = ggml_add(ctx0, cur, layer.conv1_b);

                cur = llm_build_norm(ctx0, cur, hparams,
                    layer.norm2,
                    layer.norm2_b,
                    LLM_NORM_GROUP, cb, 0);

                cur = ggml_mul(ctx0, ggml_sigmoid(ctx0, cur), cur);

                cur = ggml_conv_1d_ph(ctx0, layer.conv2, cur, 1, 1);
                cur = ggml_add(ctx0, cur, layer.conv2_b);

                cur = ggml_add(ctx0, cur, inpL);
            } break;
            case 2:
            {
                cur = llm_build_norm(ctx0, cur, hparams,
                    layer.attn_norm,
                    layer.attn_norm_b,
                    LLM_NORM_GROUP, cb, 0);

                struct ggml_tensor* q;
                struct ggml_tensor* k;
                struct ggml_tensor* v;

                q = ggml_conv_1d_ph(ctx0, layer.attn_q, cur, 1, 1);
                k = ggml_conv_1d_ph(ctx0, layer.attn_k, cur, 1, 1);
                v = ggml_conv_1d_ph(ctx0, layer.attn_v, cur, 1, 1);

                q = ggml_add(ctx0, q, layer.attn_q_b);
                k = ggml_add(ctx0, k, layer.attn_k_b);
                v = ggml_add(ctx0, v, layer.attn_v_b);

                q = ggml_cont(ctx0, ggml_transpose(ctx0, q));
                k = ggml_cont(ctx0, ggml_transpose(ctx0, k));

                struct ggml_tensor* kq = ggml_mul_mat(ctx0, k, q);

                kq = ggml_soft_max(ctx0, kq, nullptr, 1.0f / sqrtf(float(hparams.posnet.n_embd)), 0.0f);

                cur = ggml_mul_mat(ctx0, kq, v);

                cur = ggml_conv_1d_ph(ctx0, layer.attn_o, cur, 1, 1);
                cur = ggml_add(ctx0, cur, layer.attn_o_b);

                cur = ggml_add(ctx0, cur, inpL);
            } break;
            case 5:
            {
                cur = llm_build_norm(ctx0, cur, hparams,
                    layer.norm,
                    layer.norm_b,
                    LLM_NORM_GROUP, cb, 0);
            } break;
            default: GGML_ABORT("unknown posnet layer");
            };
        }

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = llm_build_norm(ctx0, cur, hparams,
            model.tok_norm,
            model.tok_norm_b,
            LLM_NORM, cb, -1);

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        inpL = cur;

        // convnext
        for (uint32_t il = 0; il < hparams.convnext.n_layer; ++il) {
            const auto& layer = model.layers[il].convnext;

            cur = inpL;

            cur = ggml_conv_1d_dw_ph(ctx0, layer.dw, cur, 1, 1);
            cur = ggml_add(ctx0, cur, layer.dw_b);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            cur = llm_build_norm(ctx0, cur, hparams,
                layer.norm,
                layer.norm_b,
                LLM_NORM, cb, -1);

            cur = llm_build_ffn(ctx0, lctx, cur,
                layer.pw1, layer.pw1_b, NULL,
                NULL, NULL, NULL,
                layer.pw2, layer.pw2_b, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, cb, il);

            cur = ggml_mul(ctx0, cur, layer.gamma);

            cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

            inpL = ggml_add(ctx0, cur, inpL);
        }

        cur = inpL;

        cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

        cur = llm_build_norm(ctx0, cur, hparams,
            model.output_norm,
            model.output_norm_b,
            LLM_NORM, cb, -1);

        // lm_head
        cur = llm_build_lora_mm(lctx, ctx0, model.output, cur);

        cur = ggml_add(ctx0, cur, model.output_b);
        cb(cur, "result_embd", -1);

        gf->build_forward_expand(cur);

        return gf;
    }
};

static ggml_cgraph* llama_build_graph(
    llama_context& lctx,
    const llama_ubatch& ubatch,
    bool   worst_case) {
    const auto& model = lctx.model;

    // this callback allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
    llm_build_cb cb = [&](struct ggml_tensor* cur, const char* name, int il) {
        if (il >= 0) {
            cur->set_name("{}-{}", name, il);
        }
        else {
            cur->set_name(name);
        }

        if (!lctx.cparams.offload_kqv) {
            if (strcmp(name, "kqv_merged_cont") == 0) {
                // all nodes between the KV store and the attention output are run on the CPU
                lctx.sched->set_tensor_backend(cur, lctx.backend_cpu);
            }
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = lctx.model.n_gpu_layers > (int)lctx.model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto& dev_layer = lctx.model.dev_layer.at(il);
                for (auto& backend : lctx.backends) {
                    if (backend->get_device() == dev_layer.dev) {
                        // TODO
#if 0
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            lctx.sched->set_tensor_backend(cur, backend.get());
                        }
#endif
                    }
                }
            }
        }
    };

    ggml_cgraph* result = NULL;

    struct llm_build_context llm(lctx, ubatch, cb, worst_case);

    llm.init();

    switch (model.arch) {
    case LLM_ARCH_LLAMA:
    case LLM_ARCH_MINICPM:
    case LLM_ARCH_GRANITE:
    case LLM_ARCH_GRANITE_MOE:
    {
        result = llm.build_llama();
    } break;
    case LLM_ARCH_DECI:
    {
        result = llm.build_deci();
    } break;
    case LLM_ARCH_BAICHUAN:
    {
        result = llm.build_baichuan();
    } break;
    case LLM_ARCH_FALCON:
    {
        result = llm.build_falcon();
    } break;
    case LLM_ARCH_GROK:
    {
        result = llm.build_grok();
    } break;
    case LLM_ARCH_STARCODER:
    {
        result = llm.build_starcoder();
    } break;
    case LLM_ARCH_REFACT:
    {
        result = llm.build_refact();
    } break;
    case LLM_ARCH_BERT:
    case LLM_ARCH_JINA_BERT_V2:
    case LLM_ARCH_NOMIC_BERT:
    {
        result = llm.build_bert();
    } break;
    case LLM_ARCH_BLOOM:
    {
        result = llm.build_bloom();
    } break;
    case LLM_ARCH_MPT:
    {
        result = llm.build_mpt();
    } break;
    case LLM_ARCH_STABLELM:
    {
        result = llm.build_stablelm();
    } break;
    case LLM_ARCH_QWEN:
    {
        result = llm.build_qwen();
    } break;
    case LLM_ARCH_QWEN2:
    {
        result = llm.build_qwen2();
    } break;
    case LLM_ARCH_QWEN2VL:
    {
        lctx.n_pos_per_token = 4;
        result = llm.build_qwen2vl();
    } break;
    case LLM_ARCH_QWEN2MOE:
    {
        result = llm.build_qwen2moe();
    } break;
    case LLM_ARCH_PHI2:
    {
        result = llm.build_phi2();
    } break;
    case LLM_ARCH_PHI3:
    {
        result = llm.build_phi3();
    } break;
    case LLM_ARCH_PLAMO:
    {
        result = llm.build_plamo();
    } break;
    case LLM_ARCH_GPT2:
    {
        result = llm.build_gpt2();
    } break;
    case LLM_ARCH_CODESHELL:
    {
        result = llm.build_codeshell();
    } break;
    case LLM_ARCH_ORION:
    {
        result = llm.build_orion();
    } break;
    case LLM_ARCH_INTERNLM2:
    {
        result = llm.build_internlm2();
    } break;
    case LLM_ARCH_MINICPM3:
    {
        result = llm.build_minicpm3();
    } break;
    case LLM_ARCH_GEMMA:
    {
        result = llm.build_gemma();
    } break;
    case LLM_ARCH_GEMMA2:
    {
        result = llm.build_gemma2();
    } break;
    case LLM_ARCH_STARCODER2:
    {
        result = llm.build_starcoder2();
    } break;
    case LLM_ARCH_MAMBA:
    {
        result = llm.build_mamba();
    } break;
    case LLM_ARCH_XVERSE:
    {
        result = llm.build_xverse();
    } break;
    case LLM_ARCH_COMMAND_R:
    {
        result = llm.build_command_r();
    } break;
    case LLM_ARCH_DBRX:
    {
        result = llm.build_dbrx();
    } break;
    case LLM_ARCH_OLMO:
    {
        result = llm.build_olmo();
    } break;
    case LLM_ARCH_OLMO2:
    {
        result = llm.build_olmo2();
    } break;
    case LLM_ARCH_OLMOE:
    {
        result = llm.build_olmoe();
    } break;
    case LLM_ARCH_OPENELM:
    {
        result = llm.build_openelm();
    } break;
    case LLM_ARCH_GPTNEOX:
    {
        result = llm.build_gptneox();
    } break;
    case LLM_ARCH_ARCTIC:
    {
        result = llm.build_arctic();
    } break;
    case LLM_ARCH_DEEPSEEK:
    {
        result = llm.build_deepseek();
    } break;
    case LLM_ARCH_DEEPSEEK2:
    {
        result = llm.build_deepseek2();
    } break;
    case LLM_ARCH_CHATGLM:
    {
        result = llm.build_chatglm();
    } break;
    case LLM_ARCH_BITNET:
    {
        result = llm.build_bitnet();
    } break;
    case LLM_ARCH_T5:
    {
        if (lctx.is_encoding) {
            result = llm.build_t5_enc();
        }
        else {
            result = llm.build_t5_dec();
        }
    } break;
    case LLM_ARCH_T5ENCODER:
    {
        result = llm.build_t5_enc();
    } break;
    case LLM_ARCH_JAIS:
    {
        result = llm.build_jais();
    } break;
    case LLM_ARCH_NEMOTRON:
    {
        result = llm.build_nemotron();
    } break;
    case LLM_ARCH_EXAONE:
    {
        result = llm.build_exaone();
    } break;
    case LLM_ARCH_RWKV6:
    {
        result = llm.build_rwkv6();
    } break;
    case LLM_ARCH_CHAMELEON:
    {
        result = llm.build_chameleon();
    } break;
    case LLM_ARCH_WAVTOKENIZER_DEC:
    {
        result = llm.build_wavtokenizer_dec();
    } break;
    default:
        GGML_ABORT("fatal error");
    }

    // add on pooling layer
    if (lctx.cparams.embeddings) {
        result = llm.append_pooling(result);
    }

    llm.free();

    return result;
}

llama_context* llama_new_context_with_model(
    llama_model* model,
    llama_context_params &params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (params.flash_attn && model->hparams.n_embd_head_k != model->hparams.n_embd_head_v) {
        LLAMA_LOG_WARN("%s: flash_attn requires n_embd_head_k == n_embd_head_v - forcing off\n", __func__);
        params.flash_attn = false;
    }

    if (ggml_is_quantized(params.type_v) && !params.flash_attn) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    auto ctx = std::make_unique<llama_context>(*model);

    const auto& hparams = model->hparams;
    auto& cparams = ctx->cparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    cparams.n_threads = params.n_threads;
    cparams.n_threads_batch = params.n_threads_batch;
    cparams.yarn_ext_factor = params.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor;
    cparams.yarn_beta_fast = params.yarn_beta_fast;
    cparams.yarn_beta_slow = params.yarn_beta_slow;
    cparams.defrag_thold = params.defrag_thold;
    cparams.embeddings = params.embeddings;
    cparams.offload_kqv = params.offload_kqv;
    cparams.flash_attn = params.flash_attn;
    cparams.no_perf = params.no_perf;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;
    cparams.rope_freq_base = params.rope_freq_base == 0.0f ? hparams.rope_freq_base_train : params.rope_freq_base;
    cparams.rope_freq_scale = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    // this is necessary due to kv_self.n being padded later during inference
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, llama_kv_cache_get_padding(cparams));

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = hparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    // the batch has to be at least GGML_KQ_MASK_PAD because we will be padding the KQ_mask
    // this is required by GPU kernels in order to avoid out-of-bounds accesses (e.g. ggml_flash_attn_ext)
    // ref: https://github.com/ggerganov/llama.cpp/pull/5021
    if (cparams.n_batch < GGML_KQ_MASK_PAD) {
        LLAMA_LOG_WARN("%s: n_batch is less than GGML_KQ_MASK_PAD - increasing to %d\n", __func__, GGML_KQ_MASK_PAD);
        cparams.n_batch = GGML_KQ_MASK_PAD;
    }

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_ctx_orig_yarn = params.yarn_orig_ctx != 0 ? params.yarn_orig_ctx :
        hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
        hparams.n_ctx_train;

    cparams.cb_eval = params.cb_eval;

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        }
        else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    }
    else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    const uint32_t n_ctx_per_seq = cparams.n_ctx / cparams.n_seq_max;

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n", __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n", __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_per_seq = %u\n", __func__, n_ctx_per_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n", __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n", __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: flash_attn    = %d\n", __func__, cparams.flash_attn);
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n", __func__, cparams.rope_freq_scale);

    if (n_ctx_per_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_per_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
            __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    if (n_ctx_per_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_pre_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
            __func__, n_ctx_per_seq, hparams.n_ctx_train);
    }

    ctx->logits_all = params.logits_all;

    // build worst-case graph for encoder if a model contains encoder
    ctx->is_encoding = llama_model_has_encoder(model);

    uint32_t kv_size = cparams.n_ctx;
    ggml_type type_k = params.type_k;
    ggml_type type_v = params.type_v;

    // Mamba only needs a constant number of KV cache cells per sequence
    if (llama_model_is_recurrent(model)) {
        // Mamba needs at least as many KV cells as there are sequences kept at any time
        kv_size = std::max((uint32_t)1, params.n_seq_max);
        // it's probably best to keep as much precision as possible for the states
        type_k = GGML_TYPE_F32; // required by ggml_ssm_conv for Mamba's conv_states
        type_v = GGML_TYPE_F32; // required by ggml_ssm_scan for Mamba's ssm_states
    }

    GGML_ASSERT(hparams.n_embd_head_k % ggml_blck_size(type_k) == 0);
    GGML_ASSERT(hparams.n_embd_head_v % ggml_blck_size(type_v) == 0);

    if (!hparams.vocab_only) {
        // GPU backends
        for (auto* dev : model->devices) {
            std::unique_ptr<ggml_backend> backend = dev->init_backend(nullptr);
            if (backend == nullptr) {
                LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                return nullptr;
            }
            ctx->backends.emplace_back(std::move(backend));
        }

        // add ACCEL backends (such as BLAS)
        for (auto dev : backend_devs()) {
            if (dev->get_type() == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                std::unique_ptr<ggml_backend> backend = dev->init_backend(nullptr);
                if (backend == nullptr) {
                    LLAMA_LOG_ERROR("%s: failed to initialize %s backend\n", __func__, ggml_backend_dev_name(dev));
                    return nullptr;
                }
                ctx->backends.emplace_back(std::move(backend));
            }
        }

        // add CPU backend
        ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr).release();
        if (ctx->backend_cpu == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to initialize CPU backend\n", __func__);
            return nullptr;
        }
        ctx->backends.emplace_back(ctx->backend_cpu);

        llama_set_abort_callback(ctx.get(), params.abort_callback);

        if (!llama_kv_cache_init(ctx->kv_self, ctx.get(), type_k, type_v, kv_size, cparams.offload_kqv)) {
            LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
            return nullptr;
        }

        {
            size_t memory_size_k = 0;
            size_t memory_size_v = 0;

            for (auto& k : ctx->kv_self.k_l) {
                memory_size_k += k->nbytes();
            }

            for (auto& v : ctx->kv_self.v_l) {
                memory_size_v += v->nbytes();
            }

            LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f),
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
        }

        // graph outputs buffer
        {
            // resized during inference when a batch uses more outputs
            if (llama_output_reserve(*ctx, params.n_seq_max) < params.n_seq_max) {
                LLAMA_LOG_ERROR("%s: failed to reserve initial output buffer\n", __func__);
                return nullptr;
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                ggml_backend_buffer_name(ctx->buf_output.get()),
                ggml_backend_buffer_get_size(ctx->buf_output.get()) / 1024.0 / 1024.0);
        }

        // scheduler and compute buffers
        {
            // buffer types used for the compute buffer of each backend
            std::vector<ggml_backend_buffer_type*> backend_buft;
            std::vector<ggml_backend*> backend_ptrs;
            for (auto& backend : ctx->backends) {
                auto* buft = backend->get_default_buffer_type();
                auto backend_type = backend->get_device()->get_type();
                if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model->devices.empty()) {
                    // use the host buffer of the first device CPU for faster transfer of the intermediate state
                    auto* dev = model->devices[0];
                    auto* host_buft = dev->get_host_buffer_type();
                    if (host_buft) {
                        buft = host_buft;
                    }
                }
                backend_buft.push_back(buft);
                backend_ptrs.push_back(backend.get());
            }

            // TODO: move these checks to ggml_backend_sched
            // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
            bool pipeline_parallel =
                llama_get_device_count(*model) > 1 &&
                model->n_gpu_layers > (int)model->hparams.n_layer &&
                model->split_mode == LLAMA_SPLIT_MODE_LAYER &&
                params.offload_kqv;

            // pipeline parallelism requires support for async compute and events in all devices
            if (pipeline_parallel) {
                for (auto& backend : ctx->backends) {
                    auto dev_type = backend->get_device()->get_type();
                    if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                        // ignore CPU backend
                        continue;
                    }
                    auto* dev = backend->get_device();
                    ggml_backend_dev_props props;
                    dev->get_props(&props);
                    if (!props.caps.async || !props.caps.events) {
                        // device does not support async compute or events
                        pipeline_parallel = false;
                        break;
                    }
                }
            }

            ctx->sched.reset(new ggml_backend_sched(nullptr, backend_buft.data(), backend_ptrs.size(), pipeline_parallel, true));

            if (pipeline_parallel) {
                LLAMA_LOG_INFO("%s: pipeline parallelism enabled (n_copies=%d)\n", __func__, ggml_backend_sched_get_n_copies(ctx->sched.get()));
            }

            // initialize scheduler with the worst-case graph
            uint32_t n_seqs = 1; // TODO: worst-case number of sequences
            uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);
            llama_token token = llama_token_bos(&ctx->model); // not actually used by llama_build_graph, but required to choose between token and embedding inputs graph

            llama_ubatch ubatch_pp = { true, n_tokens, n_tokens / n_seqs, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr };
            ggml_cgraph* gf_pp = llama_build_graph(*ctx, ubatch_pp, true);

            // reserve pp graph first so that buffers are only allocated once
            ctx->sched->reserve(gf_pp);
            int n_splits_pp = ctx->sched->splits.size();
            int n_nodes_pp = gf_pp->nodes.size();

            // reserve with tg graph to get the number of splits and nodes
            llama_ubatch ubatch_tg = { true, 1, 1, n_seqs, &token, nullptr, nullptr, nullptr, nullptr, nullptr };
            ggml_cgraph* gf_tg = llama_build_graph(*ctx, ubatch_tg, true);
            ctx->sched->reserve(gf_tg);
            int n_splits_tg = ctx->sched->splits.size();
            int n_nodes_tg = gf_tg->nodes.size();

            // reserve again with pp graph to avoid ggml-alloc reallocations during inference
            gf_pp = llama_build_graph(*ctx, ubatch_pp, true);
            if (!ctx->sched->reserve(gf_pp)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
                return nullptr;
            }

            for (size_t i = 0; i < backend_ptrs.size(); ++i) {
                ggml_backend* backend = backend_ptrs[i];
                ggml_backend_buffer_type* buft = backend_buft[i];
                size_t size = ctx->sched->get_buffer_size(backend);
                if (size > 1) {
                    LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                        ggml_backend_buft_name(buft),
                        size / 1024.0 / 1024.0);
                }
            }

            if (n_nodes_pp == n_nodes_tg) {
                LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
            }
            else {
                LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
            }
            if (n_splits_pp == n_splits_tg) {
                LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
            }
            else {
                LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
            }
        }
    }

    return ctx.release();
}

common_init_result common_init_from_params(common_params& params)
{
    auto mparams = common_model_params_to_llama(params);
    llama_model* model = new llama_model;
    if (!params.hf_repo.empty() && !params.hf_file.empty()) {
        // TODO
        //model = common_load_model_from_hf(params.hf_repo, params.hf_file, params.model, params.hf_token, mparams);
    }
    else if (!params.model_url.empty()) {
        // TODO
        //model = common_load_model_from_url(params.model_url, params.model, params.hf_token, mparams);
    }
    else {
        model = llama_load_model_from_file(params.model, mparams);
    }
    if (model == NULL) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.c_str());
        return {};
    }

    common_init_result iparams;
    // TODO
#if 0
    if (params.reranking) {
        bool ok = true;

        if (llama_token_bos(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have a  BOS token, reranking will not work\n", __func__);
            ok = false;
        }

        if (llama_token_eos(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have an EOS token, reranking will not work\n", __func__);
            ok = false;
        }

        if (llama_token_sep(model) == LLAMA_TOKEN_NULL) {
            LOG_WRN("%s: warning: model does not have a  SEP token, reranking will not work\n", __func__);
            ok = false;
        }

        if (!ok) {
            llama_free_model(model);

            return iparams;
        }
    }
#endif

    auto cparams = common_context_params_to_llama(params);

    llama_context* lctx = llama_new_context_with_model(model, cparams);
    if (!lctx) {
        LOG_ERR("%s: failed to create context with model '%s'\n", __func__, params.model.c_str());
        delete model;
        return iparams;
    }

    return {};
}

module;
#include <stdint.h>
#include <array>
#include <string>
#include <variant>
#include <vector>

module llm:ds;
import ggml;

constexpr std::array<char, 4> GGUF_MAGIC = { 'G', 'G' ,'U', 'F' };
constexpr size_t GGUF_DEFAULT_ALIGNMENT = 32;

using gguf_str = std::string; // GGUFv2

enum bool_value : bool {};

using gguf_value = std::variant<
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    uint64_t,
    int64_t,
    double,
    bool,
    gguf_str,
    std::vector<uint8_t>,
    std::vector<int8_t>,
    std::vector<uint16_t>,
    std::vector<int16_t>,
    std::vector<uint32_t>,
    std::vector<int32_t>,
    std::vector<float>,
    std::vector<uint64_t>,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<bool_value>, // special case, because std::vector<bool> is not a normal verctor
    std::vector<gguf_str>
>;

bool is_array(const gguf_value&);
std::string toString(const gguf_value&);
size_t getArrSize(const gguf_value&);

template <typename T>
struct typeName;
template <>
struct typeName<uint8_t>
{
    static constexpr const char* value = "u8";
};

template <>
struct typeName<int8_t>
{
    static constexpr const char* value = "i8";
};

template <>
struct typeName<uint16_t>
{
    static constexpr const char* value = "u16";
};

template <>
struct typeName<int16_t>
{
    static constexpr const char* value = "i16";
};

template <>
struct typeName<uint32_t>
{
    static constexpr const char* value = "u32";
};

template <>
struct typeName<int32_t>
{
    static constexpr const char* value = "i32";
};

template <>
struct typeName<float>
{
    static constexpr const char* value = "f32";
};

template <>
struct typeName<uint64_t>
{
    static constexpr const char* value = "u64";
};

template <>
struct typeName<int64_t>
{
    static constexpr const char* value = "i64";
};

template <>
struct typeName<double>
{
    static constexpr const char* value = "f64";
};

template <>
struct typeName<std::string>
{
    static constexpr const char* value = "str";
};

template <>
struct typeName<bool>
{
    static constexpr const char* value = "bool";
};

struct gguf_header {
    std::array<char, 4> magic;

    uint32_t version;
    uint64_t n_tensors; // GGUFv2
    uint64_t n_kv;      // GGUFv2
};

struct gguf_kv {
    gguf_str key;
    gguf_value value;
};

struct gguf_tensor_info {
    gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void* data;
    size_t size;
};

struct gguf_context {
    gguf_header header;

    std::vector<gguf_kv> kv;
    std::vector<gguf_tensor_info> infos;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset;    // offset of `data` from beginning of file
    size_t size = 0;      // size of `data` in bytes

    //uint8_t * padding;
    void* data = nullptr;
public:
    size_t get_data_offset() const;
    size_t get_tensor_offset(size_t idx) const;
    std::string_view get_tensor_name(size_t idx) const;
    // return -1 if key not found
    int find_key(std::string_view key) const;
    int get_n_kv() const;
    int get_version() const;
};

struct gguf_init_params {
    bool no_alloc;
};

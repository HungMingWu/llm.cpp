module;
#include <stdint.h>
#include <array>
#include <string>
#include <variant>
#include <vector>

module llm:ds;
import ggml;

constexpr std::array<char, 4> GGUF_MAGIC = { 'G', 'G' ,'U', 'F' };

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

struct gguf_init_params {
    bool no_alloc;
};

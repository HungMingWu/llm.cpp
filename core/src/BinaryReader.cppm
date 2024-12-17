module;
#include <format>
#include <fstream>
#include <variant>
module llm:Binaryreader;
import :ds;
import :Enumeration;
import :Util;

 class BinaryReader {
 public:
     explicit BinaryReader(std::ifstream ifs)
         : ifs_(std::move(ifs)) {
     }

     template <typename T>
     BinaryReader& operator>>(T& value) {
         ifs_.read(reinterpret_cast<char*>(&value), sizeof(T));
         if (ifs_.gcount() != sizeof(T)) {
             throw std::exception{};
         }
         return *this;
     }

     BinaryReader& operator>>(gguf_str& value) {
         uint64_t n;
         (*this) >> n;

         // early exit if string length is invalid, prevents from integer overflow
         if (n == SIZE_MAX) {
             throw make_format_runtime_error("invalid string length ({})", n);
         }
         value.resize(n);
         ifs_.read(&value[0], n);
         if (ifs_.gcount() != n) {
             throw std::exception{};
         }
         return *this;
     }

     BinaryReader& operator>>(gguf_value& value) {
         gguf_type type;
         (*this) >> type;

         switch (type) {
             case GGUF_TYPE_UINT8: {
                 uint8_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_INT8: {
                 int8_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_UINT16: {
                 uint16_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_INT16: {
                 int16_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_UINT32: {
                 uint32_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_INT32: {
                 int32_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_FLOAT32: {
                 float v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_UINT64: {
                 uint64_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_INT64: {
                 int64_t v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_FLOAT64: {
                 double v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_BOOL: {
                 bool v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_STRING: {
                 gguf_str v;
                 (*this) >> v;
                 value = v;
                 break;
             }
             case GGUF_TYPE_ARRAY: {
                 gguf_type arr_type;
                 uint64_t arr_n;
                 (*this) >> arr_type >> arr_n;
                 switch (arr_type) {
                     case GGUF_TYPE_UINT8: {
                         arrayBuildhelper<uint8_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_INT8: {
                         arrayBuildhelper<int8_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_UINT16: {
                         arrayBuildhelper<uint16_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_INT16: {
                         arrayBuildhelper<int16_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_UINT32: {
                         arrayBuildhelper<uint32_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_INT32: {
                         arrayBuildhelper<int32_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_FLOAT32: {
                         arrayBuildhelper<float>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_UINT64: {
                         arrayBuildhelper<uint64_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_INT64: {
                         arrayBuildhelper<int64_t>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_FLOAT64: {
                         arrayBuildhelper<double>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_BOOL: {
                         arrayBuildhelper<bool_value>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_STRING: {
                         arrayBuildhelper<gguf_str>(arr_n, value);
                         break;
                     }
                     case GGUF_TYPE_ARRAY:
                     default: {
                         throw make_format_runtime_error("invalid array type {}", static_cast<int>(arr_type));
                     }
                 }
                 break;
             }
         default: {
             throw make_format_runtime_error("invalid type {}", static_cast<int>(type));
	    }
         }
         return *this;
     }

     size_t getOffset() {
         return ifs_.tellg();
     }

     void setOffset(size_t offset) {
         ifs_.seekg(offset, std::ifstream::beg);
     }

 private:
     template <typename T>
     void arrayBuildhelper(uint64_t arr_n, gguf_value& out)
     {
         std::vector<T> v(arr_n);
         for (auto& item : v)
             (*this) >> item;
         out = std::move(v);
     }
     std::ifstream ifs_;
 };

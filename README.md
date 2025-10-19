### llm.cpp

An experimental C++ inference engine for large language models.
Inspired by [ggml](https://github.com/ggml-org/ggml), [llama.cpp](https://github.com/ggml-org/llama.cpp) and [chatllm.cpp](https://github.com/foldl/chatllm.cpp)
Based on C++26 Standard

### Key Feature
- C++20 Module
- C++26 Execution, Sender / Receiver based model
- C++23 MDSpan

### TODO
- C++26 SIMD (No compiler support yet)
- C++26 Static refelection (Clang P2296 branch support, but no mainstream compiler support yet)

### How to build

#### MSVC
Test based on Visual studio 17.14.17
Use Visual studio and just open the folder

#### GCC
GCC15 module support is incomplete, skip it right now

#### Clang
Test Compiler clang21.0
``` bash
$ CXX=clang CXXFLAGS="-stdlib=libstdc++" LDFLAGS="-lstdc++" cmake -S . -B build -G Ninja
$ cmake --build  build/
```

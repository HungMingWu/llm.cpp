add_library(libllm)
target_sources(libllm
  PUBLIC
    FILE_SET primary_interface
    TYPE CXX_MODULES
    FILES
      llm.cppm
  PRIVATE
    FILE_SET implementaion_units
    TYPE CXX_MODULES
    FILES
      src/enum.cppm
      src/Util.cppm
      src/ds.cppm
  PRIVATE
      src/llm.cpp
      src/unicode.cpp
      src/unicode-data.cpp
)

target_compile_features(libllm
  PUBLIC
    cxx_std_23
)

target_link_libraries(libllm
  PRIVATE
    libggml
)

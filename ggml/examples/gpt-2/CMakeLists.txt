add_library(gpt_common OBJECT)
target_sources(gpt_common
  PUBLIC
    FILE_SET primary_interface
    TYPE CXX_MODULES
    FILES
      common.cppm
  PRIVATE
      common.cpp
)

target_link_libraries(gpt_common
  PRIVATE
    libggml
)

target_compile_features(gpt_common
  PUBLIC
    cxx_std_23
)

add_executable(gpt-2-quantize)
target_sources(gpt-2-quantize
  PRIVATE
      quantize.cpp
)

target_link_libraries(gpt-2-quantize
  PRIVATE
    libggml
    gpt_common
)

add_executable(main-batched)
target_sources(main-batched
  PRIVATE
      main-batched.cpp
)

target_link_libraries(main-batched
  PRIVATE
    libggml
    gpt_common
)
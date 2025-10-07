add_executable(gen_table)

target_sources(gen_table
  PRIVATE
    table.cpp
)

target_compile_features(gen_table
  PUBLIC
    cxx_std_23
)

target_link_libraries(gen_table
  PRIVATE
    libllm
)
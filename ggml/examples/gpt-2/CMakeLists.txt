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

function(create_new_gpt2 gpt2_target target_sources)
    add_executable(${gpt2_target})
    target_sources(${gpt2_target}
      PRIVATE
      ${target_sources}.cpp
    )
    target_link_libraries(${gpt2_target}
      PRIVATE
        libggml
        gpt_common
    )
endfunction()

create_new_gpt2(gpt-2-alloc main-alloc)
create_new_gpt2(gpt-2-batched main-batched)
create_new_gpt2(gpt-2-quantize quantize)
create_new_gpt2(gpt-2-sched main-sched)
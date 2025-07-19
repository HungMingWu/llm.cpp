add_library(libtest OBJECT)
target_sources(libtest
  PUBLIC
    FILE_SET primary_test_interface
    TYPE CXX_MODULES
    FILES
      test.cppm
)

target_compile_features(libtest
  PUBLIC
    cxx_std_23
)

target_link_libraries(libtest
  PUBLIC
    libggml
)

function(create_new_test testCase)
    add_executable(${testCase})
    target_sources(${testCase}
      PRIVATE
      ${testCase}.cpp
    )
    target_link_libraries(${testCase}
      PRIVATE
        libggml
        libtest
    )
endfunction()

create_new_test(test-arange)
create_new_test(test-backend-ops)
create_new_test(test-conv1d)
create_new_test(test-conv2d-dw)
create_new_test(test-conv2d)
create_new_test(test-customop)
create_new_test(test-conv-transpose)
create_new_test(test-dup)
create_new_test(test-interpolate)
create_new_test(test-opt)
create_new_test(test-pad-reflect-1d)
create_new_test(test-pool)
create_new_test(test-rel-pos)
create_new_test(test-roll)
create_new_test(test-timestep_embedding)
function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        string(TOLOWER "${backend}" backend_target)
        string(TOUPPER "GGML_USE_${backend}" backend_use)
        message(STATUS "Including ${backend} backend")
        add_subdirectory(${backend_target})
        target_compile_definitions(libggml PUBLIC ${backend_use})
    endif()
endfunction()

add_library(libggml OBJECT)

add_subdirectory(cpu)
add_subdirectory(rpc)
add_subdirectory(stdexec)
find_package(CUDAToolkit)

configure_file (${CMAKE_CURRENT_LIST_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
if (GGML_CUDA AND CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found, build CUDA Backend")
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # native == GPUs available at build time
        # 52     == Maxwell, lowest CUDA 12 standard
        # 60     == P100, FP16 CUDA intrinsics
        # 61     == Pascal, __dp4a instruction (per-byte integer dot product)
        # 70     == V100, FP16 tensor cores
        # 75     == Turing, int8 tensor cores
        if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES "native")
        elseif(GGML_CUDA_F16 OR GGML_CUDA_DMMV_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75")
        else()
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75")
        endif()
    endif()
    set_target_properties(libggml PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    enable_language(CUDA)
    ggml_add_backend(cuda)
else()
    message("CUDA Toolkit not found")
endif()

target_sources(libggml
  PUBLIC
    FILE_SET primary_interface
    TYPE CXX_MODULES
    FILES
      alloc.cppm
      ds.cppm
      func.cppm
      ggml.cppm
      gguf.cppm
      log.cppm
      op.cppm
      op_back.cppm
      tensor.cppm
      types.cppm
      traits.cppm
      utility.cppm
  PRIVATE
    FILE_SET private_interface
    TYPE CXX_MODULES
    FILES
      host_buffer.cppm
      quants.cppm
      os/os.cppm
  PRIVATE
      allocator.cpp
      backend.cpp
      backend_registry.cpp
      buffer.cpp
      buffer_type.cpp
      cgraph.cpp
      context.cpp
      func.cpp
      gguf.cpp
      log.cpp
      op.cpp
      op_back.cpp
      quants.cpp
      sched.cpp
      tensor.cpp
      types.cpp
)

if (CMAKE_HOST_WIN32)
target_sources(libggml
  PRIVATE
      os/win32.cpp
)
elseif (CMAKE_HOST_APPLE)
elseif (CMAKE_HOST_UNIX)
target_sources(libggml
  PRIVATE
      os/unix.cpp
)
endif()

target_compile_features(libggml
  PUBLIC
    cxx_std_23
)

target_include_directories(libggml
  PRIVATE
    ./
    ${CMAKE_CURRENT_BINARY_DIR}/
)

target_link_libraries(libggml
  PRIVATE
    STDEXEC::stdexec
)

if (WIN32)
target_link_libraries(libggml
  PRIVATE
    Ws2_32
)
elseif (LINUX)
target_link_libraries(libggml
  PRIVATE
    m
)
endif()

add_subdirectory(examples)
add_subdirectory(tests)

add_executable(simple-backend)
target_sources(simple-backend
  PRIVATE
    simple-backend.cpp
)

target_link_libraries(simple-backend
  PRIVATE
    libggml
)
add_executable(magika)
target_sources(magika
  PRIVATE
    main.cpp
)

target_link_libraries(magika
  PRIVATE
    libggml
)
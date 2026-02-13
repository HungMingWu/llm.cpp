add_executable(chatllm)
target_sources(chatllm
  PUBLIC
    FILE_SET chatllm_interface
    TYPE CXX_MODULES
    FILES
      backend.cppm
      chat.cppm
      chatllm.cppm
      layers.cppm
      models.cppm
  PRIVATE
    FILE_SET chatllm_hidden_interface
    TYPE CXX_MODULES
    FILES
      chat_encoders.cppm
      custom_ops.cppm
      models/models_priv.cppm
      models/qwen.cppm
      models/qwen_asr.cppm
      models/qwen_tts.cppm
  PRIVATE
      audio_process.cpp
      backend.cpp
      chat.cpp
      chat_encoders.cpp
      layers.cpp
      main.cpp
      tokenizer.cpp
      unicode.cpp
      unicode-data.cpp
      vectorstore.cpp
      models.cpp
      vision_process.cpp
      models/models_priv.cpp
      models/qwen.cpp
      models/qwen_asr.cpp
      models/qwen_tts.cpp
)

target_link_libraries(chatllm
  PRIVATE
    libllm
)

target_compile_features(chatllm
  PUBLIC
    cxx_std_23
)

target_compile_options(chatllm
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/utf-8>
)
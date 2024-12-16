add_library(MathFunctions)
target_sources(MathFunctions
  PUBLIC
    FILE_SET primary_interface
    TYPE CXX_MODULES
    FILES
      MathFunctions.cppm
  PRIVATE
    FILE_SET implementaion_units
    TYPE CXX_MODULES
    FILES
      src/mysqrt.cppm
  PRIVATE
    src/MathFunctions.cxx
)

target_compile_features(MathFunctions
  PUBLIC
    cxx_std_20
)

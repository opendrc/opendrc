add_library(libgdsii gdsii.cpp)
target_include_directories(libgdsii PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_options(libgdsii PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
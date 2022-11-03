add_executable(overlap_query main.cpp)
target_include_directories(overlap_query PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(overlap_query PRIVATE libgdsii)
target_compile_options(overlap_query PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
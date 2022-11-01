add_library(libchecks space-check.cu width-check.cu)
target_include_directories(libchecks PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(libchecks PRIVATE ${CUDA_LIBRARIES} libcore)
target_compile_options(libchecks PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
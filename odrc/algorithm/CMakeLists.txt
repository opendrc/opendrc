add_library(libchecks space-check.cu space-check-enclosing.cu width-check.cpp width-check.cu width-xcheck.cu area-check.cpp)
target_include_directories(libchecks PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(libchecks PRIVATE ${CUDA_LIBRARIES} libcore libutil)
target_compile_options(libchecks PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)

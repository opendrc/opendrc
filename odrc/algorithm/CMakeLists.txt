add_library(libchecks space-check.cu space-check-dac23.cu width-check.cpp width-check.cu width-xcheck.cu area-check.cpp)
target_include_directories(libchecks PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(libchecks PRIVATE ${CUDA_LIBRARIES} libcore libutil)
target_compile_options(libchecks PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -O3 -std=c++17>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -std=c++17 -lcudadevrt --extended-lambda --expt-relaxed-constexpr>
)
set_property(TARGET libchecks PROPERTY CUDA_SEPARABLE_COMPILATION ON)
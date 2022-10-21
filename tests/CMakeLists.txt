add_executable(tests
  main.cpp
  core/sweepline.test.cpp
  core/sweepline.test.cu
  gdsii/gdsii.test.cpp
  utility/exception.test.cpp
  utility/logger.test.cpp
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(tests PRIVATE libcore libgdsii libutility)
target_compile_options(tests PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
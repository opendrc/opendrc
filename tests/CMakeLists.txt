add_executable(tests main.cpp infrastructure/sweepline/sweepline.test.cpp  infrastructure/sweepline/sweepline.test.cu interface/gdsii/gdsii.test.cpp)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(tests PRIVATE libgdsii libsweepline)
target_compile_options(tests PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
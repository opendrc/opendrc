add_subdirectory(algorithm)
add_subdirectory(core)
add_subdirectory(gdsii)
add_subdirectory(utility)
add_executable(odrc main.cpp)
target_include_directories(odrc PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(odrc PRIVATE libchecks libcore libgdsii libutil)
target_compile_options(odrc PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
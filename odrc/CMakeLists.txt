add_subdirectory(algorithm)
add_subdirectory(core)
add_subdirectory(db)
add_subdirectory(gdsii)
add_subdirectory(geometry)
add_subdirectory(utility)
add_executable(odrc main.cpp)
target_include_directories(odrc PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(odrc PRIVATE libchecks libcore libdb libgdsii libgeometry libutil)
target_compile_options(odrc PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -O3>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
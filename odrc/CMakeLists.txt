add_subdirectory(interface)
add_executable(odrc main.cpp)
target_include_directories(odrc PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(odrc PRIVATE libgdsii)
target_compile_options(odrc PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)
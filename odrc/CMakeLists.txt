add_subdirectory(interface)
add_executable(odrc main.cpp)
target_include_directories(odrc PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(odrc PRIVATE libgdsii)
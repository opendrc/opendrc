add_library(libcore INTERFACE)
target_include_directories(libcore INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(libcore INTERFACE ${CUDA_LIBRARIES} libutil)
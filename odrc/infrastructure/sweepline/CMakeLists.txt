add_library(libsweepline INTERFACE)
target_include_directories(libsweepline INTERFACE ${CMAKE_SOURCE_DIR})
target_link_libraries(libsweepline INTERFACE ${CUDA_LIBRARIES})
add_executable(tests main.cpp interface/gdsii/gdsii.test.cpp)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(tests PRIVATE libgdsii)
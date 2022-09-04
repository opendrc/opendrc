cmake_minimum_required(VERSION 3.9)

project(
  OpenDRC 
  VERSION 0.1.0
  DESCRIPTION "A Design Rule Checker with GPU Acceleration"
  LANGUAGES CXX)
  # LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ODRC_BUILD_TESTS "Enables builds of tests" ON)
option(ODRC_BUILD_EXAMPLES "Enables builds of examples" ON)


add_subdirectory(odrc)


if(ODRC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ODRC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
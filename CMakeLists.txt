cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  OpenDRC 
  VERSION 0.2.0
  DESCRIPTION "A Design Rule Checker with Hierarchical GPU Acceleration"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ODRC_BUILD_TESTS "Enables builds of tests" ON)
option(ODRC_BUILD_EXAMPLES "Enables builds of examples" ON)
option(ODRC_BUILD_CUDA "Enables builds of CUDA" ON)

if(ODRC_BUILD_CUDA)
  ENABLE_LANGUAGE(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(FindCUDA/select_compute_arch)
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  endif()
endif()


add_subdirectory(odrc)


if(ODRC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(ODRC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

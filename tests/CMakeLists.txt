add_executable(tests
  main.cpp
  algorithm/layout-partition.test.cpp
  core/sweepline.test.cpp
  core/sweepline.test.cu
<<<<<<< Updated upstream
  # db/cell.test.cpp
  # db/elem.test.cpp
=======
  db/db.test.cpp
  db/cell.test.cpp
  db/elem.test.cpp
>>>>>>> Stashed changes
  gdsii/gdsii.test.cpp
  geometry/box.test.cpp
  geometry/point.test.cpp
  geometry/polygon.test.cpp
  geometry/spatial.test.cpp
  utility/exception.test.cpp
  utility/logger.test.cpp
  utility/proxy.test.cpp
  utility/timer.test.cpp
)
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/thirdparty)
target_link_libraries(tests PRIVATE libcore libdb libgdsii libgeometry libutil)
target_compile_options(tests PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
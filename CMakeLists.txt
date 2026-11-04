cmake_minimum_required(VERSION 3.16)
project(fusion_tracking VERSION 1.0 LANGUAGES C CXX)

option(FUSION_BUILD_APPS "Build the command-line applications" ON)
option(FUSION_BUILD_VIEWER "Build the interactive 3D viewer (needs PCL with VTK)" ON)
option(FUSION_BUILD_TESTS "Build unit tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)

add_library(fusion_tracking
  src/config.cpp
  src/models.cpp
  src/filters/filter.cpp
  src/filters/ukf.cpp
  src/filters/ekf.cpp
  src/filters/imm.cpp
  src/tracking/hungarian.cpp
  src/tracking/tracker.cpp
  src/sim/vehicle.cpp
  src/sim/lidar_scanner.cpp
  src/sim/simulator.cpp
  src/eval/metrics.cpp
  src/experiment.cpp
)
target_include_directories(fusion_tracking PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(fusion_tracking PUBLIC Eigen3::Eigen PRIVATE yaml-cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fusion_tracking PRIVATE -Wall -Wextra -Wpedantic)
endif()
set_target_properties(fusion_tracking PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(FUSION_BUILD_APPS)
  add_executable(fusion_run apps/run.cpp)
  target_link_libraries(fusion_run PRIVATE fusion_tracking)

  add_executable(fusion_bench apps/bench.cpp)
  target_link_libraries(fusion_bench PRIVATE fusion_tracking)

  if(FUSION_BUILD_VIEWER)
    # Ubuntu's VTK package (pulled in by PCL's CMake config) references
    # MPI::MPI_C without finding MPI itself; finding it first avoids a
    # configure error (this is also why the project enables C).
    find_package(MPI QUIET COMPONENTS C)
    find_package(PCL 1.10 QUIET COMPONENTS common visualization)
    if(PCL_FOUND)
      add_executable(fusion_viewer apps/viewer.cpp)
      target_include_directories(fusion_viewer PRIVATE ${PCL_INCLUDE_DIRS})
      target_compile_definitions(fusion_viewer PRIVATE ${PCL_DEFINITIONS})
      target_link_libraries(fusion_viewer PRIVATE fusion_tracking ${PCL_LIBRARIES})
    else()
      message(STATUS "PCL visualization not found: skipping fusion_viewer")
    endif()
  endif()
endif()

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  add_custom_target(doc
    COMMAND ${CMAKE_COMMAND} -E make_directory build/doc
    COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building the API reference in build/doc/html")
endif()

if(FUSION_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

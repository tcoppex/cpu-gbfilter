cmake_minimum_required(VERSION 2.8)

project(gbfilter)

if(UNIX)
  set(SSE41_FLAGS "-msse4.1 -march=core2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -Wno-unused-result ${SSE41_FLAGS}")
  add_definitions("-DGBF_OMP_STATS")
elseif(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp /Ox")
  add_definitions("/DGBF_OMP_STATS")
else()
  message(FATAL_ERROR "Unknown operating system.")
endif()

add_executable(${CMAKE_PROJECT_NAME} gbfilter.cc)
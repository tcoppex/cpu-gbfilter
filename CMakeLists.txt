cmake_minimum_required(VERSION 2.8)

project(gbfilter)

if(CMAKE_COMPILER_IS_GNUCXX)
  set(SSE41_FLAGS "-msse4.1 -march=core2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp ${SSE41_FLAGS}")
  add_definitions("-DGBF_OMP_STATS")
elseif(MSVC)
  # MSVC's OpenMP implementation is crappy as hell
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ox")
  add_definitions("/DGBF_OMP_STATS")
else()
  if(${CMAKE_CXX_COMPILER})
    message(WARNING "Unsupported compiler")
  endif()
endif()

add_executable(${CMAKE_PROJECT_NAME} gbfilter.cc)
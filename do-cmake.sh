#! /bin/bash

USER_LOCAL=$HOME/local
SYSTEM_LOCAL=/usr/local

ccmake ../ \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D Eigen3_DIR="${SYSTEM_LOCAL}/share/eigen3/cmake" \
  -D TRNG_INCLUDE_DIR="${SYSTEM_LOCAL}/include" \
  -D TRNG_LIBRARY="${SYSTEM_LOCAL}/lib/libtrng4.a" \
  -D Trilinos_DIR="${USER_LOCAL}/lib/cmake/Trilinos" \
  -D yaml-cpp_DIR="${SYSTEM_LOCAL}/lib/cmake/yaml-cpp"

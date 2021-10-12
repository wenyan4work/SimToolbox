#! /bin/bash

export USER_LOCAL=$HOME/envs/t13_intint

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D CMAKE_BUILD_TYPE=Release \
  -D Eigen3_DIR="${USER_LOCAL}/share/eigen3/cmake" \
  -D TRNG_INCLUDE_DIR="${USER_LOCAL}/include" \
  -D TRNG_LIBRARY="${USER_LOCAL}/lib/libtrng4.a" \
  -D Trilinos_DIR="${USER_LOCAL}/lib/cmake/Trilinos" \
  -D VTK_DIR="${USER_LOCAL}/lib/cmake/vtk-9.0" \
  -D yaml-cpp_DIR="${USER_LOCAL}/share/cmake/yaml-cpp" \
../


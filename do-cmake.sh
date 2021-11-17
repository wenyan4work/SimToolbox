#! /bin/bash

export USER_LOCAL=$HOME/envs/t13

cmake \
  -D CMAKE_CXX_COMPILER=mpicxx \
  -D CMAKE_C_COMPILER=mpicc \
  -D CMAKE_BUILD_TYPE=Release \
  -D Eigen3_DIR="${USER_LOCAL}/share/eigen3/cmake" \
  -D Trilinos_DIR="${USER_LOCAL}/lib/cmake/Trilinos" \
  ../

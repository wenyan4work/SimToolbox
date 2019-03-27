cmake_minimum_required(VERSION 3.10) # same requirement as trilinos
project(SimToolbox)

# global settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(SFTPATH $ENV{HOME}/local)

find_package(MPI REQUIRED)  
# this does not change the compiler
# to use mpi compiler, do cmake -D CMAKE_CXX_COMPILER=mpicxx
# linking executable with mpi
# target_link_libraries(exe PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(exe PRIVATE MPI::MPI_CXX)

find_package(OpenMP REQUIRED)
find_package(TRNG REQUIRED)

message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
message(STATUS "SFTPATH: ${SFTPATH}")

enable_testing()

add_subdirectory(Util)

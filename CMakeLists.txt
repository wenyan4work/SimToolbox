cmake_minimum_required(VERSION 3.17)
project(SimToolbox)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Get Trilinos and Kokkos
find_package(Trilinos REQUIRED)
find_package(Kokkos REQUIRED PATHS ${Trilinos_DIR}/../Kokkos/)

# clean up the too-long list
list(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
list(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)

list(REVERSE Trilinos_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_LIBRARIES)
list(REVERSE Trilinos_LIBRARIES)

list(REVERSE Trilinos_TPL_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
list(REVERSE Trilinos_TPL_LIBRARIES)

# Echo trilinos build info just for fun
message("\nFound Trilinos!  Here are the details: ")
message("   Trilinos_DIR = ${Trilinos_DIR}")
message("   Trilinos_VERSION = ${Trilinos_VERSION}")
message("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
message("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
message("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
message("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
message("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
message("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
message("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
message("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
message("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
message("End of Trilinos details\n")

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

message(
  "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS"
)

include(FetchContent)
set(MSGPACK_CXX14
    ON
    CACHE INTERNAL "")
set(ARBORX_ENABLE_MPI
    ON
    CACHE INTERNAL "")

FetchContent_Declare(
  trng4
  GIT_REPOSITORY https://github.com/rabauke/trng4.git
  GIT_SHALLOW ON
  GIT_TAG master)

FetchContent_Declare(
  msgpack
  GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
  GIT_SHALLOW ON
  GIT_TAG cpp-4.0.3)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_SHALLOW ON
  GIT_TAG v1.9.2)

FetchContent_Declare(
  arborx
  GIT_REPOSITORY https://github.com/arborx/ArborX.git
  GIT_SHALLOW ON
  GIT_TAG v1.1)

FetchContent_Populate(trng4)
FetchContent_Populate(msgpack)
FetchContent_Populate(spdlog)
FetchContent_Populate(arborx)

message(${trng4_SOURCE_DIR}/trng ${trng4_BINARY_DIR})
message(${msgpack_SOURCE_DIR} ${msgpack_BINARY_DIR})
message(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
message(${arborx_SOURCE_DIR} ${arborx_BINARY_DIR})

add_subdirectory(${trng4_SOURCE_DIR}/trng ${trng4_BINARY_DIR} EXCLUDE_FROM_ALL)
add_subdirectory(${msgpack_SOURCE_DIR} ${msgpack_BINARY_DIR} EXCLUDE_FROM_ALL)
add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR} EXCLUDE_FROM_ALL)
add_subdirectory(${arborx_SOURCE_DIR} ${arborx_BINARY_DIR} EXCLUDE_FROM_ALL)

enable_testing()

add_subdirectory(Util)
add_subdirectory(Trilinos)
add_subdirectory(Constraint)
add_subdirectory(MPI)
# add_subdirectory(Particle)
# add_subdirectory(Sylinder)
# add_subdirectory(Boundary)

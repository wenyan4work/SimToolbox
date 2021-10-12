cmake_minimum_required(VERSION 3.17)
project(SimToolbox)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Get Trilinos as one entity
find_package(Trilinos REQUIRED)
# clean up the too-long list
list(REMOVE_DUPLICATES Trilinos_INCLUDE_DIRS)
list(REMOVE_DUPLICATES Trilinos_TPL_INCLUDE_DIRS)

list(REVERSE Trilinos_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_LIBRARIES)
list(REVERSE Trilinos_LIBRARIES)

list(REVERSE Trilinos_TPL_LIBRARIES)
list(REMOVE_DUPLICATES Trilinos_TPL_LIBRARIES)
list(REVERSE Trilinos_TPL_LIBRARIES)

# add full path to Trilinos_LIBRARIES
foreach(X IN LISTS ${Trilinos_LIBRARIES})
    set(X ${Trilinos_LIBRARY_DIRS}/${Trilinos_LIBRARIES})
endforeach()

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("End of Trilinos details\n")

# # Make sure to use same compilers and flags as Trilinos
# set(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER} )
# set(CMAKE_C_COMPILER ${Trilinos_C_COMPILER} )
# set(CMAKE_Fortran_COMPILER ${Trilinos_Fortran_COMPILER} )

# set(CMAKE_CXX_FLAGS  "${Trilinos_CXX_COMPILER_FLAGS} ${CMAKE_CXX_FLAGS}")
# set(CMAKE_C_FLAGS  "${Trilinos_C_COMPILER_FLAGS} ${CMAKE_C_FLAGS}")
# set(CMAKE_Fortran_FLAGS  "${Trilinos_Fortran_COMPILER_FLAGS} ${CMAKE_Fortran_FLAGS}")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
# example
# target_link_libraries(exe PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(exe PRIVATE MPI::MPI_CXX)

find_package(VTK REQUIRED)
find_package(TRNG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

message("Found VTK at ${VTK_DIR}")

message(
  STATUS
    "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS"
)
message(STATUS "SFTPATH: ${SFTPATH}")

enable_testing()

add_subdirectory(Util)
add_subdirectory(Trilinos)
add_subdirectory(MPI)
add_subdirectory(Collision)
add_subdirectory(Boundary)
add_subdirectory(Constraint)
add_subdirectory(Sylinder)

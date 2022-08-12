# Example of how to compile and link target_compile_options
# target_include_directories target_link_libraries

add_executable(CPSolver_test CPSolver_test.cpp CPSolver.cpp
                             ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(CPSolver_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(
  CPSolver_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  CPSolver_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                        trng4 OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME CPSolver COMMAND CPSolver_test)

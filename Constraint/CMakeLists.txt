# Example of how to compile and link
# target_compile_options
# target_include_directories
# target_link_libraries

add_executable(BCQPSolver_test BCQPSolver_test.cpp BCQPSolver.cpp ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(BCQPSolver_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(BCQPSolver_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR})
target_link_libraries(BCQPSolver_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(BCQPSolver_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(BCQPSolver_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(BCQPSolver_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(BCQPSolver_test PRIVATE MPI::MPI_CXX)
add_test(NAME BCQPSolver COMMAND BCQPSolver_test)

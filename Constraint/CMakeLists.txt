# Example of how to compile and link target_compile_options
# target_include_directories target_link_libraries

add_executable(BCQPSolver_test BCQPSolver_test.cpp BCQPSolver.cpp
                               ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(BCQPSolver_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(
  BCQPSolver_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                          ${TRNG_INCLUDE_DIR})
target_link_libraries(BCQPSolver_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(BCQPSolver_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(BCQPSolver_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(BCQPSolver_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(BCQPSolver_test PRIVATE MPI::MPI_CXX)
add_test(NAME BCQPSolver COMMAND BCQPSolver_test)

# copy the BCQPverify.py file to the build directory
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/BCQPSolver_verify.py
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/BCQPSolver_verify.py
          ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/BCQPSolver_verify.py")
# Custom target for activate the custom command above
add_custom_target(copy_BCQPSolver_verify
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/BCQPSolver_verify.py)
add_dependencies(BCQPSolver_test copy_BCQPSolver_verify)

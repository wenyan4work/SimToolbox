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

# copy the BCQPverify.py file to the build directory
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/BCQPverify.py
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/BCQPverify.py
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/BCQPverify.py")
# Custom target for activate the custom command above
add_custom_target(copy_file DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/BCQPverify.py)
add_dependencies(BCQPSolver_test copy_file)

add_executable(SubView_test SubView_test.cpp ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(SubView_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(SubView_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR})
target_link_libraries(SubView_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(SubView_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(SubView_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(SubView_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(SubView_test PRIVATE MPI::MPI_CXX)
add_test(NAME SubView COMMAND SubView_test)
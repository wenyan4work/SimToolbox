###############################################################################
add_executable(
  ConstraintCollector_test ConstraintCollector_test.cpp ConstraintCollector.cpp
                           ${CMAKE_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_include_directories(ConstraintCollector_test
                           PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  ConstraintCollector_test
  PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} msgpackc-cxx
          Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME ConstraintCollector COMMAND python ConstraintCollector_verify.py)

# copy the BCQPverify.py file to the build directory
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ConstraintCollector_verify.py
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/ConstraintCollector_verify.py
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ConstraintCollector_verify.py")

# Custom target for activate the custom command above
add_custom_target(
  copy_ConstraintCollector_verify
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/ConstraintCollector_verify.py)
add_dependencies(ConstraintCollector_test copy_ConstraintCollector_verify)

###############################################################################
add_executable(BCQPSolver_test BCQPSolver_test.cpp BCQPSolver.cpp
                               ${CMAKE_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_include_directories(BCQPSolver_test PRIVATE ${CMAKE_SOURCE_DIR}
                                                   ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  BCQPSolver_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                          OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME BCQPSolver COMMAND python BCQPSolver_verify.py)

# copy the BCQPverify.py file to the build directory
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/BCQPSolver_verify.py
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/BCQPSolver_verify.py
    ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/BCQPSolver_verify.py")
# Custom target for activate the custom command above
add_custom_target(copy_BCQPSolver_verify
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/BCQPSolver_verify.py)
add_dependencies(BCQPSolver_test copy_BCQPSolver_verify)

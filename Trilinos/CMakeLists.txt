# Example of how to compile and link
# target_compile_options
# target_include_directories
# target_link_libraries

add_executable(ZDD_test ZDD_test.cpp)
target_compile_options(ZDD_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(ZDD_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(ZDD_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(ZDD_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(ZDD_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(ZDD_test PRIVATE MPI::MPI_CXX)
add_test(NAME ZDD COMMAND ZDD_test)
# set_tests_properties(ZDD PROPERTIES
#   PASS_REGULAR_EXPRESSION "TestPassed;All ok"
# )

add_executable(TpetraUtil_test TpetraUtil_test.cpp TpetraUtil.cpp)
target_compile_options(TpetraUtil_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(TpetraUtil_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(TpetraUtil_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(TpetraUtil_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(TpetraUtil_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(TpetraUtil_test PRIVATE MPI::MPI_CXX)
add_test(NAME TpetraUtil COMMAND TpetraUtil_test)
# set_tests_properties(TpetraUtil PROPERTIES
#   PASS_REGULAR_EXPRESSION "TestPassed;All ok"
# )

add_executable(Belos_test Belos_test.cpp TpetraUtil.cpp)
target_compile_options(Belos_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(Belos_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(Belos_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(Belos_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(Belos_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(Belos_test PRIVATE MPI::MPI_CXX)

# copy the Belos_verify.py file to the build directory
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Belos_verify.py
  COMMAND ${CMAKE_COMMAND} -E copy
  ${CMAKE_CURRENT_SOURCE_DIR}/Belos_verify.py
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Belos_verify.py")
# Custom target for activate the custom command above
add_custom_target(copy_Belos_verify DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Belos_verify.py)
add_dependencies(Belos_test copy_Belos_verify)

add_test(NAME Belos COMMAND python Belos_verify.py)
set_tests_properties(Belos PROPERTIES PASS_REGULAR_EXPRESSION "Converged")

add_executable(SubView_test SubView_test.cpp ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(SubView_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(SubView_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR})
target_link_libraries(SubView_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(SubView_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(SubView_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(SubView_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(SubView_test PRIVATE MPI::MPI_CXX)
add_test(NAME SubView COMMAND SubView_test)
# Example of how to compile and link target_compile_options
# target_include_directories target_link_libraries

#########################################################################
add_executable(ZDD_test ZDD_test.cpp)
target_compile_options(ZDD_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(
  ZDD_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                   ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(ZDD_test ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                      OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME ZDD COMMAND ZDD_test)

#########################################################################
add_executable(TpetraUtil_test TpetraUtil_test.cpp TpetraUtil.cpp)
target_include_directories(
  TpetraUtil_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                          ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(TpetraUtil_test ${Trilinos_LIBRARIES}
                      ${Trilinos_TPL_LIBRARIES} OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME TpetraUtil COMMAND TpetraUtil_test)

#########################################################################
add_executable(Belos_test Belos_test.cpp TpetraUtil.cpp)
target_include_directories(
  Belos_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                     ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(Belos_test ${Trilinos_LIBRARIES}
                      ${Trilinos_TPL_LIBRARIES} OpenMP::OpenMP_CXX MPI::MPI_CXX)

# copy the Belos_verify.py file to the build directory
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Belos_test.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Belos_test.py
          ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Belos_test.py")
# Custom target for activate the custom command above
add_custom_target(copy_Belos_test
                  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Belos_test.py)
add_dependencies(Belos_test copy_Belos_test)

add_test(NAME Belos COMMAND python Belos_test.py)
set_tests_properties(Belos PROPERTIES PASS_REGULAR_EXPRESSION "Converged")

#########################################################################
add_executable(SubView_test SubView_test.cpp
                            ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(SubView_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(
  SubView_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                       ${TRNG_INCLUDE_DIR})
target_link_libraries(
  SubView_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                       ${TRNG_LIBRARY} OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME SubView COMMAND SubView_test)

#########################################################################
add_executable(TMAP_test TMAP_test.cpp
                         ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_compile_options(TMAP_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(
  TMAP_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                    ${TRNG_INCLUDE_DIR})
target_link_libraries(
  TMAP_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                    ${TRNG_LIBRARY} OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME TMAP COMMAND mpirun -n 4 ./TMAP_test)

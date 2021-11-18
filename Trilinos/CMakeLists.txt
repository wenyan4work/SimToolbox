function(exetest exename)
  add_executable(${exename}_test ${exename}_test.cpp TpetraUtil.cpp)
  target_include_directories(
    ${exename}_test PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                            ${Trilinos_TPL_INCLUDE_DIRS} ${EXT_INCLUDE})
  target_link_libraries(
    ${exename}_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                            trng4 OpenMP::OpenMP_CXX MPI::MPI_CXX)
  if(ipo_support)
    set_property(TARGET ${exename}_test PROPERTY INTERPROCEDURAL_OPTIMIZATION
                                                 TRUE)
  endif()
  add_test(NAME ${exename} COMMAND mpirun -n 4 ${exename}_test)
endfunction()

#########################################################################
exetest(TMAP)
exetest(TCMAT)
exetest(SubView)
exetest(Kokkos)
exetest(Zoltan2)

#########################################################################
add_executable(Belos_test Belos_test.cpp TpetraUtil.cpp)
target_include_directories(
  Belos_test PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                     ${Trilinos_TPL_INCLUDE_DIRS} ${EXT_INCLUDE})
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

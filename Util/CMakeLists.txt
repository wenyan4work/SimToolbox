function(exetest exename)
  add_executable(${exename}_test ${exename}_test.cpp)
  target_include_directories(${exename}_test PRIVATE ${CMAKE_SOURCE_DIR}
                                                     ${EXT_INCLUDE})
  target_link_libraries(${exename}_test PRIVATE trng4 OpenMP::OpenMP_CXX
                                                MPI::MPI_CXX)
  add_test(NAME ${exename} COMMAND mpirun -n 4 ${exename}_test)
  set_tests_properties(${exename} PROPERTIES PASS_REGULAR_EXPRESSION
                                             "TestPassed;All ok")
endfunction()

exetest(GeoUtil)
exetest(SortUtil)
exetest(TRngPool)

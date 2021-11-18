add_executable(Boundary_test Boundary_test.cpp Boundary.cpp)
target_include_directories(Boundary_test PRIVATE ${CMAKE_SOURCE_DIR}
                                                 ${EXT_INCLUDE})
target_link_libraries(Boundary_test PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX)
add_test(NAME Boundary COMMAND Boundary_test)
set_tests_properties(Boundary PROPERTIES PASS_REGULAR_EXPRESSION
                                         "TestPassed;All ok")

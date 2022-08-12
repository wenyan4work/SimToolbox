add_executable(Boundary_test Boundary_test.cpp Boundary.cpp)
target_compile_options(Boundary_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(Boundary_test PRIVATE ${PROJECT_SOURCE_DIR}
                                                 yaml-cpp)
target_link_libraries(Boundary_test PRIVATE OpenMP::OpenMP_CXX
                                            yaml-cpp Eigen3::Eigen)
add_test(NAME Boundary COMMAND Boundary_test)
set_tests_properties(Boundary PROPERTIES PASS_REGULAR_EXPRESSION
                                         "TestPassed;All ok")

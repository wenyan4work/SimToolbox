# Example of how to compile and link
# target_compile_options
# target_include_directories
# target_link_libraries

add_executable(GeoUtil_test GeoUtil_test.cpp)
target_compile_options(GeoUtil_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(GeoUtil_test PRIVATE ${PROJECT_SOURCE_DIR} ${TRNG_INCLUDE_DIR})
target_link_libraries(GeoUtil_test PRIVATE ${TRNG_LIBRARY} OpenMP::OpenMP_CXX)
add_test(NAME GeoUtil COMMAND GeoUtil_test)
set_tests_properties(GeoUtil PROPERTIES
  PASS_REGULAR_EXPRESSION "TestPassed;All ok"
)

add_executable(SortUtil_test SortUtil_test.cpp)
target_compile_options(SortUtil_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(SortUtil_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(SortUtil_test PRIVATE MPI::MPI_CXX)
add_test(NAME SortUtil COMMAND SortUtil_test)
set_tests_properties(SortUtil PROPERTIES
  PASS_REGULAR_EXPRESSION "TestPassed;All ok"
)

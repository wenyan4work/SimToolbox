project(Util)

add_executable(GeoUtil_test GeoUtil_test.cpp)

add_executable(SortUtil_test SortUtil_test.cpp)
target_compile_options(SortUtil_test PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(SortUtil_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(SortUtil_test PRIVATE MPI::MPI_CXX)
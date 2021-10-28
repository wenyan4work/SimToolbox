add_executable(verifyAPI main.cpp)
target_link_libraries(verifyAPI PRIVATE ArborX::ArborX OpenMP::OpenMP_CXX
                                        MPI::MPI_CXX)
target_include_directories(verifyAPI PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME verifyAPI COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyAPI")

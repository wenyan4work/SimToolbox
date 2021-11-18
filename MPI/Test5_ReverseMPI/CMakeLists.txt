add_executable(verifyReverse verify.cpp)
target_link_libraries(verifyReverse PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} 
                                            ArborX::ArborX OpenMP::OpenMP_CXX
                                            MPI::MPI_CXX)
target_include_directories(verifyReverse PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME ReverseMPI COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyReverse")

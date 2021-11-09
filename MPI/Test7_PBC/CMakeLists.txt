add_executable(verifyPBC main.cpp)
target_link_libraries(verifyPBC PRIVATE PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} 
                                                ArborX::ArborX OpenMP::OpenMP_CXX
                                                MPI::MPI_CXX)
target_include_directories(verifyPBC PRIVATE ${CMAKE_SOURCE_DIR})
add_test(NAME verifyPBC COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyPBC")

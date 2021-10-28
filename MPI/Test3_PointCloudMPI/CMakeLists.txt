add_executable(verifyMPI verify.cpp)
target_link_libraries(verifyMPI PRIVATE ArborX::ArborX OpenMP::OpenMP_CXX
                                        MPI::MPI_CXX)

add_test(
  NAME PointCloudMPI
  COMMAND
    bash -c "
    python3 GenTestDataset.py \
    && export OMP_NUM_THREADS=4 \
    && mpirun -n 4 ./verifyMPI > verify.log \
    && python3 Verify.py")
set_tests_properties(PointCloudMPI PROPERTIES FAIL_REGULAR_EXPRESSION
                                              "[^a-z]Error;ERROR;Failed")

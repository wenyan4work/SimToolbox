add_executable(verify verify.cpp)
target_link_libraries(verify PRIVATE ArborX::ArborX OpenMP::OpenMP_CXX
                                     MPI::MPI_CXX)

add_test(
  NAME PointCloud
  COMMAND
    bash -c "
    python3 GenTestDataset.py \
    && export OMP_NUM_THREADS=4 \
    && mpirun -n 2 ./verify > verify.log \
    && python3 Verify.py")
set_tests_properties(PointCloud PROPERTIES FAIL_REGULAR_EXPRESSION
                                           "[^a-z]Error;ERROR;Failed")

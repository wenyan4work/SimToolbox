add_executable(verifyForward verify.cpp)
target_include_directories(
  verifyForward PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                        ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(
  verifyForward PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                        ArborX::ArborX OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME ForwardMPI COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyForward")

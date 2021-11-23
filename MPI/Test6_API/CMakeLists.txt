add_executable(verifyAPI main.cpp)
target_include_directories(
  verifyAPI PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS})
target_link_libraries(
  verifyAPI
  PRIVATE
  PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ArborX::ArborX
          OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME verifyAPI COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyAPI")

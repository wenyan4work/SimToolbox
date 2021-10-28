# Example of how to compile and link target_compile_options
# target_include_directories target_link_libraries

add_executable(CommMPI_test CommMPI_test.cpp)
target_include_directories(CommMPI_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(CommMPI_test PRIVATE spdlog::spdlog
                                           OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME CommMPI
         COMMAND bash -c
                 "export OMP_NUM_THREADS=3 && mpirun -n 4 CommMPI_test")

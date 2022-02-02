add_executable(Particle_test Particle_test.cpp)
target_include_directories(
  Particle_test PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                        ${Trilinos_TPL_INCLUDE_DIRS} ${EXT_INCLUDE})
target_link_libraries(Particle_test PRIVATE Eigen3::Eigen OpenMP::OpenMP_CXX
                                            MPI::MPI_CXX)
add_test(NAME Particle COMMAND Particle_test)

add_executable(test_InstallArborX main.cpp)

target_link_libraries(test_InstallArborX ArborX::ArborX OpenMP::OpenMP_CXX
                      MPI::MPI_CXX)

add_test(InstallArborX test_InstallArborX)

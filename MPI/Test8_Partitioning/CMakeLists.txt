add_executable(verifyPar main.cpp ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_include_directories(
  verifyPar PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS} ${EXT_INCLUDE})
target_link_libraries(
  verifyPar
  PRIVATE
  PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ArborX::ArborX
          OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME verifyPar COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyPar")

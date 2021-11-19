add_executable(verifyVision main.cpp ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_include_directories(
  verifyVision PRIVATE ${CMAKE_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                    ${Trilinos_TPL_INCLUDE_DIRS} ${EXT_INCLUDE})
target_link_libraries(
  verifyVision
  PRIVATE
  PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ArborX::ArborX
          OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME verifyVision COMMAND bash -c "export OMP_NUM_THREADS=2 \
            && mpirun -n 4 ./verifyVision && python3 visualize.py")

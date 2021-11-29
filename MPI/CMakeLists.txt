add_executable(CommMPI_test CommMPI_test.cpp)
target_include_directories(
  CommMPI_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
                       ${EXT_INCLUDE})
target_link_libraries(CommMPI_test PRIVATE OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME CommMPI
         COMMAND bash -c "export OMP_NUM_THREADS=3 && mpirun -n 4 CommMPI_test")

add_executable(Benchmark.X Srim_Benchmark.cpp
                           ${CMAKE_SOURCE_DIR}/Trilinos/TpetraUtil.cpp)
target_include_directories(
  Benchmark.X
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}
          ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS} ${EXT_INCLUDE})
target_link_libraries(
  Benchmark.X PRIVATE ArborX::ArborX ${Trilinos_LIBRARIES}
                      ${Trilinos_TPL_LIBRARIES} OpenMP::OpenMP_CXX MPI::MPI_CXX)

file(GLOB_RECURSE resources "./*.*")
foreach(resource ${resources})
  get_filename_component(filename ${resource} NAME)
  get_filename_component(dir ${resource} DIRECTORY)
  get_filename_component(dirname ${dir} NAME)

  set(output "${CMAKE_CURRENT_BINARY_DIR}/${dirname}/${filename}")

  add_custom_command(
    COMMENT "Moving updated resource-file '${resource}'"
    OUTPUT ${output}
    DEPENDS ${resource}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${resource} ${output})
  add_custom_target(${dirname}_${filename} ALL DEPENDS ${resource} ${output})

endforeach()

add_subdirectory(Test1_InstallArborX)
add_subdirectory(Test2_PointCloud)
add_subdirectory(Test3_PointCloudMPI)
add_subdirectory(Test4_ForwardMPI)
add_subdirectory(Test5_ReverseMPI)
add_subdirectory(Test6_API)
add_subdirectory(Test7_PBC)
add_subdirectory(Test8_Partitioning)
add_subdirectory(Test9_Partitioning_Visualization)

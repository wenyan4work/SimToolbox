# Example of how to compile and link target_compile_options
# target_include_directories target_link_libraries

add_executable(CommMPI_test CommMPI_test.cpp)
target_compile_options(CommMPI_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(
  CommMPI_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  CommMPI_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                      trng4 OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME CommMPI COMMAND CommMPI_test)

# executable of sample code
add_executable(MixPairTest MixPairInteraction_test.cpp)
target_compile_options(MixPairTest PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(
  MixPairTest PRIVATE PARTICLE_SIMULATOR_THREAD_PARALLEL
                      PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(MixPairTest PRIVATE ${PROJECT_SOURCE_DIR}
                                               ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  MixPairTest PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                      OpenMP::OpenMP_CXX MPI::MPI_CXX)

# executable of test
# Scan through resource folder for updated files and copy if none existing or changed
file(GLOB_RECURSE resources "./Test*/*.*")
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

add_test(
  NAME MixPair
  COMMAND
    sh -c "cd ./TestMixPairInteraction/ \
    && python GenTestDataset.py \
    && export OMP_NUM_THREADS=4 \
    && mpirun -n 2 ../MixPairTest > Test.log \
    && python Verify.py")
set_tests_properties(MixPair PROPERTIES FAIL_REGULAR_EXPRESSION
                                        "[^a-z]Error;ERROR;Failed")

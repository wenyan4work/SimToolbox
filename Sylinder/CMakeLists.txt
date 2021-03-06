add_executable(SylinderNear_test SylinderNear_test.cpp)
target_compile_options(SylinderNear_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(SylinderNear_test PRIVATE ${PROJECT_SOURCE_DIR}
                                                     ${Trilinos_INCLUDE_DIRS})
target_link_libraries(
  SylinderNear_test PRIVATE ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES}
                            Eigen3::Eigen OpenMP::OpenMP_CXX MPI::MPI_CXX)
add_test(NAME SylidnerNear COMMAND SylinderNear_test)

add_executable(
  SylinderSystem_main
  SylinderSystem_main.cpp
  SylinderSystem.cpp
  SylinderConfig.cpp
  Sylinder.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintOperator.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintSolver.cpp
  ${PROJECT_SOURCE_DIR}/Util/Base64.cpp)
target_compile_options(SylinderSystem_main PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(
  SylinderSystem_main PRIVATE PARTICLE_SIMULATOR_THREAD_PARALLEL
                              PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(
  SylinderSystem_main PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS}
                              ${TRNG_INCLUDE_DIR} ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(
  SylinderSystem_main
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          ${YAML_CPP_LIBRARIES}
          ${TRNG_LIBRARY}
          VTK::IOXML
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

add_executable(
  SylinderSystem_test_api
  SylinderSystem_test_api.cpp
  SylinderSystem.cpp
  SylinderConfig.cpp
  Sylinder.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintOperator.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintSolver.cpp
  ${PROJECT_SOURCE_DIR}/Util/Base64.cpp)
target_compile_options(SylinderSystem_test_api PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(
  SylinderSystem_test_api PRIVATE PARTICLE_SIMULATOR_THREAD_PARALLEL
                                  PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(
  SylinderSystem_test_api
  PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR}
          ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(
  SylinderSystem_test_api
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          ${YAML_CPP_LIBRARIES}
          ${TRNG_LIBRARY}
          VTK::IOXML
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

# Scan through resource folder for updated files and copy if none existing or changed
file(GLOB_RECURSE resources "./Test*/*.*")
foreach(resource ${resources})
  get_filename_component(filename ${resource} NAME)
  get_filename_component(dir ${resource} DIRECTORY)
  get_filename_component(dirname ${dir} NAME)

  set(output "${CMAKE_CURRENT_BINARY_DIR}/TestCases/${dirname}/${filename}")

  add_custom_command(
    COMMENT "Moving updated resource-file '${resource}'"
    OUTPUT ${output}
    DEPENDS ${resource}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${resource} ${output})
  add_custom_target(${dirname}_${filename} ALL DEPENDS ${resource} ${output})

endforeach()

add_test(
  NAME Confinement
  COMMAND
    sh -c "cd TestCases/Test1_Confinement/ \
    && export OMP_NUM_THREADS=2 \
    && mpirun -n 2 ../../SylinderSystem_main > Confinement.log \
    && python Verify.py")
set_tests_properties(Confinement PROPERTIES FAIL_REGULAR_EXPRESSION
                                            "[^a-z]Error;ERROR;Failed")

add_test(
  NAME MixLink
  COMMAND
    sh -c "cd TestCases/Test2_MixLink/ \
    && export OMP_NUM_THREADS=2 \
    && mpirun -n 2 ../../SylinderSystem_main > MixLink.log \
    && python Verify.py")
set_tests_properties(MixLink PROPERTIES FAIL_REGULAR_EXPRESSION
                                        "[^a-z]Error;ERROR;Failed")

add_test(
  NAME TestAPI
  COMMAND
    sh -c "cd TestCases/Test3_TestAPI/ \
    && export OMP_NUM_THREADS=1 \
    && mpirun -n 4 ../../SylinderSystem_test_api > TestAPI.log \
    && python Verify.py")
set_tests_properties(TestAPI PROPERTIES FAIL_REGULAR_EXPRESSION
                                        "[^a-z]Error;ERROR;Failed")

add_test(
  NAME StressLC
  COMMAND
    sh -c "cd TestCases/Test4_StressLC/ \
    && export OMP_NUM_THREADS=3 \
    && mpirun -n 2 ../../SylinderSystem_main > StressLC.log \
    && python StressVerify.py")
set_tests_properties(StressLC PROPERTIES FAIL_REGULAR_EXPRESSION
                                         "[^a-z]Error;ERROR;Failed")

add_test(
  NAME StressSphere
  COMMAND
    sh -c "cd TestCases/Test5_StressSphere/ \
    && export OMP_NUM_THREADS=2 \
    && mpirun -n 4 ../../SylinderSystem_main > StressSphere.log \
    && python StressVerify.py")
set_tests_properties(StressSphere PROPERTIES FAIL_REGULAR_EXPRESSION
                                             "[^a-z]Error;ERROR;Failed")

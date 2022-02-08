add_executable(SystemConfig_test SystemConfig_test.cpp SystemConfig.cpp
                                 ../Boundary/Boundary.cpp)
target_include_directories(SystemConfig_test PRIVATE ${CMAKE_SOURCE_DIR}
                                                     ${EXT_INCLUDE})
target_link_libraries(SystemConfig_test PRIVATE Eigen3::Eigen
                                                OpenMP::OpenMP_CXX)

add_executable(
  SylinderSystem.X
  SylinderSystem_main.cpp
  SystemConfig.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintOperator.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintSolver.cpp)
target_include_directories(
  SylinderSystem.X #
  PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/Particle ${EXT_INCLUDE}
          ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR})
target_link_libraries(
  SylinderSystem.X #
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          ${TRNG_LIBRARY}
          trng4
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

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

add_test(NAME ParseConfig COMMAND sh -c "cd TestCases/Test1_ParseConfig/ \
    && ../../SystemConfig_test > ParseConfig.log")
set_tests_properties(ParseConfig PROPERTIES FAIL_REGULAR_EXPRESSION
                                            "[^a-z]Error;ERROR;Failed")

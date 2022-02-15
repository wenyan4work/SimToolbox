add_executable(SystemConfig_test SystemConfig_test.cpp SystemConfig.cpp
                                 ../Boundary/Boundary.cpp)
target_include_directories(SystemConfig_test PRIVATE ${CMAKE_SOURCE_DIR}
                                                     ${EXT_INCLUDE})
target_link_libraries(SystemConfig_test PRIVATE Eigen3::Eigen
                                                OpenMP::OpenMP_CXX)

add_executable(
  SphereSystem.X
  SphereSystem_main.cpp
  SystemConfig.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintOperator.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintSolver.cpp)
target_include_directories(
  SphereSystem.X #
  PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/Particle ${EXT_INCLUDE}
          ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE_DIR})
target_link_libraries(
  SphereSystem.X #
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          ${TRNG_LIBRARY}
          trng4
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

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

file(GLOB_RECURSE resources "./Test*/*")

foreach(resource ${resources})
  message(${resource})
  file(RELATIVE_PATH relresource ${CMAKE_CURRENT_SOURCE_DIR} ${resource})
  get_filename_component(filename ${relresource} NAME)
  get_filename_component(dir ${relresource} DIRECTORY)

  set(output "${CMAKE_CURRENT_BINARY_DIR}/TestCases/${dir}/${filename}")

  add_custom_command(
    COMMENT "Moving updated resource-file '${resource}'"
    OUTPUT ${output}
    DEPENDS ${resource}
    COMMAND ${CMAKE_COMMAND} -E copy ${resource} ${output})
  string(REPLACE "/" "_" targetname ${relresource})
  add_custom_target(${targetname} ALL DEPENDS ${resource} ${output})

endforeach()

add_test(NAME ParseConfig COMMAND sh -c "cd TestCases/Test1_ParseConfig/ \
    && ../../SystemConfig_test > ParseConfig.log")
set_tests_properties(ParseConfig PROPERTIES FAIL_REGULAR_EXPRESSION
                                            "[^a-z]Error;ERROR;Failed")

add_test(
  NAME Resume
  COMMAND
    sh -c "cd TestCases/Test2_Resume/ \
          && cd result && python3 trim_data.py \
          && cd ../ \
          && ../../SylinderSystem.X > SylinderSystem.log")
set_tests_properties(ParseConfig PROPERTIES FAIL_REGULAR_EXPRESSION
                                            "[^a-z]Error;ERROR;Failed")

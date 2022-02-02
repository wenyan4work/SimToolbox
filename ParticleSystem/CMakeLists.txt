add_executable(SystemConfig_test SystemConfig_test.cpp SystemConfig.cpp
                                 ../Boundary/Boundary.cpp)
target_include_directories(SystemConfig_test PRIVATE ${CMAKE_SOURCE_DIR}
                                                     ${EXT_INCLUDE})
target_link_libraries(SystemConfig_test PRIVATE Eigen3::Eigen
                                                OpenMP::OpenMP_CXX)

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

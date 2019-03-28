# Example of how to compile and link
# target_compile_options
# target_include_directories
# target_link_libraries

add_executable(CommMPI_test CommMPI_test.cpp)
target_compile_options(CommMPI_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(CommMPI_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE})
target_link_libraries(CommMPI_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(CommMPI_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(CommMPI_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(CommMPI_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(CommMPI_test PRIVATE MPI::MPI_CXX)
add_test(NAME CommMPI COMMAND CommMPI_test)
# set_tests_properties(ZDD PROPERTIES
#   PASS_REGULAR_EXPRESSION "TestPassed;All ok"
# )

add_executable(MixPairInteraction_test MixPairInteraction_test.cpp)
target_compile_options(MixPairInteraction_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(MixPairInteraction_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS} ${TRNG_INCLUDE})
target_link_libraries(MixPairInteraction_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(MixPairInteraction_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(MixPairInteraction_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(MixPairInteraction_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(MixPairInteraction_test PRIVATE MPI::MPI_CXX)
add_test(NAME MixPairInteraction COMMAND MixPairInteraction_test)
# set_tests_properties(ZDD PROPERTIES
#   PASS_REGULAR_EXPRESSION "TestPassed;All ok"
# )


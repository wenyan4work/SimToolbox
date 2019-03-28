add_executable(SylinderNear_test SylinderNear_test.cpp)
target_compile_options(SylinderNear_test PRIVATE ${OpenMP_CXX_FLAGS})
target_include_directories(SylinderNear_test PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})
target_link_libraries(SylinderNear_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(SylinderNear_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(SylinderNear_test PRIVATE Eigen3::Eigen)
target_link_libraries(SylinderNear_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(SylinderNear_test PRIVATE MPI::MPI_CXX)
add_test(NAME SylidnerNear COMMAND SylinderNear_test)

add_executable(SylinderSystem_test 
SylinderSystem_test.cpp 
SylinderSystem.cpp 
SylinderConfig.cpp 
Sylinder.cpp 
${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp 
${PROJECT_SOURCE_DIR}/Collision/CollisionSolver.cpp
${PROJECT_SOURCE_DIR}/Collision/CPSolver.cpp
${PROJECT_SOURCE_DIR}/Util/Base64.cpp)
target_compile_options(SylinderSystem_test PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(SylinderSystem_test PRIVATE
PARTICLE_SIMULATOR_THREAD_PARALLEL 
PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(SylinderSystem_test 
PRIVATE ${PROJECT_SOURCE_DIR} 
${Trilinos_INCLUDE_DIRS} 
${TRNG_INCLUDE_DIR} 
${YAML_CPP_INCLUDE_DIR})
target_link_libraries(SylinderSystem_test PRIVATE ${Trilinos_LIBRARIES})
target_link_libraries(SylinderSystem_test PRIVATE ${Trilinos_TPL_LIBRARIES})
target_link_libraries(SylinderSystem_test PRIVATE ${YAML_CPP_LIBRARIES})
target_link_libraries(SylinderSystem_test PRIVATE ${TRNG_LIBRARY})
target_link_libraries(SylinderSystem_test PRIVATE Eigen3::Eigen)
target_link_libraries(SylinderSystem_test PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(SylinderSystem_test PRIVATE MPI::MPI_CXX)
add_test(NAME SylinderSystem COMMAND SylinderSystem_test)
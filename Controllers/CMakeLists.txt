add_executable(
  main
  main.cpp
  DryPhysicsController.cpp
  ${PROJECT_SOURCE_DIR}/Particle/ParticleSystem.cpp
  ${PROJECT_SOURCE_DIR}/Particle/ParticleConfig.cpp
  ${PROJECT_SOURCE_DIR}/Particle/Particle.cpp
  ${PROJECT_SOURCE_DIR}/Trilinos/TpetraUtil.cpp
  ${PROJECT_SOURCE_DIR}/Boundary/Boundary.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/BCQPSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/Constraint.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintCollector.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/ConstraintJacobianOp.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/PGDConstraintSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/NOXConstraintSolver.cpp
  ${PROJECT_SOURCE_DIR}/Constraint/NOXConstraintEvaluator.cpp
  ${PROJECT_SOURCE_DIR}/Util/Base64.cpp)
target_compile_options(main PRIVATE ${OpenMP_CXX_FLAGS})
target_compile_definitions(
  main PRIVATE PARTICLE_SIMULATOR_THREAD_PARALLEL
                              PARTICLE_SIMULATOR_MPI_PARALLEL)
target_include_directories(
  main PRIVATE ${PROJECT_SOURCE_DIR} ${Trilinos_INCLUDE_DIRS})

target_link_libraries(
  main
  PRIVATE ${Trilinos_LIBRARIES}
          ${Trilinos_TPL_LIBRARIES}
          yaml-cpp
          trng4
          VTK::IOXML
          Eigen3::Eigen
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX)

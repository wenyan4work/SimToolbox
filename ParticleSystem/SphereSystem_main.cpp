/**
 * @file SphereSystem_main.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-02-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "ParticleSystem.hpp"
#include "Sphere.hpp"

#include "Util/Logger.hpp"

#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  {
    std::string configFile = "RunConfig.toml";
    std::string posFile = "SphereInitial.dat";
    auto configPtr = std::make_shared<SystemConfig>(configFile);
    ParticleSystem<Sphere> system;
    system.writeBox();

    // main time loop
    while (system.stepRunning()) {
      system.stepPrepare();
      system.stepCalcMotion();
      system.stepUpdatePtcl();
      system.writeData();
      system.stepMovePtcl();

      system.statPolarity();
      system.statStressConB();
      system.statStressConU();
      system.printTimingSummary(true);
    }
  }
  // mpi finalize
  // let the root rank wait for other
  MPI_Finalize();
  return 0;
}
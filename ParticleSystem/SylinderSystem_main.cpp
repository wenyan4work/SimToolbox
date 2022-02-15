/**
 * @file SylinderSystem_main.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-02-07
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "ParticleSystem.hpp"
#include "Spherocylinder.hpp"

#include "Util/Logger.hpp"

#include <mpi.h>
#include <omp.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  Logger::setup_mpi_spdlog();
  {
    std::string configFile = "RunConfig.toml";
    std::string posFile = "SylinderInitial.dat";
    auto configPtr = std::make_shared<const SystemConfig>(configFile);
    configPtr->echo();
    ParticleSystem<Sylinder> system;
    system.initialize(configPtr, posFile);
    if (configPtr->resume) {
      // complete the resumed step
      system.stepMovePtcl();
    }
    system.writeBox();

    // main time loop
    while (system.stepRunning()) {
      system.stepPrepare();
      system.stepCalcMotion();
      system.stepUpdatePtcl();
      if (system.stepWriting()) {
        system.writeData();
        system.writeDataEOT();
      }
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
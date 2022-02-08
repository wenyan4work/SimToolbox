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
    std::string runConfig = "RunConfig.yaml";
    std::string posFile = "SylinderInitial.dat";
    std::string restartFile = "TimeStepInfo.txt";
    ParticleSystem<Sylinder> system;

    // main time loop
    while (system.running()) {
      system.prepareStep();
      system.runStep();
      system.calcPolarity();
      system.calcStressConB();
      system.calcStressConU();
      system.printTimingSummary(true);
    }
  }
  // mpi finalize
  // let the root rank wait for other
  MPI_Finalize();
  return 0;
}
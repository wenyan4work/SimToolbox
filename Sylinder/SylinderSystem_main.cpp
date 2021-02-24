/**
 * @file SylinderSystem_main.cpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief main driver of SylinderSystem class
 * @version 0.1
 * @date 2021-02-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include <mpi.h>

#include "SylinderSystem.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    PS::Initialize(argc, argv);
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    {
        // create a system and distribute it to all ranks
        // MPI is initialized inside PS::Initialize()
        std::string runConfig = "RunConfig.yaml";
        std::string posFile = "SylinderInitial.dat";
        SylinderSystem system(runConfig, posFile, argc, argv);

        // main time loop
        while (system.getStepCount() * system.runConfig.dt < system.runConfig.timeTotal) {
            system.prepareStep();
            system.runStep();
            Teuchos::TimeMonitor::summarize();
        }
    }
    // mpi finalize
    // let the root rank wait for other
    PS::Finalize();
    MPI_Finalize();
    return 0;
}
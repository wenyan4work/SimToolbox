#include "Sylinder/SylinderSystem.hpp"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    {
        std::string configFile("runConfig.yaml");
        std::string posFile("posInitial.dat");
        SylinderSystem sylinderSystem(configFile, posFile, argc, argv);
        sylinderSystem.prepareStep();
        sylinderSystem.runStep();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
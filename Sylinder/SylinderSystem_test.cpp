#include "Sylinder/SylinderSystem.hpp"

void test(int argc, char **argv) {
    // auto runConfig = SylinderConfig("runConfig_test.yaml");
    auto runConfig = SylinderConfig();

    runConfig.simBoxLow[0] = runConfig.simBoxLow[1] = runConfig.simBoxLow[2] = 0;
    runConfig.initBoxLow[0] = runConfig.initBoxLow[1] = runConfig.initBoxLow[2] = 0;
    runConfig.simBoxHigh[0] = runConfig.simBoxHigh[1] = runConfig.simBoxHigh[2] = 20.0;
    runConfig.initBoxHigh[0] = runConfig.initBoxHigh[1] = runConfig.initBoxHigh[2] = 20.0;
    runConfig.sylinderNumber = 1024;
    runConfig.wallLowZ = true;
    runConfig.wallHighZ = true;
    runConfig.dt = 0.001;
    runConfig.timeSnap = 0.001;
    runConfig.simBoxPBC[0] = true;
    runConfig.simBoxPBC[1] = true;
    runConfig.simBoxPBC[2] = true;

    SylinderSystem sylinderSystem(runConfig, "posInitial.dat", argc, argv);
    sylinderSystem.setTimer(true);
    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        sylinderSystem.runStep();
    }

    std::vector<Sylinder> newSylinder;
    for (int i = 0; i < 10; i++) {
        newSylinder.emplace_back(i, runConfig.sylinderDiameter / 2, runConfig.sylinderDiameter / 2,
                                 runConfig.sylinderLength / 2, runConfig.sylinderLength / 2);
    }
    sylinderSystem.addNewSylinder(newSylinder);

    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        sylinderSystem.runStep();
    }

    double localLow[3];
    double localHigh[3];
    double globalLow[3];
    double globalHigh[3];
    sylinderSystem.calcBoundingBox(localLow, localHigh, globalLow, globalHigh);
    std::cout << Emap3(localLow).transpose() << std::endl;
    std::cout << Emap3(localHigh).transpose() << std::endl;
    std::cout << Emap3(globalLow).transpose() << std::endl;
    std::cout << Emap3(globalHigh).transpose() << std::endl;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    test(argc, argv);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
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
    runConfig.simBoxPBC[2] = false;

    SylinderSystem sylinderSystem(runConfig, "posInitial.dat", argc, argv);
    sylinderSystem.setTimer(true);
    std::vector<double> forceNonBrown(sylinderSystem.getContainer().getNumberOfParticleLocal() * 6, 0.0);
    for (int i = 0; i < runConfig.sylinderNumber; i++) {
        forceNonBrown[6 * i + 2] = -10; // const gravity
    }

    // run 10 steps
    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        sylinderSystem.setForceNonBrown(forceNonBrown);
        sylinderSystem.runStep();
    }

    // add 10 sylinders to the top
    std::vector<Sylinder> newSylinder;
    for (int i = 0; i < 10; i++) {
        newSylinder.emplace_back(i, runConfig.sylinderDiameter / 2, runConfig.sylinderDiameter / 2,
                                 runConfig.sylinderLength / 2, runConfig.sylinderLength / 2);
        newSylinder.back().pos[0] = sin(i) * runConfig.initBoxHigh[0];
        newSylinder.back().pos[1] = sin(i) * runConfig.initBoxHigh[1];
        newSylinder.back().pos[2] = 0.9 * runConfig.initBoxHigh[2];
    }
    sylinderSystem.addNewSylinder(newSylinder);

    // run 10 more steps
    const int nLocal = sylinderSystem.getContainer().getNumberOfParticleLocal();
    forceNonBrown.resize(nLocal * 6, 0.0);
    for (int i = 0; i < nLocal; i++) {
        forceNonBrown[6 * i + 2] = -10; // const gravity
    }

    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        sylinderSystem.setForceNonBrown(forceNonBrown);
        sylinderSystem.runStep();
    }

    Teuchos::TimeMonitor::summarize();

    double localLow[3];
    double localHigh[3];
    double globalLow[3];
    double globalHigh[3];
    sylinderSystem.calcBoundingBox(localLow, localHigh, globalLow, globalHigh);
    std::cout << "local bounding box: " << std::endl;
    std::cout << Emap3(localLow).transpose() << std::endl;
    std::cout << Emap3(localHigh).transpose() << std::endl;
    std::cout << "global bounding box: " << std::endl;
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
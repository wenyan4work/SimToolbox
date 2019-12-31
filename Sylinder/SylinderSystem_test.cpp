#include "Sylinder/SylinderSystem.hpp"

void testSedimentation(int argc, char **argv) {
    auto runConfig = SylinderConfig("SylinderSystem_test_runConfig.yaml");

    SylinderSystem sylinderSystem(runConfig, "posInitial.dat", argc, argv);
    sylinderSystem.setTimer(true);

    // run 10 steps
    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        int nLocal = sylinderSystem.getContainer().getNumberOfParticleLocal();
        std::vector<double> forceNonBrown(nLocal * 6, 0.0);
        for (int i = 0; i < nLocal; i++) {
            forceNonBrown[6 * i + 2] = -10; // const gravity
        }
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
    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        int nLocal = sylinderSystem.getContainer().getNumberOfParticleLocal();
        std::vector<double> forceNonBrown(nLocal * 6, 0.0);
        for (int i = 0; i < nLocal; i++) {
            forceNonBrown[6 * i + 2] = -10; // const gravity
        }
        sylinderSystem.setForceNonBrown(forceNonBrown);
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

void testLink(int argc, char **argv) {}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    testSedimentation(argc, argv);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
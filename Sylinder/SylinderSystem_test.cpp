#include "Sylinder/SylinderSystem.hpp"

void testSedimentation(int argc, char **argv) {
    auto runConfig = SylinderConfig("SylinderSystem_test_runConfig.yaml");

    constexpr int numQuadPt = 10;
    SylinderSystem<numQuadPt> sylinderSystem(runConfig, "posInitial.dat", argc, argv);
    sylinderSystem.setTimer(true);
    auto &rngPoolPtr = sylinderSystem.getRngPoolPtr();

    // run 10 steps for relaxation
    for (int i = 0; i < 10; i++) {
        sylinderSystem.prepareStep();
        sylinderSystem.runStep();
    }

    // add linked sylinders
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // using rank as group id
    const int nLocalNew = 20;
    std::vector<Sylinder<numQuadPt>> newSylinder(nLocalNew);
    std::vector<Link> linkage(nLocalNew);
    for (int i = 0; i < nLocalNew; i++) {
        newSylinder[i] = Sylinder<numQuadPt>(i, runConfig.sylinderDiameter / 2, runConfig.sylinderDiameter / 2,
                                  runConfig.sylinderLength, runConfig.sylinderLength);
        linkage[i].group = rank;
    }
    // sequentially linked
    for (int i = 0; i < nLocalNew - 1; i++) {
        linkage[i].next = i + 1;
        linkage[i + 1].prev = i - 1;
    }
    // specify linked sylinders
    newSylinder[0].pos[0] = rngPoolPtr->getU01(0) * runConfig.simBoxHigh[0];
    newSylinder[0].pos[1] = rngPoolPtr->getU01(0) * runConfig.simBoxHigh[1];
    newSylinder[0].pos[2] = rngPoolPtr->getU01(0) * runConfig.simBoxHigh[2];
    Equatn qtemp;
    EquatnHelper::setUnitRandomEquatn(qtemp, rngPoolPtr->getU01(0), rngPoolPtr->getU01(0), rngPoolPtr->getU01(0));
    Emapq(newSylinder[0].orientation).coeffs() = qtemp.coeffs();
    for (int i = 1; i < nLocalNew; i++) {
        const auto &prsy = newSylinder[i - 1];
        Evec3 prvec = ECmapq(prsy.orientation) * Evec3(0, 0, 1);
        Evec3 prend = prvec * 0.5 * prsy.length + ECmap3(prsy.pos);
        auto &sy = newSylinder[i];
        Equatn qtemp;
        EquatnHelper::setUnitRandomEquatn(qtemp, rngPoolPtr->getU01(0), rngPoolPtr->getU01(0), rngPoolPtr->getU01(0));
        Emapq(sy.orientation).coeffs() = qtemp.coeffs();
        Evec3 syvec = Emapq(sy.orientation) * Evec3(0, 0, 1);
        Evec3 sypos = prend + (prsy.radius + sy.radius) * prvec + sy.length * 0.5 * syvec;
        Emap3(sy.pos) = sypos;
    }

    sylinderSystem.addNewSylinder(newSylinder, linkage);

    // run as given in runConfig file
    const int nSteps = runConfig.timeTotal / runConfig.dt;
    for (int i = 0; i < nSteps; i++) {
        sylinderSystem.prepareStep();
        auto &sylinderContainer = sylinderSystem.getContainer();
        int nLocal = sylinderContainer.getNumberOfParticleLocal();
        std::vector<double> forceNonBrown(nLocal * 6, 0.0);
        for (int i = 0; i < nLocal; i++) {
            if (sylinderContainer[i].link.group != GEO_INVALID_INDEX &&
                sylinderContainer[i].link.next == GEO_INVALID_INDEX) {
                forceNonBrown[6 * i + 1] = 10; // y 
                forceNonBrown[6 * i + 2] = 10; // z
            }
        }
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
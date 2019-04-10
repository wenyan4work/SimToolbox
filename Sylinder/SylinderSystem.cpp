#include "SylinderSystem.hpp"

#include "Util/EquatnHelper.hpp"
#include "Util/GeoUtil.hpp"
#include "Util/IOHelper.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <vector>

#include <mpi.h>
#include <omp.h>

SylinderSystem::SylinderSystem(const std::string &configFile, const std::string &posFile, int argc, char **argv) {
    initialize(SylinderConfig(configFile), posFile, argc, argv);
}

SylinderSystem::SylinderSystem(const SylinderConfig &runConfig_, const std::string &posFile, int argc, char **argv) {
    initialize(runConfig_, posFile, argc, argv);
}

void SylinderSystem::initialize(const SylinderConfig &runConfig_, const std::string &posFile, int argc, char **argv) {
    runConfig = runConfig_;
    stepCount = 0;
    snapID = 0; // the first snapshot starts from 0 in writeResult

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);
    commRcp = getMPIWORLDTCOMM();

    // set openmp
    if (runConfig.ompThreads > 0) {
        omp_set_num_threads(runConfig.ompThreads);
    }

    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(runConfig.rngSeed);
    collisionSolverPtr = std::make_shared<CollisionSolver>();
    collisionCollectorPtr = std::make_shared<CollisionCollector>();

    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    sylinderContainer.initialize();
    sylinderContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more sample for better balance

    if (IOHelper::fileExist(posFile)) {
        setInitialFromFile(posFile);
    } else {
        setInitialFromConfig();
    }

    showOnScreenRank0(); // at this point all sylinders located on rank 0

    commRcp->barrier();
    decomposeDomain();
    exchangeSylinder(); // distribute to ranks, initial domain decomposition

    setTreeSylinder();

    setPosWithWall();

    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); // prepare the output directory
        writeBox();
    }

    // 100 NON-B steps to resolve initial configuration collisions
    // no output
    printf("-------------------------------------\n");
    printf("-Initial Collision Resolution Begin--\n");
    printf("-------------------------------------\n");
    for (int i = 0; i < 100; i++) {
        prepareStep();
        calcVelocityKnown();
        resolveCollision();
        stepEuler();
    }
    printf("--Initial Collision Resolution End---\n");
    printf("-------------------------------------\n");

    calcVolFrac();
    printf("SylinderSystem Initialized. %d sylinders on process %d\n", sylinderContainer.getNumberOfParticleLocal(),
           commRcp->getRank());
}

void SylinderSystem::setTreeSylinder() {
    // initialize tree
    // always keep tree max_glb_num_ptcl to be twice the global actual particle number.
    const int nGlobal = sylinderContainer.getNumberOfParticleGlobal();
    if (nGlobal > 1.5 * treeSylinderNumber || !treeSylinderNearPtr) {
        // a new larger tree
        treeSylinderNearPtr.reset();
        treeSylinderNearPtr = std::make_unique<TreeSylinderNear>();
        treeSylinderNearPtr->initialize(2 * nGlobal);
        treeSylinderNumber = nGlobal;
    }
}

void SylinderSystem::getOrient(Equatn &orient, const double px, const double py, const double pz, const int threadId) {
    Evec3 pvec;
    if (px < -1 || px > 1) {
        pvec[0] = rngPoolPtr->getU01(threadId);
    } else {
        pvec[0] = px;
    }
    if (py < -1 || py > 1) {
        pvec[1] = rngPoolPtr->getU01(threadId);
    } else {
        pvec[1] = py;
    }
    if (pz < -1 || pz > 1) {
        pvec[2] = rngPoolPtr->getU01(threadId);
    } else {
        pvec[2] = pz;
    }

    // px,py,pz all random, pick uniformly in orientation space
    if (px != pvec[0] && py != pvec[1] && pz != pvec[2]) {
        EquatnHelper::setUnitRandomEquatn(orient, rngPoolPtr->getU01(threadId), rngPoolPtr->getU01(threadId),
                                          rngPoolPtr->getU01(threadId));
        return;
    } else {
        orient = Equatn::FromTwoVectors(Evec3(0, 0, 1), pvec);
    }
}

void SylinderSystem::setInitialFromConfig() {
    // this function init all sylinders on rank 0
    if (runConfig.sylinderLengthSigma > 0) {
        rngPoolPtr->setLogNormalParameters(runConfig.sylinderLength, runConfig.sylinderLengthSigma);
    }

    if (commRcp->getRank() != 0) {
        sylinderContainer.setNumberOfParticleLocal(0);
    } else {
        const double boxEdge[3] = {runConfig.initBoxHigh[0] - runConfig.initBoxLow[0],
                                   runConfig.initBoxHigh[1] - runConfig.initBoxLow[1],
                                   runConfig.initBoxHigh[2] - runConfig.initBoxLow[2]};
        const double minBoxEdge = std::min(std::min(boxEdge[0], boxEdge[1]), boxEdge[2]);
        const double maxLength = minBoxEdge * 0.5;
        const double radius = runConfig.sylinderDiameter / 2;
        const int nSylinderLocal = runConfig.sylinderNumber;
        sylinderContainer.setNumberOfParticleLocal(nSylinderLocal);

#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nSylinderLocal; i++) {
                double length;
                if (runConfig.sylinderLengthSigma > 0) {
                    do { // generate random length
                        length = rngPoolPtr->getLN(threadId);
                    } while (length >= maxLength);
                } else {
                    length = runConfig.sylinderLength;
                }
                double pos[3];
                for (int k = 0; k < 3; k++) {
                    pos[k] = rngPoolPtr->getU01(threadId) * boxEdge[k] + runConfig.initBoxLow[k];
                }
                Equatn orientq;
                getOrient(orientq, runConfig.initOrient[0], runConfig.initOrient[1], runConfig.initOrient[2], threadId);
                double orientation[4];
                Emapq(orientation).coeffs() = orientq.coeffs();
                sylinderContainer[i] = Sylinder(i, radius, radius, length, length, pos, orientation);
                sylinderContainer[i].clear();
            }
        }
    }

    if (runConfig.initCircularX) {
        setInitialCircularCrossSection();
    }
}

// void SylinderSystem::getRandPointInCircle(const double &radius, double &x, double &y, const int &threadId) {
//     double theta = 2 * Pi * rngPoolPtr->getU01(threadId);   /* angle is uniform */
//     double r = radius * sqrt(rngPoolPtr->getU01(threadId)); /* radius proportional to sqrt(U), U~U(0,1) */
//     x = r * cos(theta);
//     y = r * sin(theta);
// }

void SylinderSystem::setInitialCircularCrossSection() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    double radiusCrossSec = 0;            // x, y, z, axis radius
    Evec3 centerCrossSec = Evec3::Zero(); // x, y, z, axis center.
    // x axis
    centerCrossSec = Evec3(0, (runConfig.initBoxHigh[1] - runConfig.initBoxLow[1]) * 0.5 + runConfig.initBoxLow[1],
                           (runConfig.initBoxHigh[2] - runConfig.initBoxLow[2]) * 0.5 + runConfig.initBoxLow[2]);
    radiusCrossSec = 0.5 * std::min(runConfig.initBoxHigh[2] - runConfig.initBoxLow[2],
                                    runConfig.initBoxHigh[1] - runConfig.initBoxLow[1]);
#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            double y = sylinderContainer[i].pos[1];
            double z = sylinderContainer[i].pos[2];
            // replace y,z with position in the circle
            getRandPointInCircle(radiusCrossSec, rngPoolPtr->getU01(threadId), rngPoolPtr->getU01(threadId), y, z);
            sylinderContainer[i].pos[1] = y + centerCrossSec[1];
            sylinderContainer[i].pos[2] = z + centerCrossSec[2];
        }
    }
}

void SylinderSystem::calcVolFrac() {
    // calc volume fraction of sphero cylinders
    // step 1, calc local total volume
    double volLocal = 0;
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for reduction(+ : volLocal)
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        volLocal += 3.1415926535 * (0.25 * sy.length * pow(sy.radius * 2, 2) + pow(sy.radius * 2, 3) / 6);
    }
    double volGlobal = 0;

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 1, &volLocal, &volGlobal);

    // step 2, reduce to root and compute total volume
    if (commRcp->getRank() == 0) {
        double boxVolume = (runConfig.simBoxHigh[0] - runConfig.simBoxLow[0]) *
                           (runConfig.simBoxHigh[1] - runConfig.simBoxLow[1]) *
                           (runConfig.simBoxHigh[2] - runConfig.simBoxLow[2]);
        std::cout << "Volume Sylinder = " << volGlobal << std::endl;
        std::cout << "Volume fraction = " << volGlobal / boxVolume << std::endl;
    }
}

void SylinderSystem::setInitialFromFile(const std::string &filename) {
    if (commRcp->getRank() != 0) {
        sylinderContainer.setNumberOfParticleLocal(0);
    } else {
        std::ifstream myfile(filename);
        std::string line;
        std::getline(myfile, line); // read two header lines
        std::getline(myfile, line);

        std::vector<Sylinder> sylinderReadFromFile;
        while (std::getline(myfile, line)) {
            char typeChar;
            std::istringstream liness(line);
            liness >> typeChar;
            if (typeChar == 'C') {
                Sylinder newBody;
                int gid;
                double mx, my, mz;
                double px, py, pz;
                double radius;
                liness >> gid >> radius >> mx >> my >> mz >> px >> py >> pz;
                Emap3(newBody.pos) = Evec3((mx + px) / 2, (my + py) / 2, (mz + pz) / 2);
                newBody.gid = gid;
                newBody.length = sqrt((px - mx) * (px - mx) + (py - my) * (py - my) + (pz - mz) * (pz - mz));
                Evec3 direction(px - mx, py - my, pz - mz);
                Emapq(newBody.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);

                newBody.radius = radius;
                newBody.radiusCollision = radius;
                newBody.lengthCollision = newBody.length;
                sylinderReadFromFile.push_back(newBody);
                typeChar = 'N';
            }
        }
        myfile.close();

        // sort body, gid ascending;
        std::cout << "Sylinder number in file: " << sylinderReadFromFile.size() << std::endl;
        std::sort(sylinderReadFromFile.begin(), sylinderReadFromFile.end(),
                  [](const Sylinder &t1, const Sylinder &t2) { return t1.gid < t2.gid; });

        // set local
        const int nRead = sylinderReadFromFile.size();
        sylinderContainer.setNumberOfParticleLocal(nRead);
#pragma omp parallel for
        for (int i = 0; i < nRead; i++) {
            sylinderContainer[i] = sylinderReadFromFile[i];
            sylinderContainer[i].clear();
        }
    }
}

std::string SylinderSystem::getCurrentResultFolder() {
    const int num = std::max(400 / commRcp->getSize(), 1); // limit max number of files per folder
    int k = snapID / num;
    int low = k * num, high = k * num + num - 1;
    std::string baseFolder =
        "./result/result" + std::to_string(low) + std::string("-") + std::to_string(high) + std::string("/");
    return baseFolder;
}

void SylinderSystem::writeAscii(const std::string &baseFolder) {
    // write a single ascii .dat file
    const int nGlobal = sylinderContainer.getNumberOfParticleGlobal();

    std::string name = baseFolder + std::string("SylinderAscii_") + std::to_string(snapID) + ".dat";
    SylinderAsciiHeader header;
    header.nparticle = nGlobal;
    header.time = stepCount * runConfig.dt;
    sylinderContainer.writeParticleAscii(name.c_str(), header);
}

void SylinderSystem::writeVTK(const std::string &baseFolder) {
    const int rank = commRcp->getRank();
    const int size = commRcp->getSize();
    Sylinder::writeVTP<PS::ParticleSystem<Sylinder>>(sylinderContainer, sylinderContainer.getNumberOfParticleLocal(),
                                                     baseFolder, std::to_string(snapID), rank);
    collisionCollectorPtr->writeVTP(baseFolder, std::to_string(snapID), rank);
    if (rank == 0) {
        Sylinder::writePVTP(baseFolder, std::to_string(snapID), size); // write parallel head
        collisionCollectorPtr->writePVTP(baseFolder, std::to_string(snapID), size);
    }
}

void SylinderSystem::writeBox() {
    FILE *boxFile = fopen("./result/simBox.vtk", "w");
    fprintf(boxFile, "# vtk DataFile Version 3.0\n");
    fprintf(boxFile, "vtk file\n");
    fprintf(boxFile, "ASCII\n");
    fprintf(boxFile, "DATASET RECTILINEAR_GRID\n");
    fprintf(boxFile, "DIMENSIONS 2 2 2\n");
    fprintf(boxFile, "X_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", runConfig.simBoxLow[0], runConfig.simBoxHigh[0]);
    fprintf(boxFile, "Y_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", runConfig.simBoxLow[1], runConfig.simBoxHigh[1]);
    fprintf(boxFile, "Z_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", runConfig.simBoxLow[2], runConfig.simBoxHigh[2]);
    fprintf(boxFile, "CELL_DATA 1\n");
    fprintf(boxFile, "POINT_DATA 8\n");
    fclose(boxFile);
}

void SylinderSystem::writeResult() {
    std::string baseFolder = getCurrentResultFolder();
    IOHelper::makeSubFolder(baseFolder);
    writeAscii(baseFolder);
    writeVTK(baseFolder);
    snapID++;
}

void SylinderSystem::showOnScreenRank0() {
    if (commRcp->getRank() == 0) {
        printf("-----------SylinderSystem Settings-----------\n");
        runConfig.dump();
        printf("-----------Sylinder Configurations-----------\n");
        const int nLocal = sylinderContainer.getNumberOfParticleLocal();
        for (int i = 0; i < nLocal; i++) {
            sylinderContainer[i].dumpSylinder();
        }
    }
    commRcp->barrier();
}

void SylinderSystem::setDomainInfo() {
    const int pbcX = (runConfig.simBoxPBC[0] ? 1 : 0);
    const int pbcY = (runConfig.simBoxPBC[1] ? 1 : 0);
    const int pbcZ = (runConfig.simBoxPBC[2] ? 1 : 0);
    const int pbcFlag = 100 * pbcX + 10 * pbcY + pbcZ;

    switch (pbcFlag) {
    case 0:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_OPEN);
        break;
    case 1:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_Z);
        break;
    case 10:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_Y);
        break;
    case 100:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_X);
        break;
    case 11:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_YZ);
        break;
    case 101:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XZ);
        break;
    case 110:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XY);
        break;
    case 111:
        dinfo.setBoundaryCondition(PS::BOUNDARY_CONDITION_PERIODIC_XYZ);
        break;
    }

    PS::F64vec3 rootDomainLow;
    PS::F64vec3 rootDomainHigh;
    for (int k = 0; k < 3; k++) {
        rootDomainLow[k] = runConfig.simBoxLow[k];
        rootDomainHigh[k] = runConfig.simBoxHigh[k];
    }
    // for (int k = 0; k < 3; k++) {
    //     if (runConfig.simBoxPBC[k]) {
    //         rootDomainLow[k] = runConfig.simBoxLow[k];
    //         rootDomainHigh[k] = runConfig.simBoxHigh[k];
    //     } else {
    //         rootDomainLow[k] = -std::numeric_limits<double>::max() / 100;
    //         rootDomainHigh[k] = std::numeric_limits<double>::max() / 100;
    //     }
    // }

    dinfo.setPosRootDomain(rootDomainLow, rootDomainHigh); // rootdomain must be specified after PBC
}

void SylinderSystem::decomposeDomain() {
    applyBoxBC();
    dinfo.decomposeDomainAll(sylinderContainer);
}

void SylinderSystem::exchangeSylinder() {
    sylinderContainer.exchangeParticle(dinfo);
    updateSylinderRank();
}

void SylinderSystem::calcMobMatrix() {
    // diagonal hydro mobility operator
    // 3*3 block for translational + 3*3 block for rotational.
    // 3 nnz per row, 18 nnz per tubule

    const double Pi = 3.14159265358979323846;
    const double mu = runConfig.viscosity;

    const int nLocal = sylinderMapRcp->getNodeNumElements();
    TEUCHOS_ASSERT(nLocal == sylinderContainer.getNumberOfParticleLocal());
    const int localSize = nLocal * 6; // local row number

    Kokkos::View<size_t *> rowPointers("rowPointers", localSize + 1);
    rowPointers[0] = 0;
    for (int i = 1; i <= localSize; i++) {
        rowPointers[i] = rowPointers[i - 1] + 3;
    }
    Kokkos::View<int *> columnIndices("columnIndices", rowPointers[localSize]);
    Kokkos::View<double *> values("values", rowPointers[localSize]);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = sylinderContainer[i];

        // calculate the Mob Trans and MobRot
        Emat3 MobTrans; //            double MobTrans[3][3];
        Emat3 MobRot;   //            double MobRot[3][3];
        Emat3 qq;
        Emat3 Imqq;
        Evec3 q = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        qq = q * q.transpose();
        Imqq = Emat3::Identity() - qq;
        // std::cout << cell.orientation.w() << std::endl;
        // std::cout << qq << std::endl;
        // std::cout << Imqq << std::endl;

        const double length = sy.length;
        const double diameter = sy.radius * 2;
        const double b = -(1 + 2 * log(diameter * 0.5 / (length)));
        const double dragPara = 8 * Pi * length * mu / (2 * b);
        const double dragPerp = 8 * Pi * length * mu / (b + 2);
        const double dragRot = 2 * Pi * mu * length * length * length / (3 * (b + 2));
        const double dragParaInv = 1 / dragPara;
        const double dragPerpInv = 1 / dragPerp;
        const double dragRotInv = 1 / dragRot;

        MobTrans = dragParaInv * qq + dragPerpInv * Imqq;
        MobRot = dragRotInv * qq + dragRotInv * Imqq;
        // MobRot regularized to remove null space.
        // here it becomes identity matrix,
        // no effect on geometric constraints
        // no problem for axissymetric slender body.
        // this simplifies the rotational Brownian calculations.

        // column index is local index
        columnIndices[18 * i] = 6 * i; // line 1 of Mob Trans
        columnIndices[18 * i + 1] = 6 * i + 1;
        columnIndices[18 * i + 2] = 6 * i + 2;
        columnIndices[18 * i + 3] = 6 * i; // line 2 of Mob Trans
        columnIndices[18 * i + 4] = 6 * i + 1;
        columnIndices[18 * i + 5] = 6 * i + 2;
        columnIndices[18 * i + 6] = 6 * i; // line 3 of Mob Trans
        columnIndices[18 * i + 7] = 6 * i + 1;
        columnIndices[18 * i + 8] = 6 * i + 2;
        columnIndices[18 * i + 9] = 6 * i + 3; // line 1 of Mob Rot
        columnIndices[18 * i + 10] = 6 * i + 4;
        columnIndices[18 * i + 11] = 6 * i + 5;
        columnIndices[18 * i + 12] = 6 * i + 3; // line 2 of Mob Rot
        columnIndices[18 * i + 13] = 6 * i + 4;
        columnIndices[18 * i + 14] = 6 * i + 5;
        columnIndices[18 * i + 15] = 6 * i + 3; // line 3 of Mob Rot
        columnIndices[18 * i + 16] = 6 * i + 4;
        columnIndices[18 * i + 17] = 6 * i + 5;

        values[18 * i] = MobTrans(0, 0); // line 1 of Mob Trans
        values[18 * i + 1] = MobTrans(0, 1);
        values[18 * i + 2] = MobTrans(0, 2);
        values[18 * i + 3] = MobTrans(1, 0); // line 2 of Mob Trans
        values[18 * i + 4] = MobTrans(1, 1);
        values[18 * i + 5] = MobTrans(1, 2);
        values[18 * i + 6] = MobTrans(2, 0); // line 3 of Mob Trans
        values[18 * i + 7] = MobTrans(2, 1);
        values[18 * i + 8] = MobTrans(2, 2);
        values[18 * i + 9] = MobRot(0, 0); // line 1 of Mob Rot
        values[18 * i + 10] = MobRot(0, 1);
        values[18 * i + 11] = MobRot(0, 2);
        values[18 * i + 12] = MobRot(1, 0); // line 2 of Mob Rot
        values[18 * i + 13] = MobRot(1, 1);
        values[18 * i + 14] = MobRot(1, 2);
        values[18 * i + 15] = MobRot(2, 0); // line 3 of Mob Rot
        values[18 * i + 16] = MobRot(2, 1);
        values[18 * i + 17] = MobRot(2, 2);
    }

    // mobMat is block-diagonal, so domainMap=rangeMap
    mobilityMatrixRcp =
        Teuchos::rcp(new TCMAT(sylinderMobilityMapRcp, sylinderMobilityMapRcp, rowPointers, columnIndices, values));
    mobilityMatrixRcp->fillComplete(sylinderMobilityMapRcp, sylinderMobilityMapRcp); // domainMap, rangeMap

#ifdef DEBUGLCPCOL
    std::cout << "MobMat Constructed: " << mobilityMatrixRcp->description() << std::endl;
    dumpTCMAT(mobilityMatrixRcp, "MobMat.mtx");
#endif
}

void SylinderSystem::calcMobOperator() {
    calcMobMatrix();
    mobilityOperatorRcp = mobilityMatrixRcp;
}

void SylinderSystem::calcVelocityKnown() {
    // allocate and zero out
    // velocityKnown = velocityBrown + velocityNonBrown + mobility * forceNonBrown
    velocityKnownRcp = Teuchos::rcp<TV>(new TV(sylinderMobilityMapRcp, true));
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(nLocal * 6 == velocityKnownRcp->getLocalLength());

    if (!forceNonBrownRcp.is_null()) {
        TEUCHOS_ASSERT(!mobilityOperatorRcp.is_null());
        mobilityOperatorRcp->apply(*forceNonBrownRcp, *velocityKnownRcp);
    }

    if (!velocityNonBrownRcp.is_null()) {
        velocityKnownRcp->update(1.0, *velocityNonBrownRcp, 1.0);
    }

    // write back total non Brownian velocity
    // combine and sync the velNonB set in two places
    auto velPtr = velocityKnownRcp->getLocalView<Kokkos::HostSpace>();
    velocityKnownRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        // translational
        sy.velNonB[0] += velPtr(6 * i + 0, 0);
        sy.velNonB[1] += velPtr(6 * i + 1, 0);
        sy.velNonB[2] += velPtr(6 * i + 2, 0);
        velPtr(6 * i + 0, 0) = sy.velNonB[0];
        velPtr(6 * i + 1, 0) = sy.velNonB[1];
        velPtr(6 * i + 2, 0) = sy.velNonB[2];
        // rotational
        sy.omegaNonB[0] += velPtr(6 * i + 3, 0);
        sy.omegaNonB[1] += velPtr(6 * i + 4, 0);
        sy.omegaNonB[2] += velPtr(6 * i + 5, 0);
        velPtr(6 * i + 3, 0) = sy.omegaNonB[0];
        velPtr(6 * i + 4, 0) = sy.omegaNonB[1];
        velPtr(6 * i + 5, 0) = sy.omegaNonB[2];
    }

    if (!velocityBrownRcp.is_null()) {
        velocityKnownRcp->update(1.0, *velocityBrownRcp, 1.0);
    }
}

void SylinderSystem::stepEuler() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    const double dt = runConfig.dt;
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        // translation
        for (int k = 0; k < 3; k++) {
            sy.vel[k] = sy.velNonB[k] + sy.velBrown[k] + sy.velCol[k];
            sy.omega[k] = sy.omegaNonB[k] + sy.omegaBrown[k] + sy.omegaCol[k];
        }
        sy.stepEuler(dt);
    }
}

void SylinderSystem::resolveCollision() {
    // collect constraints
    collectPairCollision();
    collectWallCollision();

    // solve collision
    // positive buffer value means collision radius is effectively smaller
    // i.e., less likely to collide
    const double buffer = 0;
    collisionSolverPtr->setup(*(collisionCollectorPtr->collisionPoolPtr), sylinderMobilityMapRcp, runConfig.dt, buffer);
    collisionSolverPtr->setControlLCP(runConfig.colResTol, runConfig.colMaxIte, runConfig.colNewtonRefine);
    collisionSolverPtr->solveCollision(mobilityOperatorRcp, velocityKnownRcp);
    collisionSolverPtr->writebackGamma(*(collisionCollectorPtr->collisionPoolPtr));

    // save results
    forceColRcp = collisionSolverPtr->getForceCol();
    velocityColRcp = collisionSolverPtr->getVelocityCol();

    saveVelocityCollision();
}

void SylinderSystem::updateSylinderMap() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    // setup the new sylinderMap
    sylinderMapRcp = getTMAPFromLocalSize(nLocal, commRcp);
    sylinderMobilityMapRcp = getTMAPFromLocalSize(nLocal * 6, commRcp);

    // setup the globalIndex
    int globalIndexBase = sylinderMapRcp->getMinGlobalIndex(); // this is a contiguous map
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderContainer[i].globalIndex = i + globalIndexBase;
    }
}

bool SylinderSystem::getIfWriteResultCurrentStep() {
    return (stepCount % static_cast<int>(runConfig.timeSnap / runConfig.dt) == 0);
}

void SylinderSystem::prepareStep() {
    applyBoxBC();

    if (stepCount % 50 == 0) {
        decomposeDomain();
    }

    exchangeSylinder();

    updateSylinderMap();

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderContainer[i].radiusCollision = sylinderContainer[i].radius * runConfig.sylinderDiameterColRatio;
        sylinderContainer[i].lengthCollision = sylinderContainer[i].length * runConfig.sylinderLengthColRatio;
        sylinderContainer[i].clear();
    }

    calcMobOperator();

    forceNonBrownRcp.reset();
    velocityNonBrownRcp.reset();
    velocityBrownRcp.reset();
}

void SylinderSystem::setForceNonBrown(const std::vector<double> &forceNonBrown) {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(forceNonBrown.size() == 6 * nLocal);
    TEUCHOS_ASSERT(sylinderMobilityMapRcp->getNodeNumElements() == 6 * nLocal);
    forceNonBrownRcp = getTVFromVector(forceNonBrown, commRcp);
}

void SylinderSystem::setVelocityNonBrown(const std::vector<double> &velNonBrown) {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velNonBrown.size() == 6 * nLocal);
    TEUCHOS_ASSERT(sylinderMobilityMapRcp->getNodeNumElements() == 6 * nLocal);
    velocityNonBrownRcp = getTVFromVector(velNonBrown, commRcp);
}

void SylinderSystem::runStep() {

    if (runConfig.KBT > 0) {
        calcVelocityBrown();
    }

    calcVelocityKnown(); // velocityKnown = velocityBrown + velocityNonBrown + mobility * forceNonBrown

    resolveCollision();

    if (getIfWriteResultCurrentStep()) {
        // write result before moving. guarantee data written is consistent to geometry
        writeResult();
    }

    stepEuler();

    stepCount++;
}

void SylinderSystem::saveVelocityCollision() {
    forceColRcp = collisionSolverPtr->getForceCol();
    velocityColRcp = collisionSolverPtr->getVelocityCol();
    auto velocityPtr = velocityColRcp->getLocalView<Kokkos::HostSpace>();
    velocityColRcp->modify<Kokkos::HostSpace>();

    const int sylinderLocalNumber = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velocityPtr.dimension_0() == sylinderLocalNumber * 6);
    TEUCHOS_ASSERT(velocityPtr.dimension_1() == 1);

#pragma omp parallel for
    for (int i = 0; i < sylinderLocalNumber; i++) {
        auto &sy = sylinderContainer[i];
        sy.velCol[0] = velocityPtr(6 * i, 0);
        sy.velCol[1] = velocityPtr(6 * i + 1, 0);
        sy.velCol[2] = velocityPtr(6 * i + 2, 0);
        sy.omegaCol[0] = velocityPtr(6 * i + 3, 0);
        sy.omegaCol[1] = velocityPtr(6 * i + 4, 0);
        sy.omegaCol[2] = velocityPtr(6 * i + 5, 0);
    }
}

void SylinderSystem::calcVelocityBrown() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    const double Pi = 3.1415926535897932384626433;
    const double mu = runConfig.viscosity;
    const double dt = runConfig.dt;
    const double delta = dt * 0.1; // a small parameter used in RFD algorithm
    const double kBT = runConfig.KBT;
    const double kBTfactor = sqrt(2 * kBT / dt);

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            // constants
            const double length = sy.length;
            const double diameter = sy.radius * 2;
            const double b = -(1 + 2 * log(diameter * 0.5 / (length)));
            const double invDragPara = 1 / (8 * Pi * length * mu / (2 * b));
            const double invDragPerp = 1 / (8 * Pi * length * mu / (b + 2));
            const double invDragRot = 1 / (2 * Pi * mu * length * length * length / (3 * (b + 2)));

            // convert FDPS vec3 to Evec3
            Evec3 direction = Emapq(sy.orientation) * Evec3(0, 0, 1);

            // RFD from Delong, JCP, 2015
            // slender fiber has 0 rot drag, regularize with identity rot mobility
            // trans mobility is this
            Evec3 q = direction;
            Emat3 Nmat = (invDragPara - invDragPerp) * (q * q.transpose()) + (invDragPerp)*Emat3::Identity();
            Emat3 Nmatsqrt = Nmat.llt().matrixL();

            // velocity
            Evec3 Wrot(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wpos(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wrfdrot(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wrfdpos(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));

            Equatn orientRFD = Emapq(sy.orientation);
            EquatnHelper::rotateEquatn(orientRFD, Wrfdrot, delta);
            q = orientRFD * Evec3(0, 0, 1);
            Emat3 Nmatrfd = (invDragPara - invDragPerp) * (q * q.transpose()) + (invDragPerp)*Emat3::Identity();

            Evec3 vel = kBTfactor * (Nmatsqrt * Wpos);           // Gaussian noise
            vel += (kBT / delta) * ((Nmatrfd - Nmat) * Wrfdpos); // rfd drift. seems no effect in this case
            Evec3 omega = sqrt(invDragRot) * kBTfactor * Wrot;   // regularized identity rotation drag

            Emap3(sy.velBrown) = vel;
            Emap3(sy.omegaBrown) = omega;
        }
    }

    velocityBrownRcp = Teuchos::rcp<TV>(new TV(sylinderMobilityMapRcp, true));
    auto velocityPtr = velocityBrownRcp->getLocalView<Kokkos::HostSpace>();
    velocityBrownRcp->modify<Kokkos::HostSpace>();

    TEUCHOS_ASSERT(velocityPtr.dimension_0() == nLocal * 6);
    TEUCHOS_ASSERT(velocityPtr.dimension_1() == 1);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = sylinderContainer[i];
        velocityPtr(6 * i, 0) = sy.velBrown[0];
        velocityPtr(6 * i + 1, 0) = sy.velBrown[1];
        velocityPtr(6 * i + 2, 0) = sy.velBrown[2];
        velocityPtr(6 * i + 3, 0) = sy.omegaBrown[0];
        velocityPtr(6 * i + 4, 0) = sy.omegaBrown[1];
        velocityPtr(6 * i + 5, 0) = sy.omegaBrown[2];
    }
}

void SylinderSystem::collectWallCollision() {
    auto collisionPoolPtr = collisionCollectorPtr->collisionPoolPtr; // shared_ptr
    const int nThreads = collisionPoolPtr->size();
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    if (runConfig.wallLowZ) {
        // process collisions with bottom wall
        const double wallBot = runConfig.simBoxLow[2];
#pragma omp parallel num_threads(nThreads)
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nLocal; i++) {
                const auto &sy = sylinderContainer[i];
                const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
                const Evec3 Pm = ECmap3(sy.pos) - direction * (sy.lengthCollision * 0.5);
                const Evec3 Pp = ECmap3(sy.pos) + direction * (sy.lengthCollision * 0.5);
                const double distm = Pm[2] - wallBot - sy.radius;
                const double distp = Pp[2] - wallBot - sy.radius;
                // if collision, norm is always (0,0,1), loc could be Pm, Pp, or middle
                Evec3 colLoc;
                double phi0;
                bool wallCollide = false;
                if (distm < distp && distm < 0) {
                    colLoc = Pm;
                    phi0 = distm;
                    wallCollide = true;
                } else if (distm > distp && distp < 0) {
                    colLoc = Pp;
                    phi0 = distp;
                    wallCollide = true;
                } else if (distm == distp && distm < 0) {
                    colLoc = (Pm + Pp) * 0.5; // middle point
                    phi0 = distm;
                    wallCollide = true;
                }
                if (wallCollide != true) {
                    continue;
                }
                // add a new collision block. this block has only 6 non zero entries.
                // passing sy.gid+1/globalIndex+1 as a 'fake' colliding body j, which is actually not used in the solver
                // when oneside=true, out of range index is ignored
                (*collisionPoolPtr)[threadId].emplace_back(
                    phi0, -phi0, sy.gid, sy.gid + 1, sy.globalIndex, sy.globalIndex + 1, Evec3(0, 0, 1), Evec3(0, 0, 0),
                    colLoc - ECmap3(sy.pos), Evec3(0, 0, 0), colLoc, Evec3(colLoc[0], colLoc[1], wallBot), true);
            }
        }
    }

    if (runConfig.wallHighZ) {
        const double wallTop = runConfig.simBoxHigh[2];
        // process collisions with top wall
#pragma omp parallel num_threads(nThreads)
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nLocal; i++) {
                const auto &sy = sylinderContainer[i];
                const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
                const Evec3 Pm = ECmap3(sy.pos) - direction * (sy.lengthCollision * 0.5);
                const Evec3 Pp = ECmap3(sy.pos) + direction * (sy.lengthCollision * 0.5);
                const double distm = wallTop - Pm[2] - sy.radius;
                const double distp = wallTop - Pp[2] - sy.radius;
                // if collision, norm is always (0,0,-1), loc could be Pm, Pp, or middle
                Evec3 colLoc;
                double phi0;
                bool wallCollide = false;
                if (distm < distp && distm < 0) {
                    colLoc = Pm;
                    phi0 = distm;
                    wallCollide = true;
                } else if (distm > distp && distp < 0) {
                    colLoc = Pp;
                    phi0 = distp;
                    wallCollide = true;
                } else if (distm == distp && distm < 0) {
                    colLoc = (Pm + Pp) * 0.5; // middle point
                    phi0 = distm;
                    wallCollide = true;
                }
                if (wallCollide != true) {
                    continue;
                }
                // add a new collision block. this block has only 6 non zero entries.
                // passing sy.gid+1/globalIndex+1 as a 'fake' colliding body j, which is actually not used in the solver
                // when oneside=true, out of range index is ignored
                (*collisionPoolPtr)[threadId].emplace_back(phi0, -phi0, sy.gid, sy.gid + 1, sy.globalIndex,
                                                           sy.globalIndex + 1, Evec3(0, 0, -1), Evec3(0, 0, 0),
                                                           colLoc - ECmap3(sy.pos), Evec3(0, 0, 0), colLoc,
                                                           Evec3(colLoc[0], colLoc[1], wallTop), true);
            }
        }
    }

    return;
}

void SylinderSystem::collectPairCollision() {
    auto &collector = *collisionCollectorPtr;
    collector.clear();

    CalcSylinderNearForce calcColFtr(collisionCollectorPtr->collisionPoolPtr);

    TEUCHOS_ASSERT(treeSylinderNearPtr);
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    setTreeSylinder();
    treeSylinderNearPtr->calcForceAll(calcColFtr, sylinderContainer, dinfo);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderContainer[i].sepmin = (treeSylinderNearPtr->getForce(i)).sepmin;
    }
}

std::pair<int, int> SylinderSystem::getMaxGid() {
    int maxGidLocal = 0;
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    for (int i = 0; i < nLocal; i++) {
        maxGidLocal = std::max(maxGidLocal, sylinderContainer[i].gid);
    }

    int maxGidGlobal = maxGidLocal;
    Teuchos::reduceAll(*commRcp, Teuchos::MaxValueReductionOp<int, int>(), 1, &maxGidLocal, &maxGidGlobal);
    if (commRcp->getRank() == 0)
        printf("rank: %d,maxGidLocal: %d,maxGidGlobal %d\n", commRcp->getRank(), maxGidLocal, maxGidGlobal);

    return std::pair<int, int>(maxGidLocal, maxGidGlobal);
}

void SylinderSystem::calcBoundingBox(double localLow[3], double localHigh[3], double globalLow[3],
                                     double globalHigh[3]) {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    double lx, ly, lz;
    lx = ly = lz = std::numeric_limits<double>::max();
    double hx, hy, hz;
    hx = hy = hz = std::numeric_limits<double>::min();

    for (int i = 0; i < nLocal; i++) {
        const auto &sy = sylinderContainer[i];
        const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        Evec3 pm = ECmap3(sy.pos) - (sy.length * 0.5) * direction;
        Evec3 pp = ECmap3(sy.pos) + (sy.length * 0.5) * direction;
        lx = std::min(std::min(lx, pm[0]), pp[0]);
        ly = std::min(std::min(ly, pm[1]), pp[1]);
        lz = std::min(std::min(lz, pm[2]), pp[2]);
        hx = std::max(std::max(hx, pm[0]), pp[0]);
        hy = std::max(std::max(hy, pm[1]), pp[1]);
        hz = std::max(std::max(hz, pm[2]), pp[2]);
    }

    localLow[0] = lx;
    localLow[1] = ly;
    localLow[2] = lz;
    localHigh[0] = hx;
    localHigh[1] = hy;
    localHigh[2] = hz;

    for (int k = 0; k < 3; k++) {
        globalLow[k] = localLow[k];
        globalHigh[k] = localHigh[k];
    }

    Teuchos::reduceAll(*commRcp, Teuchos::MinValueReductionOp<int, double>(), 3, localLow, globalLow);
    Teuchos::reduceAll(*commRcp, Teuchos::MaxValueReductionOp<int, double>(), 3, localHigh, globalHigh);

    return;
}

void SylinderSystem::updateSylinderRank() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    const int rank = commRcp->getRank();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderContainer[i].rank = rank;
    }
}

void SylinderSystem::applyBoxBC() { sylinderContainer.adjustPositionIntoRootDomain(dinfo); }

void SylinderSystem::calcColStress() {
    //     // average all two-side collisions
    //     const auto &colPool = *(collisionCollectorPtr->collisionPoolPtr);
    //     const int poolSize = colPool.size();

    //     // reduction of stress
    //     std::vector<Emat3> stressPool(poolSize);
    // #pragma omp parallel for schedule(static, 1)
    //     for (int que = 0; que < poolSize; que++) {
    //         Emat3 stress = Emat3::Zero();
    //         for (const auto &col : colPool[que]) {
    //             if (col.oneSide == false && col.gamma > 0)
    //                 stress = stress + (col.stress * col.gamma);
    //         }
    //         stressPool[que] = stress;
    //     }

    //     Emat3 sumStress = Emat3::Zero();
    //     for (int i = 0; i < poolSize; i++) {
    //         sumStress = sumStress + stressPool[i];
    //     }

    Emat3 meanStress = Emat3::Zero();
    collisionCollectorPtr->computeCollisionStress(meanStress, false);

    // scale to nkBT
    const double scaleFactor = 1 / (sylinderMapRcp->getGlobalNumElements() * runConfig.KBT);
    meanStress *= scaleFactor;
    // mpi reduction
    double meanStressLocal[9];
    double meanStressGlobal[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            meanStressLocal[i * 3 + j] = meanStress(i, j);
            meanStressGlobal[i * 3 + j] = 0;
        }
    }

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 9, meanStressLocal, meanStressGlobal);

    if (commRcp->getRank() == 0)
        printf("RECORD: ColXF %7g,%7g,%7g,%7g,%7g,%7g,%7g,%7g,%7g\n",         //
               meanStressGlobal[0], meanStressGlobal[1], meanStressGlobal[2], //
               meanStressGlobal[3], meanStressGlobal[4], meanStressGlobal[5], //
               meanStressGlobal[6], meanStressGlobal[7], meanStressGlobal[8]);
}

void SylinderSystem::setPosWithWall() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    const double buffer = 1e-4;
    // directly move sylinders to avoid overlapping with the wall
    if (runConfig.wallLowZ) {
        const double wallBot = runConfig.simBoxLow[2];
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            const Evec3 direction = Emapq(sy.orientation) * Evec3(0, 0, 1);
            const Evec3 Pm = Emap3(sy.pos) - direction * (sy.lengthCollision * 0.5);
            const Evec3 Pp = Emap3(sy.pos) + direction * (sy.lengthCollision * 0.5);
            const double distm = Pm[2] - sy.radius - wallBot;
            const double distp = Pp[2] - sy.radius - wallBot;
            if (distm < distp && distm < 0) {
                Emap3(sy.pos) += Evec3(0, 0, -distm + buffer);
            } else if (distp <= distm && distp < 0) {
                Emap3(sy.pos) += Evec3(0, 0, -distp + buffer);
            }
        }
    }

    if (runConfig.wallHighZ) {
        const double wallTop = runConfig.simBoxHigh[2];
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            const Evec3 direction = Emapq(sy.orientation) * Evec3(0, 0, 1);
            const Evec3 Pm = Emap3(sy.pos) - direction * (sy.lengthCollision * 0.5);
            const Evec3 Pp = Emap3(sy.pos) + direction * (sy.lengthCollision * 0.5);
            const double distm = wallTop - (Pm[2] + sy.radius);
            const double distp = wallTop - (Pp[2] + sy.radius);
            if (distm < distp && distm < 0) {
                Emap3(sy.pos) -= Evec3(0, 0, -distm + buffer);
            } else if (distp <= distm && distp < 0) {
                Emap3(sy.pos) -= Evec3(0, 0, -distp + buffer);
            }
        }
    }
}

void SylinderSystem::addNewSylinder(std::vector<Sylinder> &newSylinder) {
    // assign unique new gid for old cells on all ranks
    std::pair<int, int> maxGid = getMaxGid();
    const int maxGidLocal = maxGid.first;
    const int maxGidGlobal = maxGid.second;
    const int newNumberOnLocal = newSylinder.size();
    const auto &newMapRcp = getTMAPFromLocalSize(newNumberOnLocal, commRcp);

    // a large enough buffer on every rank
    std::vector<int> newID(newMapRcp->getGlobalNumElements(), 0);
    std::vector<int> newNumber(commRcp->getSize(), 0);
    std::vector<int> displ(commRcp->getSize(), 0);

    // assign random id on rank 0
    if (commRcp->getRank() == 0) {
        std::iota(newID.begin(), newID.end(), 0);
        std::random_shuffle(newID.begin(), newID.end());
    }
    // collect number of ids from all ranks to rank0
    MPI_Gather(&newNumberOnLocal, 1, MPI_INT, newNumber.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (commRcp->getRank() == 0) {
        std::partial_sum(newNumber.cbegin(), newNumber.cend() - 1, displ.begin() + 1);
    }

    std::vector<int> newIDRecv(newNumberOnLocal, 0);
    // scatter from rank 0 to every rank
    MPI_Scatterv(newID.data(), newNumber.data(), displ.data(), MPI_INT, //
                 newIDRecv.data(), newNumberOnLocal, MPI_INT, 0, MPI_COMM_WORLD);

    // set new gid
    for (int i = 0; i < newNumberOnLocal; i++) {
        newSylinder[i].gid = newIDRecv[i] + 1 + maxGidGlobal;
    }
    // add new cell to old cell
    for (int i = 0; i < newNumberOnLocal; i++) {
        sylinderContainer.addOneParticle(newSylinder[i]);
    }
}
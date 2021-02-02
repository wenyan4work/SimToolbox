#include "ParticleSystem.hpp"

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

template <class Ptcl>
ParticleSystem<Ptcl>::ParticleSystem(const std::string &configFile, const std::string &posFile, int argc, char **argv) {
    initialize(ParticleConfig(configFile), posFile, argc, argv);
}

template <class Ptcl>
ParticleSystem<Ptcl>::ParticleSystem(const ParticleConfig &runConfig_, const std::string &posFile, int argc,
                                     char **argv) {
    initialize(runConfig_, posFile, argc, argv);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::initialize(const ParticleConfig &runConfig_, const std::string &posFile, int argc,
                                      char **argv) {
    runConfig = runConfig_;
    stepCount = 0;
    snapID = 0; // the first snapshot starts from 0 in writeResult

    // store the random seed
    restartRngSeed = runConfig.rngSeed;

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);
    commRcp = getMPIWORLDTCOMM();
    showOnScreenRank0();

    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(runConfig.rngSeed);
    conSolverPtr = std::make_shared<ConstraintSolver>();
    conCollectorPtr = std::make_shared<ConstraintCollector>();

    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    ptclContainer.initialize();
    ptclContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more sample for better balance

    if (IOHelper::fileExist(posFile)) {
        setInitialFromFile(posFile);
    } else {
        setInitialFromConfig();
    }

    // at this point all sylinders located on rank 0
    commRcp->barrier();
    decomposeDomain();
    exchangeParticle(); // distribute to ranks, initial domain decomposition

    ptclNearDataDirectoryPtr = std::make_shared<ZDD<PtclNearEP>>(ptclContainer.getNumberOfParticleLocal());

    treeParticleNumber = 0;
    setTreeParticle();

    calcVolFrac();

    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); // prepare the output directory
        writeBox();
    }

    if (!runConfig.particleFixed) {
        // 100 NON-B steps to resolve initial configuration collisions
        // no output
        if (commRcp->getRank() == 0) {
            printf("-------------------------------------\n");
            printf("-Initial Collision Resolution Begin--\n");
            printf("-------------------------------------\n");
        }
        for (int i = 0; i < 100; i++) {
            prepareStep();
            calcVelocityNonCon();
            resolveConstraints();
            saveForceVelocityConstraints();
            sumForceVelocity();
            stepEuler();
        }
        if (commRcp->getRank() == 0) {
            printf("--Initial Collision Resolution End---\n");
            printf("-------------------------------------\n");
        }
    }

    printf("ParticleSystem Initialized. %d sylinders on process %d\n", sylinderContainer.getNumberOfParticleLocal(),
           commRcp->getRank());
}

template <class Ptcl>
void ParticleSystem<Ptcl>::reinitialize(const ParticleConfig &runConfig_, const std::string &restartFile, int argc,
                                        char **argv, bool eulerStep) {
    runConfig = runConfig_;

    // Read the timestep information and pvtp filenames from restartFile
    std::string pvtpFileName;
    std::ifstream myfile(restartFile);

    myfile >> restartRngSeed;
    myfile >> stepCount;
    myfile >> snapID;
    myfile >> pvtpFileName;

    // increment the rngSeed forward by one to ensure randomness compared to previous run
    restartRngSeed++;

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);
    commRcp = getMPIWORLDTCOMM();
    showOnScreenRank0();

    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(restartRngSeed);
    conSolverPtr = std::make_shared<ConstraintSolver>();
    conCollectorPtr = std::make_shared<ConstraintCollector>();

    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    sylinderContainer.initialize();
    sylinderContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more samples for better balance

    setInitialFromVTKFile(pvtpFileName);

    // VTK data is wrote before the Euler step, thus we need to run one Euler step below
    if (eulerStep)
        stepEuler();

    stepCount++;
    snapID++;

    // at this point all sylinders located on rank 0
    commRcp->barrier();
    applyBoxBC();
    decomposeDomain();
    exchangeParticle(); // distribute to ranks, initial domain decomposition
    updateParticleMap();

    sylinderNearDataDirectoryPtr = std::make_shared<ZDD<ParticleNearEP>>(sylinderContainer.getNumberOfParticleLocal());

    treeParticleNumber = 0;
    setTreeParticle();
    calcVolFrac();

    printf("ParticleSystem Reinitialized. %d sylinders on process %d\n", sylinderContainer.getNumberOfParticleLocal(),
           commRcp->getRank());
}

template <class Ptcl>
void ParticleSystem<Ptcl>::setTreeParticle() {
    // initialize tree
    // always keep tree max_glb_num_ptcl to be twice the global actual particle number.
    const int nGlobal = sylinderContainer.getNumberOfParticleGlobal();
    if (nGlobal > 1.5 * treeParticleNumber || !treeParticleNearPtr) {
        // a new larger tree
        treeParticleNearPtr.reset();
        treeParticleNearPtr = std::make_unique<TreeParticleNear>();
        treeParticleNearPtr->initialize(2 * nGlobal);
        treeParticleNumber = nGlobal;
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::getOrient(Equatn &orient, const double px, const double py, const double pz,
                                     const int threadId) {
    Evec3 pvec;
    if (px < -1 || px > 1) {
        pvec[0] = 2 * rngPoolPtr->getU01(threadId) - 1;
    } else {
        pvec[0] = px;
    }
    if (py < -1 || py > 1) {
        pvec[1] = 2 * rngPoolPtr->getU01(threadId) - 1;
    } else {
        pvec[1] = py;
    }
    if (pz < -1 || pz > 1) {
        pvec[2] = 2 * rngPoolPtr->getU01(threadId) - 1;
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

template <class Ptcl>
void ParticleSystem<Ptcl>::setInitialFromConfig() {
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
        const int nParticleLocal = runConfig.sylinderNumber;
        sylinderContainer.setNumberOfParticleLocal(nParticleLocal);

#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nParticleLocal; i++) {
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
                sylinderContainer[i] = Particle(i, radius, radius, length, length, pos, orientation);
                sylinderContainer[i].clear();
            }
        }
    }

    if (runConfig.initCircularX) {
        setInitialCircularCrossSection();
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::setInitialCircularCrossSection() {
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

template <class Ptcl>
void ParticleSystem<Ptcl>::calcVolFrac() {
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
        std::cout << "Volume Particle = " << volGlobal << std::endl;
        std::cout << "Volume fraction = " << volGlobal / boxVolume << std::endl;
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::setInitialFromFile(const std::string &filename) {
    if (commRcp->getRank() != 0) {
        sylinderContainer.setNumberOfParticleLocal(0);
    } else {
        std::ifstream myfile(filename);
        std::string line;
        std::getline(myfile, line); // read two header lines
        std::getline(myfile, line);

        std::vector<Particle> sylinderReadFromFile;
        while (std::getline(myfile, line)) {
            char typeChar;
            std::istringstream liness(line);
            liness >> typeChar;
            if (typeChar == 'C') {
                Particle newBody;
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
        std::cout << "Particle number in file: " << sylinderReadFromFile.size() << std::endl;
        std::sort(sylinderReadFromFile.begin(), sylinderReadFromFile.end(),
                  [](const Particle &t1, const Particle &t2) { return t1.gid < t2.gid; });

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

template <class Ptcl>
void ParticleSystem<Ptcl>::setInitialFromVTKFile(const std::string &pvtpFileName) {
    if (commRcp->getRank() != 0) {
        sylinderContainer.setNumberOfParticleLocal(0);
    } else {
        std::string baseFolder = getCurrentResultFolder();

        // Read the pvtp file and automatically merge the vtks files into a single polydata
        vtkSmartPointer<vtkXMLPPolyDataReader> reader1 = vtkSmartPointer<vtkXMLPPolyDataReader>::New();
        std::cout << "Reading " << baseFolder + pvtpFileName << std::endl;
        reader1->SetFileName((baseFolder + pvtpFileName).c_str());
        reader1->Update();

        // Extract the polydata (At this point, the polydata is unsorted)
        vtkSmartPointer<vtkPolyData> polydata1 = reader1->GetOutput();

        // Extract the point/cell data
        vtkSmartPointer<vtkPoints> posData = polydata1->GetPoints();
        vtkSmartPointer<vtkDataArray> gidData = polydata1->GetCellData()->GetArray("gid");
        vtkSmartPointer<vtkDataArray> groupData = polydata1->GetCellData()->GetArray("group");
        vtkSmartPointer<vtkDataArray> lengthData = polydata1->GetCellData()->GetArray("length");
        vtkSmartPointer<vtkDataArray> lengthCollisionData = polydata1->GetCellData()->GetArray("lengthCollision");
        vtkSmartPointer<vtkDataArray> radiusData = polydata1->GetCellData()->GetArray("radius");
        vtkSmartPointer<vtkDataArray> radiusCollisionData = polydata1->GetCellData()->GetArray("radiusCollision");
        vtkSmartPointer<vtkDataArray> znormData = polydata1->GetCellData()->GetArray("znorm");
        vtkSmartPointer<vtkDataArray> velData = polydata1->GetCellData()->GetArray("vel");
        vtkSmartPointer<vtkDataArray> omegaData = polydata1->GetCellData()->GetArray("omega");

        // Store the data within a temporary vector of Particles
        const int sylinderNumberInFile = posData->GetNumberOfPoints() / 2; // two points per sylinder
        std::vector<Particle> sylinderReadFromFile(sylinderNumberInFile);

#pragma omp parallel for
        for (int i = 0; i < sylinderNumberInFile; i++) {
            auto &newBody = sylinderReadFromFile[i];
            double leftEndpointPos[3] = {0, 0, 0};
            double rightEndpointPos[3] = {0, 0, 0};
            posData->GetPoint(i * 2, leftEndpointPos);
            posData->GetPoint(i * 2 + 1, rightEndpointPos);
            newBody.pos[0] = (leftEndpointPos[0] + rightEndpointPos[0]) / 2;
            newBody.pos[1] = (leftEndpointPos[1] + rightEndpointPos[1]) / 2;
            newBody.pos[2] = (leftEndpointPos[2] + rightEndpointPos[2]) / 2;
            newBody.gid = gidData->GetComponent(i, 0);
            newBody.link.group = groupData->GetComponent(i, 0);
            newBody.length = lengthData->GetComponent(i, 0);
            newBody.lengthCollision = lengthCollisionData->GetComponent(i, 0);
            newBody.radius = radiusData->GetComponent(i, 0);
            newBody.radiusCollision = radiusCollisionData->GetComponent(i, 0);
            const Evec3 direction(znormData->GetComponent(i, 0), znormData->GetComponent(i, 1),
                                  znormData->GetComponent(i, 2));
            Emapq(newBody.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);
            newBody.vel[0] = velData->GetComponent(i, 0);
            newBody.vel[1] = velData->GetComponent(i, 1);
            newBody.vel[2] = velData->GetComponent(i, 2);
            newBody.omega[0] = omegaData->GetComponent(i, 0);
            newBody.omega[1] = omegaData->GetComponent(i, 1);
            newBody.omega[2] = omegaData->GetComponent(i, 2);
        }

        // sort the vector of Particles by gid ascending;
        std::cout << "Particle number in file: " << sylinderReadFromFile.size() << std::endl;
        std::sort(sylinderReadFromFile.begin(), sylinderReadFromFile.end(),
                  [](const Particle &t1, const Particle &t2) { return t1.gid < t2.gid; });

        // set local
        const int nRead = sylinderReadFromFile.size();
        sylinderContainer.setNumberOfParticleLocal(nRead);
#pragma omp parallel for
        for (int i = 0; i < nRead; i++) {
            sylinderContainer[i] = sylinderReadFromFile[i];
            // sylinderContainer[i].clear(); (Do not clear the sylinderContainer as the velocity data is needed to take
            // a EulerStep)
        }
    }
}

template <class Ptcl>
std::string ParticleSystem<Ptcl>::getCurrentResultFolder() {
    return getResultFolderWithID(this->snapID);
}

template <class Ptcl>
std::string ParticleSystem<Ptcl>::getResultFolderWithID(int snapID_) {
    const int num = std::max(400 / commRcp->getSize(), 1); // limit max number of files per folder
    int k = snapID_ / num;
    int low = k * num, high = k * num + num - 1;
    std::string baseFolder =
        "./result/result" + std::to_string(low) + std::string("-") + std::to_string(high) + std::string("/");
    return baseFolder;
}

template <class Ptcl>
void ParticleSystem<Ptcl>::writeAscii(const std::string &baseFolder) {
    // write a single ascii .dat file
    const int nGlobal = sylinderContainer.getNumberOfParticleGlobal();

    std::string name = baseFolder + std::string("ParticleAscii_") + std::to_string(snapID) + ".dat";
    ParticleAsciiHeader header;
    header.nparticle = nGlobal;
    header.time = stepCount * runConfig.dt;
    sylinderContainer.writeParticleAscii(name.c_str(), header);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::writeTimeStepInfo(const std::string &baseFolder) {
    if (commRcp->getRank() == 0) {
        // write a single txt file containing timestep and most recent pvtp file names
        std::string name = baseFolder + std::string("../../TimeStepInfo.txt");
        std::string pvtpFileName = std::string("Particle_") + std::to_string(snapID) + std::string(".pvtp");

        FILE *restartFile = fopen(name.c_str(), "w");
        fprintf(restartFile, "%u\n", restartRngSeed);
        fprintf(restartFile, "%u\n", stepCount);
        fprintf(restartFile, "%u\n", snapID);
        fprintf(restartFile, "%s\n", pvtpFileName.c_str());
        fclose(restartFile);
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::writeVTK(const std::string &baseFolder) {
    const int rank = commRcp->getRank();
    const int size = commRcp->getSize();
    Particle::writeVTP<PS::ParticleSystem<Particle>>(sylinderContainer, sylinderContainer.getNumberOfParticleLocal(),
                                                     baseFolder, std::to_string(snapID), rank);
    conCollectorPtr->writeVTP(baseFolder, "", std::to_string(snapID), rank);
    if (rank == 0) {
        Particle::writePVTP(baseFolder, std::to_string(snapID), size); // write parallel head
        conCollectorPtr->writePVTP(baseFolder, "", std::to_string(snapID), size);
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::writeBox() {
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

template <class Ptcl>
void ParticleSystem<Ptcl>::writeResult() {
    std::string baseFolder = getCurrentResultFolder();
    IOHelper::makeSubFolder(baseFolder);
    writeAscii(baseFolder);
    writeVTK(baseFolder);
    writeTimeStepInfo(baseFolder);
    snapID++;
}

template <class Ptcl>
void ParticleSystem<Ptcl>::showOnScreenRank0() {
    if (commRcp->getRank() == 0) {
        printf("-----------ParticleSystem Settings-----------\n");
        runConfig.dump();
        // printf("-----------Particle Configurations-----------\n");
        // const int nLocal = sylinderContainer.getNumberOfParticleLocal();
        // for (int i = 0; i < nLocal; i++) {
        //     sylinderContainer[i].dumpParticle();
        // }
    }
    // commRcp->barrier();
}

template <class Ptcl>
void ParticleSystem<Ptcl>::setDomainInfo() {
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

    dinfo.setPosRootDomain(rootDomainLow, rootDomainHigh); // rootdomain must be specified after PBC
}

template <class Ptcl>
void ParticleSystem<Ptcl>::decomposeDomain() {
    applyBoxBC();
    dinfo.decomposeDomainAll(sylinderContainer);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::exchangeParticle() {
    sylinderContainer.exchangeParticle(dinfo);
    updateParticleRank();
}

template <class Ptcl>
void ParticleSystem<Ptcl>::calcMobMatrix() {
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

template <class Ptcl>
void ParticleSystem<Ptcl>::calcMobOperator() {
    calcMobMatrix();
    mobilityOperatorRcp = mobilityMatrixRcp;
}

template <class Ptcl>
void ParticleSystem<Ptcl>::calcVelocityNonCon() {
    // velocityNonCon = velocityBrown + velocityPartNonBrown + mobility * forcePartNonBrown
    // if monolayer, set velBrownZ =0, velPartNonBrownZ =0, forcePartNonBrownZ =0
    velocityNonConRcp = Teuchos::rcp<TV>(new TV(sylinderMobilityMapRcp, true)); // allocate and zero out
    auto velNCPtr = velocityNonConRcp->getLocalView<Kokkos::HostSpace>();

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(nLocal * 6 == velocityNonConRcp->getLocalLength());

    if (!forcePartNonBrownRcp.is_null()) {
        // apply mobility
        TEUCHOS_ASSERT(!mobilityOperatorRcp.is_null());
        mobilityOperatorRcp->apply(*forcePartNonBrownRcp, *velocityNonConRcp);
        if (runConfig.monolayer) {
#pragma omp parallel for
            for (int i = 0; i < nLocal; i++) {
                velNCPtr(6 * i + 2, 0) = 0; // vz
                velNCPtr(6 * i + 3, 0) = 0; // omegax
                velNCPtr(6 * i + 4, 0) = 0; // omegay
            }
        }
        // write back to Particle members
        auto forcePtr = forcePartNonBrownRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            // torque
            sy.forceNonB[0] = forcePtr(6 * i + 0, 0);
            sy.forceNonB[1] = forcePtr(6 * i + 1, 0);
            sy.forceNonB[2] = forcePtr(6 * i + 2, 0);
            sy.torqueNonB[0] = forcePtr(6 * i + 3, 0);
            sy.torqueNonB[1] = forcePtr(6 * i + 4, 0);
            sy.torqueNonB[2] = forcePtr(6 * i + 5, 0);
        }
    }

    if (!velocityPartNonBrownRcp.is_null()) {
        if (runConfig.monolayer) {
            auto velNBPtr = velocityPartNonBrownRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
            for (int i = 0; i < nLocal; i++) {
                velNBPtr(6 * i + 2, 0) = 0; // vz
                velNBPtr(6 * i + 3, 0) = 0; // omegax
                velNBPtr(6 * i + 4, 0) = 0; // omegay
            }
        }
        velocityNonConRcp->update(1.0, *velocityPartNonBrownRcp, 1.0);
    }

    // write back total non Brownian velocity
    // combine and sync the velNonB set in by setForceNonBrown() and setVelocityNonBrown()
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        // velocity
        sy.velNonB[0] = velNCPtr(6 * i + 0, 0);
        sy.velNonB[1] = velNCPtr(6 * i + 1, 0);
        sy.velNonB[2] = velNCPtr(6 * i + 2, 0);
        sy.omegaNonB[0] = velNCPtr(6 * i + 3, 0);
        sy.omegaNonB[1] = velNCPtr(6 * i + 4, 0);
        sy.omegaNonB[2] = velNCPtr(6 * i + 5, 0);
    }

    // add Brownian motion
    if (!velocityBrownRcp.is_null()) {
        if (runConfig.monolayer) {
            auto velBPtr = velocityBrownRcp->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
            for (int i = 0; i < nLocal; i++) {
                velBPtr(6 * i + 2, 0) = 0; // vz
                velBPtr(6 * i + 3, 0) = 0; // omegax
                velBPtr(6 * i + 4, 0) = 0; // omegay
            }
        }
        velocityNonConRcp->update(1.0, *velocityBrownRcp, 1.0);
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::sumForceVelocity() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        for (int k = 0; k < 3; k++) {
            sy.vel[k] = sy.velNonB[k] + sy.velBrown[k] + sy.velCol[k] + sy.velBi[k];
            sy.omega[k] = sy.omegaNonB[k] + sy.omegaBrown[k] + sy.omegaCol[k] + sy.omegaBi[k];
            sy.force[k] = sy.forceNonB[k] + sy.forceCol[k] + sy.forceBi[k];
            sy.torque[k] = sy.torqueNonB[k] + sy.torqueCol[k] + sy.torqueBi[k];
        }
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::stepEuler() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    const double dt = runConfig.dt;

    if (!runConfig.sylinderFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            sy.stepEuler(dt);
        }
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::resolveConstraints() {

    Teuchos::RCP<Teuchos::Time> collectColTimer =
        Teuchos::TimeMonitor::getNewCounter("ParticleSystem::CollectCollision");
    if (enableTimer) {
        collectColTimer->enable();
    } else {
        collectColTimer->disable();
    }

    printRank0("start collect collisions");
    {
        Teuchos::TimeMonitor mon(*collectColTimer);
        collectPairCollision();
        collectBoundaryCollision();
    }

    // solve collision
    // positive buffer value means collision radius is effectively smaller
    // i.e., less likely to collide
    Teuchos::RCP<Teuchos::Time> solveTimer = Teuchos::TimeMonitor::getNewCounter("ParticleSystem::SolveConstraints");
    if (enableTimer) {
        solveTimer->enable();
    } else {
        solveTimer->disable();
    }
    {
        Teuchos::TimeMonitor mon(*solveTimer);
        const double buffer = 0;
        // printRank0("constraint solver setup");
        conSolverPtr->setup(*conCollectorPtr, mobilityOperatorRcp, velocityNonConRcp, runConfig.dt);
        // printRank0("set control");
        conSolverPtr->setControlParams(runConfig.conResTol, runConfig.conMaxIte, runConfig.conSolverChoice);
        // printRank0("solve");
        conSolverPtr->solveConstraints();
        // printRank0("writeback");
        conSolverPtr->writebackGamma();
    }

    saveForceVelocityConstraints();
}

template <class Ptcl>
void ParticleSystem<Ptcl>::updatePtclMap() {
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

template <class Ptcl>
bool ParticleSystem<Ptcl>::getIfWriteResultCurrentStep() {
    return (stepCount % static_cast<int>(runConfig.timeSnap / runConfig.dt) == 0);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::prepareStep() {
    applyBoxBC();

    if (stepCount % 50 == 0) {
        decomposeDomain();
    }

    exchangeParticle();

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        sy.clear();
        sy.radiusCollision = sylinderContainer[i].radius * runConfig.sylinderDiameterColRatio;
        sy.lengthCollision = sylinderContainer[i].length * runConfig.sylinderLengthColRatio;
        sy.rank = commRcp->getRank();
    }
    if (runConfig.monolayer) {
        const double monoZ = (runConfig.simBoxHigh[2] + runConfig.simBoxLow[2]) / 2;
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            sy.pos[2] = monoZ;
            Evec3 drt = Emapq(sy.orientation) * Evec3(0, 0, 1);
            drt[2] = 0;
            drt.normalize();
            Emapq(sy.orientation).setFromTwoVectors(Evec3(0, 0, 1), drt);
        }
    }

    updateParticleMap();

    calcMobOperator();

    conCollectorPtr->clear();

    forcePartNonBrownRcp.reset();
    velocityPartNonBrownRcp.reset();
    velocityNonBrownRcp.reset();
    velocityBrownRcp.reset();
}

template <class Ptcl>
void ParticleSystem<Ptcl>::setForceNonBrown(const std::vector<double> &forceNonBrown) {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(forceNonBrown.size() == 6 * nLocal);
    TEUCHOS_ASSERT(sylinderMobilityMapRcp->getNodeNumElements() == 6 * nLocal);
    forcePartNonBrownRcp = getTVFromVector(forceNonBrown, commRcp);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::setVelocityNonBrown(const std::vector<double> &velNonBrown) {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velNonBrown.size() == 6 * nLocal);
    TEUCHOS_ASSERT(sylinderMobilityMapRcp->getNodeNumElements() == 6 * nLocal);
    velocityPartNonBrownRcp = getTVFromVector(velNonBrown, commRcp);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::runStep() {

    if (runConfig.KBT > 0) {
        calcVelocityBrown();
    }

    calcVelocityNonCon();

    resolveConstraints();

    sumForceVelocity();

    if (getIfWriteResultCurrentStep()) {
        // write result before moving. guarantee data written is consistent to geometry
        writeResult();
    }

    stepEuler();

    stepCount++;
}

template <class Ptcl>
void ParticleSystem<Ptcl>::saveForceVelocityConstraints() {
    // save results
    forceUniRcp = conSolverPtr->getForceUni();
    velocityUniRcp = conSolverPtr->getVelocityUni();
    forceBiRcp = conSolverPtr->getForceBi();
    velocityBiRcp = conSolverPtr->getVelocityBi();

    auto velUniPtr = velocityUniRcp->getLocalView<Kokkos::HostSpace>();
    auto velBiPtr = velocityBiRcp->getLocalView<Kokkos::HostSpace>();
    auto forceUniPtr = forceUniRcp->getLocalView<Kokkos::HostSpace>();
    auto forceBiPtr = forceBiRcp->getLocalView<Kokkos::HostSpace>();

    const int sylinderLocalNumber = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velUniPtr.dimension_0() == sylinderLocalNumber * 6);
    TEUCHOS_ASSERT(velUniPtr.dimension_1() == 1);
    TEUCHOS_ASSERT(velBiPtr.dimension_0() == sylinderLocalNumber * 6);
    TEUCHOS_ASSERT(velBiPtr.dimension_1() == 1);

#pragma omp parallel for
    for (int i = 0; i < sylinderLocalNumber; i++) {
        auto &sy = sylinderContainer[i];
        sy.velCol[0] = velUniPtr(6 * i + 0, 0);
        sy.velCol[1] = velUniPtr(6 * i + 1, 0);
        sy.velCol[2] = velUniPtr(6 * i + 2, 0);
        sy.omegaCol[0] = velUniPtr(6 * i + 3, 0);
        sy.omegaCol[1] = velUniPtr(6 * i + 4, 0);
        sy.omegaCol[2] = velUniPtr(6 * i + 5, 0);
        sy.velBi[0] = velBiPtr(6 * i + 0, 0);
        sy.velBi[1] = velBiPtr(6 * i + 1, 0);
        sy.velBi[2] = velBiPtr(6 * i + 2, 0);
        sy.omegaBi[0] = velBiPtr(6 * i + 3, 0);
        sy.omegaBi[1] = velBiPtr(6 * i + 4, 0);
        sy.omegaBi[2] = velBiPtr(6 * i + 5, 0);

        sy.forceCol[0] = forceUniPtr(6 * i + 0, 0);
        sy.forceCol[1] = forceUniPtr(6 * i + 1, 0);
        sy.forceCol[2] = forceUniPtr(6 * i + 2, 0);
        sy.torqueCol[0] = forceUniPtr(6 * i + 3, 0);
        sy.torqueCol[1] = forceUniPtr(6 * i + 4, 0);
        sy.torqueCol[2] = forceUniPtr(6 * i + 5, 0);
        sy.forceBi[0] = forceBiPtr(6 * i + 0, 0);
        sy.forceBi[1] = forceBiPtr(6 * i + 1, 0);
        sy.forceBi[2] = forceBiPtr(6 * i + 2, 0);
        sy.torqueBi[0] = forceBiPtr(6 * i + 3, 0);
        sy.torqueBi[1] = forceBiPtr(6 * i + 4, 0);
        sy.torqueBi[2] = forceBiPtr(6 * i + 5, 0);
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::calcVelocityBrown() {
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

template <class Ptcl>
void ParticleSystem<Ptcl>::collectBoundaryCollision() {
    auto collisionPoolPtr = conCollectorPtr->constraintPoolPtr; // shared_ptr
    const int nThreads = collisionPoolPtr->size();
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    // process collisions with all boundaries
    for (const auto &bPtr : runConfig.boundaryPtr) {
#pragma omp parallel num_threads(nThreads)
        {
            const int threadId = omp_get_thread_num();
            auto &que = (*collisionPoolPtr)[threadId];
#pragma omp for
            for (int i = 0; i < nLocal; i++) {
                const auto &sy = sylinderContainer[i];
                const Evec3 center = ECmap3(sy.pos);
                const Equatn orientation = ECmapq(sy.orientation);
                const Evec3 direction = orientation * Evec3(0, 0, 1);
                const double length = sy.lengthCollision;
                const Evec3 Qm = center - direction * (length * 0.5);
                const Evec3 Qp = center + direction * (length * 0.5);

                auto checkEnd = [&](const Evec3 &Query) {
                    // check one end
                    double Proj[3], delta[3];
                    bPtr->project(Query.data(), Proj, delta);
                    // if (!bPtr->check(Query.data(), Proj, delta)) {
                    //     printf("boundary projection error\n");
                    // }
                    // if inside boundary, delta = Q-Proj
                    // if outside boundary, delta = Proj-Q
                    double deltanorm = Emap3(delta).norm();
                    Evec3 norm = Emap3(delta) * (1 / deltanorm);
                    Evec3 posI = Query - center;

                    if ((Query - ECmap3(Proj)).dot(ECmap3(delta)) < 0) { // outside boundary
                        que.emplace_back(-deltanorm - sy.radiusCollision, 0, sy.gid, sy.gid, sy.globalIndex,
                                         sy.globalIndex, norm.data(), norm.data(), posI.data(), posI.data(),
                                         Query.data(), Proj, true, false, 0.0, 0.0);
                    } else if (deltanorm <
                               (1 + runConfig.sylinderColBuf * 2) * sy.radiusCollision) { // inside boundary but close
                        que.emplace_back(deltanorm - sy.radiusCollision, 0, sy.gid, sy.gid, sy.globalIndex,
                                         sy.globalIndex, norm.data(), norm.data(), posI.data(), posI.data(),
                                         Query.data(), Proj, true, false, 0.0, 0.0);
                    }
                };
                checkEnd(Qm);
                checkEnd(Qp);
            }
        }
    }
    return;
}

template <class Ptcl>
void ParticleSystem<Ptcl>::collectPairCollision() {

    CalcParticleNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.sylinderColBuf);

    TEUCHOS_ASSERT(treeParticleNearPtr);
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    setTreeParticle();
    treeParticleNearPtr->calcForceAll(calcColFtr, sylinderContainer, dinfo);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderContainer[i].sepmin = (treeParticleNearPtr->getForce(i)).sepmin;
    }

    const int nQue = conCollectorPtr->constraintPoolPtr->size();
#pragma omp parallel for
    for (int q = 0; q < nQue; q++) {
        auto &queue = conCollectorPtr->constraintPoolPtr->at(q);
        for (auto &block : queue) {
            if (block.bilateral && block.kappa < 0) {
                block.kappa = runConfig.linkKappa;
                block.gamma = block.kappa * block.delta0;
            }
        }
    }
}

template <class Ptcl>
std::pair<int, int> ParticleSystem<Ptcl>::getMaxGid() {
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

template <class Ptcl>
void ParticleSystem<Ptcl>::calcBoundingBox(double localLow[3], double localHigh[3], double globalLow[3],
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

template <class Ptcl>
void ParticleSystem<Ptcl>::updatePtclRank() {
    const int nLocal = ptclContainer.getNumberOfParticleLocal();
    const int rank = commRcp->getRank();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        ptclContainer[i].rank = rank;
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::applyBoxBC() {
    sylinderContainer.adjustPositionIntoRootDomain(dinfo);
}

template <class Ptcl>
void ParticleSystem<Ptcl>::calcConStress() {
    Emat3 sumBiStress = Emat3::Zero();
    Emat3 sumUniStress = Emat3::Zero();
    conCollectorPtr->sumLocalConstraintStress(sumUniStress, sumBiStress, false);

    // scale to nkBT
    const double scaleFactor = 1 / (sylinderMapRcp->getGlobalNumElements() * runConfig.KBT);
    sumBiStress *= scaleFactor;
    sumUniStress *= scaleFactor;
    // mpi reduction
    double uniStressLocal[9];
    double biStressLocal[9];
    double uniStressGlobal[9];
    double biStressGlobal[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            uniStressLocal[i * 3 + j] = sumUniStress(i, j);
            uniStressGlobal[i * 3 + j] = 0;
            biStressLocal[i * 3 + j] = sumBiStress(i, j);
            biStressGlobal[i * 3 + j] = 0;
        }
    }

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 9, uniStressLocal, uniStressGlobal);
    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 9, biStressLocal, biStressGlobal);

    if (commRcp->getRank() == 0) {
        printf("RECORD: ColXF,%7g,%7g,%7g,%7g,%7g,%7g,%7g,%7g,%7g\n",      //
               uniStressGlobal[0], uniStressGlobal[1], uniStressGlobal[2], //
               uniStressGlobal[3], uniStressGlobal[4], uniStressGlobal[5], //
               uniStressGlobal[6], uniStressGlobal[7], uniStressGlobal[8]);
        printf("RECORD: BiXF,%7g,%7g,%7g,%7g,%7g,%7g,%7g,%7g,%7g\n",    //
               biStressGlobal[0], biStressGlobal[1], biStressGlobal[2], //
               biStressGlobal[3], biStressGlobal[4], biStressGlobal[5], //
               biStressGlobal[6], biStressGlobal[7], biStressGlobal[8]);
    }
}

template <class Ptcl>
void ParticleSystem<Ptcl>::calcOrderParameter() {
    double px = 0, py = 0, pz = 0;    // pvec
    double Qxx = 0, Qxy = 0, Qxz = 0; // Qtensor
    double Qyx = 0, Qyy = 0, Qyz = 0; // Qtensor
    double Qzx = 0, Qzy = 0, Qzz = 0; // Qtensor

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

#pragma omp parallel for reduction(+ : px, py, pz, Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz)
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = sylinderContainer[i];
        const Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        px += direction.x();
        py += direction.y();
        pz += direction.z();
        const Emat3 Q = direction * direction.transpose() - Emat3::Identity() * (1 / 3.0);
        Qxx += Q(0, 0);
        Qxy += Q(0, 1);
        Qxz += Q(0, 2);
        Qyx += Q(1, 0);
        Qyy += Q(1, 1);
        Qyz += Q(1, 2);
        Qzx += Q(2, 0);
        Qzy += Q(2, 1);
        Qzz += Q(2, 2);
    }

    // global average
    const int nGlobal = sylinderContainer.getNumberOfParticleGlobal();
    double pQ[12] = {px, py, pz, Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz};
    MPI_Allreduce(MPI_IN_PLACE, pQ, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < 12; i++) {
        pQ[i] *= (1.0 / nGlobal);
    }

    if (commRcp()->getRank() == 0) {
        printf("RECORD: Order P,%6g,%6g,%6g,Q,%6g,%6g,%6g,%6g,%6g,%6g,%6g,%6g,%6g\n", //
               pQ[0], pQ[1], pQ[2],                                                   // pvec
               pQ[3], pQ[4], pQ[5],                                                   // Qtensor
               pQ[6], pQ[7], pQ[8],                                                   // Qtensor
               pQ[9], pQ[10], pQ[11]                                                  // Qtensor
        );
    }
}

// void template <class Ptcl>
// ParticleSystem<Ptcl>::setPosWithBoundary() {
//     //     const int nLocal = sylinderContainer.getNumberOfParticleLocal();
//     //     const double buffer = 1e-4;
//     // }

//     void template <class Ptcl>
//     ParticleSystem<Ptcl>::addNewParticle(std::vector<Particle> & newParticle, std::vector<Link> & linkage) {
//         // assign unique new gid for sylinders on all ranks
//         std::pair<int, int> maxGid = getMaxGid();
//         const int maxGidLocal = maxGid.first;
//         const int maxGidGlobal = maxGid.second;
//         const int newNumberOnLocal = newParticle.size();
//         const auto &newMapRcp = getTMAPFromLocalSize(newNumberOnLocal, commRcp);

//         // a large enough buffer on every rank
//         std::vector<int> newID(newMapRcp->getGlobalNumElements(), 0);
//         std::vector<int> newNumber(commRcp->getSize(), 0);
//         std::vector<int> displ(commRcp->getSize(), 0);

//         // assign random id on rank 0
//         if (commRcp->getRank() == 0) {
//             std::iota(newID.begin(), newID.end(), 0);
//             std::random_shuffle(newID.begin(), newID.end());
//         }
//         // collect number of ids from all ranks to rank0
//         MPI_Gather(&newNumberOnLocal, 1, MPI_INT, newNumber.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

//         if (commRcp->getRank() == 0) {
//             std::partial_sum(newNumber.cbegin(), newNumber.cend() - 1, displ.begin() + 1);
//         }

//         std::vector<int> newIDRecv(newNumberOnLocal, 0);
//         // scatter from rank 0 to every rank
//         MPI_Scatterv(newID.data(), newNumber.data(), displ.data(), MPI_INT, //
//                      newIDRecv.data(), newNumberOnLocal, MPI_INT, 0, MPI_COMM_WORLD);

//         // set new gid
//         for (int i = 0; i < newNumberOnLocal; i++) {
//             newParticle[i].gid = newIDRecv[i] + 1 + maxGidGlobal;
//         }

//         // set link connection
//         if (linkage.size() == newNumberOnLocal) {
//             for (int i = 0; i < newNumberOnLocal; i++) {
//                 newParticle[i].link.group = linkage[i].group;
//                 newParticle[i].link.prev =
//                     (linkage[i].prev == GEO_INVALID_INDEX) ? GEO_INVALID_INDEX : newParticle[linkage[i].prev].gid;
//                 newParticle[i].link.next =
//                     (linkage[i].next == GEO_INVALID_INDEX) ? GEO_INVALID_INDEX : newParticle[linkage[i].next].gid;
//             }
//         } else if (linkage.size() == 0) {
//             // no linkage, do nothing
//         } else {
//             printf("wrong linkage on rank: %d\n", commRcp->getRank());
//             std::exit(1);
//         }

//         // add new cell to old cell
//         for (int i = 0; i < newNumberOnLocal; i++) {
//             sylinderContainer.addOneParticle(newParticle[i]);
//         }
//     }

//     void template <class Ptcl>
//     ParticleSystem<Ptcl>::printRank0(const std::string &message) {
//         if (commRcp->getRank() == 0) {
//             std::cout << message << std::endl;
//         }
//     }

//     void template <class Ptcl>
//     ParticleSystem<Ptcl>::buildParticleNearDataDirectory() {
//         const size_t nLocal = sylinderContainer.getNumberOfParticleLocal();
//         auto &sylinderNearDataDirectory = *sylinderNearDataDirectoryPtr;
//         sylinderNearDataDirectory.gidOnLocal.resize(nLocal);
//         sylinderNearDataDirectory.dataOnLocal.resize(nLocal);
// #pragma omp parallel for
//         for (int i = 0; i < nLocal; i++) {
//             sylinderNearDataDirectory.gidOnLocal[i] = sylinderContainer[i].gid;
//             sylinderNearDataDirectory.dataOnLocal[i].copyFromFP(sylinderContainer[i]);
//         }

//         // build index
//         sylinderNearDataDirectory.buildIndex();
//     }

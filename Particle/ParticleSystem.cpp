#include "ParticleSystem.hpp"

#include "MPI/CommMPI.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/GeoUtil.hpp"
#include "Util/IOHelper.hpp"
#include "Util/Logger.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTypeInt32Array.h>
#include <vtkTypeUInt8Array.h>
#include <vtkXMLPPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>

#include <mpi.h>
#include <omp.h>

ParticleSystem::ParticleSystem(const ParticleConfig &runConfig_, const Teuchos::RCP<const TCOMM> &commRcp_,
                               std::shared_ptr<TRngPool> rngPoolPtr_,
                               std::shared_ptr<ConstraintCollector> conCollectorPtr_)
    : runConfig(runConfig_), rngPoolPtr(std::move(rngPoolPtr_)), conCollectorPtr(std::move(conCollectorPtr_)),
      commRcp(commRcp_) {}

void ParticleSystem::initialize(const std::string &posFile) {
    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    particleContainer.initialize();
    particleContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more sample for better balance

    if (IOHelper::fileExist(posFile)) {
        setInitialFromFile(posFile);
    } else {
        setInitialFromConfig();
    }
    setLinkMapFromFile(posFile);

    // at this point all particles located on rank 0
    commRcp->barrier();
    decomposeDomain();
    exchangeParticle(); // distribute to ranks, initial domain decomposition

    // initialize particle growth
    initParticleGrowth();

    // initialize the GID search tree
    particleNearDataDirectoryPtr = std::make_shared<ZDD<ParticleNearEP>>(particleContainer.getNumberOfParticleLocal());

    treeParticleNumber = 0;
    setTreeParticle();

    calcVolFrac();

    if (commRcp->getRank() == 0) {
        writeBox();
    }

    // make sure the current position and orientation are updated
    advanceParticles();

    spdlog::warn("ParticleSystem Initialized. {} local particles", particleContainer.getNumberOfParticleLocal());
}

void ParticleSystem::reinitialize(const std::string &pvtpFileName_) {
    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    particleContainer.initialize();
    particleContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more samples for better balance

    // reinitialize from pctp file
    std::string pvtpFileName = pvtpFileName_;
    std::string asciiFileName = pvtpFileName;
    auto pos = asciiFileName.find_last_of('.');
    asciiFileName.replace(pos, 5, std::string(".dat")); // replace '.pvtp' with '.dat'
    pos = asciiFileName.find_last_of('_');
    asciiFileName.replace(pos, 1, std::string("Ascii_")); // replace '_' with 'Ascii_'

    setInitialFromVTKFile(pvtpFileName);
    setLinkMapFromFile(asciiFileName);

    // at this point all particles located on rank 0
    commRcp->barrier();
    applyBoxBC();
    decomposeDomain();
    exchangeParticle(); // distribute to ranks, initial domain decomposition
    updateParticleMap();

    // initialize the GID search tree
    particleNearDataDirectoryPtr = std::make_shared<ZDD<ParticleNearEP>>(particleContainer.getNumberOfParticleLocal());

    treeParticleNumber = 0;
    setTreeParticle();
    calcVolFrac();

    // make sure the current position and orientation are updated
    advanceParticles();

    spdlog::warn("ParticleSystem Initialized. {} local particles", particleContainer.getNumberOfParticleLocal());
}

void ParticleSystem::initParticleGrowth() {
    // TODO: name these variables so they are directly interprettabole by their names
    if (runConfig.ptcGrowth.size() == 0) {
        return;
    }

    const int nLocal = particleContainer.getNumberOfParticleLocal();

    Growth *pgrowth;
    if (runConfig.ptcGrowth.size() == 1)
        pgrowth = &(runConfig.ptcGrowth.begin()->second);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &ptc = particleContainer[i];
        ptc.t = 0;
        if (pgrowth) {
            ptc.deltaL = pgrowth->Delta;
            ptc.sigma = pgrowth->sigma;
            ptc.tauD = pgrowth->tauD;
            ptc.tg = pgrowth->tauD * log2(1 + pgrowth->Delta / ptc.length) + pgrowth->sigma * rngPoolPtr->getN01();
        } else {
            // TODO, multiple species
        }
    }
}

void ParticleSystem::calcParticleDivision() {
    const double eps = std::numeric_limits<double>::epsilon() * 100;

    if (runConfig.ptcGrowth.size() == 0) {
        return;
    }

    const int nLocal = particleContainer.getNumberOfParticleLocal();

    const double facMut[3] = {1., 1., 1.};
    Growth *pgrowth;
    if (runConfig.ptcGrowth.size() == 1)
        pgrowth = &(runConfig.ptcGrowth.begin()->second);

    std::vector<Particle> newPtc;

    for (int i = 0; i < nLocal; i++) {
        auto &ptc = particleContainer[i];
        if (ptc.t < ptc.tg) {
            continue;
        }
        const Evec3 center = Evec3(ptc.pos[0], ptc.pos[1], ptc.pos[2]);

        Evec3 direction = ECmapq(ptc.orientation) * Evec3(0, 0, 1);
        const double currentLength = ptc.length;
        const double newLength = currentLength * 0.5 - ptc.radius;
        if (newLength <= 0) {
            continue;
        }

        // old particle, shrink and reset center, no rotation
        ptc.length = newLength;
        ptc.lengthCollision = newLength * runConfig.particleLengthColRatio;
        Emap3(ptc.pos) = center - direction * (ptc.radius + 0.5 * newLength + eps);

        // new particle
        Evec3 ncPos = center + direction * (ptc.radius + 0.5 * newLength + eps);
        newPtc.emplace_back(-1, ptc.radius, ptc.radius * runConfig.particleDiameterColRatio, newLength,
                            newLength * runConfig.particleLengthColRatio, ncPos.data(), ptc.orientation);

        newPtc.back().tauD = ptc.tauD;
        newPtc.back().sigma = ptc.sigma;
        newPtc.back().deltaL = ptc.deltaL;
        if (ptc.tauD == pgrowth->tauD && ptc.sigma == pgrowth->sigma && ptc.deltaL == pgrowth->Delta) {
            if (rngPoolPtr->getU01() < 0.01) {
                newPtc.back().tauD *= facMut[0];
                newPtc.back().sigma *= facMut[1];
                newPtc.back().deltaL *= facMut[2];
            }
        }

        // set new tg
        const double tg = ptc.tauD * log2(1 + ptc.deltaL / newLength) + ptc.sigma * rngPoolPtr->getN01();
        ptc.t = 0;
        ptc.tg = tg;
        newPtc.back().t = 0;
        newPtc.back().tg = tg;

        if (pgrowth->sftAng != 0.) {
            double alpha = pgrowth->sftAng * 3.141592653589793238 / 180;
            // rotation of second cell
            double fac = (rngPoolPtr->getU01() >= 0.5) ? 1. : -1.;
            double angle1 = atan2(direction[1], direction[0]) + alpha * fac;
            direction[0] = cos(angle1);
            direction[1] = sin(angle1);
            Equatn orientq = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);
            Emapq(ptc.orientation).coeffs() = orientq.coeffs();
        }
    }
    addNewParticle(newPtc);
}

void ParticleSystem::calcParticleGrowth(const Teuchos::RCP<const TV> &ptcStressRcp) {
    TEUCHOS_ASSERT(nonnull(ptcStressRcp));

    if (runConfig.ptcGrowth.size() == 0) {
        return;
    }
    Growth *pgrowth;
    if (runConfig.ptcGrowth.size() == 1)
        pgrowth = &(runConfig.ptcGrowth.begin()->second);

    const double dt = runConfig.dt;
    const int particleLocalNumber = particleContainer.getNumberOfParticleLocal();
    
    // get the local view of the stress vector
    auto ptcStressPtr = ptcStressRcp->getLocalView<Kokkos::HostSpace>();
    TEUCHOS_ASSERT(ptcStressPtr.extent(0) == particleLocalNumber * 9);
    TEUCHOS_ASSERT(ptcStressPtr.extent(1) == 1);

#pragma omp parallel for
    for (int i = 0; i < particleLocalNumber; i++) {
        auto &ptc = particleContainer[i];

        // get the particle virial stress
        Emat3 stress;
        for (int idx1 = 0; idx1 < 3; idx1++) {
            for (int idx2 = 0; idx2 < 3; idx2++) {
                stress(idx1, idx2) = ptcStressPtr(i * 9 + idx1 * 3 + idx2, 0);
            }
        }

        // transform the stress into the particle frame
        const Emat3 rotMat = Emapq(ptc.orientation).toRotationMatrix();
        const Emat3 rotMatInv = Emapq(ptc.orientation).inverse().toRotationMatrix();
        stress = rotMat * stress * rotMatInv;

        // output the normal stress to the terminal // TODO: couple it to growth rate
        // spdlog::warn("Normal Stress: {:g}", stress(0, 0));

        // store the normal stress
        ptc.normalStress = stress(2, 2);

        // compute the amount of growth 
        double dLdt = log(2.0) * ptc.length / ptc.tauD;

        // apply the growth to the particle
        ptc.t += dt;
        ptc.length += fabs(dLdt * dt); // never shrink
        ptc.lengthCollision = ptc.length * runConfig.particleLengthColRatio;
    }
}

void ParticleSystem::setTreeParticle() {
    // initialize tree
    // always keep tree max_glb_num_ptcl to be twice the global actual particle number.
    const int nGlobal = particleContainer.getNumberOfParticleGlobal();
    if (nGlobal > 1.5 * treeParticleNumber || !treeParticleNearPtr) {
        // a new larger tree
        treeParticleNearPtr.reset();
        treeParticleNearPtr = std::make_unique<TreeParticleNear>();
        treeParticleNearPtr->initialize(2 * nGlobal);
        treeParticleNumber = nGlobal;
    }
}

void ParticleSystem::getOrient(Equatn &orient, const double px, const double py, const double pz, const int threadId) {
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

void ParticleSystem::setInitialFromConfig() {
    // this function init all particles on rank 0
    if (runConfig.particleLengthSigma > 0) {
        rngPoolPtr->setLogNormalParameters(runConfig.particleLength, runConfig.particleLengthSigma);
    }

    if (commRcp->getRank() != 0) {
        particleContainer.setNumberOfParticleLocal(0);
    } else {
        const double boxEdge[3] = {runConfig.initBoxHigh[0] - runConfig.initBoxLow[0],
                                   runConfig.initBoxHigh[1] - runConfig.initBoxLow[1],
                                   runConfig.initBoxHigh[2] - runConfig.initBoxLow[2]};
        const double minBoxEdge = std::min(std::min(boxEdge[0], boxEdge[1]), boxEdge[2]);
        const double maxLength = minBoxEdge * 0.5;
        const double radius = runConfig.particleDiameter / 2;
        const int nParticleLocal = runConfig.particleNumber;
        particleContainer.setNumberOfParticleLocal(nParticleLocal);

#pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < nParticleLocal; i++) {
                double length;
                if (runConfig.particleLengthSigma > 0) {
                    do { // generate random length
                        length = rngPoolPtr->getLN(threadId);
                    } while (length >= maxLength);
                } else {
                    length = runConfig.particleLength;
                }
                double pos[3];
                for (int k = 0; k < 3; k++) {
                    pos[k] = rngPoolPtr->getU01(threadId) * boxEdge[k] + runConfig.initBoxLow[k];
                }
                Equatn orientq;
                getOrient(orientq, runConfig.initOrient[0], runConfig.initOrient[1], runConfig.initOrient[2], threadId);
                double orientation[4];
                Emapq(orientation).coeffs() = orientq.coeffs();
                particleContainer[i] = Particle(i, radius, radius, length, length, pos, orientation);
                particleContainer[i].clear();
            }
        }
    }

    if (runConfig.initCircularX) {
        setInitialCircularCrossSection();
    }
}

void ParticleSystem::setInitialCircularCrossSection() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
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
            double y = particleContainer[i].pos[1];
            double z = particleContainer[i].pos[2];
            // replace y,z with position in the circle
            getRandPointInCircle(radiusCrossSec, rngPoolPtr->getU01(threadId), rngPoolPtr->getU01(threadId), y, z);
            particleContainer[i].pos[1] = y + centerCrossSec[1];
            particleContainer[i].pos[2] = z + centerCrossSec[2];
        }
    }
}

void ParticleSystem::calcVolFrac() {
    // calc volume fraction of sphero cylinders
    // step 1, calc local total volume
    double volLocal = 0;
    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for reduction(+ : volLocal)
    for (int i = 0; i < nLocal; i++) {
        auto &ptc = particleContainer[i];
        volLocal += 3.1415926535 * (0.25 * ptc.length * pow(ptc.radius * 2, 2) + pow(ptc.radius * 2, 3) / 6);
    }
    double volGlobal = 0;

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 1, &volLocal, &volGlobal);

    // step 2, reduce to root and compute total volume
    double boxVolume = (runConfig.simBoxHigh[0] - runConfig.simBoxLow[0]) *
                       (runConfig.simBoxHigh[1] - runConfig.simBoxLow[1]) *
                       (runConfig.simBoxHigh[2] - runConfig.simBoxLow[2]);
    spdlog::warn("Volume Particle = {:g}", volGlobal);
    spdlog::warn("Volume fraction = {:g}", volGlobal / boxVolume);
}

void ParticleSystem::setInitialFromFile(const std::string &filename) {
    spdlog::warn("Reading file " + filename);

    auto parseParticle = [&](Particle &ptc, const std::string &line) {
        std::stringstream liness(line);
        // required data
        int gid;
        char type;
        double radius;
        double mx, my, mz;
        double px, py, pz;
        liness >> type >> gid >> radius >> mx >> my >> mz >> px >> py >> pz;
        // optional data
        int group = -1;
        liness >> group;

        Emap3(ptc.pos) = Evec3((mx + px), (my + py), (mz + pz)) * 0.5;
        ptc.gid = gid;
        ptc.group = group;
        ptc.isImmovable = (type == 'S') ? true : false;
        ptc.radius = radius;
        ptc.radiusCollision = radius;
        ptc.length = sqrt(pow(px - mx, 2) + pow(py - my, 2) + pow(pz - mz, 2));
        ptc.lengthCollision = ptc.length;
        if (ptc.length > 1e-7) {
            Evec3 direction(px - mx, py - my, pz - mz);
            Emapq(ptc.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);
        } else {
            Emapq(ptc.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), Evec3(0, 0, 1));
        }
    };

    if (commRcp->getRank() != 0) {
        particleContainer.setNumberOfParticleLocal(0);
    } else {
        std::ifstream myfile(filename);
        std::string line;
        std::getline(myfile, line); // read two header lines
        std::getline(myfile, line);

        std::deque<Particle> particleReadFromFile;
        while (std::getline(myfile, line)) {
            if (line[0] == 'C' || line[0] == 'S') {
                Particle ptc;
                parseParticle(ptc, line);
                particleReadFromFile.push_back(ptc);
            }
        }
        myfile.close();

        spdlog::debug("Particle number in file {} ", particleReadFromFile.size());

        // set on rank 0
        const int nRead = particleReadFromFile.size();
        particleContainer.setNumberOfParticleLocal(nRead);
#pragma omp parallel for
        for (int i = 0; i < nRead; i++) {
            particleContainer[i] = particleReadFromFile[i];
            particleContainer[i].clear();
        }
    }
}

// TODO: this should be part of the constraint collector (init from file functionality)
// Set constraints from file, allowing any type of constraint to be initialized
void ParticleSystem::setLinkMapFromFile(const std::string &filename) {
    spdlog::warn("Reading file " + filename);

    auto parseLink = [&](Link &link, const std::string &line) {
        std::stringstream liness(line);
        char header;
        liness >> header >> link.prev >> link.next;
        assert(header == 'L');
    };

    std::ifstream myfile(filename);
    std::string line;
    std::getline(myfile, line); // read two header lines
    std::getline(myfile, line);

    linkMap.clear();
    while (std::getline(myfile, line)) {
        if (line[0] == 'L') {
            Link link;
            parseLink(link, line);
            linkMap.emplace(link.prev, link.next);
            linkReverseMap.emplace(link.next, link.prev);
        }
    }
    myfile.close();

    spdlog::debug("Link number in file {} ", linkMap.size());
}

void ParticleSystem::setInitialFromVTKFile(const std::string &pvtpFileName) {
    spdlog::warn("Reading file " + pvtpFileName);

    if (commRcp->getRank() != 0) {
        particleContainer.setNumberOfParticleLocal(0);
    } else {

        // Read the pvtp file and automatically merge the vtks files into a single polydata
        vtkSmartPointer<vtkXMLPPolyDataReader> reader = vtkSmartPointer<vtkXMLPPolyDataReader>::New();
        reader->SetFileName(pvtpFileName.c_str());
        reader->Update();

        // Extract the polydata (At this point, the polydata is unsorted)
        vtkSmartPointer<vtkPolyData> polydata = reader->GetOutput();
        // geometry data
        vtkSmartPointer<vtkPoints> posData = polydata->GetPoints();
        // Extract the point/cell data
        // int32 types
        vtkSmartPointer<vtkTypeInt32Array> gidData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetCellData()->GetAbstractArray("gid"));
        vtkSmartPointer<vtkTypeInt32Array> groupData =
            vtkArrayDownCast<vtkTypeInt32Array>(polydata->GetCellData()->GetAbstractArray("group"));
        // unsigned char type
        vtkSmartPointer<vtkTypeUInt8Array> isImmovableData =
            vtkArrayDownCast<vtkTypeUInt8Array>(polydata->GetCellData()->GetAbstractArray("isImmovable"));
        // float/double types
        vtkSmartPointer<vtkDataArray> lengthData = polydata->GetCellData()->GetArray("length");
        vtkSmartPointer<vtkDataArray> lengthCollisionData = polydata->GetCellData()->GetArray("lengthCollision");
        vtkSmartPointer<vtkDataArray> radiusData = polydata->GetCellData()->GetArray("radius");
        vtkSmartPointer<vtkDataArray> radiusCollisionData = polydata->GetCellData()->GetArray("radiusCollision");
        vtkSmartPointer<vtkDataArray> znormData = polydata->GetCellData()->GetArray("znorm");
        vtkSmartPointer<vtkDataArray> velData = polydata->GetCellData()->GetArray("vel");
        vtkSmartPointer<vtkDataArray> omegaData = polydata->GetCellData()->GetArray("omega");
        vtkSmartPointer<vtkDataArray> tData = polydata->GetCellData()->GetArray("t");
        vtkSmartPointer<vtkDataArray> tgData = polydata->GetCellData()->GetArray("tg");
        vtkSmartPointer<vtkDataArray> tauDData = polydata->GetCellData()->GetArray("tauD");
        vtkSmartPointer<vtkDataArray> sigmaData = polydata->GetCellData()->GetArray("sigma");
        vtkSmartPointer<vtkDataArray> deltaLData = polydata->GetCellData()->GetArray("deltaL");

        const int particleNumberInFile = posData->GetNumberOfPoints() / 2; // two points per particle
        particleContainer.setNumberOfParticleLocal(particleNumberInFile);
        spdlog::debug("Particle number in file {} ", particleNumberInFile);

#pragma omp parallel for
        for (int i = 0; i < particleNumberInFile; i++) {
            auto &ptc = particleContainer[i];
            double leftEndpointPos[3] = {0, 0, 0};
            double rightEndpointPos[3] = {0, 0, 0};
            posData->GetPoint(i * 2, leftEndpointPos);
            posData->GetPoint(i * 2 + 1, rightEndpointPos);

            Emap3(ptc.pos) = (Emap3(leftEndpointPos) + Emap3(rightEndpointPos)) * 0.5;
            ptc.gid = gidData->GetComponent(i, 0);
            ptc.group = groupData->GetComponent(i, 0);
            ptc.isImmovable = isImmovableData->GetTypedComponent(i, 0) > 0 ? true : false;
            ptc.length = lengthData->GetComponent(i, 0);
            ptc.lengthCollision = lengthCollisionData->GetComponent(i, 0);
            ptc.radius = radiusData->GetComponent(i, 0);
            ptc.radiusCollision = radiusCollisionData->GetComponent(i, 0);
            const Evec3 direction(znormData->GetComponent(i, 0), znormData->GetComponent(i, 1),
                                  znormData->GetComponent(i, 2));
            Emapq(ptc.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);

            // read in growth information (if necessary) // TODO: use a growth flag to turn off and on this kind of
            // functionality
            ptc.t = tData->GetComponent(i, 0);
            ptc.tg = tgData->GetComponent(i, 0);
            ptc.tauD = tauDData->GetComponent(i, 0);
            ptc.sigma = sigmaData->GetComponent(i, 0);
            ptc.deltaL = deltaLData->GetComponent(i, 0);

            // read in velocity for correct stepping
            ptc.vel[0] = velData->GetComponent(i, 0);
            ptc.vel[1] = velData->GetComponent(i, 1);
            ptc.vel[2] = velData->GetComponent(i, 2);
            ptc.omega[0] = omegaData->GetComponent(i, 0);
            ptc.omega[1] = omegaData->GetComponent(i, 1);
            ptc.omega[2] = omegaData->GetComponent(i, 2);
        }

        // sort the vector of Particles by gid ascending;
        // std::sort(particleReadFromFile.begin(), particleReadFromFile.end(),
        //           [](const Particle &t1, const Particle &t2) { return t1.gid < t2.gid; });
    }
    commRcp->barrier();
}

void ParticleSystem::writeAscii(const int stepCount, const std::string &baseFolder, const std::string &postfix) {
    // write a single ascii .dat file
    const int nGlobal = particleContainer.getNumberOfParticleGlobal();

    std::string name = baseFolder + std::string("ParticleAscii_") + postfix + ".dat";
    ParticleAsciiHeader header;
    header.nparticle = nGlobal;
    header.time = stepCount * runConfig.dt;
    particleContainer.writeParticleAscii(name.c_str(), header);
    if (commRcp->getRank() == 0) {
        FILE *fptr = fopen(name.c_str(), "a");
        for (const auto &key_value : linkMap) {
            fprintf(fptr, "L %d %d\n", key_value.first, key_value.second);
        }
        fclose(fptr);
    }
    commRcp->barrier();
}

void ParticleSystem::writeVTK(const std::string &baseFolder, const std::string &postfix) {
    const int rank = commRcp->getRank();
    const int size = commRcp->getSize();
    Particle::writeVTP<PS::ParticleSystem<Particle>>(particleContainer, particleContainer.getNumberOfParticleLocal(),
                                                     baseFolder, postfix, rank);
    conCollectorPtr->writeVTP(baseFolder, "", postfix, rank);
    if (rank == 0) {
        Particle::writePVTP(baseFolder, postfix, size); // write parallel head
        conCollectorPtr->writePVTP(baseFolder, "", postfix, size);
    }
}

void ParticleSystem::writeBox() {
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

void ParticleSystem::writeResult(const int stepCount, const std::string &baseFolder, const std::string &postfix) {
    writeAscii(stepCount, baseFolder, postfix);
    writeVTK(baseFolder, postfix);
}

void ParticleSystem::setDomainInfo() {
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

void ParticleSystem::decomposeDomain() {
    applyBoxBC();
    dinfo.decomposeDomainAll(particleContainer);
}

void ParticleSystem::exchangeParticle() {
    particleContainer.exchangeParticle(dinfo);
    updateParticleRank();
}

// TODO: mobility class should allow for flags to decide what type of mobility matrix to use: diagonal or dense
void ParticleSystem::calcMobMatrix() {
    // diagonal hydro mobility operator
    // 3*3 block for translational + 3*3 block for rotational.
    // 3 nnz per row, 18 nnz per tubule

    const double Pi = 3.14159265358979323846;
    const double mu = runConfig.viscosity;

    const int nLocal = particleMapRcp->getNodeNumElements();
    TEUCHOS_ASSERT(nLocal == particleContainer.getNumberOfParticleLocal());
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
        const auto &ptc = particleContainer[i];

        // calculate the Mob Trans and MobRot
        Emat3 MobTrans; //            double MobTrans[3][3];
        Emat3 MobRot;   //            double MobRot[3][3];
        Emat3 qq;
        Emat3 Imqq;
        Evec3 q = ECmapq(ptc.orientation) * Evec3(0, 0, 1);
        qq = q * q.transpose();
        Imqq = Emat3::Identity() - qq;

        double dragPara = 0;
        double dragPerp = 0;
        double dragRot = 0;
        ptc.calcDragCoeff(mu, dragPara, dragPerp, dragRot);
        const double dragParaInv = ptc.isImmovable ? 0.0 : 1 / dragPara;
        const double dragPerpInv = ptc.isImmovable ? 0.0 : 1 / dragPerp;
        const double dragRotInv = ptc.isImmovable ? 0.0 : 1 / dragRot;

        MobTrans = dragParaInv * qq + dragPerpInv * Imqq;
        MobRot = dragRotInv * qq + dragRotInv * Imqq; // = dragRotInv * Identity
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
        Teuchos::rcp(new TCMAT(particleMobilityMapRcp, particleMobilityMapRcp, rowPointers, columnIndices, values));
    mobilityMatrixRcp->fillComplete(particleMobilityMapRcp, particleMobilityMapRcp); // domainMap, rangeMap

    spdlog::debug("MobMat Constructed " + mobilityMatrixRcp->description());
}

void ParticleSystem::calcMobOperator() {
    calcMobMatrix();
    mobilityOperatorRcp = mobilityMatrixRcp;
}

void ParticleSystem::sumForceVelocity() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &ptc = particleContainer[i];
        for (int k = 0; k < 3; k++) {
            ptc.vel[k] = ptc.velNonB[k] + ptc.velBrown[k] + ptc.velCon[k];
            ptc.omega[k] = ptc.omegaNonB[k] + ptc.omegaBrown[k] + ptc.omegaCon[k];
            ptc.force[k] = ptc.forceNonB[k] + ptc.forceCon[k];
            ptc.torque[k] = ptc.torqueNonB[k] + ptc.torqueCon[k];
        }
    }
}

void ParticleSystem::stepEuler(const int stepType) {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    const double dt = runConfig.dt;

    if (!runConfig.particleFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &ptc = particleContainer[i];
            ptc.stepEuler(dt, stepType);
        }
    }
}

void ParticleSystem::resetConfiguration() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();

    if (!runConfig.particleFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &ptc = particleContainer[i];
            ptc.resetConfiguration();
        }
    }
}

void ParticleSystem::advanceParticles() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();

    if (!runConfig.particleFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &ptc = particleContainer[i];
            ptc.advance();
        }
    }
}

void ParticleSystem::collectConstraints() {

    Teuchos::RCP<Teuchos::Time> collectColTimer =
        Teuchos::TimeMonitor::getNewCounter("ParticleSystem::CollectCollision");
    Teuchos::RCP<Teuchos::Time> collectLinkTimer = Teuchos::TimeMonitor::getNewCounter("ParticleSystem::CollectLink");

    spdlog::debug("start collect collisions");
    {
        Teuchos::TimeMonitor mon(*collectColTimer);
        collectPairCollision();
        collectBoundaryCollision();
    }

    spdlog::debug("start collect links");
    {
        Teuchos::TimeMonitor mon(*collectLinkTimer);
        collectLinkBilateral();
    }
}

void ParticleSystem::updateParticleMap() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    // setup the new particleMap
    particleMapRcp = getTMAPFromLocalSize(nLocal, commRcp);
    particleMobilityMapRcp = getTMAPFromLocalSize(nLocal * 6, commRcp);
    particleStressMapRcp = getTMAPFromLocalSize(nLocal * 9, commRcp);

    // setup the globalIndex
    int globalIndexBase = particleMapRcp->getMinGlobalIndex(); // this is a contiguous map
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        particleContainer[i].globalIndex = i + globalIndexBase;
    }
}

void ParticleSystem::prepareStep(const int stepCount) {
    applyBoxBC();

    if (stepCount % 50 == 0) {
        decomposeDomain();
    }

    exchangeParticle();

    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &ptc = particleContainer[i];
        ptc.clear();
        ptc.radiusCollision = particleContainer[i].radius * runConfig.particleDiameterColRatio;
        ptc.lengthCollision = particleContainer[i].length * runConfig.particleLengthColRatio;
        ptc.rank = commRcp->getRank();
        ptc.colBuf = runConfig.particleColBuf;
    }
}

void ParticleSystem::applyMonolayer() {
    const double monoZ = (runConfig.simBoxHigh[2] + runConfig.simBoxLow[2]) / 2;
    const int nLocal = particleContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &ptc = particleContainer[i];
        ptc.vel[2] = 0.0;
        ptc.omega[2] = 0.0;
        ptc.pos[2] = monoZ;
        Evec3 drt = Emapq(ptc.orientation) * Evec3(0, 0, 1);
        drt[2] = 0;
        drt.normalize();
        Emapq(ptc.orientation).setFromTwoVectors(Evec3(0, 0, 1), drt);
    }
}

void ParticleSystem::getForceVelocityNonConstraint(const Teuchos::RCP<TV> &forceNCRcp,
                                                   const Teuchos::RCP<TV> &velocityNCRcp) const {
    // store results
    auto forceNCPtr = forceNCRcp->getLocalView<Kokkos::HostSpace>();
    auto velocityNCPtr = velocityNCRcp->getLocalView<Kokkos::HostSpace>();
    forceNCRcp->modify<Kokkos::HostSpace>();
    velocityNCRcp->modify<Kokkos::HostSpace>();

    const int particleLocalNumber = particleContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velocityNCPtr.extent(0) == particleLocalNumber * 6);
    TEUCHOS_ASSERT(velocityNCPtr.extent(1) == 1);

#pragma omp parallel for
    for (int i = 0; i < particleLocalNumber; i++) {
        auto &ptc = particleContainer[i];
        velocityNCPtr(6 * i + 0, 0) = ptc.velNonB[0] + ptc.velBrown[0];
        velocityNCPtr(6 * i + 1, 0) = ptc.velNonB[1] + ptc.velBrown[1];
        velocityNCPtr(6 * i + 2, 0) = ptc.velNonB[2] + ptc.velBrown[2];
        velocityNCPtr(6 * i + 3, 0) = ptc.omegaNonB[0] + ptc.omegaBrown[0];
        velocityNCPtr(6 * i + 4, 0) = ptc.omegaNonB[1] + ptc.omegaBrown[1];
        velocityNCPtr(6 * i + 5, 0) = ptc.omegaNonB[2] + ptc.omegaBrown[2];

        forceNCPtr(6 * i + 0, 0) = ptc.forceNonB[0];
        forceNCPtr(6 * i + 1, 0) = ptc.forceNonB[1];
        forceNCPtr(6 * i + 2, 0) = ptc.forceNonB[2];
        forceNCPtr(6 * i + 3, 0) = ptc.torqueNonB[0];
        forceNCPtr(6 * i + 4, 0) = ptc.torqueNonB[1];
        forceNCPtr(6 * i + 5, 0) = ptc.torqueNonB[2];
    }
}

void ParticleSystem::saveForceVelocityConstraints(const Teuchos::RCP<const TV> &forceRcp,
                                                  const Teuchos::RCP<const TV> &velocityRcp) {
    // save results
    auto velPtr = velocityRcp->getLocalView<Kokkos::HostSpace>();
    auto forcePtr = forceRcp->getLocalView<Kokkos::HostSpace>();

    const int particleLocalNumber = particleContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velPtr.extent(0) == particleLocalNumber * 6);
    TEUCHOS_ASSERT(velPtr.extent(1) == 1);

#pragma omp parallel for
    for (int i = 0; i < particleLocalNumber; i++) {
        auto &ptc = particleContainer[i];
        ptc.velCon[0] = velPtr(6 * i + 0, 0);
        ptc.velCon[1] = velPtr(6 * i + 1, 0);
        ptc.velCon[2] = velPtr(6 * i + 2, 0);
        ptc.omegaCon[0] = velPtr(6 * i + 3, 0);
        ptc.omegaCon[1] = velPtr(6 * i + 4, 0);
        ptc.omegaCon[2] = velPtr(6 * i + 5, 0);

        ptc.forceCon[0] = forcePtr(6 * i + 0, 0);
        ptc.forceCon[1] = forcePtr(6 * i + 1, 0);
        ptc.forceCon[2] = forcePtr(6 * i + 2, 0);
        ptc.torqueCon[0] = forcePtr(6 * i + 3, 0);
        ptc.torqueCon[1] = forcePtr(6 * i + 4, 0);
        ptc.torqueCon[2] = forcePtr(6 * i + 5, 0);
    }
}

void ParticleSystem::calcVelocityBrown() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
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
            auto &ptc = particleContainer[i];
            // constants
            double dragPara = 0;
            double dragPerp = 0;
            double dragRot = 0;
            ptc.calcDragCoeff(mu, dragPara, dragPerp, dragRot);
            const double dragParaInv = ptc.isImmovable ? 0.0 : 1 / dragPara;
            const double dragPerpInv = ptc.isImmovable ? 0.0 : 1 / dragPerp;
            const double dragRotInv = ptc.isImmovable ? 0.0 : 1 / dragRot;

            // convert FDPS vec3 to Evec3
            Evec3 direction = Emapq(ptc.orientation) * Evec3(0, 0, 1);

            // RFD from Delong, JCP, 2015
            // slender fiber has 0 rot drag, regularize with identity rot mobility
            // trans mobility is this
            Evec3 q = direction;
            Emat3 Nmat = (dragParaInv - dragPerpInv) * (q * q.transpose()) + (dragPerpInv)*Emat3::Identity();
            Emat3 Nmatsqrt = Nmat.llt().matrixL();

            // velocity
            Evec3 Wrot(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wpos(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wrfdrot(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));
            Evec3 Wrfdpos(rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId), rngPoolPtr->getN01(threadId));

            Equatn orientRFD = Emapq(ptc.orientation);
            EquatnHelper::rotateEquatn(orientRFD, Wrfdrot, delta);
            q = orientRFD * Evec3(0, 0, 1);
            Emat3 Nmatrfd = (dragParaInv - dragPerpInv) * (q * q.transpose()) + (dragPerpInv)*Emat3::Identity();

            Evec3 vel = kBTfactor * (Nmatsqrt * Wpos);           // Gaussian noise
            vel += (kBT / delta) * ((Nmatrfd - Nmat) * Wrfdpos); // rfd drift. seems no effect in this case
            Evec3 omega = sqrt(dragRotInv) * kBTfactor * Wrot;   // regularized identity rotation drag

            Emap3(ptc.velBrown) = vel;
            Emap3(ptc.omegaBrown) = omega;
        }
    }
}

// TODO: move to particle wall interaction class
void ParticleSystem::collectBoundaryCollision() {
    auto constraintPoolPtr = conCollectorPtr->constraintPoolPtr; // shared_ptr
    const int nThreads = constraintPoolPtr->size();
    const int nLocal = particleContainer.getNumberOfParticleLocal();

    // process collisions with all boundaries
    for (const auto &bPtr : runConfig.boundaryPtr) {
#pragma omp parallel num_threads(nThreads)
        {
            const int threadId = omp_get_thread_num();
            auto &conQue = (*constraintPoolPtr)[threadId];
#pragma omp for
            for (int i = 0; i < nLocal; i++) {
                const auto &ptc = particleContainer[i];
                const Evec3 center = ECmap3(ptc.pos);

                // check one point
                auto checkEnd = [&](const Evec3 &Query, const double radius) {
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
                        const double sep = -deltanorm - radius;
                        Emat3 stressIJ = Emat3::Zero();
                        Constraint con;
                        noPenetrationConstraint(con,            // constraint object,
                                                sep,            // amount of overlap,
                                                ptc.gid, ptc.gid, //
                                                ptc.globalIndex, //
                                                ptc.globalIndex, //
                                                posI.data(),
                                                posI.data(),        // location of collision relative to particle center
                                                Query.data(), Proj, // location of collision in lab frame
                                                norm.data(),        // direction of collision force
                                                stressIJ.data(), true);
                        conQue.push_back(con);
                    } else if (deltanorm <
                               (1 + runConfig.particleColBuf * 2) * ptc.radiusCollision) { // inside boundary but close
                        const double sep = deltanorm - radius;
                        Emat3 stressIJ = Emat3::Zero();
                        Constraint con;
                        noPenetrationConstraint(con,            // constraint object,
                                                sep,            // amount of overlap,
                                                ptc.gid, ptc.gid, //
                                                ptc.globalIndex, //
                                                ptc.globalIndex, //
                                                posI.data(),
                                                posI.data(),        // location of collision relative to particle center
                                                Query.data(), Proj, // location of collision in lab frame
                                                norm.data(),        // direction of collision force
                                                stressIJ.data(), true);
                        conQue.push_back(con);
                    }
                };

                if (ptc.isSphere(true)) {
                    double radius = ptc.lengthCollision * 0.5 + ptc.radiusCollision;
                    checkEnd(center, radius);
                } else {
                    const Equatn orientation = ECmapq(ptc.orientation);
                    const Evec3 direction = orientation * Evec3(0, 0, 1);
                    const double length = ptc.lengthCollision;
                    const Evec3 Qm = center - direction * (length * 0.5);
                    const Evec3 Qp = center + direction * (length * 0.5);
                    checkEnd(Qm, ptc.radiusCollision);
                    checkEnd(Qp, ptc.radiusCollision);
                }
            }
        }
    }
    return;
}

std::pair<int, int> ParticleSystem::getMaxGid() {
    int maxGidLocal = 0;
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    for (int i = 0; i < nLocal; i++) {
        maxGidLocal = std::max(maxGidLocal, particleContainer[i].gid);
    }

    int maxGidGlobal = maxGidLocal;
    Teuchos::reduceAll(*commRcp, Teuchos::MaxValueReductionOp<int, int>(), 1, &maxGidLocal, &maxGidGlobal);
    spdlog::warn("rank: {}, maxGidLocal: {}, maxGidGlobal {}", commRcp->getRank(), maxGidLocal, maxGidGlobal);

    return std::pair<int, int>(maxGidLocal, maxGidGlobal);
}

void ParticleSystem::calcBoundingBox(double localLow[3], double localHigh[3], double globalLow[3],
                                     double globalHigh[3]) {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    double lx, ly, lz;
    lx = ly = lz = std::numeric_limits<double>::max();
    double hx, hy, hz;
    hx = hy = hz = std::numeric_limits<double>::min();

    for (int i = 0; i < nLocal; i++) {
        const auto &ptc = particleContainer[i];
        const Evec3 direction = ECmapq(ptc.orientation) * Evec3(0, 0, 1);
        Evec3 pm = ECmap3(ptc.pos) - (ptc.length * 0.5) * direction;
        Evec3 pp = ECmap3(ptc.pos) + (ptc.length * 0.5) * direction;
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

void ParticleSystem::updateParticleRank() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    const int rank = commRcp->getRank();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        particleContainer[i].rank = rank;
    }
}

void ParticleSystem::applyBoxBC() { particleContainer.adjustPositionIntoRootDomain(dinfo); }

void ParticleSystem::calcConStress() {
    if (runConfig.logLevel > spdlog::level::info)
        return;

    Emat3 sumConStress = Emat3::Zero();
    conCollectorPtr->sumLocalConstraintStress(sumConStress, false);

    // scale to nkBT
    const double scaleFactor = 1 / (particleMapRcp->getGlobalNumElements() * runConfig.KBT);
    sumConStress *= scaleFactor;

    // mpi reduction
    double conStressLocal[9];
    double conStressGlobal[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            conStressLocal[i * 3 + j] = sumConStress(i, j);
            conStressGlobal[i * 3 + j] = 0;
        }
    }

    Teuchos::reduceAll(*commRcp, Teuchos::SumValueReductionOp<int, double>(), 9, conStressLocal, conStressGlobal);

    spdlog::info("RECORD: ColXF,{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g}", //
                 conStressGlobal[0], conStressGlobal[1], conStressGlobal[2],   //
                 conStressGlobal[3], conStressGlobal[4], conStressGlobal[5],   //
                 conStressGlobal[6], conStressGlobal[7], conStressGlobal[8]);
}

void ParticleSystem::calcOrderParameter() {
    if (runConfig.logLevel > spdlog::level::info)
        return;

    double px = 0, py = 0, pz = 0;    // pvec
    double Qxx = 0, Qxy = 0, Qxz = 0; // Qtensor
    double Qyx = 0, Qyy = 0, Qyz = 0; // Qtensor
    double Qzx = 0, Qzy = 0, Qzz = 0; // Qtensor

    const int nLocal = particleContainer.getNumberOfParticleLocal();

#pragma omp parallel for reduction(+ : px, py, pz, Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz)
    for (int i = 0; i < nLocal; i++) {
        const auto &ptc = particleContainer[i];
        const Evec3 direction = ECmapq(ptc.orientation) * Evec3(0, 0, 1);
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
    const int nGlobal = particleContainer.getNumberOfParticleGlobal();
    double pQ[12] = {px, py, pz, Qxx, Qxy, Qxz, Qyx, Qyy, Qyz, Qzx, Qzy, Qzz};
    MPI_Allreduce(MPI_IN_PLACE, pQ, 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < 12; i++) {
        pQ[i] *= (1.0 / nGlobal);
    }

    spdlog::info("RECORD: Order P,{:g},{:g},{:g},Q,{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g}", //
                 pQ[0], pQ[1], pQ[2],                                                             // pvec
                 pQ[3], pQ[4], pQ[5],                                                             // Qtensor
                 pQ[6], pQ[7], pQ[8],                                                             // Qtensor
                 pQ[9], pQ[10], pQ[11]                                                            // Qtensor
    );
}

std::vector<int> ParticleSystem::addNewParticle(const std::vector<Particle> &newParticle) {
    // assign unique new gid for particles on all ranks
    std::pair<int, int> maxGid = getMaxGid();
    const int maxGidLocal = maxGid.first;
    const int maxGidGlobal = maxGid.second;
    const int newCountLocal = newParticle.size();

    // collect number of ids from all ranks to rank0
    std::vector<int> newCount(commRcp->getSize(), 0);
    std::vector<int> displ(commRcp->getSize() + 1, 0);
    MPI_Gather(&newCountLocal, 1, MPI_INT, newCount.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> newGid;
    if (commRcp->getRank() == 0) {
        // generate random gid on rank 0
        std::partial_sum(newCount.cbegin(), newCount.cend(), displ.begin() + 1);
        const int newCountGlobal = displ.back();
        newGid.resize(newCountGlobal, 0);
        std::iota(newGid.begin(), newGid.end(), maxGidGlobal + 1);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(newGid.begin(), newGid.end(), g);
    } else {
        newGid.resize(newCountLocal, 0);
    }

    // scatter from rank 0 to every rank
    std::vector<int> newGidRecv(newCountLocal, 0);
    MPI_Scatterv(newGid.data(), newCount.data(), displ.data(), MPI_INT, //
                 newGidRecv.data(), newCountLocal, MPI_INT, 0, MPI_COMM_WORLD);

    // set new gid
    for (int i = 0; i < newCountLocal; i++) {
        Particle ptc = newParticle[i];
        ptc.gid = newGidRecv[i];
        particleContainer.addOneParticle(ptc);
    }

    return newGidRecv;
}

void ParticleSystem::addNewLink(const std::vector<Link> &newLink) {
    // synchronize newLink to all mpi ranks
    const int newCountLocal = newLink.size();
    std::vector<int> newCount(commRcp->getSize(), 0);
    MPI_Allgather(&newCountLocal, 1, MPI_INT, newCount.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displ(commRcp->getSize() + 1, 0);
    std::partial_sum(newCount.cbegin(), newCount.cend(), displ.begin() + 1);
    std::vector<Link> newLinkRecv(displ.back());
    MPI_Allgatherv(newLink.data(), newCountLocal, createMPIStructType<Link>(), newLinkRecv.data(), newCount.data(),
                   displ.data(), createMPIStructType<Link>(), MPI_COMM_WORLD);

    // put newLinks into the map, same op on all mpi ranks
    for (const auto &ll : newLinkRecv) {
        linkMap.emplace(ll.prev, ll.next);
        linkReverseMap.emplace(ll.next, ll.prev);
    }
}

void ParticleSystem::buildparticleNearDataDirectory() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &particleNearDataDirectory = *particleNearDataDirectoryPtr;
    particleNearDataDirectory.gidOnLocal.resize(nLocal);
    particleNearDataDirectory.dataOnLocal.resize(nLocal);
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        particleNearDataDirectory.gidOnLocal[i] = particleContainer[i].gid;
        particleNearDataDirectory.dataOnLocal[i].copyFromFP(particleContainer[i]);
    }

    // build index
    particleNearDataDirectory.buildIndex();
}

void ParticleSystem::updateparticleNearDataDirectory() {
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &particleNearDataDirectory = *particleNearDataDirectoryPtr;
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        assert(particleNearDataDirectory.gidOnLocal[i] == particleContainer[i].gid);
        particleNearDataDirectory.dataOnLocal[i].copyFromFP(particleContainer[i]);
    }

    // build index (must be called even though the index is unchanged. boooooo)
    particleNearDataDirectory.buildIndex();
}

// TODO: move to particle-particle interaction class
void ParticleSystem::collectPairCollision() {
    CalcParticleNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.simBoxPBC, runConfig.simBoxLow,
                                     runConfig.simBoxHigh);

    TEUCHOS_ASSERT(treeParticleNearPtr);
    const int nLocal = particleContainer.getNumberOfParticleLocal();
    setTreeParticle();
    treeParticleNearPtr->calcForceAll(calcColFtr, particleContainer, dinfo);
}

// TODO: Move this into particle link interaction class
void ParticleSystem::collectLinkBilateral() {
    // setup bilateral link constraints
    // uses special treatment for periodic boundary conditions

    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &conPool = *(this->conCollectorPtr->constraintPoolPtr);
    if (conPool.size() != omp_get_max_threads()) {
        spdlog::critical("conPool multithread mismatch error");
        std::exit(1);
    }

    // fill the data to find
    auto &gidToFind = particleNearDataDirectoryPtr->gidToFind;
    const auto &dataToFind = particleNearDataDirectoryPtr->dataToFind;

    std::vector<int> gidDisp(nLocal + 1, 0);
    gidToFind.clear();
    gidToFind.reserve(nLocal);

    // loop over all particles
    // if linkMap[ptc.gid] not empty, find info for all next
    for (int i = 0; i < nLocal; i++) {
        const auto &ptc = particleContainer[i];
        const auto &range = linkMap.equal_range(ptc.gid);
        int count = 0;
        for (auto it = range.first; it != range.second; it++) {
            gidToFind.push_back(it->second); // next
            count++;
        }
        gidDisp[i + 1] = gidDisp[i] + count; // number of links for each local Particle
    }

    particleNearDataDirectoryPtr->find();

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        auto &conQue = conPool[threadId];
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            const auto &ptcI = particleContainer[i]; // particle
            const int lb = gidDisp[i];
            const int ub = gidDisp[i + 1];

            for (int j = lb; j < ub; j++) {
                // create a spring constraint
                const auto &ptcJ = particleNearDataDirectoryPtr->dataToFind[j]; // particleNear

                const Evec3 &centerI = ECmap3(ptcI.pos);
                Evec3 centerJ = ECmap3(ptcJ.pos);
                // apply PBC on centerJ
                for (int k = 0; k < 3; k++) {
                    if (!runConfig.simBoxPBC[k])
                        continue;
                    double trg = centerI[k];
                    double xk = centerJ[k];
                    findPBCImage(runConfig.simBoxLow[k], runConfig.simBoxHigh[k], xk, trg);
                    centerJ[k] = xk;
                    // error check
                    if (fabs(trg - xk) > 0.5 * (runConfig.simBoxHigh[k] - runConfig.simBoxLow[k])) {
                        spdlog::critical("pbc image error in bilateral links");
                        std::exit(1);
                    }
                }
                // particles are not treated as spheres for bilateral constraints
                // constraint is always added between Pp and Qm
                // constraint target length is radiusI + radiusJ + runConfig.linkGap
                const Evec3 directionI = ECmapq(ptcI.orientation) * Evec3(0, 0, 1);
                const Evec3 directionJ = ECmap3(ptcJ.direction);
                const Evec3 Pp = centerI + directionI * (0.5 * ptcI.length); // plus end
                const Evec3 Qm = centerJ - directionJ * (0.5 * ptcJ.length);
                const Evec3 Ploc = Pp;
                const Evec3 Qloc = Qm;
                const Evec3 rvec = Qloc - Ploc;
                const double rnorm = rvec.norm();
                const Evec3 normI = (Ploc - Qloc).normalized();
                const Evec3 normJ = -normI;
                const Evec3 posI = Ploc - centerI;
                const Evec3 posJ = Qloc - centerJ;
                const double restLength = runConfig.linkGap;
                const double springLength = rnorm - ptcI.radius - ptcJ.radius;

                Emat3 stressIJ = Emat3::Zero();
                CalcParticleNearForce::collideStress(directionI, directionJ, centerI, centerJ, ptcI.length, ptcJ.length,
                                                     ptcI.radius, ptcJ.radius, 1.0, Ploc, Qloc, stressIJ);

                Constraint con;
                springConstraint(con,                      // constraint object
                                 springLength, restLength, // length of spring, rest length of spring
                                 runConfig.linkKappa,      // spring constant
                                 ptcI.gid, ptcJ.gid,         //
                                 ptcI.globalIndex,          //
                                 ptcJ.globalIndex,          //
                                 posI.data(), posJ.data(), // location of collision relative to particle center
                                 Ploc.data(), Qloc.data(), // location of collision in lab frame
                                 normI.data(),             // direction of collision force
                                 stressIJ.data(), false);
                conQue.push_back(con);
            }
        }
    }
}

// TODO: Move this into particle-particle interaction class
void ParticleSystem::updatePairCollision() {
    // TODO: extend this to include updating other types of constraints. Feting via GID should happen regardless.
    //       The only difference is the update step
    std::cout << "updatePairCollision doesn't account for systems with multiple types of constraints" << std::endl;

    // update the collision constraints stored in the constraintPool
    // uses special treatment for periodic boundary conditions

    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &conPool = *(this->conCollectorPtr->constraintPoolPtr);

    if (conPool.size() != omp_get_max_threads()) {
        spdlog::critical("cPool multithread mismatch error");
        std::exit(1);
    }

    // Step 1. fill gidToFind (2 GIDs per collision constraint)

    // Step1.1. Setup the offset array to allow for parallel filling
    //      we use  multi-thread filling with nThreads = poolSize and each thread process a queue
    //      GID's are filled contiguously based on threadID
    //      threadOffset stores the index of the start of each thread's region of access in gidToFind
    // This is exactly conCollectorPtr->buildConIndex with slight modification
    const int nThreads = conPool.size();
    std::vector<int> threadOffset(nThreads + 1, 0); // last entry is the total GID's to find (2 per constraint)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        const auto &conQue = conPool[threadId];
        threadOffset[threadId + 1] = threadOffset[threadId] + 2 * conQue.size();
    }
    assert(threadOffset.back() == 2 * conCollectorPtr->getLocalNumberOfConstraints());

    // Step 1.2 loop over all constraints and add gidI and gidJ to gidToFind
    auto &gidToFind = particleNearDataDirectoryPtr->gidToFind;
    gidToFind.clear();
    gidToFind.resize(threadOffset.back());

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        const auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        const int conIndexBase = threadOffset[threadId];
        for (int j = 0; j < conNum; j++) {
            gidToFind[conIndexBase + 2 * j + 0] = conQue[j].gidI;
            gidToFind[conIndexBase + 2 * j + 1] = conQue[j].gidJ;
        }
    }

    particleNearDataDirectoryPtr->find();

    // Step 2. Update each constraint
    CalcParticleNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.simBoxPBC, runConfig.simBoxLow,
                                     runConfig.simBoxHigh);

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        const int conIndexBase = threadOffset[threadId];
        for (int j = 0; j < conNum; j++) {
            const int idx = conIndexBase + 2 * j;
            const auto &ptcI = particleNearDataDirectoryPtr->dataToFind[idx + 0]; // particleNearI
            const auto &ptcJ = particleNearDataDirectoryPtr->dataToFind[idx + 1]; // particleNearJ
            calcColFtr.updateCollisionBlock(ptcI, ptcJ, conQue[j]);
        }
    }
}

// TODO: Move this into particle-particle interaction class
void ParticleSystem::collectUnresolvedConstraints() {
    // TODO: extend this function to use a factory for updating constraints

    // Loop over all constraints, check if the constraint is satisfies;
    // if not, generate a new constraint to handle the residual

    // update the constraints stored in the constraintPool
    // uses special treatment for periodic boundary conditions

    const int nLocal = particleContainer.getNumberOfParticleLocal();
    auto &conPool = *(this->conCollectorPtr->constraintPoolPtr);

    if (conPool.size() != omp_get_max_threads()) {
        spdlog::critical("cPool multithread mismatch error");
        std::exit(1);
    }

    // Step 1. fill gidToFind (2 GIDs per constraint)

    // Step1.1. Setup the offset array to allow for parallel filling
    //      we use  multi-thread filling with nThreads = poolSize and each thread process a queue
    //      GID's are filled contiguously based on threadID
    //      threadOffset stores the index of the start of each thread's region of access in gidToFind
    // This is exactly conCollectorPtr->buildConIndex with slight modification
    const int nThreads = conPool.size();
    std::vector<int> threadOffset(nThreads + 1, 0); // last entry is the total GID's to find (2 per constraint)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        const auto &conQue = conPool[threadId];
        threadOffset[threadId + 1] = threadOffset[threadId] + 2 * conQue.size();
    }
    assert(threadOffset.back() == 2 * conCollectorPtr->getLocalNumberOfConstraints());

    // Step 1.2 loop over all constraints and add gidI and gidJ to gidToFind
    auto &gidToFind = particleNearDataDirectoryPtr->gidToFind;
    gidToFind.clear();
    gidToFind.resize(threadOffset.back());

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        const auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        const int conIndexBase = threadOffset[threadId];
        for (int j = 0; j < conNum; j++) {
            gidToFind[conIndexBase + 2 * j + 0] = conQue[j].gidI;
            gidToFind[conIndexBase + 2 * j + 1] = conQue[j].gidJ;
        }
    }

    particleNearDataDirectoryPtr->find();

    // Step 2. check each collision pair for violation
    CalcParticleNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.simBoxPBC, runConfig.simBoxLow,
                                     runConfig.simBoxHigh);

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        const int conIndexBase = threadOffset[threadId];
        for (int j = 0; j < conNum; j++) {
            const int idx = conIndexBase + 2 * j;
            const auto &ptcI = particleNearDataDirectoryPtr->dataToFind[idx + 0]; // particleNearI
            const auto &ptcJ = particleNearDataDirectoryPtr->dataToFind[idx + 1]; // particleNearJ

            // get an updated constraint between the two particles
            Constraint con;
            // TODO: replace the following with a factory based on the ID of the constraint. For now, this is ok.
            // TODO: add in ball joints and angular springs
            // TODO: how to account for nonconvex boundaries?
            // TODO: generalize this to multi-dof constraints (friction)
            if (conQue[j].oneSide) { // boundary collision
                // // check each boundary
                // for (const auto &bPtr : runConfig.boundaryPtr) {
                //     const Evec3 center = ECmap3(ptcI.pos);

                //     // check one point
                //     auto checkEnd = [&](const Evec3 &Query, const double radius) {
                //         bool collision = false;
                //         double Proj[3], delta[3];
                //         bPtr->project(Query.data(), Proj, delta);
                //         // if (!bPtr->check(Query.data(), Proj, delta)) {
                //         //     printf("boundary projection error\n");
                //         // }
                //         // if inside boundary, delta = Q-Proj
                //         // if outside boundary, delta = Proj-Q
                //         double deltanorm = Emap3(delta).norm();
                //         Evec3 norm = Emap3(delta) * (1 / deltanorm);
                //         Evec3 posI = Query - center;

                //         if ((Query - ECmap3(Proj)).dot(ECmap3(delta)) < 0) { // outside boundary
                //             collision = true;
                //             const double sep = -deltanorm - radius;
                //             Emat3 stressIJ = Emat3::Zero();
                //             noPenetrationConstraint(con,            // constraint object,
                //                                     sep,            // amount of overlap,
                //                                     ptcI.gid, ptcI.gid, //
                //                                     ptcI.globalIndex, //
                //                                     ptcI.globalIndex, //
                //                                     posI.data(),
                //                                     posI.data(),        // location of collision relative to particle
                //                                     center Query.data(), Proj, // location of collision in lab frame
                //                                     norm.data(),        // direction of collision force
                //                                     stressIJ.data(), false);
                //         }
                //         return collision;
                //     };

                //     bool unsatisfied = false;
                //     if (ptcI.isSphere(true)) {
                //         double radius = ptcI.lengthCollision * 0.5 + ptcI.radiusCollision;
                //         const bool collision = checkEnd(center, radius);
                //         unsatisfied = collision;
                //     } else {
                //         const Evec3 direction = ECmap3(ptcI.direction);
                //         const double length = ptcI.lengthCollision;
                //         const Evec3 Qm = center - direction * (length * 0.5);
                //         const Evec3 Qp = center + direction * (length * 0.5);
                //         const bool collisionM = checkEnd(Qm, ptcI.radiusCollision);
                //         const bool collisionP = checkEnd(Qp, ptcI.radiusCollision);
                //         unsatisfied = (collisionM || collisionP);
                //     }

                //     // if the updated constaint is unsatisfied add it to the que
                //     if (unsatisfied) {
                //         conQue.push_back(con);
                //     }
                // }
            } else if (conQue[j].bilaterial) { // spring
                const Evec3 &centerI = ECmap3(ptcI.pos);
                Evec3 centerJ = ECmap3(ptcJ.pos);
                // apply PBC on centerJ
                for (int k = 0; k < 3; k++) {
                    if (!runConfig.simBoxPBC[k])
                        continue;
                    double trg = centerI[k];
                    double xk = centerJ[k];
                    findPBCImage(runConfig.simBoxLow[k], runConfig.simBoxHigh[k], xk, trg);
                    centerJ[k] = xk;
                    // error check
                    if (fabs(trg - xk) > 0.5 * (runConfig.simBoxHigh[k] - runConfig.simBoxLow[k])) {
                        spdlog::critical("pbc image error in bilateral links");
                        std::exit(1);
                    }
                }
                // particles are not treated as spheres for bilateral constraints
                // constraint is always added between Pp and Qm
                // constraint target length is radiusI + radiusJ + runConfig.linkGap
                const Evec3 directionI = ECmap3(ptcI.direction);
                const Evec3 directionJ = ECmap3(ptcJ.direction);

                const Evec3 Pp = centerI + directionI * (0.5 * ptcI.length); // plus end
                const Evec3 Qm = centerJ - directionJ * (0.5 * ptcJ.length);
                const Evec3 Ploc = Pp;
                const Evec3 Qloc = Qm;
                const Evec3 rvec = Qloc - Ploc;
                const double rnorm = rvec.norm();
                const Evec3 normI = (Ploc - Qloc).normalized();
                const Evec3 normJ = -normI;
                const Evec3 posI = Ploc - centerI;
                const Evec3 posJ = Qloc - centerJ;
                const double restLength = runConfig.linkGap;
                const double springLength = rnorm - ptcI.radius - ptcJ.radius;

                Emat3 stressIJ = Emat3::Zero();
                CalcParticleNearForce::collideStress(directionI, directionJ, centerI, centerJ, ptcI.length, ptcJ.length,
                                                     ptcI.radius, ptcJ.radius, 1.0, Ploc, Qloc, stressIJ);

                springConstraint(con,                      // constraint object
                                 springLength, restLength, // length of spring, rest length of spring
                                 runConfig.linkKappa,      // spring constant
                                 ptcI.gid, ptcJ.gid,         //
                                 ptcI.globalIndex,          //
                                 ptcJ.globalIndex,          //
                                 posI.data(), posJ.data(), // location of collision relative to particle center
                                 Ploc.data(), Qloc.data(), // location of collision in lab frame
                                 normI.data(),             // direction of collision force
                                 stressIJ.data(), false);

                // if the updated constaint is unsatisfied add it to the que
                // TODO: this is wrong. We aren't trying to force the spring to be at rest, we are trying to compute the reactionary spring force
                if (std::abs(rnorm - restLength) > runConfig.conResTol) { 
                    conQue.push_back(con);
                }
            } else { // pairwise collision
                calcColFtr.updateCollisionBlock(ptcI, ptcJ, con);
                // if the updated constaint is unsatisfied add it to the que
                if (con.getSep(0) < -runConfig.conResTol) {
                    conQue.push_back(con);
                }
            }
        }
    }
}
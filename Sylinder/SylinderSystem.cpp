#include "SylinderSystem.hpp"

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

SylinderSystem::SylinderSystem(const SylinderConfig &runConfig_, const Teuchos::RCP<const TCOMM> &commRcp_,
                               std::shared_ptr<TRngPool> rngPoolPtr_,
                               std::shared_ptr<ConstraintCollector> conCollectorPtr_)
    : runConfig(runConfig_), rngPoolPtr(std::move(rngPoolPtr_)), conCollectorPtr(std::move(conCollectorPtr_)),
      commRcp(commRcp_) {}

void SylinderSystem::initialize(const std::string &posFile) {
    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    sylinderContainer.initialize();
    sylinderContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more sample for better balance

    if (IOHelper::fileExist(posFile)) {
        setInitialFromFile(posFile);
    } else {
        setInitialFromConfig();
    }
    setLinkMapFromFile(posFile);

    // at this point all sylinders located on rank 0
    commRcp->barrier();
    decomposeDomain();
    exchangeSylinder(); // distribute to ranks, initial domain decomposition

    // initialize sylinder growth
    initSylinderGrowth();

    // initialize the GID search tree
    sylinderNearDataDirectoryPtr = std::make_shared<ZDD<SylinderNearEP>>(sylinderContainer.getNumberOfParticleLocal());

    treeSylinderNumber = 0;
    setTreeSylinder();

    calcVolFrac();

    if (commRcp->getRank() == 0) {
        writeBox();
    }

    // make sure the current position and orientation are updated
    advanceParticles();

    spdlog::warn("SylinderSystem Initialized. {} local sylinders", sylinderContainer.getNumberOfParticleLocal());
}

void SylinderSystem::reinitialize(const std::string &pvtpFileName_) {
    dinfo.initialize(); // init DomainInfo
    setDomainInfo();

    sylinderContainer.initialize();
    sylinderContainer.setAverageTargetNumberOfSampleParticlePerProcess(200); // more samples for better balance

    // reinitialize from pctp file
    std::string pvtpFileName = pvtpFileName_;
    std::string asciiFileName = pvtpFileName;
    auto pos = asciiFileName.find_last_of('.');
    asciiFileName.replace(pos, 5, std::string(".dat")); // replace '.pvtp' with '.dat'
    pos = asciiFileName.find_last_of('_');
    asciiFileName.replace(pos, 1, std::string("Ascii_")); // replace '_' with 'Ascii_'

    setInitialFromVTKFile(pvtpFileName);
    setLinkMapFromFile(asciiFileName);

    // at this point all sylinders located on rank 0
    commRcp->barrier();
    applyBoxBC();
    decomposeDomain();
    exchangeSylinder(); // distribute to ranks, initial domain decomposition
    updateSylinderMap();

    // initialize the GID search tree
    sylinderNearDataDirectoryPtr = std::make_shared<ZDD<SylinderNearEP>>(sylinderContainer.getNumberOfParticleLocal());

    treeSylinderNumber = 0;
    setTreeSylinder();
    calcVolFrac();

    // make sure the current position and orientation are updated
    advanceParticles();

    spdlog::warn("SylinderSystem Initialized. {} local sylinders", sylinderContainer.getNumberOfParticleLocal());
}

void SylinderSystem::initSylinderGrowth() {
    // TODO: name these variables so they are directly interprettabole by their names
    if (runConfig.ptcGrowth.size() == 0) {
        return;
    }

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    Growth *pgrowth;
    if (runConfig.ptcGrowth.size() == 1)
        pgrowth = &(runConfig.ptcGrowth.begin()->second);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        sy.t = 0;
        if (pgrowth) {
            sy.deltaL = pgrowth->Delta;
            sy.sigma = pgrowth->sigma;
            sy.tauD = pgrowth->tauD;
            sy.tg = pgrowth->tauD * log2(1 + pgrowth->Delta / sy.length) + pgrowth->sigma * rngPoolPtr->getN01();
        } else {
            // TODO, multiple species
        }
    }
}

void SylinderSystem::calcSylinderDivision() {
    const double eps = std::numeric_limits<double>::epsilon() * 100;

    if (runConfig.ptcGrowth.size() == 0) {
        return;
    }

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    const double facMut[3] = {1., 1., 1.};
    Growth *pgrowth;
    if (runConfig.ptcGrowth.size() == 1)
        pgrowth = &(runConfig.ptcGrowth.begin()->second);

    std::vector<Sylinder> newPtc;

    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        if (sy.t < sy.tg) {
            continue;
        }
        const Evec3 center = Evec3(sy.pos[0], sy.pos[1], sy.pos[2]);

        Evec3 direction = ECmapq(sy.orientation) * Evec3(0, 0, 1);
        const double currentLength = sy.length;
        const double newLength = currentLength * 0.5 - sy.radius;
        if (newLength <= 0) {
            continue;
        }

        // old sylinder, shrink and reset center, no rotation
        sy.length = newLength;
        sy.lengthCollision = newLength * runConfig.sylinderLengthColRatio;
        Emap3(sy.pos) = center - direction * (sy.radius + 0.5 * newLength + eps);

        // new sylinder
        Evec3 ncPos = center + direction * (sy.radius + 0.5 * newLength + eps);
        newPtc.emplace_back(-1, sy.radius, sy.radius * runConfig.sylinderDiameterColRatio, newLength,
                            newLength * runConfig.sylinderLengthColRatio, ncPos.data(), sy.orientation);

        newPtc.back().tauD = sy.tauD;
        newPtc.back().sigma = sy.sigma;
        newPtc.back().deltaL = sy.deltaL;
        if (sy.tauD == pgrowth->tauD && sy.sigma == pgrowth->sigma && sy.deltaL == pgrowth->Delta) {
            if (rngPoolPtr->getU01() < 0.01) {
                newPtc.back().tauD *= facMut[0];
                newPtc.back().sigma *= facMut[1];
                newPtc.back().deltaL *= facMut[2];
            }
        }

        // set new tg
        const double tg = sy.tauD * log2(1 + sy.deltaL / newLength) + sy.sigma * rngPoolPtr->getN01();
        sy.t = 0;
        sy.tg = tg;
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
            Emapq(sy.orientation).coeffs() = orientq.coeffs();
        }
    }
    addNewSylinder(newPtc);
}

void SylinderSystem::calcSylinderGrowth(const Teuchos::RCP<const TV> &ptcStressRcp) {
    TEUCHOS_ASSERT(nonnull(ptcStressRcp));

    if (runConfig.ptcGrowth.size() == 0) {
        return;
    }
    Growth *pgrowth;
    if (runConfig.ptcGrowth.size() == 1)
        pgrowth = &(runConfig.ptcGrowth.begin()->second);

    const double dt = runConfig.dt;
    const int sylinderLocalNumber = sylinderContainer.getNumberOfParticleLocal();
    
    // get the local view of the stress vector
    auto ptcStressPtr = ptcStressRcp->getLocalView<Kokkos::HostSpace>();
    TEUCHOS_ASSERT(ptcStressPtr.extent(0) == sylinderLocalNumber * 9);
    TEUCHOS_ASSERT(ptcStressPtr.extent(1) == 1);

#pragma omp parallel for
    for (int i = 0; i < sylinderLocalNumber; i++) {
        auto &sy = sylinderContainer[i];

        // get the particle virial stress
        Emat3 stress;
        for (int idx1 = 0; idx1 < 3; idx1++) {
            for (int idx2 = 0; idx2 < 3; idx2++) {
                stress(idx1, idx2) = ptcStressPtr(i * 9 + idx1 * 3 + idx2, 0);
            }
        }

        // transform the stress into the particle frame
        const Emat3 rotMat = Emapq(sy.orientation).toRotationMatrix();
        const Emat3 rotMatInv = Emapq(sy.orientation).inverse().toRotationMatrix();
        stress = rotMat * stress * rotMatInv;

        // output the normal stress to the terminal // TODO: couple it to growth rate
        spdlog::warn("Normal Stress: {:g}", stress(0, 0));

        // compute the amount of growth 
        double dLdt = log(2.0) * sy.length / sy.tauD;

        // apply the growth to the particle
        sy.t += dt;
        sy.length += fabs(dLdt * dt); // never shrink
        sy.lengthCollision = sy.length * runConfig.sylinderLengthColRatio;
    }
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
    double boxVolume = (runConfig.simBoxHigh[0] - runConfig.simBoxLow[0]) *
                       (runConfig.simBoxHigh[1] - runConfig.simBoxLow[1]) *
                       (runConfig.simBoxHigh[2] - runConfig.simBoxLow[2]);
    spdlog::warn("Volume Sylinder = {:g}", volGlobal);
    spdlog::warn("Volume fraction = {:g}", volGlobal / boxVolume);
}

void SylinderSystem::setInitialFromFile(const std::string &filename) {
    spdlog::warn("Reading file " + filename);

    auto parseSylinder = [&](Sylinder &sy, const std::string &line) {
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

        Emap3(sy.pos) = Evec3((mx + px), (my + py), (mz + pz)) * 0.5;
        sy.gid = gid;
        sy.group = group;
        sy.isImmovable = (type == 'S') ? true : false;
        sy.radius = radius;
        sy.radiusCollision = radius;
        sy.length = sqrt(pow(px - mx, 2) + pow(py - my, 2) + pow(pz - mz, 2));
        sy.lengthCollision = sy.length;
        if (sy.length > 1e-7) {
            Evec3 direction(px - mx, py - my, pz - mz);
            Emapq(sy.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);
        } else {
            Emapq(sy.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), Evec3(0, 0, 1));
        }
    };

    if (commRcp->getRank() != 0) {
        sylinderContainer.setNumberOfParticleLocal(0);
    } else {
        std::ifstream myfile(filename);
        std::string line;
        std::getline(myfile, line); // read two header lines
        std::getline(myfile, line);

        std::deque<Sylinder> sylinderReadFromFile;
        while (std::getline(myfile, line)) {
            if (line[0] == 'C' || line[0] == 'S') {
                Sylinder sy;
                parseSylinder(sy, line);
                sylinderReadFromFile.push_back(sy);
            }
        }
        myfile.close();

        spdlog::debug("Sylinder number in file {} ", sylinderReadFromFile.size());

        // set on rank 0
        const int nRead = sylinderReadFromFile.size();
        sylinderContainer.setNumberOfParticleLocal(nRead);
#pragma omp parallel for
        for (int i = 0; i < nRead; i++) {
            sylinderContainer[i] = sylinderReadFromFile[i];
            sylinderContainer[i].clear();
        }
    }
}

// TODO: this should be part of the constraint collector (init from file functionality)
// Set constraints from file, allowing any type of constraint to be initialized
void SylinderSystem::setLinkMapFromFile(const std::string &filename) {
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

void SylinderSystem::setInitialFromVTKFile(const std::string &pvtpFileName) {
    spdlog::warn("Reading file " + pvtpFileName);

    if (commRcp->getRank() != 0) {
        sylinderContainer.setNumberOfParticleLocal(0);
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

        const int sylinderNumberInFile = posData->GetNumberOfPoints() / 2; // two points per sylinder
        sylinderContainer.setNumberOfParticleLocal(sylinderNumberInFile);
        spdlog::debug("Sylinder number in file {} ", sylinderNumberInFile);

#pragma omp parallel for
        for (int i = 0; i < sylinderNumberInFile; i++) {
            auto &sy = sylinderContainer[i];
            double leftEndpointPos[3] = {0, 0, 0};
            double rightEndpointPos[3] = {0, 0, 0};
            posData->GetPoint(i * 2, leftEndpointPos);
            posData->GetPoint(i * 2 + 1, rightEndpointPos);

            Emap3(sy.pos) = (Emap3(leftEndpointPos) + Emap3(rightEndpointPos)) * 0.5;
            sy.gid = gidData->GetComponent(i, 0);
            sy.group = groupData->GetComponent(i, 0);
            sy.isImmovable = isImmovableData->GetTypedComponent(i, 0) > 0 ? true : false;
            sy.length = lengthData->GetComponent(i, 0);
            sy.lengthCollision = lengthCollisionData->GetComponent(i, 0);
            sy.radius = radiusData->GetComponent(i, 0);
            sy.radiusCollision = radiusCollisionData->GetComponent(i, 0);
            const Evec3 direction(znormData->GetComponent(i, 0), znormData->GetComponent(i, 1),
                                  znormData->GetComponent(i, 2));
            Emapq(sy.orientation) = Equatn::FromTwoVectors(Evec3(0, 0, 1), direction);

            // read in growth information (if necessary) // TODO: use a growth flag to turn off and on this kind of
            // functionality
            sy.t = tData->GetComponent(i, 0);
            sy.tg = tgData->GetComponent(i, 0);
            sy.tauD = tauDData->GetComponent(i, 0);
            sy.sigma = sigmaData->GetComponent(i, 0);
            sy.deltaL = deltaLData->GetComponent(i, 0);

            // read in velocity for correct stepping
            sy.vel[0] = velData->GetComponent(i, 0);
            sy.vel[1] = velData->GetComponent(i, 1);
            sy.vel[2] = velData->GetComponent(i, 2);
            sy.omega[0] = omegaData->GetComponent(i, 0);
            sy.omega[1] = omegaData->GetComponent(i, 1);
            sy.omega[2] = omegaData->GetComponent(i, 2);
        }

        // sort the vector of Sylinders by gid ascending;
        // std::sort(sylinderReadFromFile.begin(), sylinderReadFromFile.end(),
        //           [](const Sylinder &t1, const Sylinder &t2) { return t1.gid < t2.gid; });
    }
    commRcp->barrier();
}

void SylinderSystem::writeAscii(const int stepCount, const std::string &baseFolder, const std::string &postfix) {
    // write a single ascii .dat file
    const int nGlobal = sylinderContainer.getNumberOfParticleGlobal();

    std::string name = baseFolder + std::string("SylinderAscii_") + postfix + ".dat";
    SylinderAsciiHeader header;
    header.nparticle = nGlobal;
    header.time = stepCount * runConfig.dt;
    sylinderContainer.writeParticleAscii(name.c_str(), header);
    if (commRcp->getRank() == 0) {
        FILE *fptr = fopen(name.c_str(), "a");
        for (const auto &key_value : linkMap) {
            fprintf(fptr, "L %d %d\n", key_value.first, key_value.second);
        }
        fclose(fptr);
    }
    commRcp->barrier();
}

void SylinderSystem::writeVTK(const std::string &baseFolder, const std::string &postfix) {
    const int rank = commRcp->getRank();
    const int size = commRcp->getSize();
    Sylinder::writeVTP<PS::ParticleSystem<Sylinder>>(sylinderContainer, sylinderContainer.getNumberOfParticleLocal(),
                                                     baseFolder, postfix, rank);
    conCollectorPtr->writeVTP(baseFolder, "", postfix, rank);
    if (rank == 0) {
        Sylinder::writePVTP(baseFolder, postfix, size); // write parallel head
        conCollectorPtr->writePVTP(baseFolder, "", postfix, size);
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

void SylinderSystem::writeResult(const int stepCount, const std::string &baseFolder, const std::string &postfix) {
    writeAscii(stepCount, baseFolder, postfix);
    writeVTK(baseFolder, postfix);
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

// TODO: mobility class should allow for flags to decide what type of mobility matrix to use: diagonal or dense
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

        double dragPara = 0;
        double dragPerp = 0;
        double dragRot = 0;
        sy.calcDragCoeff(mu, dragPara, dragPerp, dragRot);
        const double dragParaInv = sy.isImmovable ? 0.0 : 1 / dragPara;
        const double dragPerpInv = sy.isImmovable ? 0.0 : 1 / dragPerp;
        const double dragRotInv = sy.isImmovable ? 0.0 : 1 / dragRot;

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
        Teuchos::rcp(new TCMAT(sylinderMobilityMapRcp, sylinderMobilityMapRcp, rowPointers, columnIndices, values));
    mobilityMatrixRcp->fillComplete(sylinderMobilityMapRcp, sylinderMobilityMapRcp); // domainMap, rangeMap

    spdlog::debug("MobMat Constructed " + mobilityMatrixRcp->description());
}

void SylinderSystem::calcMobOperator() {
    calcMobMatrix();
    mobilityOperatorRcp = mobilityMatrixRcp;
}

void SylinderSystem::sumForceVelocity() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        for (int k = 0; k < 3; k++) {
            sy.vel[k] = sy.velNonB[k] + sy.velBrown[k] + sy.velCon[k];
            sy.omega[k] = sy.omegaNonB[k] + sy.omegaBrown[k] + sy.omegaCon[k];
            sy.force[k] = sy.forceNonB[k] + sy.forceCon[k];
            sy.torque[k] = sy.torqueNonB[k] + sy.torqueCon[k];
        }
    }
}

void SylinderSystem::stepEuler(const int stepType) {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    const double dt = runConfig.dt;

    if (!runConfig.sylinderFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            sy.stepEuler(dt, stepType);
        }
    }
}

void SylinderSystem::resetConfiguration() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    if (!runConfig.sylinderFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            sy.resetConfiguration();
        }
    }
}

void SylinderSystem::advanceParticles() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    if (!runConfig.sylinderFixed) {
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
            auto &sy = sylinderContainer[i];
            sy.advance();
        }
    }
}

void SylinderSystem::collectConstraints() {

    Teuchos::RCP<Teuchos::Time> collectColTimer =
        Teuchos::TimeMonitor::getNewCounter("SylinderSystem::CollectCollision");
    Teuchos::RCP<Teuchos::Time> collectLinkTimer = Teuchos::TimeMonitor::getNewCounter("SylinderSystem::CollectLink");

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

void SylinderSystem::updateSylinderMap() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    // setup the new sylinderMap
    sylinderMapRcp = getTMAPFromLocalSize(nLocal, commRcp);
    sylinderMobilityMapRcp = getTMAPFromLocalSize(nLocal * 6, commRcp);
    sylinderStressMapRcp = getTMAPFromLocalSize(nLocal * 9, commRcp);

    // setup the globalIndex
    int globalIndexBase = sylinderMapRcp->getMinGlobalIndex(); // this is a contiguous map
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderContainer[i].globalIndex = i + globalIndexBase;
    }
}

void SylinderSystem::prepareStep(const int stepCount) {
    applyBoxBC();

    if (stepCount % 50 == 0) {
        decomposeDomain();
    }

    exchangeSylinder();

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        sy.clear();
        sy.radiusCollision = sylinderContainer[i].radius * runConfig.sylinderDiameterColRatio;
        sy.lengthCollision = sylinderContainer[i].length * runConfig.sylinderLengthColRatio;
        sy.rank = commRcp->getRank();
        sy.colBuf = runConfig.sylinderColBuf;
    }
}

void SylinderSystem::applyMonolayer() {
    const double monoZ = (runConfig.simBoxHigh[2] + runConfig.simBoxLow[2]) / 2;
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        auto &sy = sylinderContainer[i];
        sy.vel[2] = 0.0;
        sy.omega[2] = 0.0;
        sy.pos[2] = monoZ;
        Evec3 drt = Emapq(sy.orientation) * Evec3(0, 0, 1);
        drt[2] = 0;
        drt.normalize();
        Emapq(sy.orientation).setFromTwoVectors(Evec3(0, 0, 1), drt);
    }
}

void SylinderSystem::getForceVelocityNonConstraint(const Teuchos::RCP<TV> &forceNCRcp,
                                                   const Teuchos::RCP<TV> &velocityNCRcp) const {
    // store results
    auto forceNCPtr = forceNCRcp->getLocalView<Kokkos::HostSpace>();
    auto velocityNCPtr = velocityNCRcp->getLocalView<Kokkos::HostSpace>();
    forceNCRcp->modify<Kokkos::HostSpace>();
    velocityNCRcp->modify<Kokkos::HostSpace>();

    const int sylinderLocalNumber = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velocityNCPtr.extent(0) == sylinderLocalNumber * 6);
    TEUCHOS_ASSERT(velocityNCPtr.extent(1) == 1);

#pragma omp parallel for
    for (int i = 0; i < sylinderLocalNumber; i++) {
        auto &sy = sylinderContainer[i];
        velocityNCPtr(6 * i + 0, 0) = sy.velNonB[0] + sy.velBrown[0];
        velocityNCPtr(6 * i + 1, 0) = sy.velNonB[1] + sy.velBrown[1];
        velocityNCPtr(6 * i + 2, 0) = sy.velNonB[2] + sy.velBrown[2];
        velocityNCPtr(6 * i + 3, 0) = sy.omegaNonB[0] + sy.omegaBrown[0];
        velocityNCPtr(6 * i + 4, 0) = sy.omegaNonB[1] + sy.omegaBrown[1];
        velocityNCPtr(6 * i + 5, 0) = sy.omegaNonB[2] + sy.omegaBrown[2];

        forceNCPtr(6 * i + 0, 0) = sy.forceNonB[0];
        forceNCPtr(6 * i + 1, 0) = sy.forceNonB[1];
        forceNCPtr(6 * i + 2, 0) = sy.forceNonB[2];
        forceNCPtr(6 * i + 3, 0) = sy.torqueNonB[0];
        forceNCPtr(6 * i + 4, 0) = sy.torqueNonB[1];
        forceNCPtr(6 * i + 5, 0) = sy.torqueNonB[2];
    }
}

void SylinderSystem::saveForceVelocityConstraints(const Teuchos::RCP<const TV> &forceRcp,
                                                  const Teuchos::RCP<const TV> &velocityRcp) {
    // save results
    auto velPtr = velocityRcp->getLocalView<Kokkos::HostSpace>();
    auto forcePtr = forceRcp->getLocalView<Kokkos::HostSpace>();

    const int sylinderLocalNumber = sylinderContainer.getNumberOfParticleLocal();
    TEUCHOS_ASSERT(velPtr.extent(0) == sylinderLocalNumber * 6);
    TEUCHOS_ASSERT(velPtr.extent(1) == 1);

#pragma omp parallel for
    for (int i = 0; i < sylinderLocalNumber; i++) {
        auto &sy = sylinderContainer[i];
        sy.velCon[0] = velPtr(6 * i + 0, 0);
        sy.velCon[1] = velPtr(6 * i + 1, 0);
        sy.velCon[2] = velPtr(6 * i + 2, 0);
        sy.omegaCon[0] = velPtr(6 * i + 3, 0);
        sy.omegaCon[1] = velPtr(6 * i + 4, 0);
        sy.omegaCon[2] = velPtr(6 * i + 5, 0);

        sy.forceCon[0] = forcePtr(6 * i + 0, 0);
        sy.forceCon[1] = forcePtr(6 * i + 1, 0);
        sy.forceCon[2] = forcePtr(6 * i + 2, 0);
        sy.torqueCon[0] = forcePtr(6 * i + 3, 0);
        sy.torqueCon[1] = forcePtr(6 * i + 4, 0);
        sy.torqueCon[2] = forcePtr(6 * i + 5, 0);
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
            double dragPara = 0;
            double dragPerp = 0;
            double dragRot = 0;
            sy.calcDragCoeff(mu, dragPara, dragPerp, dragRot);
            const double dragParaInv = sy.isImmovable ? 0.0 : 1 / dragPara;
            const double dragPerpInv = sy.isImmovable ? 0.0 : 1 / dragPerp;
            const double dragRotInv = sy.isImmovable ? 0.0 : 1 / dragRot;

            // convert FDPS vec3 to Evec3
            Evec3 direction = Emapq(sy.orientation) * Evec3(0, 0, 1);

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

            Equatn orientRFD = Emapq(sy.orientation);
            EquatnHelper::rotateEquatn(orientRFD, Wrfdrot, delta);
            q = orientRFD * Evec3(0, 0, 1);
            Emat3 Nmatrfd = (dragParaInv - dragPerpInv) * (q * q.transpose()) + (dragPerpInv)*Emat3::Identity();

            Evec3 vel = kBTfactor * (Nmatsqrt * Wpos);           // Gaussian noise
            vel += (kBT / delta) * ((Nmatrfd - Nmat) * Wrfdpos); // rfd drift. seems no effect in this case
            Evec3 omega = sqrt(dragRotInv) * kBTfactor * Wrot;   // regularized identity rotation drag

            Emap3(sy.velBrown) = vel;
            Emap3(sy.omegaBrown) = omega;
        }
    }
}

// TODO: move to particle wall interaction class
void SylinderSystem::collectBoundaryCollision() {
    auto constraintPoolPtr = conCollectorPtr->constraintPoolPtr; // shared_ptr
    const int nThreads = constraintPoolPtr->size();
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();

    // process collisions with all boundaries
    for (const auto &bPtr : runConfig.boundaryPtr) {
#pragma omp parallel num_threads(nThreads)
        {
            const int threadId = omp_get_thread_num();
            auto &conQue = (*constraintPoolPtr)[threadId];
#pragma omp for
            for (int i = 0; i < nLocal; i++) {
                const auto &sy = sylinderContainer[i];
                const Evec3 center = ECmap3(sy.pos);

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
                                                sy.gid, sy.gid, //
                                                sy.globalIndex, //
                                                sy.globalIndex, //
                                                posI.data(),
                                                posI.data(),        // location of collision relative to particle center
                                                Query.data(), Proj, // location of collision in lab frame
                                                norm.data(),        // direction of collision force
                                                stressIJ.data(), true);
                        conQue.push_back(con);
                    } else if (deltanorm <
                               (1 + runConfig.sylinderColBuf * 2) * sy.radiusCollision) { // inside boundary but close
                        const double sep = deltanorm - radius;
                        Emat3 stressIJ = Emat3::Zero();
                        Constraint con;
                        noPenetrationConstraint(con,            // constraint object,
                                                sep,            // amount of overlap,
                                                sy.gid, sy.gid, //
                                                sy.globalIndex, //
                                                sy.globalIndex, //
                                                posI.data(),
                                                posI.data(),        // location of collision relative to particle center
                                                Query.data(), Proj, // location of collision in lab frame
                                                norm.data(),        // direction of collision force
                                                stressIJ.data(), true);
                        conQue.push_back(con);
                    }
                };

                if (sy.isSphere(true)) {
                    double radius = sy.lengthCollision * 0.5 + sy.radiusCollision;
                    checkEnd(center, radius);
                } else {
                    const Equatn orientation = ECmapq(sy.orientation);
                    const Evec3 direction = orientation * Evec3(0, 0, 1);
                    const double length = sy.lengthCollision;
                    const Evec3 Qm = center - direction * (length * 0.5);
                    const Evec3 Qp = center + direction * (length * 0.5);
                    checkEnd(Qm, sy.radiusCollision);
                    checkEnd(Qp, sy.radiusCollision);
                }
            }
        }
    }
    return;
}

std::pair<int, int> SylinderSystem::getMaxGid() {
    int maxGidLocal = 0;
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    for (int i = 0; i < nLocal; i++) {
        maxGidLocal = std::max(maxGidLocal, sylinderContainer[i].gid);
    }

    int maxGidGlobal = maxGidLocal;
    Teuchos::reduceAll(*commRcp, Teuchos::MaxValueReductionOp<int, int>(), 1, &maxGidLocal, &maxGidGlobal);
    spdlog::warn("rank: {}, maxGidLocal: {}, maxGidGlobal {}", commRcp->getRank(), maxGidLocal, maxGidGlobal);

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

void SylinderSystem::calcConStress() {
    if (runConfig.logLevel > spdlog::level::info)
        return;

    Emat3 sumConStress = Emat3::Zero();
    conCollectorPtr->sumLocalConstraintStress(sumConStress, false);

    // scale to nkBT
    const double scaleFactor = 1 / (sylinderMapRcp->getGlobalNumElements() * runConfig.KBT);
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

void SylinderSystem::calcOrderParameter() {
    if (runConfig.logLevel > spdlog::level::info)
        return;

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

    spdlog::info("RECORD: Order P,{:g},{:g},{:g},Q,{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g},{:g}", //
                 pQ[0], pQ[1], pQ[2],                                                             // pvec
                 pQ[3], pQ[4], pQ[5],                                                             // Qtensor
                 pQ[6], pQ[7], pQ[8],                                                             // Qtensor
                 pQ[9], pQ[10], pQ[11]                                                            // Qtensor
    );
}

std::vector<int> SylinderSystem::addNewSylinder(const std::vector<Sylinder> &newSylinder) {
    // assign unique new gid for sylinders on all ranks
    std::pair<int, int> maxGid = getMaxGid();
    const int maxGidLocal = maxGid.first;
    const int maxGidGlobal = maxGid.second;
    const int newCountLocal = newSylinder.size();

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
        Sylinder sy = newSylinder[i];
        sy.gid = newGidRecv[i];
        sylinderContainer.addOneParticle(sy);
    }

    return newGidRecv;
}

void SylinderSystem::addNewLink(const std::vector<Link> &newLink) {
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

void SylinderSystem::buildsylinderNearDataDirectory() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    auto &sylinderNearDataDirectory = *sylinderNearDataDirectoryPtr;
    sylinderNearDataDirectory.gidOnLocal.resize(nLocal);
    sylinderNearDataDirectory.dataOnLocal.resize(nLocal);
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        sylinderNearDataDirectory.gidOnLocal[i] = sylinderContainer[i].gid;
        sylinderNearDataDirectory.dataOnLocal[i].copyFromFP(sylinderContainer[i]);
    }

    // build index
    sylinderNearDataDirectory.buildIndex();
}

void SylinderSystem::updatesylinderNearDataDirectory() {
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    auto &sylinderNearDataDirectory = *sylinderNearDataDirectoryPtr;
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        assert(sylinderNearDataDirectory.gidOnLocal[i] == sylinderContainer[i].gid);
        sylinderNearDataDirectory.dataOnLocal[i].copyFromFP(sylinderContainer[i]);
    }

    // build index (must be called even though the index is unchanged. boooooo)
    sylinderNearDataDirectory.buildIndex();
}

// TODO: move to particle-particle interaction class
void SylinderSystem::collectPairCollision() {
    CalcSylinderNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.simBoxPBC, runConfig.simBoxLow,
                                     runConfig.simBoxHigh);

    TEUCHOS_ASSERT(treeSylinderNearPtr);
    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    setTreeSylinder();
    treeSylinderNearPtr->calcForceAll(calcColFtr, sylinderContainer, dinfo);
}

// TODO: Move this into particle link interaction class
void SylinderSystem::collectLinkBilateral() {
    // setup bilateral link constraints
    // uses special treatment for periodic boundary conditions

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
    auto &conPool = *(this->conCollectorPtr->constraintPoolPtr);
    if (conPool.size() != omp_get_max_threads()) {
        spdlog::critical("conPool multithread mismatch error");
        std::exit(1);
    }

    // fill the data to find
    auto &gidToFind = sylinderNearDataDirectoryPtr->gidToFind;
    const auto &dataToFind = sylinderNearDataDirectoryPtr->dataToFind;

    std::vector<int> gidDisp(nLocal + 1, 0);
    gidToFind.clear();
    gidToFind.reserve(nLocal);

    // loop over all sylinders
    // if linkMap[sy.gid] not empty, find info for all next
    for (int i = 0; i < nLocal; i++) {
        const auto &sy = sylinderContainer[i];
        const auto &range = linkMap.equal_range(sy.gid);
        int count = 0;
        for (auto it = range.first; it != range.second; it++) {
            gidToFind.push_back(it->second); // next
            count++;
        }
        gidDisp[i + 1] = gidDisp[i] + count; // number of links for each local Sylinder
    }

    sylinderNearDataDirectoryPtr->find();

#pragma omp parallel
    {
        const int threadId = omp_get_thread_num();
        auto &conQue = conPool[threadId];
#pragma omp for
        for (int i = 0; i < nLocal; i++) {
            const auto &syI = sylinderContainer[i]; // sylinder
            const int lb = gidDisp[i];
            const int ub = gidDisp[i + 1];

            for (int j = lb; j < ub; j++) {
                // create a spring constraint
                const auto &syJ = sylinderNearDataDirectoryPtr->dataToFind[j]; // sylinderNear

                const Evec3 &centerI = ECmap3(syI.pos);
                Evec3 centerJ = ECmap3(syJ.pos);
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
                // sylinders are not treated as spheres for bilateral constraints
                // constraint is always added between Pp and Qm
                // constraint target length is radiusI + radiusJ + runConfig.linkGap
                const Evec3 directionI = ECmapq(syI.orientation) * Evec3(0, 0, 1);
                const Evec3 directionJ = ECmap3(syJ.direction);
                const Evec3 Pp = centerI + directionI * (0.5 * syI.length); // plus end
                const Evec3 Qm = centerJ - directionJ * (0.5 * syJ.length);
                const Evec3 Ploc = Pp;
                const Evec3 Qloc = Qm;
                const Evec3 rvec = Qloc - Ploc;
                const double rnorm = rvec.norm();
                const Evec3 normI = (Ploc - Qloc).normalized();
                const Evec3 normJ = -normI;
                const Evec3 posI = Ploc - centerI;
                const Evec3 posJ = Qloc - centerJ;
                const double restLength = runConfig.linkGap;
                const double springLength = rnorm - syI.radius - syJ.radius;

                Emat3 stressIJ = Emat3::Zero();
                CalcSylinderNearForce::collideStress(directionI, directionJ, centerI, centerJ, syI.length, syJ.length,
                                                     syI.radius, syJ.radius, 1.0, Ploc, Qloc, stressIJ);

                Constraint con;
                springConstraint(con,                      // constraint object
                                 springLength, restLength, // length of spring, rest length of spring
                                 runConfig.linkKappa,      // spring constant
                                 syI.gid, syJ.gid,         //
                                 syI.globalIndex,          //
                                 syJ.globalIndex,          //
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
void SylinderSystem::updatePairCollision() {
    // TODO: extend this to include updating other types of constraints. Feting via GID should happen regardless.
    //       The only difference is the update step
    std::cout << "updatePairCollision doesn't account for systems with multiple types of constraints" << std::endl;

    // update the collision constraints stored in the constraintPool
    // uses special treatment for periodic boundary conditions

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
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
    auto &gidToFind = sylinderNearDataDirectoryPtr->gidToFind;
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

    sylinderNearDataDirectoryPtr->find();

    // Step 2. Update each constraint
    CalcSylinderNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.simBoxPBC, runConfig.simBoxLow,
                                     runConfig.simBoxHigh);

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        const int conIndexBase = threadOffset[threadId];
        for (int j = 0; j < conNum; j++) {
            const int idx = conIndexBase + 2 * j;
            const auto &syI = sylinderNearDataDirectoryPtr->dataToFind[idx + 0]; // sylinderNearI
            const auto &syJ = sylinderNearDataDirectoryPtr->dataToFind[idx + 1]; // sylinderNearJ
            calcColFtr.updateCollisionBlock(syI, syJ, conQue[j]);
        }
    }
}

// TODO: Move this into particle-particle interaction class
void SylinderSystem::collectUnresolvedConstraints() {
    // TODO: extend this function to use a factory for updating constraints

    // Loop over all constraints, check if the constraint is satisfies;
    // if not, generate a new constraint to handle the residual

    // update the constraints stored in the constraintPool
    // uses special treatment for periodic boundary conditions

    const int nLocal = sylinderContainer.getNumberOfParticleLocal();
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
    auto &gidToFind = sylinderNearDataDirectoryPtr->gidToFind;
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

    sylinderNearDataDirectoryPtr->find();

    // Step 2. check each collision pair for violation
    CalcSylinderNearForce calcColFtr(conCollectorPtr->constraintPoolPtr, runConfig.simBoxPBC, runConfig.simBoxLow,
                                     runConfig.simBoxHigh);

    // multi-thread filling. nThreads = poolSize, each thread process a queue
#pragma omp parallel for num_threads(nThreads)
    for (int threadId = 0; threadId < nThreads; threadId++) {
        auto &conQue = conPool[threadId];
        const int conNum = conQue.size();
        const int conIndexBase = threadOffset[threadId];
        for (int j = 0; j < conNum; j++) {
            const int idx = conIndexBase + 2 * j;
            const auto &syI = sylinderNearDataDirectoryPtr->dataToFind[idx + 0]; // sylinderNearI
            const auto &syJ = sylinderNearDataDirectoryPtr->dataToFind[idx + 1]; // sylinderNearJ

            // get an updated constraint between the two particles
            Constraint con;
            // TODO: replace the following with a factory based on the ID of the constraint. For now, this is ok.
            // TODO: add in ball joints and angular springs
            // TODO: how to account for nonconvex boundaries?
            // TODO: generalize this to multi-dof constraints
            if (conQue[j].oneSide) { // boundary collision
                // // check each boundary
                // for (const auto &bPtr : runConfig.boundaryPtr) {
                //     const Evec3 center = ECmap3(syI.pos);

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
                //                                     syI.gid, syI.gid, //
                //                                     syI.globalIndex, //
                //                                     syI.globalIndex, //
                //                                     posI.data(),
                //                                     posI.data(),        // location of collision relative to particle
                //                                     center Query.data(), Proj, // location of collision in lab frame
                //                                     norm.data(),        // direction of collision force
                //                                     stressIJ.data(), false);
                //         }
                //         return collision;
                //     };

                //     bool unsatisfied = false;
                //     if (syI.isSphere(true)) {
                //         double radius = syI.lengthCollision * 0.5 + syI.radiusCollision;
                //         const bool collision = checkEnd(center, radius);
                //         unsatisfied = collision;
                //     } else {
                //         const Evec3 direction = ECmap3(syI.direction);
                //         const double length = syI.lengthCollision;
                //         const Evec3 Qm = center - direction * (length * 0.5);
                //         const Evec3 Qp = center + direction * (length * 0.5);
                //         const bool collisionM = checkEnd(Qm, syI.radiusCollision);
                //         const bool collisionP = checkEnd(Qp, syI.radiusCollision);
                //         unsatisfied = (collisionM || collisionP);
                //     }

                //     // if the updated constaint is unsatisfied add it to the que
                //     if (unsatisfied) {
                //         conQue.push_back(con);
                //     }
                // }
            } else if (conQue[j].bilaterial) { // spring
                const Evec3 &centerI = ECmap3(syI.pos);
                Evec3 centerJ = ECmap3(syJ.pos);
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
                // sylinders are not treated as spheres for bilateral constraints
                // constraint is always added between Pp and Qm
                // constraint target length is radiusI + radiusJ + runConfig.linkGap
                const Evec3 directionI = ECmap3(syI.direction);
                const Evec3 directionJ = ECmap3(syJ.direction);

                const Evec3 Pp = centerI + directionI * (0.5 * syI.length); // plus end
                const Evec3 Qm = centerJ - directionJ * (0.5 * syJ.length);
                const Evec3 Ploc = Pp;
                const Evec3 Qloc = Qm;
                const Evec3 rvec = Qloc - Ploc;
                const double rnorm = rvec.norm();
                const Evec3 normI = (Ploc - Qloc).normalized();
                const Evec3 normJ = -normI;
                const Evec3 posI = Ploc - centerI;
                const Evec3 posJ = Qloc - centerJ;
                const double restLength = runConfig.linkGap;
                const double springLength = rnorm - syI.radius - syJ.radius;

                Emat3 stressIJ = Emat3::Zero();
                CalcSylinderNearForce::collideStress(directionI, directionJ, centerI, centerJ, syI.length, syJ.length,
                                                     syI.radius, syJ.radius, 1.0, Ploc, Qloc, stressIJ);

                springConstraint(con,                      // constraint object
                                 springLength, restLength, // length of spring, rest length of spring
                                 runConfig.linkKappa,      // spring constant
                                 syI.gid, syJ.gid,         //
                                 syI.globalIndex,          //
                                 syJ.globalIndex,          //
                                 posI.data(), posJ.data(), // location of collision relative to particle center
                                 Ploc.data(), Qloc.data(), // location of collision in lab frame
                                 normI.data(),             // direction of collision force
                                 stressIJ.data(), false);

                // if the updated constaint is unsatisfied add it to the que
                if (std::abs(rnorm - restLength) > runConfig.conResTol) {
                    conQue.push_back(con);
                }
            } else { // pairwise collision
                calcColFtr.updateCollisionBlock(syI, syJ, con);
                // if the updated constaint is unsatisfied add it to the que
                if (con.getSep(0) < -runConfig.conResTol) {
                    conQue.push_back(con);
                }
            }
        }
    }
}
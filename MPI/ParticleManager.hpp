/*
 Tempalte class to manage particles distributed on mpi ranks
*/

#ifndef PARTICLEMANAGER_HPP_
#define PARTICLEMANAGER_HPP_

#define PARTICLE_SIMULATOR_THREAD_PARALLEL
#define PARTICLE_SIMULATOR_MPI_PARALLEL
#include "FDPS/particle_simulator.hpp"

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

// internal data for each object, for use in FDPS
struct Neighbor {
    int species;
    int rank;
    int localIndex;
    int globalId;
    int pbcShift[3]; // may be the periodic image particle set by FDPS
    // the shift of neighbor relative to the target particle. each axis takes a value [-1,0,1].
    Neighbor() = default;
    Neighbor(int s_, int r_, int lI_, int gI_, double pbcShift_[3])
        : species(s_), rank(r_), localIndex(lI_), globalId(gI_) {
        for (int i = 0; i < 3; i++) {
            pbcShift[i] = pbcShift_[i];
        }
    }
};

struct Force {
    void clear() {}
};

struct Particle {
    // this is trivially_copyable
    int species = -1;      //
    int rank = -1;         // mpi rank owning this particle
    int localFPIndex = -1; // the local index in the container particle vector
    int globalId = -1;     // global unique ID
    double RSearch = 0;    // search radius
    PS::F64vec3 pos;       // coord supplied, not necessarily in the original box

    std::vector<Neighbor> *nbListPtr = nullptr; // invalidated during MPI transfer

  public:
    inline double getRSearch() const { return RSearch; }
    inline const PS::F64vec3 &getPos() const { return pos; }
    inline void setPos(const PS::F64vec3 &newPos) { pos = newPos; }

    void dump() const {
        printf("species %d, rank %d, localIndex %d, globalId %d, RSearch %f, pos %f,%f,%f \n", species, rank,
               localFPIndex, globalId, RSearch, pos[0], pos[1], pos[2]);
    }
};

class FindNeighbor {
  public:
    bool pbc[3];
    double boxLow[3];
    double boxHigh[3];
    double boxEdge[3];

    // constructor
    FindNeighbor(const bool pbc_[3], const double boxLow_[3], const double boxHigh_[3]) {
        for (int i = 0; i < 3; i++) {
            pbc[i] = pbc_[i];
            boxLow[i] = boxLow_[i];
            boxHigh[i] = boxHigh_[i];
            boxEdge[i] = boxHigh[i] - boxLow[i];
        }
    }

    // copy constructor
    FindNeighbor(const FindNeighbor &other) {
        for (int i = 0; i < 3; i++) {
            pbc[i] = other.pbc[i];
            boxLow[i] = other.boxLow[i];
            boxHigh[i] = other.boxHigh[i];
            boxEdge[i] = other.boxEdge[i];
        }
    }

    FindNeighbor(FindNeighbor &&other) {
        for (int i = 0; i < 3; i++) {
            pbc[i] = other.pbc[i];
            boxLow[i] = other.boxLow[i];
            boxHigh[i] = other.boxHigh[i];
            boxEdge[i] = other.boxEdge[i];
        }
    }

    void operator()(const Particle *const trgPtr, const PS::S32 Ntrg, const Particle *const srcPtr, const PS::S32 Nsrc,
                    Force *const nbCollectorPtr) {

        const int myThreadId = omp_get_thread_num();

        for (PS::S32 i = 0; i < Ntrg; ++i) {
            nbCollectorPtr[i].clear();
            auto &parTrg = trgPtr[i];
            auto &nbList = *(parTrg.nbListPtr);
            const auto &posT = parTrg.getPos();
            for (PS::S32 j = 0; j < Nsrc; ++j) {
                auto &parSrc = srcPtr[j];
                const auto &posS = parSrc.getPos();
                auto vecTS = posS - posT;
                double r2 = vecTS.x * vecTS.x + vecTS.y * vecTS.y + vecTS.z * vecTS.z;
                double r = sqrt(r2);
                // detemine pbc shift
                int pShift[3] = {0, 0, 0};
                for (int p = 0; p < 3; p++) {
                    if (!pbc[p]) { // non periodic, no need to fix
                        continue;
                    }
                    int pT = floor((posT[i] - boxLow[i]) / boxEdge[i]);
                    int pS = floor((posS[i] - boxLow[i]) / boxEdge[i]);
                    pShift[i] = pS - pT;
                }

                if (r < parTrg.getRSearch() + parSrc.getRSearch()) {
                    nbList.emplace_back(parSrc.species, parSrc.rank, parSrc.localFPIndex, parSrc.globalId, pShift);
                }
            }
        }
    }
};

/* ParticleManager Class
 * holds information for near neighbor interaction
 * User holds information for particle vector, TrgEP/SrcEP vector, and interactor
 * */
template <class... Species>
class ParticleManager {

  private:
    std::tuple<std::vector<Species> *...> particleVectorPointers;

    double boxLow[3] = {0, 0, 0};
    double boxHigh[3] = {1, 1, 1};
    bool pbc[3] = {false, false, false};

    // internal FDPS stuff
    using TreeType = PS::TreeForForceShort<Force, Particle, Particle>::Scatter;
    using SystemType = PS::ParticleSystem<Particle>;
    SystemType system;
    PS::DomainInfo dinfo;
    std::unique_ptr<TreeType> treeNeighborPtr;
    int numberParticleInTreeNeighbor = 0;

    // neighbor list
    std::vector<std::vector<Neighbor>> neighborList; // the list of neighbors for each particle in system

  public:
    ParticleManager(int argc, char **argv) {
        PS::Initialize(argc, argv);
        system.initialize();
        dinfo.initialize();
    };

    ~ParticleManager() { PS::Finalize(); };

    template <int N, class ParType>
    void initSpecies(std::vector<ParType> *particleVector); // init pointer in tuple

    template <int N>
    void showSpecies(); // show particles pointed in tuple

    template <int N>
    void addSpeciesToIndex(); //  put particle information to FDPS system

    template <int N>
    void adjustSpeciesToRootBox(); // reset pos into pbc box

    void clearIndex();

    template <int N>
    void partition();

    void buildIndex();

    void setBox(const bool pbc_[3], const double boxLow_[3], const double boxHigh_[3]);

    void dumpSystem();
};

template <class... Species>
template <int N>
void ParticleManager<Species...>::partition() {
    // TODO:
    system.adjustPositionIntoRootDomain(dinfo);
    dinfo.collectSampleParticle(system);
    dinfo.decomposeDomain();
    system.exchangeParticle(dinfo);
};

template <class... Species>
void ParticleManager<Species...>::buildIndex() {
    const int nParGlobal = system.getNumberOfParticleGlobal();
    // make a large enough tree
    if (!treeNeighborPtr || (nParGlobal > numberParticleInTreeNeighbor * 1.5)) {
        treeNeighborPtr = std::make_unique<TreeType>();
    }
    // build tree
    treeNeighborPtr->initialize(2 * nParGlobal, 0.7, 8, 64);
    numberParticleInTreeNeighbor = nParGlobal;

    // TODO:
    system.adjustPositionIntoRootDomain(dinfo);
};

template <class... Species>
void ParticleManager<Species...>::setBox(const bool pbc_[3], const double boxLow_[3], const double boxHigh_[3]) {
    const int pbcX = (pbc_[0] ? 1 : 0);
    const int pbcY = (pbc_[1] ? 1 : 0);
    const int pbcZ = (pbc_[2] ? 1 : 0);
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

    for (int i = 0; i < 3; i++) {
        boxLow[i] = boxLow_[i];
        boxHigh[i] = boxHigh_[i];
    }

    // rootdomain must be specified after PBC
    dinfo.setPosRootDomain(PS::F64vec3(boxLow[0], boxLow[1], boxLow[2]), //
                           PS::F64vec3(boxHigh[0], boxHigh[1], boxHigh[2]));
}

template <class... Species>
template <int N, class ParType>
void ParticleManager<Species...>::initSpecies(std::vector<ParType> *particleVector) {
    // setup pointer
    std::get<N>(particleVectorPointers) = particleVector;
}

template <class... Species>
template <int N>
void ParticleManager<Species...>::showSpecies() {
    auto &vec = *std::get<N>(particleVectorPointers);
    printf("Particle Kind %d\n", N);
    for (auto &par : vec) {
        auto gid = par.getGid();
        auto kind = N;
        auto RSearch = par.getRSearch();
        auto posPtr = par.getPos();
        printf("%8d,%8d,%6f,%12f,%12f,%12f\n", gid, kind, RSearch, posPtr[0], posPtr[1], posPtr[2]);
    }
}

template <class... Species>
void ParticleManager<Species...>::clearIndex() {
    system.setNumberOfParticleLocal(0);
}

template <class... Species>
template <int N>
void ParticleManager<Species...>::addSpeciesToIndex() {
    // Assumption: setNumberOfParticleLocal() does not invalidate the particles already in the index.
    const int rank = PS::Comm::getRank();
    const auto &vec = *std::get<N>(particleVectorPointers);
    const int nLocalAdd = vec.size();
    const int nLocalCurrent = system.getNumberOfParticleLocal();
    system.setNumberOfParticleLocal(nLocalCurrent + nLocalAdd);
#pragma omp parallel for
    for (int i = 0; i < nLocalAdd; i++) {
        system[nLocalCurrent + i].species = N;
        system[nLocalCurrent + i].rank = rank;
        system[nLocalCurrent + i].localIndex = i;
        system[nLocalCurrent + i].globalId = vec[i].getGid();
        system[nLocalCurrent + i].RSearch = vec[i].getRSearch();
        const double *fppos = vec[i].getPos();
        system[nLocalCurrent + i].pos[0] = fppos[0];
        system[nLocalCurrent + i].pos[1] = fppos[1];
        system[nLocalCurrent + i].pos[2] = fppos[2];
    }
}

template <class... Species>
template <int N>
void ParticleManager<Species...>::adjustSpeciesToRootBox() {
    const auto &vec = *std::get<N>(particleVectorPointers);
    const int nLocal = vec.size();
    const double boxEdge[3];
    for (int k = 0; k < 3; k++) {
        boxEdge[k] = boxHigh[k] - boxLow[k];
    }
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
        double pos[3] = {0, 0, 0};
        for (int p = 0; p < 3; p++) {
            pos[p] = vec[i].getPos[p];
            if (pbc[p]) {
                if (pos[p] < boxLow[p]) {
                }
            }
        }
    }
}

template <class... Species>
void ParticleManager<Species...>::dumpSystem() {
    printf("------------Dump FDPS particle system-----------\n");
    printf("rank %d, nProcs %d\n", PS::Comm::getRank(), PS::Comm::getNumberOfProc());
    printf("dinfo bc %d \n", dinfo.getBoundaryCondition());
    auto &box = dinfo.getPosRootDomain();
    printf("boxLow %f,%f,%f boxHigh %f,%f,%f\n", box.low_.x, box.low_.y, box.low_.z, box.high_.x, box.high_.y,
           box.high_.z);

    const int nLocal = system.getNumberOfParticleLocal();
    for (int i = 0; i < nLocal; i++) {
        system[i].dump();
    }
};

#endif

/**
 * A particle type should define the following (inline) member functions:
 * int getGid();
 * double getRSearch();
 * double* getPos();
 */

/**
 * A trgEP type should define the following (inline) member functions:
 * int getGid();
 * double* getPos();
 */

/**
 * Case 1, global particle numbers are not changing
 * Case 2, global particle numbers are changing
 */

// // update Essential Information for one interaction
// // pairList stores the index in srcEPVec for each target in trgEPVec
// // by default, the pairList is reused unless it is empty or size does not match trgEPVec
// // if maxGlobalRSearch=0, a mpi collective operation is called to find the maximum RSearch globally
// // trgEPVec matches exactly the order and information of locally owned particles
// // the first part of srcEPVec matches exactly the order and information of locally owned particles
// // the second part of srcEPVec contains particles received and unpacked from other ranks.
// template <int Trg, class TrgEP, int Src, class SrcEP>
// void buildInteractionEP(std::vector<std::vector<size_t>> &pairList, std::vector<TrgEP> &trgEPVec,
//                         std::vector<SrcEP> &srcEPVec, bool updateList = false, double maxGlobalRSearch = 0);

// // evaluate interaction with known interaction list
// // interactor has a function defined as
// // interact(TrgEP& trg, const std::vector<int> & srcEPIndex, const std::vector<SrcEP> & src)
// template <int Trg, class TrgEP, int Src, class SrcEP, class Interactor>
// void evaluateInteraction(std::vector<TrgEP> &trgEPVec, std::vector<SrcEP> &srcEPVec, Interactor &interactor);
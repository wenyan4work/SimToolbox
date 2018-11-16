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

/**
 * A particle type should define the following (inline) member functions:
 * int getGid();
 * int getKind();
 * double getRSearch();
 * double* getPos();
 */

/**
 * A trgEP type should define the following (inline) member functions:
 * int getGid();
 * int getKind();
 * double* getPos();
 */



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

  public:
    ParticleManager(int argc, char **argv) { PS::Initialize(argc, argv); }; // constructor
    ~ParticleManager() { PS::Finalize(); };                                 // destructor

    // init pointer in tuple
    template <int N, class ParType>
    void initKind(std::vector<ParType> *particleVector);

    // show particles pointed in tuple
    template <int N>
    void showKind();

    // update EPential Information for one interaction
    // pairList stores the index in srcEPVec for each target in trgEPVec
    // by default, the pairList is reused unless it is empty or size does not match trgEPVec
    // if maxGlobalRSearch=0, a mpi collective operation is called to find the maximum RSearch globally
    // trgEPVec matches exactly the order and information of locally owned particles
    // the first part of srcEPVec matches exactly the order and information of locally owned particles
    // the second part of srcEPVec contains particles received and unpacked from other ranks.
    template <int Trg, class TrgEP, int Src, class SrcEP>
    void buildInteractionEP(std::vector<std::vector<size_t>> &pairList, std::vector<TrgEP> &trgEPVec,
                            std::vector<SrcEP> &srcEPVec, bool updateList = false, double maxGlobalRSearch = 0);

    // evaluate interaction with known interaction list
    // interactor has a function defined as
    // interact(TrgEP& trg, const std::vector<int> & srcEPIndex, const std::vector<SrcEP> & src)
    template <int Trg, class TrgEP, int Src, class SrcEP, class Interactor>
    void evaluateInteraction(std::vector<TrgEP> &trgEPVec, std::vector<SrcEP> &srcEPVec, Interactor &interactor);

    // update the internal particle list according to the modified std::vector<ParType> *particleVector
    template <int N>
    void updateAddRemoveParticles();

  private:
    // internal data for each object, for use in FDPS
    struct ParData {
        // this is trivially_copyable
        int kind;        //
        int rank;        // mpi rank owning this particle
        int localIndex;  // the local index in the container particle vector
        size_t globalId; // global unique ID
        double RSearch;  // search radius
        double pos[3];   // coord supplied, not necessarily in the original box

      public:
        inline double getRSearch() { return RSearch; }
        inline const PS::F64vec3 &getPos() { return PS::F64vec3(pos[0], pos[1], pos[2]); }
        // inline PS::F64vec3 getPos() { return PS::F64vec3(pos[0], pos[1], pos[2]); }
        inline void setPos(const PS::F64vec3 &newPos) {
            pos[0] = newPos.x;
            pos[1] = newPos.y;
            pos[2] = newPos.z;
        }
    };

    class recordNeighbor {
      public:
        void operator()(/*TODO:*/){

        };
    };

    // internal FDPS stuff
    // PS::ParticleSystem<ParData> sysParData;
    // PS::DomainInfo domainInfo;
    // PS::TreeForForceShort<recordNeighbor, ParData, ParData> treeNeighbor;

    // MPI Comm stuff
    template <class ParType>
    void ForwardScatter(const std::vector<ParType> &inVec, std::vector<ParType> &outVec,
                        const std::vector<int> &recvRanks) const;

    template <class ParType>
    void ReverseScatter(const std::vector<ParType> &inVec, std::vector<ParType> &outVec,
                        const std::vector<int> &sendRanks) const;

    // Interaction List building stuff
};

template <class... Species>
template <int N, class ParType>
void ParticleManager<Species...>::initKind(std::vector<ParType> *particleVector) {
    // step 1, setup pointer
    std::get<N>(particleVectorPointers) = particleVector;
    // step 2, add kind N to the FDPS ParticleSystem
}

template <class... Species>
template <int N>
void ParticleManager<Species...>::showKind() {
    auto &vec = *std::get<N>(particleVectorPointers);
    printf("Particle Kind %d\n", N);
    for (auto &par : vec) {
        auto gid = par.getGid();
        auto kind = par.getKind();
        auto RSearch = par.getRSearch();
        auto posPtr = par.getPos();
        printf("%8d,%8d,%6f,%12f,%12f,%12f\n", gid, kind, RSearch, posPtr[0], posPtr[1], posPtr[2]);
    }
}

template <class... Species>
template <int Trg, class TrgEP, int Src, class SrcEP>
void ParticleManager<Species...>::buildInteractionEP(std::vector<std::vector<size_t>> &pairList,
                                                     std::vector<TrgEP> &trgEPVec, std::vector<SrcEP> &srcEPVec,
                                                     bool updateList, double maxGlobalRSearch) {
    return;
}

template <class... Species>
template <int Trg, class TrgEP, int Src, class SrcEP, class Interactor>
void ParticleManager<Species...>::evaluateInteraction(std::vector<TrgEP> &trgEPVec, std::vector<SrcEP> &srcEPVec,
                                                      Interactor &interactor) {
    return;
}

#endif
/*
 Tempalte class to manage particles distributed on mpi ranks
*/
#include "FDPS/particle_simulator.hpp"

#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

// traits, required interface to be implemented for each Species type

/* ParticleManager Class
 * holds information for near neighbor interaction
 * User holds information for particle vector, TrgEss/SrcEss vector, and interactor
 * */
template <class... Species>
class ParticleManager {
  private:
    std::tuple<std::vector<Species> *...> particleVectorPointers;

    double boxLow[3];
    double boxHigh[3];

  public:
    ParticleManager() = default;  // constructor
    ~ParticleManager() = default; // destructor

    // init pointer in tuple
    template <int N, class Kind>
    void initKind(std::vector<Kind> *particleVector);

    // show particles pointed in tuple
    template <int N>
    void showKind();

    // update Essential Information for one interaction
    // pairList stores the index in srcEssVec for each target in trgEssVec
    // by default, the pairList is reused unless it is empty or size does not match trgEssVec
    // trgEssVec matches exactly the order and information of locally owned particles
    // the first part of srcEssVec matches exactly the order and information of locally owned particles
    // the second part of srcEssVec contains particles received and unpacked from other ranks.
    template <int Trg, class TrgEss, int Src, class SrcEss>
    void buildInteractionEss(std::vector<std::vector<size_t>> &pairList, std::vector<TrgEss> &trgEssVec,
                             std::vector<SrcEss> &srcEssVec, bool updateList = false){};

    // evaluate interaction with known interaction list
    // interactor has a function defined as interact(TrgEss& trg, const SrcEss& src)
    template <int Trg, class TrgEss, int Src, class SrcEss, class Interactor>
    void evaluateInteraction(std::vector<TrgEss> &trgEssVec, std::vector<SrcEss> &srcEssVec, Interactor &interactor){};

  private:
    // internal data for each object, for use in FDPS
    struct ParData {
        int kind;        //
        int rank;        // mpi rank owning this particle
        size_t globalId; // global unique ID
        double RSearch;  // search radius
        double pos[3];   // coord supplied, not necessarily in the original box
    };
};

template <class... Species>
template <int N, class Kind>
void ParticleManager<Species...>::initKind(std::vector<Kind> *particleVector) {
    std::get<N>(particleVectorPointers) = particleVector;
}

template <class... Species>
template <int N>
void ParticleManager<Species...>::showKind() {
    auto &vec = *std::get<N>(particleVectorPointers);
    printf("Particle Kind %d\n", N);
    for (auto &par : vec) {
        auto gid = par.getGid();
        auto RSearch = par.getRSearch();
        printf("%8d:\t%6f\n", gid, RSearch);
    }
}
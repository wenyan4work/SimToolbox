/*
 Tempalte class to manage mixed types particle interactions in FDPS
*/

#ifndef MIXPAIRINTERACTION_HPP_
#define MIXPAIRINTERACTION_HPP_

#define PARTICLE_SIMULATOR_THREAD_PARALLEL
#define PARTICLE_SIMULATOR_MPI_PARALLEL
#include "FDPS/particle_simulator.hpp"

#include <memory>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

template <class FPT, class FPS, class EPT, class EPS, class Force>
class MixPairInteraction {

    static_assert(std::is_trivially_copyable<EPT>::value);
    static_assert(std::is_trivially_copyable<EPS>::value);
    static_assert(std::is_trivially_copyable<Force>::value);
    static_assert(std::is_default_constructible<EPT>::value);
    static_assert(std::is_default_constructible<EPS>::value);
    static_assert(std::is_default_constructible<Force>::value);

    template <class EPT, class EPS>
    struct MixFP {
        EPT epTrg;
        EPS epSrc;

        bool trgFlag;
        double RSearch;
        PS::F64vec3 pos;

        inline double getRSearch() const { return RSearch; }
        inline const PS::F64vec3 &getPos() const { return pos; }
        inline void setPos(const PS::F64vec3 &newPos) { pos = newPos; }
    };

    template <class EPT>
    struct MixEPI {
        EPT epTrg;
        bool trgFlag;

        template <class EPS>
        void CopyFromFP(const MixFP<EPT, EPS> &fp) {
            epTrg = fp.epTrg;
            trgFlag = fp.trgFlag;
        }
    };

    template <class EPS>
    struct MixEPJ {
        EPS epSrc;
        bool srcFlag;

        template <class EPT>
        void CopyFromFP(const MixFP<EPT, EPS> &fp) {
            epSrc = fp.epSrc;
            srcFlag = !fp.trgFlag;
        }
    };

    // hold information
    const PS::ParticleSystem<FPT> &systemTrg;
    const PS::ParticleSystem<FPS> &systemSrc;
    const PS::DomainInfo &dinfo;

    // internal FDPS stuff

    PS::ParticleSystem<MixFP<EPT, EPS>> systemMix;

    std::unique_ptr<typename PS::TreeForForceShort<Force, MixEPI<EPT>, MixEPJ<EPS>>::Scatter> treeNeighborPtr;
    int numberParticleInTreeNeighbor;

    // C-tor
    MixPairInteraction(PS::ParticleSystem<FPT> &systemTrg_, PS::ParticleSystem<FPS> &systemSrc_, PS::DomainInfo &dinfo_)
        : systemTrg(systemTrg_), systemSrc(systemSrc_), dinfo(dinfo_) {
        systemMix.initialize();
        numberParticleInTreeNeighbor = 0;
    };

    ~MixPairInteraction() = default;

    void buildSystemMix() {
        const int nLocalTrg = systemTrg.getNumberOfParticlesLocal();
        const int nLocalSrc = systemSrc.getNumberOfParticlesLocal();
        systemMix.setNumberOfParticlesLocal(nLocalTrg + nLocalSrc);
#pragma omp parallel for
        for (size_t i = 0; i < nLocalTrg; i++) {
            systemMix[i].epTrg.CopyFromFP(systemTrg[i]);
            systemMix[i].trgFlag = true;
            systemMix[i].RSearch = systemTrg[i].getRSearch();
            systemMix[i].pos = systemTrg[i].getPos();
        }
#pragma omp parallel for
        for (size_t i = 0; i < nLocalSrc; i++) {
            systemMix[i + nLocalTrg].epSrc.CopyFromFP(systemSrc[i]);
            systemMix[i].trgFlag = false;
            systemMix[i].RSearch = systemSRc[i].getRSearch();
            systemMix[i].pos = systemSrc[i].getPos();
        }
    }
};

#endif

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

// the field pos should always be valid
// epTrg/epSrc data is valid only when the corresponding flag is set true
// check before return

template <class EPT, class EPS, class Force>
struct MixFP {
    EPT epTrg; // results will be write back to this struct
    EPS epSrc;
    Force force;

    bool trgFlag; //
    double pos[3];

    PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }

    void copyFromForce(Force &f) { force = f; }
};

template <class EPT>
struct MixEPI {
    EPT epTrg;

    bool trgFlag;
    double pos[3];

    double getRSearch() const { return trgFlag ? epTrg.getRSearch() : 0; }

    PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }

    template <class EPS, class Force>
    void copyFromFP(const MixFP<EPT, EPS, Force> &fp) {
        trgFlag = fp.trgFlag;
        for (int i = 0; i < 3; i++) {
            pos[i] = fp.pos[i];
        }
        if (trgFlag) {
            epTrg = fp.epTrg;
        }
    }
};

template <class EPS>
struct MixEPJ {
    EPS epSrc;

    bool srcFlag;
    double pos[3];

    double getRSearch() const { return srcFlag ? epSrc.getRSearch() : 0; }

    PS::F64vec3 getPos() const { return PS::F64vec3(pos[0], pos[1], pos[2]); }
    void setPos(const PS::F64vec3 &newPos) {
        pos[0] = newPos.x;
        pos[1] = newPos.y;
        pos[2] = newPos.z;
    }

    template <class EPT, class Force>
    void copyFromFP(const MixFP<EPT, EPS, Force> &fp) {
        srcFlag = !fp.trgFlag;
        for (int i = 0; i < 3; i++) {
            pos[i] = fp.pos[i];
        }
        if (srcFlag) {
            epSrc = fp.epSrc;
        }
    }
};

// user should define something like this
// this is an example
template <class EPT, class EPS, class Force>
class CalcMixPairForce {

  public:
    void operator()(const MixEPI<EPT> *const trgPtr, const PS::S32 nTrg, const MixEPJ<EPS> *const srcPtr,
                    const PS::S32 nSrc, Force *const mixForcePtr) {
        for (int t = 0; t < nTrg; t++) {
            auto &trg = trgPtr[t];
            if (!trg.trgFlag) {
                continue;
            }
            for (int s = 0; s < nSrc; s++) {
                auto &src = srcPtr[s];
                if (!src.srcFlag) {
                    continue;
                }
                // actual interaction
                force(trg, src, mixForcePtr[t]);
            }
        }
    }
};

/**
 * This class does not exchange particles between mpi ranks
 *
 *
 *
 */
template <class FPT, class FPS, class EPT, class EPS, class Force>
class MixPairInteraction {

    static_assert(std::is_trivially_copyable<FPT>::value);
    static_assert(std::is_trivially_copyable<FPS>::value);
    static_assert(std::is_trivially_copyable<EPT>::value);
    static_assert(std::is_trivially_copyable<EPS>::value);
    static_assert(std::is_trivially_copyable<Force>::value);

    static_assert(std::is_default_constructible<FPT>::value);
    static_assert(std::is_default_constructible<FPS>::value);
    static_assert(std::is_default_constructible<EPT>::value);
    static_assert(std::is_default_constructible<EPS>::value);
    static_assert(std::is_default_constructible<Force>::value);

    // hold information
    const PS::ParticleSystem<FPT> &systemTrg;
    const PS::ParticleSystem<FPS> &systemSrc;
    PS::DomainInfo &dinfo;

    // result
    std::vector<Force> forceResult;

    // internal FDPS stuff
    using EPIType = MixEPI<EPT>;
    using EPJType = MixEPJ<EPS>;
    using FPType = MixFP<EPT, EPS, Force>;

    using SystemType = typename PS::ParticleSystem<FPType>;
    using TreeType = typename PS::TreeForForceShort<Force, EPIType, EPJType>::Scatter;
    // scatter mode, search radius determined by EPJ

    SystemType systemMix;
    std::unique_ptr<TreeType> treeMixPtr;
    int numberParticleInTree;

  public:
    // C-tor
    MixPairInteraction(PS::ParticleSystem<FPT> &systemTrg_, PS::ParticleSystem<FPS> &systemSrc_, PS::DomainInfo &dinfo_)
        : systemTrg(systemTrg_), systemSrc(systemSrc_), dinfo(dinfo_) {
        systemMix.initialize();
        numberParticleInTree = 0;
    };

    ~MixPairInteraction() = default;

    void updateSystem();

    void updateTree();

    template <class CalcMixForce>
    void computeForce(CalcMixForce &calcMixForceFtr);

    const std::vector<Force> &getForceResult() { return forceResult; }
};

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::updateSystem() {
    const int nLocalTrg = systemTrg.getNumberOfParticleLocal();
    const int nLocalSrc = systemSrc.getNumberOfParticleLocal();
    systemMix.setNumberOfParticleLocal(nLocalTrg + nLocalSrc);
#pragma omp parallel for
    for (size_t i = 0; i < nLocalTrg; i++) {
        systemMix[i].epTrg.copyFromFP(systemTrg[i]);
        systemMix[i].trgFlag = true;
        systemMix[i].setPos(systemTrg[i].getPos());
    }
#pragma omp parallel for
    for (size_t i = 0; i < nLocalSrc; i++) {
        systemMix[i + nLocalTrg].epSrc.copyFromFP(systemSrc[i]);
        systemMix[i].trgFlag = false;
        systemMix[i].setPos(systemSrc[i].getPos());
    }
    systemMix.adjustPositionIntoRootDomain(dinfo);
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::updateTree() {
    const int nParGlobal = systemMix.getNumberOfParticleGlobal();
    // make a large enough tree
    if (!treeMixPtr || (nParGlobal > numberParticleInTree * 1.5)) {
        treeMixPtr = std::make_unique<TreeType>();
    }
    // build tree
    treeMixPtr->initialize(2 * nParGlobal, 0.7, 32, 64);
    numberParticleInTree = nParGlobal;
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
template <class CalcMixForce>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::computeForce(CalcMixForce &calcMixForceFtr) {
    treeMixPtr->calcForceAllAndWriteBack(calcMixForceFtr, systemMix, dinfo);

    forceResult.resize(systemTrg.getNumberOfParticleLocal());
    const int nTrg = forceResult.size();
#pragma omp parallel for
    for (int i = 0; i < nTrg; i++) {
        forceResult[i] = systemMix[i].force;
    }
}

#endif

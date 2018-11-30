/*
 Tempalte class to manage mixed types particle interactions in FDPS
*/

#ifndef MIXPAIRINTERACTION_HPP_
#define MIXPAIRINTERACTION_HPP_

#include "FDPS/particle_simulator.hpp"

#include <memory>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

// the field pos should always be valid
// epTrg/epSrc data is valid only when the corresponding flag is set true
// check before return

template <class EPT, class EPS>
struct MixFP {
    bool trgFlag;
    double maxRSearch;
    EPT epTrg;
    EPS epSrc;

    PS::F64vec3 getPos() const { return trgFlag ? epTrg.getPos() : epSrc.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { trgFlag ? epTrg.setPos(newPos) : epSrc.setPos(newPos); }

    int getGid() { return trgFlag ? epTrg.getGid() : epSrc.getGid(); }
    // void copyFromForce(Force &f) { force = f; }
};

template <class EPT>
struct MixEPI {
    bool trgFlag;
    EPT epTrg;

    PS::F64vec3 getPos() const { return epTrg.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { epTrg.setPos(newPos); }

    template <class EPS>
    void copyFromFP(const MixFP<EPT, EPS> &fp) {
        trgFlag = fp.trgFlag;
        if (trgFlag) {
            epTrg = fp.epTrg;
        }
        setPos(fp.getPos());
    }
};

template <class EPS>
struct MixEPJ {
    bool srcFlag;
    double maxRSearch; // same for all EPI and EPJ
    EPS epSrc;

    double getRSearch() const { return maxRSearch; }

    PS::F64vec3 getPos() const { return epSrc.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { epSrc.setPos(newPos); }

    template <class EPT>
    void copyFromFP(const MixFP<EPT, EPS> &fp) {
        srcFlag = !fp.trgFlag;
        maxRSearch = fp.maxRSearch;
        if (srcFlag) {
            epSrc = fp.epSrc;
        }
        epSrc.setPos(fp.getPos());
    }
};

// user should define something like this
// this is an example
template <class EPT, class EPS, class Force>
class CalcMixPairForceExample {

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

    static_assert(std::is_trivially_copyable<FPT>::value, "FPT is potentially unsafe for memcpy\n");
    static_assert(std::is_trivially_copyable<FPS>::value, "FPS is potentially unsafe for memcpy\n");
    static_assert(std::is_trivially_copyable<EPT>::value, "EPT is potentially unsafe for memcpy\n");
    static_assert(std::is_trivially_copyable<EPS>::value, "EPS is potentially unsafe for memcpy\n");
    static_assert(std::is_trivially_copyable<Force>::value, "Force is potentially unsafe for memcpy\n");

    static_assert(std::is_default_constructible<FPT>::value, "FPT is not defalut constructible\n");
    static_assert(std::is_default_constructible<FPS>::value, "FPS is not defalut constructible\n");
    static_assert(std::is_default_constructible<EPT>::value, "EPT is not defalut constructible\n");
    static_assert(std::is_default_constructible<EPS>::value, "EPS is not defalut constructible\n");
    static_assert(std::is_default_constructible<Force>::value, "Force is not defalut constructible\n");

    // hold information
    const PS::ParticleSystem<FPT> &systemTrg;
    const PS::ParticleSystem<FPS> &systemSrc;
    PS::DomainInfo &dinfo;

    // result
    std::vector<Force> forceResult;

    // internal FDPS stuff
    using EPIType = MixEPI<EPT>;
    using EPJType = MixEPJ<EPS>;
    using FPType = MixFP<EPT, EPS>;

    static_assert(std::is_trivially_copyable<EPIType>::value, "");
    static_assert(std::is_trivially_copyable<EPJType>::value, "");
    static_assert(std::is_trivially_copyable<FPType>::value, "");
    static_assert(std::is_default_constructible<EPIType>::value, "");
    static_assert(std::is_default_constructible<EPJType>::value, "");
    static_assert(std::is_default_constructible<FPType>::value, "");

    using SystemType = typename PS::ParticleSystem<FPType>;
    using TreeType = typename PS::TreeForForceShort<Force, EPIType, EPJType>::Scatter;
    // gather mode, search radius determined by EPI
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

    void setMaxRSearch();

    void dumpSystem();

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
        const int mixIndex = i + nLocalTrg;
        systemMix[mixIndex].epSrc.copyFromFP(systemSrc[i]);
        systemMix[mixIndex].trgFlag = false;
        systemMix[mixIndex].setPos(systemSrc[i].getPos());
    }

    systemMix.adjustPositionIntoRootDomain(dinfo);
    setMaxRSearch();
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::dumpSystem() {
    const int nLocalMix = systemMix.getNumberOfParticleLocal();

    for (int i = 0; i < nLocalMix; i++) {
        const auto &pos = systemMix[i].getPos();
        printf("%d,%d,%lf,%lf,%lf\n", systemMix[i].trgFlag, systemMix[i].getGid(), pos.x, pos.y, pos.z);
    }
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::setMaxRSearch() {
    const int nLocalTrg = systemTrg.getNumberOfParticleLocal();
    const int nLocalSrc = systemSrc.getNumberOfParticleLocal();
    const int nLocalMix = systemMix.getNumberOfParticleLocal();

    double maxRSearchTrg = 0;
    double maxRSearchSrc = 0;

#pragma omp parallel for reduction(max : maxRSearchTrg)
    for (int i = 0; i < nLocalTrg; i++) {
        maxRSearchTrg = std::max(maxRSearchTrg, systemTrg[i].getRSearch());
    }
#pragma omp parallel for reduction(max : maxRSearchSrc)
    for (int i = 0; i < nLocalSrc; i++) {
        maxRSearchSrc = std::max(maxRSearchSrc, systemSrc[i].getRSearch());
    }

    double maxRSearchAll[2] = {maxRSearchTrg, maxRSearchSrc};
    MPI_Allreduce(MPI_IN_PLACE, maxRSearchAll, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double maxRSearch = maxRSearchAll[0] + maxRSearchAll[1];

#ifndef DNDEBUG
    printf("maxRSearch %lf\n", maxRSearch);
#endif

#pragma omp parallel for
    for (int i = 0; i < nLocalMix; i++) {
        systemMix[i].maxRSearch = maxRSearch;
    }
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::updateTree() {
    const int nParGlobal = systemMix.getNumberOfParticleGlobal();
    // make a large enough tree
    if (!treeMixPtr || (nParGlobal > numberParticleInTree * 1.5)) {
        treeMixPtr = std::make_unique<TreeType>();
    }
    // build tree
    // be careful if tuning the tree default parameters
    treeMixPtr->initialize(2 * nParGlobal);
    numberParticleInTree = nParGlobal;
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
template <class CalcMixForce>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::computeForce(CalcMixForce &calcMixForceFtr) {
#ifndef DNDEBUG
    dumpSystem();
#endif
    treeMixPtr->calcForceAll(calcMixForceFtr, systemMix, dinfo);

    forceResult.resize(systemTrg.getNumberOfParticleLocal());
    const int nTrg = forceResult.size();
#pragma omp parallel for
    for (int i = 0; i < nTrg; i++) {
        forceResult[i] = treeMixPtr->getForce(i);
    }
}

#endif

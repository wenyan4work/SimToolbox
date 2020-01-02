/**
 * @file MixPairInteraction.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief interaction for two FDPS systems of different particle types
 * @version 0.1
 * @date 2019-01-10
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef MIXPAIRINTERACTION_HPP_
#define MIXPAIRINTERACTION_HPP_

#include "FDPS/particle_simulator.hpp"
#include "Util/GeoCommon.h"

#include <memory>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <omp.h>

/**
 * Note about RSearch:
 * getRSearch() not needed for FP type
 * getRSearch() necessary for EPI type if Gather
 * getRSearch() necessary for EPJ type if Scatter
 * getRSearch() necessary for BOTH EPI and EPJ types if Symmetry
 * getRSearch() should not return zero in any cases
 *
 */

/**
 * @brief Mix Full Particle type.
 * One MixFP is created at every location of src and trg systems.
 * number of MixFP = number of src + number of trg
 * the actual behavior of one MixFP is controlled by the trgFlag switch
 * the field pos should always be valid
 * epTrg/epSrc data is valid only when the corresponding flag is set true
 * check before return
 * @tparam EPT
 * @tparam EPS
 */
template <class EPT, class EPS>
struct MixFP {
    bool trgFlag; ///< switch. if true, this MixFP represents a trg particle.
    double maxRSearch;
    EPT epTrg; ///< data for trg type. valid if trgFlag==true
    EPS epSrc; ///< data for src type. valid if trgFlag==false

    /**
     * @brief Get position.
     * required interface for FDPS
     *
     * @return PS::F64vec3
     */
    PS::F64vec3 getPos() const { return trgFlag ? epTrg.getPos() : epSrc.getPos(); }
    /**
     * @brief Set position
     * required interface for FDPS
     *
     * @param newPos
     */
    void setPos(const PS::F64vec3 &newPos) { trgFlag ? epTrg.setPos(newPos) : epSrc.setPos(newPos); }

    int getGid() const { return trgFlag ? epTrg.getGid() : epSrc.getGid(); }
    // void copyFromForce(Force &f) { force = f; }

    PS::F64 getRSearch() const { return trgFlag ? epTrg.getRSearch() : epSrc.getRSearch(); }
};

/**
 * @brief EssentialParticleI (trg) type for use in FDPS tree
 *  contains only target data. invalid and unsed if trgFlag==false
 * @tparam EPT
 */
template <class EPT>
struct MixEPI {
    bool trgFlag;
    double maxRSearch;
    EPT epTrg;

    double getRSearch() const { return trgFlag ? epTrg.getRSearch() : maxRSearch * 0.5; }
    int getGid() const { return trgFlag ? epTrg.getGid() : GEO_INVALID_INDEX; }

    PS::F64vec3 getPos() const { return epTrg.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { epTrg.setPos(newPos); }

    template <class EPS>
    void copyFromFP(const MixFP<EPT, EPS> &fp) {
        trgFlag = fp.trgFlag;
        maxRSearch = fp.maxRSearch;
        if (trgFlag) {
            epTrg = fp.epTrg;
        } else {
            // if not a trg particle, pos should be always valid
            setPos(fp.getPos());
        }
    }
};

/**
 * @brief EssentialParticleJ (src) type for use in FDPS tree
 *  contains only src data. invalid and unsed if srcFlag==false
 * @tparam EPS
 */
template <class EPS>
struct MixEPJ {
    bool srcFlag;
    double maxRSearch;
    EPS epSrc;

    double getRSearch() const { return srcFlag ? epSrc.getRSearch() : maxRSearch * 0.5; }
    int getGid() const { return srcFlag ? epSrc.getGid() : GEO_INVALID_INDEX; }

    PS::F64vec3 getPos() const { return epSrc.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { epSrc.setPos(newPos); }

    template <class EPT>
    void copyFromFP(const MixFP<EPT, EPS> &fp) {
        srcFlag = !fp.trgFlag;
        maxRSearch = fp.maxRSearch;
        if (srcFlag) {
            epSrc = fp.epSrc;
        } else {
            // if not a src particle, pos should be always valid
            setPos(fp.getPos());
        }
    }
};

/**
 * @brief An example interaction class to be passed to computeForce() function
 *
 * @tparam EPT
 * @tparam EPS
 * @tparam Force
 */
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
 * @brief interaction between two FDPS systems of different particle types
 * This class does not exchange particles between mpi ranks
 *
 * @tparam FPT
 * @tparam FPS
 * @tparam EPT
 * @tparam EPS
 * @tparam Force
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

    // result
    std::vector<Force> forceResult; ///< computed force result

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
    // symmetry mode, search radius determined by larger of EPI and EPJ

    SystemType systemMix;                 ///< mixed FDPS system
    std::unique_ptr<TreeType> treeMixPtr; ///< tree for pair interaction
    int numberParticleInTree;             ///< number of particles in tree
    int nLocalTrg, nLocalSrc;

  public:
    /**
     * @brief Construct a new MixPairInteraction object
     *
     */
    MixPairInteraction() = default;

    /**
     * @brief initializer
     * must be called once after constructor
     *
     * @param systemTrg_
     * @param systemSrc_
     * @param dinfo_
     */
    void initialize() {
        // systemTrgPtr = systemTrgPtr_;
        // systemSrcPtr = systemSrcPtr_;
        // dinfoPtr = dinfoPtr_;
        systemMix.initialize();
        numberParticleInTree = 0;
    };

    /**
     * @brief Destroy the MixPairInteraction object
     *
     */
    ~MixPairInteraction() = default;

    /**
     * @brief update systemMix
     *
     */
    void updateSystem(const PS::ParticleSystem<FPT> &systemTrg, const PS::ParticleSystem<FPS> &systemSrc,
                      PS::DomainInfo &dinfo);

    /**
     * @brief update treeMix
     *
     */
    void updateTree();

    /**
     * @brief Set the MaxRSearch looping over all obj in systemMix
     *
     */
    void setMaxRSearch();

    /**
     * @brief print the particles in systemMix
     * for debug use
     *
     */
    void dumpSystem();

    /**
     * @brief calculate interaction with CalcMixForce object
     *
     * @tparam CalcMixForce
     * @param calcMixForceFtr
     */
    template <class CalcMixForce>
    void computeForce(CalcMixForce &calcMixForceFtr, PS::DomainInfo &dinfo);

    /**
     * @brief Get the calculated force
     *
     * @return const std::vector<Force>&
     */
    const std::vector<Force> &getForceResult() { return forceResult; }
};

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::updateSystem(const PS::ParticleSystem<FPT> &systemTrg,
                                                                 const PS::ParticleSystem<FPS> &systemSrc,
                                                                 PS::DomainInfo &dinfo) {
    nLocalTrg = systemTrg.getNumberOfParticleLocal();
    nLocalSrc = systemSrc.getNumberOfParticleLocal();

    // fill systemMix
    systemMix.setNumberOfParticleLocal(nLocalTrg + nLocalSrc);
#pragma omp parallel for
    for (size_t i = 0; i < nLocalTrg; i++) {
        systemMix[i].epTrg.copyFromFP(systemTrg[i]);
        systemMix[i].trgFlag = true;
        // systemMix[i].setPos(systemTrg[i].getPos());
    }
#pragma omp parallel for
    for (size_t i = 0; i < nLocalSrc; i++) {
        const int mixIndex = i + nLocalTrg;
        systemMix[mixIndex].epSrc.copyFromFP(systemSrc[i]);
        systemMix[mixIndex].trgFlag = false;
        // systemMix[mixIndex].setPos(systemSrc[i].getPos());
    }

    systemMix.adjustPositionIntoRootDomain(dinfo);

    // set maxRSearch
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
    const int nLocalMix = systemMix.getNumberOfParticleLocal();
    double maxRSearchMix = 0;

#pragma omp parallel for reduction(max : maxRSearchMix)
    for (int i = 0; i < nLocalMix; i++) {
        maxRSearchMix = std::max(maxRSearchMix, systemMix[i].getRSearch());
    }

    MPI_Allreduce(MPI_IN_PLACE, &maxRSearchMix, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // printf("Global maxRSearch %lf\n", maxRSearchMix);

#pragma omp parallel for
    for (int i = 0; i < nLocalMix; i++) {
        systemMix[i].maxRSearch = maxRSearchMix;
    }
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::updateTree() {
    const int nParGlobal = systemMix.getNumberOfParticleGlobal();
    // make a large enough tree
    if (!treeMixPtr || (nParGlobal > numberParticleInTree * 1.5)) {
        treeMixPtr.reset();
        treeMixPtr = std::make_unique<TreeType>();
        // build tree
        // be careful if tuning the tree default parameters
        treeMixPtr->initialize(2 * nParGlobal);
        numberParticleInTree = nParGlobal;
    }
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
template <class CalcMixForce>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::computeForce(CalcMixForce &calcMixForceFtr, PS::DomainInfo &dinfo) {
    // dumpSystem();
    treeMixPtr->calcForceAll(calcMixForceFtr, systemMix, dinfo);

    forceResult.resize(nLocalTrg);
#pragma omp parallel for
    for (int i = 0; i < nLocalTrg; i++) {
        forceResult[i] = treeMixPtr->getForce(i);
    }
}

#endif

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
 * EPI and EPJ created from the same FP should return the same getRSearch()
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
    EPT epTrg;    ///< data for trg type. valid if trgFlag==true
    EPS epSrc;    ///< data for src type. valid if trgFlag==false

    double getRadius() const { return trgFlag ? epTrg.getRSearch() : epSrc.getRSearch(); }

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
};

/**
 * @brief EssentialParticleI (trg) type for use in FDPS tree
 *  contains only target data. invalid and unused if trgFlag==false
 * @tparam EPT
 */
template <class EPT>
struct MixEPI {
    bool trgFlag;
    double radius;
    EPT epTrg;

    double getRSearch() const { return radius; }

    PS::F64vec3 getPos() const { return epTrg.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { epTrg.setPos(newPos); }

    template <class EPS>
    void copyFromFP(const MixFP<EPT, EPS> &fp) {
        trgFlag = fp.trgFlag;
        radius = fp.getRadius();
        if (trgFlag) {
            epTrg = fp.epTrg;
        } else {
            // pos should be always valid even if not a trg particle
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
    double radius;
    EPS epSrc;

    double getRSearch() const { return radius; }

    PS::F64vec3 getPos() const { return epSrc.getPos(); }
    void setPos(const PS::F64vec3 &newPos) { epSrc.setPos(newPos); }

    template <class EPT>
    void copyFromFP(const MixFP<EPT, EPS> &fp) {
        srcFlag = !fp.trgFlag;
        radius = fp.getRadius();
        if (srcFlag) {
            epSrc = fp.epSrc;
        } else {
            // pos should be always valid even if not a src particle
            setPos(fp.getPos());
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

    // internal FDPS stuff
    using EPIType = MixEPI<EPT>;
    using EPJType = MixEPJ<EPS>;
    using FPType = MixFP<EPT, EPS>;
    using SystemType = typename PS::ParticleSystem<FPType>;
    using TreeType = typename PS::TreeForForceShort<Force, EPIType, EPJType>::Symmetry;
    // gather mode, search radius determined by EPI
    // scatter mode, search radius determined by EPJ
    // symmetry mode, search radius determined by larger of EPI and EPJ

    static_assert(std::is_trivially_copyable<EPIType>::value, "");
    static_assert(std::is_trivially_copyable<EPJType>::value, "");
    static_assert(std::is_trivially_copyable<FPType>::value, "");
    static_assert(std::is_default_constructible<EPIType>::value, "");
    static_assert(std::is_default_constructible<EPJType>::value, "");
    static_assert(std::is_default_constructible<FPType>::value, "");

    SystemType systemMix;                 ///< mixed FDPS system
    std::unique_ptr<TreeType> treeMixPtr; ///< tree for pair interaction
    int numberParticleInTree;             ///< number of particles in tree
    int nLocalTrg, nLocalSrc;

    std::vector<Force> forceResult; ///< computed force result

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
    }
#pragma omp parallel for
    for (size_t i = 0; i < nLocalSrc; i++) {
        const int mixIndex = i + nLocalTrg;
        systemMix[mixIndex].epSrc.copyFromFP(systemSrc[i]);
        systemMix[mixIndex].trgFlag = false;
    }

    systemMix.exchangeParticle(dinfo);
    systemMix.adjustPositionIntoRootDomain(dinfo);
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::dumpSystem() {
    const int nLocalMix = systemMix.getNumberOfParticleLocal();

    for (int i = 0; i < nLocalMix; i++) {
        const auto &pos = systemMix[i].getPos();
        const auto r = systemMix[i].trgFlag ? //
                           systemMix[i].epTrg.getRSearch()
                                            : //
                           systemMix[i].epSrc.getRSearch();
        printf("%d,%g,%g,%g,%g\n", systemMix[i].trgFlag, pos.x, pos.y, pos.z, r);
    }
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::updateTree() {
    const int nParGlobal = systemMix.getNumberOfParticleGlobal();
    bool newTree = PS::Comm::synchronizeConditionalBranchAND(!treeMixPtr || (nParGlobal > numberParticleInTree * 1.5));
    // make a large enough tree
    if (newTree) {
        treeMixPtr.reset();
        treeMixPtr = std::make_unique<TreeType>();
        // build tree
        // be careful if tuning the tree default parameters
        // treeMixPtr->initialize(4 * nParGlobal, 1.0, 32, 128);
        treeMixPtr->initialize(2 * nParGlobal);
        numberParticleInTree = nParGlobal;
    }
    PS::Comm::barrier();
}

template <class FPT, class FPS, class EPT, class EPS, class Force>
template <class CalcMixForce>
void MixPairInteraction<FPT, FPS, EPT, EPS, Force>::computeForce(CalcMixForce &calcMixForceFtr, PS::DomainInfo &dinfo) {
    treeMixPtr->calcForceAll(calcMixForceFtr, systemMix, dinfo);

    forceResult.resize(nLocalTrg);
#pragma omp parallel for
    for (int i = 0; i < nLocalTrg; i++) {
        forceResult[i] = treeMixPtr->getForce(i);
    }
}

#endif

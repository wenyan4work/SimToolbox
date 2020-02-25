/**
 * @file ConstraintBlock.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief
 * @version 0.1
 * @date 2019-11-04
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef CONSTRAINTBLOCK_HPP_
#define CONSTRAINTBLOCK_HPP_

#include "Util/EigenDef.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

/**
 * @brief collision constraint information block
 *
 * Each block stores the information for one collision constraint.
 * The blocks are collected by ConstraintCollector and then used to construct the sparse fcTrans matrix
 */
struct ConstraintBlock {
  public:
    double delta0 = 0;                    ///< constraint initial value
    double gamma = 0;                     ///< force magnitude, could be an initial guess
    double gammaLB = 0;                   ///< lower bound of gamma for unilateral constraints
    int gidI = GEO_INVALID_INDEX;         ///< unique global ID of particle I
    int gidJ = GEO_INVALID_INDEX;         ///< unique global ID of particle J
    int globalIndexI = GEO_INVALID_INDEX; ///< global index of particle I
    int globalIndexJ = GEO_INVALID_INDEX; ///< global index of particle J
    bool oneSide = false;                 ///< flag for one side constraint. body J does not appear in mobility matrix
    bool bilateral = false;               ///< if this is a bilateral constraint or not
    double kappa = 0;                     ///< spring constant. =0 means no spring
    double normI[3] = {0, 0, 0};
    double normJ[3] = {0, 0, 0}; ///< surface norm vector at the location of constraints (minimal separation).
    double posI[3] = {0, 0, 0};
    double posJ[3] = {0, 0, 0}; ///< the relative constraint position on bodies I and J.
    double labI[3] = {0, 0, 0};
    double labJ[3] = {0, 0, 0}; ///< the labframe location of collision points endI and endJ
    double stress[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    ///< stress 3x3 matrix (row-major) for unit constraint force gamma

    /**
     * @brief Construct a new empty collision block
     *
     */
    ConstraintBlock() = default;

    /**
     * @brief Construct a new ConstraintBlock object
     *
     * @param delta0_ current value of the constraint function
     * @param gamma_
     * @param gidI_
     * @param gidJ_
     * @param globalIndexI_
     * @param globalIndexJ_
     * @param normI_
     * @param normJ_
     * @param posI_
     * @param posJ_
     * @param labI_
     * @param labJ_
     * @param oneSide_ flag for one side constarint
     * @param bilateral_ flag for bilateral constraint
     * @param kappa_ flag for kappa of bilateral constraint
     * @param gammaLB_ lower bound of gamma for unilateral constraints
     */
    ConstraintBlock(double delta0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                    const double normI_[3], const double normJ_[3], const double posI_[3], const double posJ_[3],
                    const double labI_[3], const double labJ_[3], bool oneSide_, bool bilateral_, double kappa_,
                    double gammaLB_)
        : delta0(delta0_), gamma(gamma_), gidI(gidI_), gidJ(gidJ_), globalIndexI(globalIndexI_),
          globalIndexJ(globalIndexJ_), oneSide(oneSide_), bilateral(bilateral_), kappa(kappa_), gammaLB(gammaLB_) {
        for (int d = 0; d < 3; d++) {
            normI[d] = normI_[d];
            normJ[d] = normJ_[d];
            posI[d] = posI_[d];
            posJ[d] = posJ_[d];
            labI[d] = labI_[d];
            labJ[d] = labJ_[d];
        }
        std::fill(stress, stress + 9, 0);
    }

    void setStress(const Emat3 &stress_) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress[i * 3 + j] = stress_(i, j);
            }
        }
    }

    void setStress(const double *stress_) {
        for (int i = 0; i < 9; i++) {
            stress[i] = stress_[i];
        }
    }

    const double *getStress() const { return stress; }

    void getStress(Emat3 &stress_) const {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress_(i, j) = stress[i * 3 + j];
            }
        }
    }
};

static_assert(std::is_trivially_copyable<ConstraintBlock>::value, "");
static_assert(std::is_default_constructible<ConstraintBlock>::value, "");

using ConstraintBlockQue = std::vector<ConstraintBlock>;     ///< a queue contains blocks collected by one thread
using ConstraintBlockPool = std::vector<ConstraintBlockQue>; ///< a pool contains queues on different threads

#endif
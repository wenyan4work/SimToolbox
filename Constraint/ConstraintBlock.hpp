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

#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"
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
    double delta0 = 0;                      ///< constraint initial value
    double gamma = 0;                       ///< force magnitude, could be an initial guess
    int gidI = 0, gidJ = 0;                 ///< global ID of the two constrained objects
    int globalIndexI = 0, globalIndexJ = 0; ///< the global index of the two objects in mobility matrix
    bool oneSide = false;                   ///< flag for one side constraint. body J does not appear in mobility matrix
    double kappa = 0;                       ///< spring constant. =0 means no spring
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
     * @brief Construct a new Constraint Block object
     *
     * @param phi0_ current value of the constraint
     * @param gamma_ initial guess of constraint force magnitude
     * @param gidI_
     * @param gidJ_
     * @param globalIndexI_
     * @param globalIndexJ_
     * @param normI_
     * @param normJ_
     * @param posI_
     * @param posJ_
     * @param oneSide_ flag for one side collision
     *
     * If oneside = true, the gidJ, globalIndexJ, normJ, posJ will be ignored when constructing the fcTrans matrix
     * so any value of gidJ, globalIndexJ, normJ, posJ can be used in that case.
     *
     */
    ConstraintBlock(double delta0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                    const Evec3 &normI_, const Evec3 &normJ_, const Evec3 &posI_, const Evec3 &posJ_,
                    const Evec3 &labI_, const Evec3 &labJ_, bool oneSide_ = false, double kappa_=0)
        : ConstraintBlock(delta0_, gamma_, gidI_, gidJ_, globalIndexI_, globalIndexJ_, normI_.data(), normJ_.data(),
                          posI_.data(), posJ_.data(), labI_.data(), labJ_.data(), oneSide_, kappa_) {}

    ConstraintBlock(double delta0_, double gamma_, int gidI_, int gidJ_, int globalIndexI_, int globalIndexJ_,
                    const double normI_[3], const double normJ_[3], const double posI_[3], const double posJ_[3],
                    const double labI_[3], const double labJ_[3], bool oneSide_ = false, double kappa_ = 0)
        : delta0(delta0_), gamma(gamma_), gidI(gidI_), gidJ(gidJ_), globalIndexI(globalIndexI_),
          globalIndexJ(globalIndexJ_), oneSide(oneSide_), kappa(kappa_) {
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

#endif
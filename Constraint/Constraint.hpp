/**
 * @file Constraint.hpp
 * @author Bryce Palmer (brycepalmer96@gmail.com)
 * @brief
 * @version 0.1
 * @date 8/23/2022
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef CONSTRAINT_HPP_
#define CONSTRAINT_HPP_

#include "Util/EigenDef.hpp"
#include "Util/GeoCommon.h"
#include "Util/IOHelper.hpp"

#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

/**
 * @brief abstract constraint object
 *
 * Each constraint has some number of unknown DOF, each of which can be coupled together
 *
 * Constraints can (currently) have the following types:
 *   id=0: no penetration         | prevents penetration
 *   id=1: hookean spring         | resists relative translational motion between two points
 *   id=2: angular hookean spring | resists relative rotational motion between two vectors
 *   id=3: ball and socket        | prevents the separation of two points
 *
 */
struct Constraint {
  private:
    // private members to be accessed with getters and set with setters
    // currently, all constraints use at most 3 dof, so arrays will be sized to fix this maxima
    // TODO: template this class using maxNumRecursions_ (Harder than it sounds)
    static const int maxNumRecursions_ = 5;
    int recursionCounter_ = 0;                             ///< count which recursion we are in
    double seps0_[maxNumRecursions_] = {0};                 ///< initial constrainted quantity
    double gammas_[maxNumRecursions_] = {0};                ///< unknown dof for this constraint
    double labI_[maxNumRecursions_ * 3] = {0};              ///< the labframe location of constraint on particle I
    double labJ_[maxNumRecursions_ * 3] = {0};              ///< the labframe location of constraint on particle J
    double unscaledForceComI_[maxNumRecursions_ * 3] = {0}; ///< com force induced by this constraint on particle I for
                                                           ///< unit constraint Lagrange multiplier gamma
    double unscaledForceComJ_[maxNumRecursions_ * 3] = {0}; ///< com force induced by this constraint on particle J for
                                                           ///< unit constraint Lagrange multiplier gamma
    double unscaledTorqueComI_[maxNumRecursions_ * 3] = {0}; ///< com torque induced by this constraint on particle I for
                                                            ///< unit constraint Lagrange multiplier gamma
    double unscaledTorqueComJ_[maxNumRecursions_ * 3] = {0}; ///< com torque induced by this constraint on particle J for
                                                            ///< unit constraint Lagrange multiplier gamma
    double stress_[maxNumRecursions_ * 9] = {0}; ///< virial stress induced by these constraints
  public:
    int numRecursions = -1;  ///< number of constrained recursions
    int id = -1;             ///< identifier specifying the type of constraint this is
    bool oneSide = false;    ///< flag for one side constraint. when true, body J does not appear in mobility matrix
    bool bilaterial = false; ///< flag for bilaterial constraint. If not bilaterial, then assume unilaterial
    int gidI = GEO_INVALID_INDEX;         ///< unique global ID of particle I
    int gidJ = GEO_INVALID_INDEX;         ///< unique global ID of particle J
    int globalIndexI = GEO_INVALID_INDEX; ///< global index of particle I
    int globalIndexJ = GEO_INVALID_INDEX; ///< global index of particle J
    double diagonal = 0;

    std::function<bool(const double sep, const double gamma)> isConstrained;
    std::function<double(const double sep, const double gamma)> getValue;

    double getInitialSep(const int idxRecursion) const { return seps0_[idxRecursion]; }
    double getGamma(const int idxRecursion) const { return gammas_[idxRecursion]; }
    void getLabI(const int idxRecursion, double labI[3]) const {
        for (int i = 0; i < 3; i++) {
            labI[i] = labI_[3 * idxRecursion + i];
        }
    }
    void getLabJ(const int idxRecursion, double labJ[3]) const {
        for (int i = 0; i < 3; i++) {
            labJ[i] = labJ_[3 * idxRecursion + i];
        }
    }
    void getUnscaledForceComI(const int idxRecursion, double unscaledForceComI[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledForceComI[i] = unscaledForceComI_[3 * idxRecursion + i];
        }
    }
    void getUnscaledForceComJ(const int idxRecursion, double unscaledForceComJ[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledForceComJ[i] = unscaledForceComJ_[3 * idxRecursion + i];
        }
    }
    void getUnscaledTorqueComI(const int idxRecursion, double unscaledTorqueComI[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComI[i] = unscaledTorqueComI_[3 * idxRecursion + i];
        }
    }
    void getUnscaledTorqueComJ(const int idxRecursion, double unscaledTorqueComJ[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComJ[i] = unscaledTorqueComJ_[3 * idxRecursion + i];
        }
    }
    void getStress(const int idxRecursion, double stress[9]) const {
        for (int i = 0; i < 9; i++) {
            stress[i] = stress_[9 * idxRecursion + i];
        }
    }
    void getStress(const int idxRecursion, Emat3 &stress) const {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress(i, j) = stress_[idxRecursion * 9 + i * 3 + j];
            }
        }
    }

    void setGamma(const int idxRecursion, const double gamma) { gammas_[idxRecursion] = gamma; }
    void setInitialSep(const int idxRecursion, const double sep0) { seps0_[idxRecursion] = sep0; }

    void initialize(const double gammaGuess, const double initialSep, const double labI[3], const double labJ[3],
                    const double unscaledForceComI[3], const double unscaledForceComJ[3],
                    const double unscaledTorqueComI[3], const double unscaledTorqueComJ[3], const double stress[9]) {
        recursionCounter_ = 0;
        gammas_[recursionCounter_] = gammaGuess;
        seps0_[recursionCounter_] = initialSep;
        for (int i = 0; i < 3; i++) {
            labI_[3 * recursionCounter_ + i] = labI[i];
            labJ_[3 * recursionCounter_ + i] = labJ[i];
            unscaledForceComI_[3 * recursionCounter_ + i] = unscaledForceComI[i];
            unscaledForceComJ_[3 * recursionCounter_ + i] = unscaledForceComJ[i];
            unscaledTorqueComI_[3 * recursionCounter_ + i] = unscaledTorqueComI[i];
            unscaledTorqueComJ_[3 * recursionCounter_ + i] = unscaledTorqueComJ[i];
        }
        for (int i = 0; i < 9; i++) {
            stress_[9 * recursionCounter_ + i] = stress[i];
        }
    }

    void addRecursion(const double gammaGuess, const double initialSep, const double labI[3], const double labJ[3],
                      const double unscaledForceComI[3], const double unscaledForceComJ[3],
                      const double unscaledTorqueComI[3], const double unscaledTorqueComJ[3], const double stress[9]) {
        recursionCounter_++;
        assert(recursionCounter_ < numRecursions);

        gammas_[recursionCounter_] = gammaGuess;
        seps0_[recursionCounter_] = initialSep;
        for (int i = 0; i < 3; i++) {
            labI_[3 * recursionCounter_ + i] = labI[i];
            labJ_[3 * recursionCounter_ + i] = labJ[i];
            unscaledForceComI_[3 * recursionCounter_ + i] = unscaledForceComI[i];
            unscaledForceComJ_[3 * recursionCounter_ + i] = unscaledForceComJ[i];
            unscaledTorqueComI_[3 * recursionCounter_ + i] = unscaledTorqueComI[i];
            unscaledTorqueComJ_[3 * recursionCounter_ + i] = unscaledTorqueComJ[i];
        }
        for (int i = 0; i < 9; i++) {
            stress_[9 * recursionCounter_ + i] = stress[i];
        }
    }

    void resetRecursions() {
        recursionCounter_ = 0;
        for (int idx = 1; idx < numRecursions; idx++) {
            gammas_[idx] = 0.0;
            seps0_[idx] = 0.0;
            for (int i = 0; i < 3; i++) {
                labI_[3 * idx + i] = 0.0;
                labJ_[3 * idx + i] = 0.0;
                unscaledForceComI_[3 * idx + i] = 0.0;
                unscaledForceComJ_[3 * idx + i] = 0.0;
                unscaledTorqueComI_[3 * idx + i] = 0.0;
                unscaledTorqueComJ_[3 * idx + i] = 0.0;
            }
            for (int i = 0; i < 9; i++) {
                stress_[9 * idx + i] = 0.0;
            }
        }
    }

    void reverseIJ() {
        std::swap(gidI, gidJ);
        std::swap(globalIndexI, globalIndexJ);
        for (int i = 0; i < 9; i++) {
            std::swap(unscaledForceComI_[i], unscaledForceComJ_[i]);
            std::swap(unscaledTorqueComI_[i], unscaledTorqueComJ_[i]);
            std::swap(labI_[i], labJ_[i]);
        }
    }
};

// public constructors for different types of constraints
void noPenetrationConstraint(Constraint &con, const int numRecursions, const double sepDistance, const int gidI,
                             const int gidJ, const int globalIndexI, const int globalIndexJ, const double posI[3],
                             const double posJ[3], const double labI[3], const double labJ[3], const double normI[3],
                             const double stressIJ[9], const bool oneSide, const bool recursionFlag);

void springConstraint(Constraint &con, const int numRecursions, const double sepDistance, const double restLength,
                      const double springConstant, const int gidI, const int gidJ, const int globalIndexI,
                      const int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                      const double labJ[3], const double normI[3], const double stressIJ[9], const bool oneSide,
                      const bool recursionFlag);

void angularSpringConstraint(Constraint &con, const int numRecursions, const double sepDistance, const double restAngle,
                             const double springConstant, const int gidI, const int gidJ, const int globalIndexI,
                             const int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                             const double labJ[3], const double normI[3], const double stressIJ[9], const bool oneSide,
                             const bool recursionFlag);

void pivotConstraint(Constraint &con, const int numRecursions, const double sepDistance, const int gidI, const int gidJ,
                     const int globalIndexI, const int globalIndexJ, const double posI[3], const double posJ[3],
                     const double labI[3], const double labJ[3], const double normI[3], const double stressIJ[9],
                     const bool oneSide, const bool recursionFlag);

// static_assert(std::is_trivially_copyable<Constraint>::value, ""); // TODO: must it be?
static_assert(std::is_default_constructible<Constraint>::value, "");

using ConstraintQue = std::vector<Constraint>;     ///< a vect contains constraints collected by one thread
using ConstraintPool = std::vector<ConstraintQue>; ///< a pool contains queues on different threads

#endif
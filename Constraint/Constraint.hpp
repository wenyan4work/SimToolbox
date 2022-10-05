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
 * Currently, only frictional constraints will have more than one DOF
 * We have yet to implement collision constraints, so we fix maxNumDOF_ to be 1.
 *
 * Constraints can (currently) have the following types:
 *   id=0: no penetration         | dof 1 | prevents penetration
 *   id=1: hookean spring         | dof 1 | resists relative translational motion between two points
 *   id=2: angular hookean spring | dof 1 | resists relative rotational motion between two vectors
 *   id=3: ball and socket        | dof 1 | prevents the separation of two points
 *
 */
struct Constraint {
  private:
    // private members to be accessed with getters and set with setters
    // currently, constraints use at most 1 dof, so arrays will be sized to fix this maxima
    // TODO: template this class using maxNumDOF_ (Harder than it sounds)
    static const int maxNumDOF_ = 1;
    double seps_[maxNumDOF_] = {0};                   ///< initial constrainted quantity
    double gammas_[maxNumDOF_] = {0};                 ///< unknown dof for this constraint
    double gammaGuesses_[maxNumDOF_] = {0};           ///< approximate unknown dof for this constraint
    double labI_[maxNumDOF_ * 3] = {0};               ///< the labframe location of constraint on particle I
    double labJ_[maxNumDOF_ * 3] = {0};               ///< the labframe location of constraint on particle J
    double unscaledForceComI_[maxNumDOF_ * 3] = {0};  ///< com force induced by this constraint on particle I for
                                                      ///< unit constraint Lagrange multiplier gamma
    double unscaledForceComJ_[maxNumDOF_ * 3] = {0};  ///< com force induced by this constraint on particle J for
                                                      ///< unit constraint Lagrange multiplier gamma
    double unscaledTorqueComI_[maxNumDOF_ * 3] = {0}; ///< com torque induced by this constraint on particle I for
                                                      ///< unit constraint Lagrange multiplier gamma
    double unscaledTorqueComJ_[maxNumDOF_ * 3] = {0}; ///< com torque induced by this constraint on particle J for
                                                      ///< unit constraint Lagrange multiplier gamma
    double stress_[maxNumDOF_ * 9] = {0};             ///< virial stress induced by these constraints
  public:
    int numDOF = -1;         ///< number of constrained dof
    int id = -1;             ///< identifier specifying the type of constraint this is
    bool oneSide = false;    ///< flag for one side constraint. when true, body J does not appear in mobility matrix
    bool bilaterial = false; ///< flag for bilaterial constraint. If not bilaterial, then assume unilaterial
    int gidI = GEO_INVALID_INDEX;         ///< unique global ID of particle I
    int gidJ = GEO_INVALID_INDEX;         ///< unique global ID of particle J
    int globalIndexI = GEO_INVALID_INDEX; ///< global index of particle I
    int globalIndexJ = GEO_INVALID_INDEX; ///< global index of particle J
    double diagonal = 0;

    std::function<bool(const Constraint &con, const double sep, const double gamma)> isConstrained;
    std::function<double(const Constraint &con, const double sep, const double gamma)> getValue;

    double getSep(const int idxDOF) const { return seps_[idxDOF]; }
    double getGamma(const int idxDOF) const { return gammas_[idxDOF]; }
    double getGammaGuess(const int idxDOF) const { return gammaGuesses_[idxDOF]; }
    void getLabI(const int idxDOF, double labI[3]) const {
        for (int i = 0; i < 3; i++) {
            labI[i] = labI_[3 * idxDOF + i];
        }
    }
    void getLabJ(const int idxDOF, double labJ[3]) const {
        for (int i = 0; i < 3; i++) {
            labJ[i] = labJ_[3 * idxDOF + i];
        }
    }
    void getUnscaledForceComI(const int idxDOF, double unscaledForceComI[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledForceComI[i] = unscaledForceComI_[3 * idxDOF + i];
        }
    }
    void getUnscaledForceComJ(const int idxDOF, double unscaledForceComJ[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledForceComJ[i] = unscaledForceComJ_[3 * idxDOF + i];
        }
    }
    void getUnscaledTorqueComI(const int idxDOF, double unscaledTorqueComI[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComI[i] = unscaledTorqueComI_[3 * idxDOF + i];
        }
    }
    void getUnscaledTorqueComJ(const int idxDOF, double unscaledTorqueComJ[3]) const {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComJ[i] = unscaledTorqueComJ_[3 * idxDOF + i];
        }
    }
    void getStress(const int idxDOF, double stress[9]) const {
        for (int i = 0; i < 9; i++) {
            stress[i] = stress_[9 * idxDOF + i];
        }
    }
    void getStress(const int idxDOF, Emat3 &stress) const {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress(i, j) = stress_[idxDOF * 9 + i * 3 + j];
            }
        }
    }

    void setSep(const int idxDOF, const double sep) { seps_[idxDOF] = sep; }
    void setGamma(const int idxDOF, const double gamma) { gammas_[idxDOF] = gamma; }
    void setGammaGuess(const int idxDOF, const double gammaGuess) { gammaGuesses_[idxDOF] = gammaGuess; }

    void initializeDOF(const int idxDOF, const double gammaGuess, const double initialSep, const double labI[3],
                       const double labJ[3], const double unscaledForceComI[3], const double unscaledForceComJ[3],
                       const double unscaledTorqueComI[3], const double unscaledTorqueComJ[3], const double stress[9]) {
        gammaGuesses_[idxDOF] = gammaGuess;
        seps_[idxDOF] = initialSep;
        for (int i = 0; i < 3; i++) {
            labI_[3 * idxDOF + i] = labI[i];
            labJ_[3 * idxDOF + i] = labJ[i];
            unscaledForceComI_[3 * idxDOF + i] = unscaledForceComI[i];
            unscaledForceComJ_[3 * idxDOF + i] = unscaledForceComJ[i];
            unscaledTorqueComI_[3 * idxDOF + i] = unscaledTorqueComI[i];
            unscaledTorqueComJ_[3 * idxDOF + i] = unscaledTorqueComJ[i];
        }
        for (int i = 0; i < 9; i++) {
            stress_[9 * idxDOF + i] = stress[i];
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
void noPenetrationConstraint(Constraint &con, const double sepDistance, const int gidI, const int gidJ,
                             const int globalIndexI, const int globalIndexJ, const double posI[3], const double posJ[3],
                             const double labI[3], const double labJ[3], const double normI[3],
                             const double stressIJ[9], const bool oneSide);

void springConstraint(Constraint &con, const double sepDistance, const double restLength, const double springConstant,
                      const int gidI, const int gidJ, const int globalIndexI, const int globalIndexJ,
                      const double posI[3], const double posJ[3], const double labI[3], const double labJ[3],
                      const double normI[3], const double stressIJ[9], const bool oneSide);

void angularSpringConstraint(Constraint &con, const double sepDistance, const double restAngle,
                             const double springConstant, const int gidI, const int gidJ, const int globalIndexI,
                             const int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                             const double labJ[3], const double normI[3], const double stressIJ[9], const bool oneSide);

void pivotConstraint(Constraint &con, const double sepDistance, const int gidI, const int gidJ, const int globalIndexI,
                     const int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                     const double labJ[3], const double normI[3], const double stressIJ[9], const bool oneSide);

// static_assert(std::is_trivially_copyable<Constraint>::value, ""); // TODO: must it be? I don't believe so.
static_assert(std::is_default_constructible<Constraint>::value, "");

using ConstraintQue = std::vector<Constraint>;     ///< a vect contains constraints collected by one thread
using ConstraintPool = std::vector<ConstraintQue>; ///< a pool contains queues on different threads

#endif
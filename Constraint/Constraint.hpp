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
 *   id=0: collision              | 3 constrained DOF | prevents penetration and enforces tangency of colliding surfaces
 *   id=1: no penetration         | 1 constrained DOF | prevents penetration
 *   id=2: hookean spring         | 3 constrained DOF | resists relative translational motion between two points
 *   id=3: angular hookean spring | 3 constrained DOF | resists relative rotational motion between two vectors
 *   id=4: ball and socket        | 3 constrained DOF | prevents the separation of two points
 *
 */
struct Constraint {
  private:
    // private members to be accessed with getters and set with setters
    // currently, all constraints use at most 3 dof, so arrays will be sized to fix this maxima
    double seps_[3] = {0};               ///< constrainted quantity
    double gammas_[3] = {0};             ///< unknown dof for this constraint
    double unscaledForceComI_[9] = {0};  ///< com force induced by this constraint on particle I for unit constraint
                                         ///< Lagrange multiplier gamma
    double unscaledForceComJ_[9] = {0};  ///< com force induced by this constraint on particle J for unit constraint
                                         ///< Lagrange multiplier gamma
    double unscaledTorqueComI_[9] = {0}; ///< com torque induced by this constraint on particle I for unit constraint
                                         ///< Lagrange multiplier gamma
    double unscaledTorqueComJ_[9] = {0}; ///< com torque induced by this constraint on particle J for unit constraint
                                         ///< Lagrange multiplier gamma
    double stress_[27] = {0};            ///< virial stress induced by these constraints
  public:
    int id = -1;          ///< identifier specifying the type of constraint this is
    int numDOF = -1;      ///< number of constrained degrees of freedom
    bool oneSide = false; ///< flag for one side constraint. when true, body J does not appear in mobility matrix
    int gidI = GEO_INVALID_INDEX;         ///< unique global ID of particle I
    int gidJ = GEO_INVALID_INDEX;         ///< unique global ID of particle J
    int globalIndexI = GEO_INVALID_INDEX; ///< global index of particle I
    int globalIndexJ = GEO_INVALID_INDEX; ///< global index of particle J
    double labI[3] = {0};                 ///< the labframe location of constraint on particle I
    double labJ[3] = {0};                 ///< the labframe location of constraint on particle J
    double diagonal = 0; 

    std::function<std::vector<bool>(const double *seps, const double *gammas)> isConstrained;
    std::function<std::vector<double>(const double *seps, const double *gammas)> getValues;

    double getSep(const int idxDOF) const { return seps_[idxDOF]; };
    double getGamma(const int idxDOF) const { return gammas_[idxDOF]; };
    double getValue(const int idxDOF) const {
      return getValues(seps_, gammas_)[idxDOF];
    }
    double getState(const int idxDOF) const {
      return isConstrained(seps_, gammas_)[idxDOF];
    }

    void getUnscaledForceComI(const int idxDOF, double *unscaledForceComI) const {
        for (int i = 0; i < 3; i++) {
            unscaledForceComI[i] = unscaledForceComI_[3 * idxDOF + i];
        }
    }
    void getUnscaledForceComJ(const int idxDOF, double *unscaledForceComJ) const {
        for (int i = 0; i < 3; i++) {
            unscaledForceComJ[i] = unscaledForceComJ_[3 * idxDOF + i];
        }
    }
    void getUnscaledTorqueComI(const int idxDOF, double *unscaledTorqueComI) const {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComI[i] = unscaledTorqueComI_[3 * idxDOF + i];
        }
    }
    void getUnscaledTorqueComJ(const int idxDOF, double *unscaledTorqueComJ) const {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComJ[i] = unscaledTorqueComJ_[3 * idxDOF + i];
        }
    }
    void getStress(const int idxDOF, double *stress) const {
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
    

    void setGamma(const int idxDOF, const double gamma) { gammas_[idxDOF] = gamma; }
    void setSep(const int idxDOF, const double sep) { seps_[idxDOF] = sep; }
    void setUnscaledForceComI(const int idxDOF, const double *unscaledForceComI) {
        for (int i = 0; i < 3; i++) {
            unscaledForceComI_[3 * idxDOF + i] = unscaledForceComI[i];
        }
    }
    void setUnscaledForceComJ(const int idxDOF, const double *unscaledForceComJ) {
        for (int i = 0; i < 3; i++) {
            unscaledForceComJ_[3 * idxDOF + i] = unscaledForceComJ[i];
        }
    }
    void setUnscaledTorqueComI(const int idxDOF, const double *unscaledTorqueComI) {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComI_[3 * idxDOF + i] = unscaledTorqueComI[i];
        }
    }
    void setUnscaledTorqueComJ(const int idxDOF, const double *unscaledTorqueComJ) {
        for (int i = 0; i < 3; i++) {
            unscaledTorqueComJ_[3 * idxDOF + i] = unscaledTorqueComJ[i];
        }
    }
    void setStress(const int idxDOF, const double *stress) {
        for (int i = 0; i < 9; i++) {
            stress_[i] = stress[9 * idxDOF + i];
        }
    }
    void setStress(const int idxDOF, const Emat3 &stress) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                stress_[idxDOF * 9 + i * 3 + j] = stress(i, j);
            }
        }
    }

    void reverseIJ() {
        std::swap(gidI, gidJ);
        std::swap(globalIndexI, globalIndexJ);
        for (int i = 0; i < 3; i++) {
            std::swap(unscaledForceComI_[i], unscaledForceComJ_[i]);
            std::swap(unscaledTorqueComI_[i], unscaledTorqueComJ_[i]);
            std::swap(labI[i], labJ[i]);
        }
    }
};

// public constructors for different types of constraints
Constraint collisionConstraint(double sepDistance, int gidI, int gidJ, int globalIndexI, int globalIndexJ,
                               const double posI[3], const double posJ[3], const double labI[3], const double labJ[3],
                               const double normI[3], const double tangent1I[3], const double tangent2I[3],
                               bool oneSide);

Constraint noPenetrationConstraint(double sepDistance, int gidI, int gidJ, int globalIndexI, int globalIndexJ,
                                   const double posI[3], const double posJ[3], const double labI[3],
                                   const double labJ[3], const double normI[3], bool oneSide);

Constraint springConstraint(double sepDistance, double restLength, double springConstant, int gidI, int gidJ, int globalIndexI,
                            int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                            const double labJ[3], const double normI[3], bool oneSide);

Constraint angularSpringConstraint(double sepDistance, double restAngle, double springConstant, int gidI, int gidJ, int globalIndexI,
                                   int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                                   const double labJ[3], const double normI[3], bool oneSide);

Constraint pivotConstraint(double sepDistance, int gidI, int gidJ, int globalIndexI, int globalIndexJ,
                           const double posI[3], const double posJ[3], const double labI[3], const double labJ[3],
                           const double normI[3], bool oneSide);

// static_assert(std::is_trivially_copyable<Constraint>::value, ""); // TODO: must it be?
static_assert(std::is_default_constructible<Constraint>::value, "");

using ConstraintQue = std::vector<Constraint>;     ///< a vect contains constraints collected by one thread
using ConstraintPool = std::vector<ConstraintQue>; ///< a pool contains queues on different threads

#endif
/**
 * @file Constraint.cpp
 * @author Bryce Palmer (brycepalmer96@gmail.com)
 * @brief
 * @version 0.1
 * @date 8/23/2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "Constraint.hpp"

// public constructors for different types of constraints
void noPenetrationConstraint(Constraint &con, const double sepDistance, const int gidI,
                             const int gidJ, const int globalIndexI, const int globalIndexJ, const double posI[3],
                             const double posJ[3], const double labI[3], const double labJ[3], const double normI[3],
                             const double stressIJ[9], const bool oneSide) {
    // set base information
    con.id = 0;
    con.numDOF = 1;

    // set constraint info
    con.diagonal = 0;
    con.oneSide = oneSide;
    con.bilaterial = false;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    std::function<bool(const Constraint &con, const double sep, const double gamma)> isConstrained = [](const Constraint &con, const double sep, const double gamma) {
        return sep < gamma; // min-map
    };
    std::function<double(const Constraint &con, const double sep, const double gamma)> getValue = [](const Constraint &con, const double sep, const double gamma) {
        return std::min(sep, gamma); // min-map
    };
    con.isConstrained = std::move(isConstrained);
    con.getValue = std::move(getValue);

    // store the recursion data
    const Evec3 unscaledForceComI(normI[0], normI[1], normI[2]);
    const Evec3 unscaledForceComJ = -unscaledForceComI;
    const Evec3 unscaledTorqueComI(normI[2] * posI[1] - normI[1] * posI[2], normI[0] * posI[2] - normI[2] * posI[0],
                                   normI[1] * posI[0] - normI[0] * posI[1]);
    const Evec3 unscaledTorqueComJ(-normI[2] * posJ[1] + normI[1] * posJ[2], -normI[0] * posJ[2] + normI[2] * posJ[0],
                                   -normI[1] * posJ[0] + normI[0] * posJ[1]);
    const double gammaGuess = sepDistance < 0 ? -sepDistance : 0;
    con.initializeDOF(0, gammaGuess, sepDistance, labI, labJ, unscaledForceComI.data(), unscaledForceComJ.data(),
                    unscaledTorqueComI.data(), unscaledTorqueComJ.data(), stressIJ);
}

void springConstraint(Constraint &con, const double sepDistance, const double restLength,
                      const double springConstant, const int gidI, const int gidJ, const int globalIndexI,
                      const int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                      const double labJ[3], const double normI[3], const double stressIJ[9], const bool oneSide) {
    // set base information
    con.id = 1;
    con.numDOF = 1;

    // set constraint info
    con.diagonal = 1.0 / springConstant;
    con.oneSide = oneSide;
    con.bilaterial = true;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    std::function<bool(const Constraint &con, const double sep, const double gamma)> isConstrained = [](const Constraint &con, const double sep, const double gamma) {
        return true;
    };
    std::function<double(const Constraint &con, const double sep, const double gamma)> getValue = [](const Constraint &con, const double sep, const double gamma) {
        return con.diagonal * sep + gamma;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValue = std::move(getValue);

    // store the recursion data
    const Evec3 unscaledForceComI(normI[0], normI[1], normI[2]);
    const Evec3 unscaledForceComJ = -unscaledForceComI;
    const Evec3 unscaledTorqueComI(normI[2] * posI[1] - normI[1] * posI[2], normI[0] * posI[2] - normI[2] * posI[0],
                                   normI[1] * posI[0] - normI[0] * posI[1]);
    const Evec3 unscaledTorqueComJ(-normI[2] * posJ[1] + normI[1] * posJ[2], -normI[0] * posJ[2] + normI[2] * posJ[0],
                                   -normI[1] * posJ[0] + normI[0] * posJ[1]);
    const double gammaGuess = sepDistance - restLength;
    con.initializeDOF(0, gammaGuess, -gammaGuess, labI, labJ, unscaledForceComI.data(), unscaledForceComJ.data(),
            unscaledTorqueComI.data(), unscaledTorqueComJ.data(), stressIJ);
}

void angularSpringConstraint(Constraint &con, const double sepAngle, const double restAngle,
                             const double springConstant, const int gidI, const int gidJ, const int globalIndexI,
                             const int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                             const double labJ[3], const double normI[3], const double stressIJ[9], const bool oneSide) {
    // set base information
    con.id = 2;
    con.numDOF = 1;

    // set constraint info
    con.diagonal = 1.0 / springConstant;
    con.oneSide = oneSide;
    con.bilaterial = true;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    std::function<bool(const Constraint &con, const double sep, const double gamma)> isConstrained = [](const Constraint &con, const double sep, const double gamma) {
        return true;
    };
    std::function<double(const Constraint &con, const double sep, const double gamma)> getValue = [](const Constraint &con, const double sep, const double gamma) {
        return sep;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValue = std::move(getValue);

    // store the recursion data
    // TODO: update these forces and torques
    const Evec3 unscaledForceComI(normI[0], normI[1], normI[2]);
    const Evec3 unscaledForceComJ = -unscaledForceComI;
    const Evec3 unscaledTorqueComI(normI[2] * posI[1] - normI[1] * posI[2], normI[0] * posI[2] - normI[2] * posI[0],
                                   normI[1] * posI[0] - normI[0] * posI[1]);
    const Evec3 unscaledTorqueComJ(-normI[2] * posJ[1] + normI[1] * posJ[2], -normI[0] * posJ[2] + normI[2] * posJ[0],
                                   -normI[1] * posJ[0] + normI[0] * posJ[1]);
    const double gammaGuess = sepAngle - restAngle;
    con.initializeDOF(0, gammaGuess, -gammaGuess, labI, labJ, unscaledForceComI.data(), unscaledForceComJ.data(),
            unscaledTorqueComI.data(), unscaledTorqueComJ.data(), stressIJ);
}

void pivotConstraint(Constraint &con, const double sepDistance, const int gidI, const int gidJ,
                     const int globalIndexI, const int globalIndexJ, const double posI[3], const double posJ[3],
                     const double labI[3], const double labJ[3], const double normI[3], const double stressIJ[9],
                     const bool oneSide) {
    // set base information
    con.id = 3;
    con.numDOF = 1;

    // set constraint info
    con.diagonal = 0;
    con.oneSide = oneSide;
    con.bilaterial = true;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    std::function<bool(const Constraint &con, const double sep, const double gamma)> isConstrained = [](const Constraint &con, const double sep, const double gamma) {
        return true;
    };
    std::function<double(const Constraint &con, const double sep, const double gamma)> getValue = [](const Constraint &con, const double sep, const double gamma) {
        return sep;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValue = std::move(getValue);

    // store the recursion data
    const Evec3 unscaledForceComI(normI[0], normI[1], normI[2]);
    const Evec3 unscaledForceComJ = -unscaledForceComI;
    const Evec3 unscaledTorqueComI(normI[2] * posI[1] - normI[1] * posI[2], normI[0] * posI[2] - normI[2] * posI[0],
                                   normI[1] * posI[0] - normI[0] * posI[1]);
    const Evec3 unscaledTorqueComJ(-normI[2] * posJ[1] + normI[1] * posJ[2], -normI[0] * posJ[2] + normI[2] * posJ[0],
                                   -normI[1] * posJ[0] + normI[0] * posJ[1]);
    const double gammaGuess = sepDistance;
    con.initializeDOF(0, gammaGuess, sepDistance, labI, labJ, unscaledForceComI.data(), unscaledForceComJ.data(),
            unscaledTorqueComI.data(), unscaledTorqueComJ.data(), stressIJ);
}

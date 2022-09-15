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
Constraint collisionConstraint(double sepDistance, int gidI, int gidJ, int globalIndexI, int globalIndexJ,
                               const double posI[3], const double posJ[3], const double labI[3], const double labJ[3],
                               const double normI[3], const double tangent1I[3], const double tangent2I[3],
                               bool oneSide) {
    // set base information
    Constraint con;
    con.id = 0;
    con.numDOF = 3;

    // set constraint info
    con.diagonal = 0; 
    con.oneSide = oneSide;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    for (int i = 0; i < 3; i++) {
        con.labI[i] = labI[i];
        con.labJ[i] = labJ[i];
    }
    std::function<std::vector<bool>(const double *seps, const double *gammas)> isConstrained = [](const double *seps, const double *gammas) { 
        std::vector<bool> result(3);
        result[0] = !(gammas[0] < seps[0]); // min-map
        result[1] = result[0];
        result[2] = result[0];
        return result;
    };
    std::function<std::vector<double>(const double *seps, const double *gammas)> getValues = [](const double *seps, const double *gammas) {
        std::vector<double> result(3);
        if (gammas[0] < seps[0]) {
            result[0] = gammas[0];
            result[1] = gammas[1];
            result[2] = gammas[2];
        } else {
            result[0] = seps[0];
            result[1] = seps[1];
            result[2] = seps[2];
        }
        return result;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValues = std::move(getValues);

    // store the dof data
    {
        // no-penetration
        const Evec3 unscaledForceComI(normI[0], normI[1], normI[2]);
        const Evec3 unscaledForceComJ = -unscaledForceComI;
        const Evec3 unscaledTorqueComI(normI[2] * posI[1] - normI[1] * posI[2], normI[0] * posI[2] - normI[2] * posI[0],
                                       normI[1] * posI[0] - normI[0] * posI[1]);
        const Evec3 unscaledTorqueComJ(-normI[2] * posJ[1] + normI[1] * posJ[2],
                                       -normI[0] * posJ[2] + normI[2] * posJ[0],
                                       -normI[1] * posJ[0] + normI[0] * posJ[1]);
        const double gammaGuess = sepDistance < 0 ? -sepDistance : 0;
        con.setGamma(0, gammaGuess);
        con.setSep(0, sepDistance);
        con.setUnscaledForceComI(0, unscaledForceComI.data());
        con.setUnscaledForceComJ(0, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(0, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(0, unscaledTorqueComJ.data());
    }
    {
        // tangent constraint 1
        const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(normI[2] * tangent1I[1] - normI[1] * tangent1I[2],
                                       normI[0] * tangent1I[2] - normI[2] * tangent1I[0],
                                       normI[1] * tangent1I[0] - normI[0] * tangent1I[1]);
        const Evec3 unscaledTorqueComJ(-normI[2] * tangent1I[1] + normI[1] * tangent1I[2],
                                       -normI[0] * tangent1I[2] + normI[2] * tangent1I[0],
                                       -normI[1] * tangent1I[0] + normI[0] * tangent1I[1]);
        const double gammaGuess = 0.0;
        con.setGamma(1, gammaGuess);
        con.setSep(1, 0.0);
        con.setUnscaledForceComI(1, unscaledForceComI.data());
        con.setUnscaledForceComJ(1, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(1, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(1, unscaledTorqueComJ.data());
    }
    {
        // tangent constraint 1
        const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(normI[2] * tangent2I[1] - normI[1] * tangent2I[2],
                                       normI[0] * tangent2I[2] - normI[2] * tangent2I[0],
                                       normI[1] * tangent2I[0] - normI[0] * tangent2I[1]);
        const Evec3 unscaledTorqueComJ(-normI[2] * tangent2I[1] + normI[1] * tangent2I[2],
                                       -normI[0] * tangent2I[2] + normI[2] * tangent2I[0],
                                       -normI[1] * tangent2I[0] + normI[0] * tangent2I[1]);
        const double gammaGuess = 0.0;
        con.setGamma(2, gammaGuess);
        con.setSep(2, 0.0);
        con.setUnscaledForceComI(2, unscaledForceComI.data());
        con.setUnscaledForceComJ(2, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(2, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(2, unscaledTorqueComJ.data());
    }
    return con;
}

Constraint noPenetrationConstraint(double sepDistance, int gidI, int gidJ, int globalIndexI, int globalIndexJ,
                                   const double posI[3], const double posJ[3], const double labI[3],
                                   const double labJ[3], const double normI[3], bool oneSide) {
    // set base information
    Constraint con;
    con.id = 0;
    con.numDOF = 1;

    // set constraint info
    con.diagonal = 0; 
    con.oneSide = oneSide;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    for (int i = 0; i < 3; i++) {
        con.labI[i] = labI[i];
        con.labJ[i] = labJ[i];
    }
    std::function<std::vector<bool>(const double *seps, const double *gammas)> isConstrained = [](const double *seps, const double *gammas) { 
        std::vector<bool> result(1);
        result[0] = seps[0] < gammas[0]; // min-map
        return result;
    };
    std::function<std::vector<double>(const double *seps, const double *gammas)> getValues = [](const double *seps, const double *gammas) {
        std::vector<double> result(1);
        if (seps[0] < gammas[0]) {
            result[0] = seps[0];
        } else {
            result[0] = gammas[0];
        }
        return result;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValues = std::move(getValues);

    // store the dof data
    const Evec3 unscaledForceComI(normI[0], normI[1], normI[2]);
    const Evec3 unscaledForceComJ = -unscaledForceComI;
    const Evec3 unscaledTorqueComI(normI[2] * posI[1] - normI[1] * posI[2], normI[0] * posI[2] - normI[2] * posI[0],
                                   normI[1] * posI[0] - normI[0] * posI[1]);
    const Evec3 unscaledTorqueComJ(-normI[2] * posJ[1] + normI[1] * posJ[2], -normI[0] * posJ[2] + normI[2] * posJ[0],
                                   -normI[1] * posJ[0] + normI[0] * posJ[1]);
    const double gammaGuess = sepDistance < 0 ? -sepDistance : 0;
    con.setGamma(0, gammaGuess);
    con.setSep(0, 0.0);
    con.setUnscaledForceComI(0, unscaledForceComI.data());
    con.setUnscaledForceComJ(0, unscaledForceComJ.data());
    con.setUnscaledTorqueComI(0, unscaledTorqueComI.data());
    con.setUnscaledTorqueComJ(0, unscaledTorqueComJ.data());
    return con;
}

Constraint springConstraint(double sepDistance, double restLength, double springConstant, int gidI, int gidJ, int globalIndexI,
                            int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                            const double labJ[3], const double normI[3], bool oneSide) {
    // set base information
    Constraint con;
    con.id = 0;
    con.numDOF = 3;

    // set constraint info
    con.diagonal = 1.0 / springConstant;
    con.oneSide = oneSide;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    for (int i = 0; i < 3; i++) {
        con.labI[i] = labI[i];
        con.labJ[i] = labJ[i];
    }
    std::function<std::vector<bool>(const double *seps, const double *gammas)> isConstrained = [](const double *seps, const double *gammas) { 
        std::vector<bool> result(3);
        result[0] = true;
        result[1] = true;
        result[2] = true;
        return result;
    };
    std::function<std::vector<double>(const double *seps, const double *gammas)> getValues = [](const double *seps, const double *gammas) {
        std::vector<double> result(3);
        result[0] = seps[0];
        result[1] = seps[1];
        result[2] = seps[2];
        return result;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValues = std::move(getValues);


    // store the dof data
    // TODO: update this to be the real angular spring constraint with unknown angle
    //       potential solution is to have the unknowns be force magnatude, force theta, force phi
    {
        // x-direction constraint
        const Evec3 unscaledForceComI(1.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(-1.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(0.0, posI[2], posI[1]);
        const Evec3 unscaledTorqueComJ(0.0, -posJ[2], -posJ[1]);
        const double gammaGuess = normI[0] * (sepDistance - restLength);
        con.setGamma(0, gammaGuess);
        con.setSep(0, 0.0);
        con.setUnscaledForceComI(0, unscaledForceComI.data());
        con.setUnscaledForceComJ(0, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(0, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(0, unscaledTorqueComJ.data());
    }
    {
        // y-direction constraint
        const Evec3 unscaledForceComI(0.0, 1.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, -1.0, 0.0);
        const Evec3 unscaledTorqueComI(posI[2], 0.0, posI[0]);
        const Evec3 unscaledTorqueComJ(-posJ[2], 0.0, -posJ[0]);
        const double gammaGuess = normI[1] * (sepDistance - restLength);
        con.setGamma(1, gammaGuess);
        con.setSep(1, 0.0);
        con.setUnscaledForceComI(1, unscaledForceComI.data());
        con.setUnscaledForceComJ(1, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(1, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(1, unscaledTorqueComJ.data());
    }
    {
        // z-direction constraint
        const Evec3 unscaledForceComI(0.0, 0.0, 1.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, -1.0);
        const Evec3 unscaledTorqueComI(posI[1], posI[0], 0.0);
        const Evec3 unscaledTorqueComJ(-posJ[1], -posJ[0], 0.0);
        const double gammaGuess = normI[2] * (sepDistance - restLength);
        con.setGamma(2, gammaGuess);
        con.setSep(2, 0.0);
        con.setUnscaledForceComI(2, unscaledForceComI.data());
        con.setUnscaledForceComJ(2, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(2, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(2, unscaledTorqueComJ.data());
    }
    return con;
}

Constraint angularSpringConstraint(double sepDistance, double restAngle, double springConstant, int gidI, int gidJ, int globalIndexI,
                                   int globalIndexJ, const double posI[3], const double posJ[3], const double labI[3],
                                   const double labJ[3], const double normI[3], bool oneSide) {
    // set base information
    Constraint con;
    con.id = 0;
    con.numDOF = 3;

    // set constraint info
    con.diagonal = 1.0 / springConstant;
    con.oneSide = oneSide;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    for (int i = 0; i < 3; i++) {
        con.labI[i] = labI[i];
        con.labJ[i] = labJ[i];
    }
    std::function<std::vector<bool>(const double *seps, const double *gammas)> isConstrained = [](const double *seps, const double *gammas) { 
        std::vector<bool> result(3);
        result[0] = true;
        result[1] = true;
        result[2] = true;
        return result;
    };
    std::function<std::vector<double>(const double *seps, const double *gammas)> getValues = [](const double *seps, const double *gammas) {
        std::vector<double> result(3);
        result[0] = seps[0];
        result[1] = seps[1];
        result[2] = seps[2];
        return result;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValues = std::move(getValues);

    // store the dof data
    // TODO: update this to be the real angular spring constraint with unknown angle
    // potential solution is to have the unknowns be torque magnatude, torque theta, torque phi
    {
        // x-direction constraint
        const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(0.0, posI[2], posI[1]);
        const Evec3 unscaledTorqueComJ(0.0, -posJ[2], -posJ[1]);
        const double gammaGuess = normI[0] * (sepDistance - restAngle);
        con.setGamma(0, gammaGuess);
        con.setSep(0, 0.0);
        con.setUnscaledForceComI(0, unscaledForceComI.data());
        con.setUnscaledForceComJ(0, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(0, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(0, unscaledTorqueComJ.data());
    }
    {
        // y-direction constraint
        const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(posI[2], 0.0, posI[0]);
        const Evec3 unscaledTorqueComJ(-posJ[2], 0.0, -posJ[0]);
        const double gammaGuess = normI[1] * (sepDistance - restAngle);
        con.setGamma(1, gammaGuess);
        con.setSep(1, 0.0);
        con.setUnscaledForceComI(1, unscaledForceComI.data());
        con.setUnscaledForceComJ(1, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(1, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(1, unscaledTorqueComJ.data());
    }
    {
        // z-direction constraint
        const Evec3 unscaledForceComI(0.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(posI[1], posI[0], 0.0);
        const Evec3 unscaledTorqueComJ(-posJ[1], -posJ[0], 0.0);
        const double gammaGuess = normI[2] * (sepDistance - restAngle);
        con.setGamma(2, gammaGuess);
        con.setSep(2, 0.0);
        con.setUnscaledForceComI(2, unscaledForceComI.data());
        con.setUnscaledForceComJ(2, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(2, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(2, unscaledTorqueComJ.data());
    }
    return con;
}

Constraint pivotConstraint(double sepDistance, int gidI, int gidJ, int globalIndexI, int globalIndexJ,
                           const double posI[3], const double posJ[3], const double labI[3], const double labJ[3],
                           const double normI[3], bool oneSide) {
    // set base information
    Constraint con;
    con.id = 0;
    con.numDOF = 3;

    // set constraint info
    con.diagonal = 0; 
    con.oneSide = oneSide;
    con.gidI = gidI;
    con.gidJ = gidJ;
    con.globalIndexI = globalIndexI;
    con.globalIndexJ = globalIndexJ;
    for (int i = 0; i < 3; i++) {
        con.labI[i] = labI[i];
        con.labJ[i] = labJ[i];
    }
    std::function<std::vector<bool>(const double *seps, const double *gammas)> isConstrained = [](const double *seps, const double *gammas) { 
        std::vector<bool> result(3);
        result[0] = true;
        result[1] = true;
        result[2] = true;
        return result;
    };
    std::function<std::vector<double>(const double *seps, const double *gammas)> getValues = [](const double *seps, const double *gammas) {
        std::vector<double> result(3);
        result[0] = seps[0];
        result[1] = seps[1];
        result[2] = seps[2];
        return result;
    };
    con.isConstrained = std::move(isConstrained);
    con.getValues = std::move(getValues);

    // store the dof data
    {
        // x-direction constraint
        const Evec3 unscaledForceComI(1.0, 0.0, 0.0);
        const Evec3 unscaledForceComJ(-1.0, 0.0, 0.0);
        const Evec3 unscaledTorqueComI(0.0, posI[2], posI[1]);
        const Evec3 unscaledTorqueComJ(0.0, -posJ[2], -posJ[1]);
        const double gammaGuess = normI[0] * sepDistance;
        con.setGamma(0, gammaGuess);
        con.setSep(0, 0.0);
        con.setUnscaledForceComI(0, unscaledForceComI.data());
        con.setUnscaledForceComJ(0, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(0, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(0, unscaledTorqueComJ.data());
    }
    {
        // y-direction constraint
        const Evec3 unscaledForceComI(0.0, 1.0, 0.0);
        const Evec3 unscaledForceComJ(0.0, -1.0, 0.0);
        const Evec3 unscaledTorqueComI(posI[2], 0.0, posI[0]);
        const Evec3 unscaledTorqueComJ(-posJ[2], 0.0, -posJ[0]);
        const double gammaGuess = normI[1] * sepDistance;
        con.setGamma(1, gammaGuess);
        con.setSep(1, 0.0);
        con.setUnscaledForceComI(1, unscaledForceComI.data());
        con.setUnscaledForceComJ(1, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(1, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(1, unscaledTorqueComJ.data());
    }
    {
        // z-direction constraint
        const Evec3 unscaledForceComI(0.0, 0.0, 1.0);
        const Evec3 unscaledForceComJ(0.0, 0.0, -1.0);
        const Evec3 unscaledTorqueComI(posI[1], posI[0], 0.0);
        const Evec3 unscaledTorqueComJ(-posJ[1], -posJ[0], 0.0);
        const double gammaGuess = normI[2] * sepDistance;
        con.setGamma(2, gammaGuess);
        con.setSep(2, 0.0);
        con.setUnscaledForceComI(2, unscaledForceComI.data());
        con.setUnscaledForceComJ(2, unscaledForceComJ.data());
        con.setUnscaledTorqueComI(2, unscaledTorqueComI.data());
        con.setUnscaledTorqueComJ(2, unscaledTorqueComJ.data());
    }
    return con;
}
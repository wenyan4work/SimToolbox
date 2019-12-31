/**
 * @file SylinderConfig.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief read configuration parameters from a yaml file
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */
#ifndef SYLINDERCONFIG_HPP_
#define SYLINDERCONFIG_HPP_

#include <iostream>

/**
 * @brief read configuration parameters from a yaml file
 *
 */
class SylinderConfig {
  public:
    // parallel setting
    unsigned int rngSeed = 0;

    // domain setting
    double simBoxHigh[3] = {100.0, 100.0, 100.0}; ///< simulation box size
    double simBoxLow[3] = {0, 0, 0};              ///< simulation box size
    bool simBoxPBC[3] = {false, false, false};    ///< flag of true/false of periodic in that direction
    bool wallLowZ = false;                        ///< a collision wall in xy-plane at z=simBoxLow[2]
    bool wallHighZ = false;                       ///< a collision wall in xy-plane at z=simBoxHigh[2]

    double initBoxHigh[3] = {1.0, 1.0, 1.0}; ///< initialize sylinders within this box
    double initBoxLow[3] = {0, 0, 0};        ///< initialize sylinders within this box
    double initOrient[3] = {2.0, 2.0, 2.0};  ///< initial orientation for each sylinder. >1 <-1 means random
    bool initCircularX = false;              ///< set the initial cross-section as a circle in the yz-plane

    // physical constant
    double viscosity = 0.01; ///< unit pN/(um^2 s), water ~ 0.0009
    double KBT = 0.00411;    ///< pN.um, at 300K

    // sylinder settings
    bool sylinderFixed = false;     ///< sylinders do not move
    int sylinderNumber = 100;       ///< initial number of sylinders
    double sylinderLength = 2.0;    ///< sylinder length (mean if sigma>0)
    double sylinderLengthSigma = 0; ///< sylinder length lognormal distribution sigma
    double sylinderDiameter = 1.0;  ///< sylinder diameter

    // collision radius and diameter
    double sylinderDiameterColRatio = 1.0; ///< collision diameter = ratio * real diameter
    double sylinderLengthColRatio = 1.0;   ///< collision length = ratio * real length

    // time stepping
    double dt = 0.01;        ///< timestep size
    double timeTotal = 0.01; ///< total simulation time
    double timeSnap = 0.01;  ///< snapshot time. save one group of data for each snapshot

    // constraint solver
    bool usePotential = false; ///< TODO: use repulsive potential
    double conResTol = 1e-5;   ///< constraint solver residual
    int conMaxIte = 1e5;       ///< constraint solver maximum iteration
    int conSolverChoice = 0;   ///< choose a iterative solver. 0 for BBPGD, 1 for APGD, etc

    SylinderConfig() = default;
    SylinderConfig(std::string filename);
    ~SylinderConfig() = default;

    void dump() const;
};

#endif
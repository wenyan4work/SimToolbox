/**
 * @file SystemConfig.hpp
 * @author Wen Yan (wenyan4work@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2021-02-02
 * 
 * @copyright Copyright (c) 2021
 * 
 */
#ifndef SYSTEMCONFIG_HPP_
#define SYSTEMCONFIG_HPP_

#include "Boundary/Boundary.hpp"

#include <iostream>
#include <memory>

/**
 * @brief read configuration parameters from a yaml file
 *
 */
class SystemConfig {
  public:
    // parallel setting
    unsigned int rngSeed = 0;

    // domain setting
    double simBoxHigh[3]; ///< simulation box size
    double simBoxLow[3];  ///< simulation box size
    bool simBoxPBC[3];    ///< flag of true/false of periodic in that direction
    bool monolayer;       ///< flag for simulating monolayer on x-y plane

    double initBoxHigh[3]; ///< initialize sylinders within this box
    double initBoxLow[3];  ///< initialize sylinders within this box
    double initOrient[3];  ///< initial orientation for each sylinder. >1 <-1 means random
    bool initCircularX;    ///< set the initial cross-section as a circle in the yz-plane

    // physical constant
    double viscosity; ///< unit pN/(um^2 s), water ~ 0.0009
    double KBT;       ///< pN.um, 0.00411 at 300K
    double linkKappa; ///< pN/um stiffness of sylinder links

    // time stepping
    double dt;        ///< timestep size
    double timeTotal; ///< total simulation time
    double timeSnap;  ///< snapshot time. save one group of data for each snapshot

    // constraint solver
    double conResTol;    ///< constraint solver residual
    int conMaxIte;       ///< constraint solver maximum iteration
    int conSolverChoice; ///< choose a iterative solver. 0 for BBPGD, 1 for APGD, etc

    std::vector<std::shared_ptr<Boundary>> boundaryPtr;

    SystemConfig() = default;
    SystemConfig(std::string filename);
    ~SystemConfig() = default;

    void dump() const;
};

#endif
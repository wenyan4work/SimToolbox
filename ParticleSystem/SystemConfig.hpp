#ifndef SYSTEMCONFIG_HPP_
#define SYSTEMCONFIG_HPP_

#include "Boundary/Boundary.hpp"

#include <iostream>
#include <memory>

/**
 * @brief setup configuration parameters
 *
 */
class SystemConfig {
public:
  unsigned int rngSeed; ///< random number seed
  int logLevel; ///< follows SPDLOG level enum, see Util/Logger.hpp for details
  int timerLevel = 0; ///< how detailed the timer should be

  // domain setting
  double simBoxHigh[3];   ///< simulation box size
  double simBoxLow[3];    ///< simulation box size
  bool simBoxPBC[3];      ///< flag of true/false of periodic in that direction
  bool monolayer = false; ///< flag for simulating monolayer on x-y plane

  // physical constant
  double viscosity; ///< unit pN/(um^2 s), water ~ 0.0009
  double KBT;       ///< pN.um, 0.00411 at 300K

  // particle settings
  bool particleFixed = false;    ///< particles do not move
  double particleBufferAABB = 0; ///< added buffer for particle AABB

  // time stepping
  double dt;        ///< timestep size
  double timeTotal; ///< total simulation time
  double timeSnap;  ///< snapshot time. save one group of data for each snapshot

  // constraint solver
  double conResTol;    ///< constraint solver residual
  int conMaxIte;       ///< constraint solver maximum iteration
  int conSolverChoice; ///< choose a iterative solver. 0 for BBPGD, 1 for APGD,
                       ///< etc

  SystemConfig() = default;
  ~SystemConfig() = default;

  SystemConfig(std::string filename);  ///< parse a file
  SystemConfig(int argc, char **argv); ///< parse cmd args

  void echo() const;
};

#endif
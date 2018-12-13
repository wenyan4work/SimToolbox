/**
 * @file SystemConfig.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief read system configuration from a yaml file
 * @version 1.0
 * @date 2018-12-13
 *
 * @copyright Copyright (c) 2018
 *
 */

#ifndef SYSTEMCONFIG_HPP_
#define SYSTEMCONFIG_HPP_

#include <iostream>

class SystemConfig {
  public:
    // parallel setting
    int ompThreads;
    unsigned int rngSeed;

    // domain setting
    double simBoxHigh[3]; // simulation box size
    double simBoxLow[3];  // simulation box size
    bool simBoxPBC[3];    // flag of true/false of periodic in that direction

    double initBoxHigh[3];            // initial size
    double initBoxLow[3];             // initial size
    double initOrient[3];             // initial orientation for each tubule
    bool initCircularCrossSection[3]; // set the initial x, y, z axis cross-section as a circular

    // physical constant
    double viscosity; // pN/(um^2 s)
    double KBT;       // pN.um

    // number
    int number;

    // time stepping
    double dt;
    double timeTotal;
    double timeSnap;

    /**
     * @brief Construct a new System Config object from a yaml file
     *
     * @param filename
     */
    SystemConfig(std::string filename);

    ~SystemConfig() = default;

    void dump() const;
};

#endif
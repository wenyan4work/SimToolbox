#ifndef SYLINDERCONFIG_HPP_
#define SYLINDERCONFIG_HPP_

#include <iostream>

class SylinderConfig {
  public:
    // parallel setting
    int ompThreads = -1;
    unsigned int rngSeed = 0;

    // domain setting
    double simBoxHigh[3] = {1.0, 1.0, 1.0};    // simulation box size
    double simBoxLow[3] = {0, 0, 0};           // simulation box size
    bool simBoxPBC[3] = {false, false, false}; // flag of true/false of periodic in that direction
    bool wallLowZ = false;
    bool wallHighZ = false;

    double initBoxHigh[3] = {1.0, 1.0, 1.0}; // initial size
    double initBoxLow[3] = {0, 0, 0};        // initial size
    double initOrient[3] = {2.0, 2.0, 2.0};  // initial orientation for each sylinder. >1 <-1 means random
    bool initCircularCrossSection[3] = {false, false, false}; // set the initial cross-section as a circular

    // physical constant
    double viscosity = 1.0; // pN/(um^2 s)
    double KBT = 0.00411;   // pN.um

    int sylinderNumber = 100;
    double sylinderLength = 2.0;
    double sylinderLengthSigma = 0;
    double sylinderDiameter = 1.0;

    // time stepping
    double dt = 0.01;
    double timeTotal = 0.01;
    double timeSnap = 0.01;

    // collision solver
    double colResTol = 0.00001;
    int colMaxIte = 8000;
    bool colNewtonRefine = false;

    SylinderConfig() = default;
    SylinderConfig(std::string filename);
    ~SylinderConfig() = default;

    void dump() const;
};

#endif
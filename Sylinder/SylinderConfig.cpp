#include "SylinderConfig.hpp"

#include <yaml-cpp/yaml.h>

SylinderConfig::SylinderConfig(std::string filename) {

    YAML::Node config = YAML::LoadFile("runConfig.yaml");

    ompThreads = config["ompThreads"].as<int>();
    rngSeed = config["rngSeed"].as<int>();

    YAML::Node seq;
    seq = config["simBoxLow"];
    for (int i = 0; i < 3; i++) {
        simBoxLow[i] = seq[i].as<double>();
    }
    seq = config["simBoxHigh"];
    for (int i = 0; i < 3; i++) {
        simBoxHigh[i] = seq[i].as<double>();
    }
    seq = config["simBoxPBC"];
    for (int i = 0; i < 3; i++) {
        simBoxPBC[i] = seq[i].as<bool>();
    }

    wallLowZ = config["wallLowZ"].as<bool>();
    wallHighZ = config["wallHighZ"].as<bool>();

    seq = config["initBoxLow"];
    for (int i = 0; i < 3; i++) {
        initBoxLow[i] = seq[i].as<double>();
    }
    seq = config["initBoxHigh"];
    for (int i = 0; i < 3; i++) {
        initBoxHigh[i] = seq[i].as<double>();
    }
    seq = config["initOrient"];
    for (int i = 0; i < 3; i++) {
        initOrient[i] = seq[i].as<double>();
    }

    initCircularX = config["initCircularX"].as<bool>();

    viscosity = config["viscosity"].as<double>();
    KBT = config["KBT"].as<double>();

    sylinderNumber = config["sylinderNumber"].as<int>();
    sylinderLength = config["sylinderLength"].as<double>();
    sylinderLengthSigma = config["sylinderLengthSigma"].as<double>();
    sylinderDiameter = config["sylinderDiameter"].as<double>();
    sylinderDiameterColRatio = config["sylinderDiameterColRatio"].as<double>();
    sylinderLengthColRatio = config["sylinderLengthColRatio"].as<double>();

    dt = config["dt"].as<double>();
    timeTotal = config["timeTotal"].as<double>();
    timeSnap = config["timeSnap"].as<double>();

    colResTol = config["colResTol"].as<double>();
    colMaxIte = config["colMaxIte"].as<int>();
    colNewtonRefine = config["colNewtonRefine"].as<bool>();
}

void SylinderConfig::dump() const {
    {
        printf("-------------------------------------------\n");
        printf("Run Setting: \n");
        printf("OpenMP thread number: %d\n", ompThreads);
        printf("Random number seed: %d\n", rngSeed);
        printf("Simulation box Low: %g,%g,%g\n", simBoxLow[0], simBoxLow[1], simBoxLow[2]);
        printf("Simulation box High: %g,%g,%g\n", simBoxHigh[0], simBoxHigh[1], simBoxHigh[2]);
        printf("Periodicity: %d,%d,%d\n", simBoxPBC[0], simBoxPBC[1], simBoxPBC[2]);
        printf("Initialization box Low: %g,%g,%g\n", initBoxLow[0], initBoxLow[1], initBoxLow[2]);
        printf("Initialization box High: %g,%g,%g\n", initBoxHigh[0], initBoxHigh[1], initBoxHigh[2]);
        printf("Initialization orientation: %g,%g,%g\n", initOrient[0], initOrient[1], initOrient[2]);
        printf("Initialization circular cross: %d\n", initCircularX);
        printf("Time step size: %g\n", dt);
        printf("Total Time: %g\n", timeTotal);
        printf("Snap Time: %g\n", timeSnap);
        printf("-------------------------------------------\n");
    }
    {
        printf("Physical setting: \n");
        printf("viscosity: %g\n", viscosity);
        printf("kBT: %g\n", KBT);
        printf("Sylinder Number: %d\n", sylinderNumber);
        printf("Sylinder Length: %g\n", sylinderLength);
        printf("Sylinder Length Sigma: %g\n", sylinderLengthSigma);
        printf("Sylinder Diameter: %g\n", sylinderDiameter);
        printf("-------------------------------------------\n");
        printf("Solver Setting:\n");
        printf("Collision Solver Residual Tolerance: %g\n", colResTol);
        printf("Collision Solver Max Iteration: %d\n", colMaxIte);
        printf("Collision Solver use Newton Refinement: %d\n", colNewtonRefine);
        printf("Sylinder Length Collition Ratio: %g\n", sylinderLengthColRatio);
        printf("Sylinder Diameter Collision Ratio: %g\n", sylinderDiameterColRatio);
        printf("-------------------------------------------\n");
    }
}
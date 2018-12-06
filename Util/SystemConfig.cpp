#include "SystemConfig.hpp"

#include <fstream>
#include <yaml-cpp/yaml.h>

SystemConfig::SystemConfig(std::string filename) {

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
    seq = config["initCircularCrossSection"];
    for (int i = 0; i < 3; i++) {
        initCircularCrossSection[i] = seq[i].as<bool>();
    }

    viscosity = config["viscosity"].as<double>();
    KBT = config["KBT"].as<double>();

    number = config["number"].as<int>();

    dt = config["dt"].as<double>();
    timeTotal = config["timeTotal"].as<double>();
    timeSnap = config["timeSnap"].as<double>();
}

void SystemConfig::dump() const {
    {
        printf("-------------------------------------------\n");
        printf("Run Setting: \n");
        printf("OpenMP thread number: %d\n", ompThreads);
        printf("Random number seed: %d\n", rngSeed);
        printf("Simulation box Low: %lf,%lf,%lf\n", simBoxLow[0], simBoxLow[1], simBoxLow[2]);
        printf("Simulation box High: %lf,%lf,%lf\n", simBoxHigh[0], simBoxHigh[1], simBoxHigh[2]);
        printf("Periodicity: %d,%d,%d\n", simBoxPBC[0], simBoxPBC[1], simBoxPBC[2]);
        printf("Initialization box Low: %lf,%lf,%lf\n", initBoxLow[0], initBoxLow[1], initBoxLow[2]);
        printf("Initialization box High: %lf,%lf,%lf\n", initBoxHigh[0], initBoxHigh[1], initBoxHigh[2]);
        printf("Initialization orientation: %lf,%lf,%lf\n", initOrient[0], initOrient[1], initOrient[2]);
        printf("Initialization circular cross: %d,%d,%d\n", initCircularCrossSection[0], initCircularCrossSection[1],
               initCircularCrossSection[2]);
        printf("Time step size: %lf\n", dt);
        printf("Total Time: %lf\n", timeTotal);
        printf("Snap Time: %lf\n", timeSnap);
        printf("-------------------------------------------\n");
    }
    {
        printf("Physical setting: \n");
        printf("viscosity: %lf\n", viscosity);
        printf("kBT: %lf\n", KBT);
        printf("Number: %d\n", number);
        printf("-------------------------------------------\n");
    }
}
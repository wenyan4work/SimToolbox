#include "SylinderConfig.hpp"

#include "Util/YamlHelper.hpp"

SylinderConfig::SylinderConfig(std::string filename) {

    YAML::Node config = YAML::LoadFile(filename);

    readConfig(config, VARNAME(rngSeed), rngSeed, "");
    readConfig(config, VARNAME(simBoxLow), simBoxLow, 3, "");
    readConfig(config, VARNAME(simBoxHigh), simBoxHigh, 3, "");
    readConfig(config, VARNAME(simBoxPBC), simBoxPBC, 3, "");
    readConfig(config, VARNAME(monolayer), monolayer, "");

    readConfig(config, VARNAME(initBoxLow), initBoxLow, 3, "");
    readConfig(config, VARNAME(initBoxHigh), initBoxHigh, 3, "");
    readConfig(config, VARNAME(initOrient), initOrient, 3, "");

    readConfig(config, VARNAME(initCircularX), initCircularX, "");

    readConfig(config, VARNAME(viscosity), viscosity, "");
    readConfig(config, VARNAME(KBT), KBT, "");

    readConfig(config, VARNAME(sylinderFixed), sylinderFixed, "");
    readConfig(config, VARNAME(sylinderNumber), sylinderNumber, "");
    readConfig(config, VARNAME(sylinderLength), sylinderLength, "");
    readConfig(config, VARNAME(sylinderLengthSigma), sylinderLengthSigma, "");
    readConfig(config, VARNAME(sylinderDiameter), sylinderDiameter, "");
    readConfig(config, VARNAME(sylinderDiameterColRatio), sylinderDiameterColRatio, "");
    readConfig(config, VARNAME(sylinderLengthColRatio), sylinderLengthColRatio, "");
    readConfig(config, VARNAME(sylinderColBuf), sylinderColBuf, "");

    readConfig(config, VARNAME(dt), dt, "");
    readConfig(config, VARNAME(timeTotal), timeTotal, "");
    readConfig(config, VARNAME(timeSnap), timeSnap, "");

    readConfig(config, VARNAME(conResTol), conResTol, "");
    readConfig(config, VARNAME(conMaxIte), conMaxIte, "");
    readConfig(config, VARNAME(conSolverChoice), conSolverChoice, "");
    readConfig(config, VARNAME(linkKappa), linkKappa, "");

    boundaryPtr.clear();
    printf("b\n");

    if (config["boundaries"]) {
        YAML::Node boundaries = config["boundaries"];
        printf("b2\n");
        for (const auto &b : boundaries) {
            std::string name = b["type"].as<std::string>();
            std::cout << name << std::endl;
            if (name == "wall") {
                boundaryPtr.push_back(std::make_shared<Wall>(b));
            } else if (name == "tube") {
                boundaryPtr.push_back(std::make_shared<Tube>(b));
            } else if (name == "sphere") {
                boundaryPtr.push_back(std::make_shared<SphereShell>(b));
            }
        }
    }
}

void SylinderConfig::dump() const {
    {
        printf("-------------------------------------------\n");
        printf("Run Setting: \n");
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
        printf("Link Kappa: %g\n", linkKappa);
        printf("Sylinder Number: %d\n", sylinderNumber);
        printf("Sylinder Length: %g\n", sylinderLength);
        printf("Sylinder Length Sigma: %g\n", sylinderLengthSigma);
        printf("Sylinder Diameter: %g\n", sylinderDiameter);
        printf("Sylinder Length Collision Ratio: %g\n", sylinderLengthColRatio);
        printf("Sylinder Diameter Collision Ratio: %g\n", sylinderDiameterColRatio);
        printf("Sylinder Collision Buffer: %g\n", sylinderColBuf);
        printf("-------------------------------------------\n");
        printf("Constraint Solver Setting:\n");
        printf("Residual Tolerance: %g\n", conResTol);
        printf("Max Iteration: %d\n", conMaxIte);
        printf("Solver Choice: %d\n", conSolverChoice);
        printf("-------------------------------------------\n");
    }
    {
        for (const auto &b : boundaryPtr) {
            b->echo();
        }
    }
}
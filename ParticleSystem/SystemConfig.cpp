#include "SystemConfig.hpp"

#include "Util/Logger.hpp"
#include "Util/TomlHelper.hpp"

SystemConfig::SystemConfig(std::string filename) {

  auto config = toml::parse(filename);

  // required parameters
  readConfig(config, VARNAME(rngSeed), rngSeed, "");
  readConfig(config, VARNAME(simBoxLow), simBoxLow, 3, "");
  readConfig(config, VARNAME(simBoxHigh), simBoxHigh, 3, "");
  readConfig(config, VARNAME(simBoxPBC), simBoxPBC, 3, "");

  readConfig(config, VARNAME(viscosity), viscosity, "");
  readConfig(config, VARNAME(KBT), KBT, "");

  readConfig(config, VARNAME(dt), dt, "");
  readConfig(config, VARNAME(timeTotal), timeTotal, "");
  readConfig(config, VARNAME(timeSnap), timeSnap, "");

  readConfig(config, VARNAME(conResTol), conResTol, "");
  readConfig(config, VARNAME(conMaxIte), conMaxIte, "");
  readConfig(config, VARNAME(conSolverChoice), conSolverChoice, "");

  // optional parameters
  logLevel = spdlog::level::info; // default to info
  readConfig(config, VARNAME(logLevel), logLevel, true);

  timerLevel = logLevel; // default to info
  readConfig(config, VARNAME(timerLevel), timerLevel, true);

  monolayer = false;
  readConfig(config, VARNAME(monolayer), monolayer, true);

  particleFixed = false;
  readConfig(config, VARNAME(particleFixed), particleFixed, true);

  particleBufferAABB = 0.3;
  readConfig(config, VARNAME(particleBufferAABB), particleBufferAABB);
}

void SystemConfig::echo() const {
  {
    printf("-------------------------------------------\n");
    printf("Run Setting: \n");
    printf("Random number seed: %d\n", rngSeed);
    printf("Log Level: %d\n", logLevel);
    printf("Timer Level: %d\n", timerLevel);
    printf("Simulation box Low: %g,%g,%g\n", simBoxLow[0], simBoxLow[1],
           simBoxLow[2]);
    printf("Simulation box High: %g,%g,%g\n", simBoxHigh[0], simBoxHigh[1],
           simBoxHigh[2]);
    printf("Periodicity: %d,%d,%d\n", simBoxPBC[0], simBoxPBC[1], simBoxPBC[2]);
    printf("Time step size: %g\n", dt);
    printf("Total Time: %g\n", timeTotal);
    printf("Snap Time: %g\n", timeSnap);
    printf("-------------------------------------------\n");
  }
  {
    printf("-------------------------------------------\n");
    printf("For drag and collision: Sylinders with length < diameter are "
           "treated as spheres\n");
    printf("-------------------------------------------\n");
  }
  {
    printf("Physical setting: \n");
    printf("viscosity: %g\n", viscosity);
    printf("kBT: %g\n", KBT);
    printf("Particle Fixed: %s", particleFixed ? "true" : "false");
    printf("Particle AABB Buffer: %g\n", particleBufferAABB);
    printf("-------------------------------------------\n");
    printf("Constraint Solver Setting:\n");
    printf("Residual Tolerance: %g\n", conResTol);
    printf("Max Iteration: %d\n", conMaxIte);
    printf("Solver Choice: %d\n", conSolverChoice);
    printf("-------------------------------------------\n");
  }
}
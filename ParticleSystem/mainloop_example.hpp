#ifndef MAINLOOP_EXAMPLE_HPP_
#define MAINLOOP_EXAMPLE_HPP_
#include "ParticleSystem.hpp"

template <class ParticleType>
void mainloop(std::string configFile, std::string posFile) {
  auto configPtr = std::make_shared<const SystemConfig>(configFile);
  configPtr->echo();
  ParticleSystem<ParticleType> system;
  system.initialize(configPtr, posFile);
  if (configPtr->resume) {
    // complete the resumed step
    system.stepMovePtcl();
  }
  system.writeBox();

  // main time loop
  while (system.stepRunning()) {
    system.stepPrepare();
    system.stepCalcMotion();
    system.stepUpdatePtcl();
    if (system.stepWriting()) {
      system.writeData();
      system.writeDataEOT();
    }
    system.stepMovePtcl();

    system.statPolarity();
    system.statStressConB();
    system.statStressConU();
    system.printTimingSummary(true);
  }
}

#endif
#ifndef PARTICLESYSTEM_HPP_
#define PARTICLESYSTEM_HPP_

#include "Particle/Particle.hpp"
#include "SystemConfig.hpp"

#include "Boundary/Boundary.hpp"
#include "Constraint/ConstraintBlock.hpp"
#include "Constraint/ConstraintSolver.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Util/EigenDef.hpp"
#include "Util/TRngPool.hpp"

#include <msgpack.hpp>

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>

template <class ParticleType>
class ParticleSystem {
private:
  long stepID; ///< timestep id. sequentially numbered from 0

  std::shared_ptr<std::ofstream> dataStreamPtr;   ///< output data
  std::shared_ptr<std::ofstream> offsetStreamPtr; ///< offset data
  std::shared_ptr<const SystemConfig> configPtr;

  // TRngPool object for thread-safe random number generation
  std::shared_ptr<TRngPool> rngPoolPtr;
  // Teuchos::MpiComm
  Teuchos::RCP<const TCOMM> commRcp;

  // pointer to ConstraintSolver
  std::shared_ptr<ConstraintSolver> conSolverPtr;
  // pointer to ConstraintCollector
  std::shared_ptr<ConstraintCollector> conCollectorPtr;

  /*********************************************
   *
   *   These are particle basic data
   *
   **********************************************/
  std::vector<ParticleType> particles;
  Teuchos::RCP<TMAP> ptclMapRcp;     ///< TMAP, contiguous and sequentially
                                     ///< ordered 1 dof per sylinder
  Teuchos::RCP<TMAP> ptclMobMapRcp;  ///< TMAP, contiguous and sequentially
                                     ///< ordered 6 dofs per sylinder
  Teuchos::RCP<TCMAT> ptclMobMatRcp; ///< block-diagonal mobility matrix
  Teuchos::RCP<TOP> ptclMobOpRcp;    ///< full mobility operator
                                     ///< (matrix-free), to be implemented

  /*********************************************
   *
   *   These are computed BEFORE the constraint solver
   *
   **********************************************/
  Teuchos::RCP<TV> forcePartNonConRcp;
  Teuchos::RCP<TV> velPartNonConRcp;
  Teuchos::RCP<TV> velNonConRcp;

  // Brownian velocity
  Teuchos::RCP<TV> velBrownRcp;

  // velTotalNonCon = velNonCon + velBrown
  Teuchos::RCP<TV> velTotalNonConRcp;

  /*********************************************
   *
   *   These are computed by the constraint solver
   *
   **********************************************/
  Teuchos::RCP<TV> forceConURcp; ///< unilateral constraint force
  Teuchos::RCP<TV> forceConBRcp; ///< bilateral constraint force
  Teuchos::RCP<TV> velConURcp;   ///< unilateral constraint velocity
  Teuchos::RCP<TV> velConBRcp;   ///< bilateral constraint velocity

public:
  ParticleSystem() = default;
  ~ParticleSystem() = default;

  ParticleSystem(const ParticleSystem &) = delete;
  ParticleSystem(ParticleSystem &&) = delete;
  ParticleSystem &operator=(const ParticleSystem &) = delete;
  ParticleSystem &operator=(ParticleSystem &&) = delete;

  /*********************************************
   *
   *   member getters
   *
   **********************************************/

  auto getStepID() { return stepID; }
  auto getConfig() { return configPtr; }
  auto getRngPool() { return rngPoolPtr; }
  auto getComm() { return commRcp; }
  auto getDataFS() { return dataStreamPtr; }

  auto getConSolver() { return conSolverPtr; }
  auto getConCollector() { return conCollectorPtr; }

  auto getPtclContainer() { return particles; }
  auto getPtclMap() { return ptclMapRcp; }
  auto getPtclMobMap() { return ptclMobMapRcp; }
  auto getPtclMobMat() { return ptclMobMatRcp; }
  auto getPtclMobOp() { return ptclMobOpRcp; }

  auto getForcePartNonCon() { return forcePartNonConRcp; }
  auto getVelPartNonCon() { return velPartNonConRcp; }
  auto getVelNonCon() { return velNonConRcp; }
  auto getVelBrown() { return velBrownRcp; }
  auto getVelTotalNonCon() { return velTotalNonConRcp; }

  auto getForceConU() { return forceConURcp; }
  auto getForceConB() { return forceConBRcp; }
  auto getVelConU() { return velConURcp; }
  auto getVelConB() { return velConBRcp; }

  /*********************************************
   *
   *   member setters
   *
   **********************************************/
  void setForcePartNonCon(const std::vector<double> &forcePartNonCon) {
    const int nLocal = particles.size();
    TEUCHOS_ASSERT(ptclMapRcp->getNodeNumElements() == nLocal);
    TEUCHOS_ASSERT(ptclMobMapRcp->getNodeNumElements() == 6 * nLocal);
    forcePartNonConRcp = getTVFromVector(forcePartNonCon, commRcp);
  }

  void setVelPartNonCon(const std::vector<double> &velPartNonCon) {
    const int nLocal = particles.size();
    TEUCHOS_ASSERT(ptclMapRcp->getNodeNumElements() == nLocal);
    TEUCHOS_ASSERT(ptclMobMapRcp->getNodeNumElements() == 6 * nLocal);
    velPartNonConRcp = getTVFromVector(velPartNonCon, commRcp);
  }

  /*********************************************
   *
   *   calculations
   *
   **********************************************/

  /**
   * @brief calculate Brownian velocity
   *
   */
  void calcVelBrown() {
    const int nLocal = particles.size();
    const double Pi = 3.1415926535897932384626433;
    const double mu = configPtr->viscosity;
    const double dt = configPtr->dt;
    const double delta = dt * 0.1; // a small parameter used in RFD algorithm
    const double kBT = configPtr->KBT;
    const double kBTfactor = sqrt(2 * kBT / dt);

    velBrownRcp = Teuchos::rcp<TV>(new TV(ptclMobMapRcp, true));
    auto velBrownPtr = velBrownRcp->getLocalView<Kokkos::HostSpace>();
    velBrownRcp->modify<Kokkos::HostSpace>();

#pragma omp parallel
    {
      const int threadId = omp_get_thread_num();
#pragma omp for
      for (int i = 0; i < nLocal; i++) {
        auto &par = particles[i];
        std::array<double, 12> rngN01s;
        for (auto &v : rngN01s) {
          v = rngPoolPtr->getN01(threadId);
        }
        Evec6 vel = par.getVelBrown(rngN01s, dt, kBT, mu);
        for (int j = 0; j < 6; j++) {
          velBrownPtr(6 * i + j, 0) = vel[j];
        }
      }
    }
  }

  /**
   * @brief calculate total non-constraint velocity
   *
   *  \f$U_{TotalNonCon} = U_{Brown} + U_{Part,NonCon} + M_{UF}
   * F_{Part,NonCon}\f$
   */
  void calcVelTotalNonCon() {
    const int nLocal = particles.size();
    TEUCHOS_ASSERT(ptclMobMapRcp->getNodeNumElements() == 6 * nLocal);
    TEUCHOS_ASSERT(!velTotalNonConRcp.is_null());
    TEUCHOS_ASSERT(velTotalNonConRcp->getLocalLength() == 6 * nLocal);
    velTotalNonConRcp->putScalar(0.0);

    // step 1, force part
    if (!forcePartNonConRcp.is_null()) {
      TEUCHOS_ASSERT(!ptclMobOpRcp.is_null());
      TEUCHOS_ASSERT(forcePartNonConRcp->getLocalLength() == 6 * nLocal);
      // U_{TotalNonCon} = M_{UF} F_{Part,NonCon}
      ptclMobOpRcp->apply(*forcePartNonConRcp, *velTotalNonConRcp);
    }

    // step 2, vel part
    if (!velPartNonConRcp.is_null()) {
      TEUCHOS_ASSERT(velPartNonConRcp->getLocalLength() == 6 * nLocal);
      // U_{TotalNonCon} += U_{Part,NonCon}
      velTotalNonConRcp->update(1.0, *velPartNonConRcp, 1.0);
    }

    // step 3, velBrown
    if (!velBrownRcp.is_null()) {
      TEUCHOS_ASSERT(velBrownRcp->getLocalLength() == 6 * nLocal);
      // U_{TotalNonCon} += U_{Brown}
      velTotalNonConRcp->update(1.0, *velBrownRcp, 1.0);
    }
  }

  /**
   * @brief update the mpi rank for particles
   *
   */
  void updatePtclRank() {
    const int nLocal = particles.size();
    const int rank = commRcp->getRank();
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
      particles[i].rank = rank;
    }
  }

  /**
   * @brief update the particle map and particle mobility map
   *
   * This function is called in prepareStep(),
   * and no adding/removing/exchanging is allowed before runStep()
   */
  void updatePtclMap() {
    const int nLocal = particles.size();
    ptclMapRcp = getTMAPFromLocalSize(nLocal, commRcp);
    ptclMobMapRcp = getTMAPFromLocalSize(nLocal * 6, commRcp);

    // setup the globalIndex
    long globalIndexBase =
        ptclMapRcp->getMinGlobalIndex(); // this is a contiguous map
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
      particles[i].globalIndex = i + globalIndexBase;
    }
  }

  /**
   * @brief apply periodic boundary condition
   *
   */
  void applyBoxPBC() { return; }

  void applyMonoLayer() {
    //         if (configPtr->monolayer) {
    //       const double monoZ =
    //           (configPtr->simBoxHigh[2] + configPtr->simBoxLow[2]) / 2;
    // #pragma omp parallel for
    //       for (int i = 0; i < nLocal; i++) {
    //         auto &par = particles[i];
    //         par.pos[2] = monoZ;
    //         Evec3 drt = Emapq(par.quaternion.data()) * Evec3(0, 0, 1);
    //         drt[2] = 0;
    //         drt.normalize();
    //         Emapq(par.quaternion.data()).setFromTwoVectors(Evec3(0, 0, 1),
    //         drt);
    //       }
    //     }

    if (configPtr->monolayer) {
      const int nLocal = particles.size();
#pragma omp parallel for
      for (int i = 0; i < nLocal; i++) {
        particles[i].vel[3] = 0; // vz
        particles[i].vel[4] = 0; // wx
        particles[i].vel[5] = 0; // wy
      }
    }
    return;
  }

  /*********************************************
   *
   *   system high level API
   *
   **********************************************/
  void initialize(std::shared_ptr<const SystemConfig> &configPtr_,
                  const std::string &posFile_) {
    commRcp = Tpetra::getDefaultComm();
    stepID = 0;

    configPtr = configPtr_;
    if (!commRcp->getRank()) {
      configPtr->echo();
    }
    if (commRcp->getRank() == 0) {
      IOHelper::makeSubFolder("./result"); // prepare the output directory
      writeBox();
    }

    Logger::set_level(configPtr->logLevel);

    conSolverPtr = std::make_shared<ConstraintSolver>();
    conCollectorPtr = std::make_shared<ConstraintCollector>();

    particles.clear();

    if (configPtr->resume) {
      rngPoolPtr = std::make_shared<TRngPool>(
          configPtr->rngSeed + 1, commRcp->getRank(), commRcp->getSize());
      resume();
    } else {
      rngPoolPtr = std::make_shared<TRngPool>(
          configPtr->rngSeed, commRcp->getRank(), commRcp->getSize());
      if (IOHelper::fileExist(posFile_)) {
        readFromDatFile(posFile_);
      }
      spdlog::warn("ParticleSystem Initialized. {} local particles",
                   particles.size());
    }

    const auto &filenames = getDataFileNames();
    std::ios_base::openmode mode;
    if (IOHelper::fileExist(filenames.first) &&
        IOHelper::fileExist(filenames.second)) {
      mode = std::ios_base::app;
    } else {
      mode = std::ios_base::out;
    }
    dataStreamPtr = std::make_shared<std::ofstream>(filenames.first, mode);
    offsetStreamPtr = std::make_shared<std::ofstream>(filenames.second, mode);
  }

  bool stepRunning() {
    return configPtr->dt * stepID < configPtr->timeTotal ? true : false;
  }

  bool stepWriting() {
    return stepID % static_cast<long>(configPtr->timeSnap / configPtr->dt) == 0
               ? true
               : false;
  }

  void stepPrepare() {
    spdlog::warn("CurrentStep {}", stepID);
    applyBoxPBC();

    updatePtclRank();
    updatePtclMap();

    const int nLocal = particles.size();
    const double buffer = configPtr->particleBufferAABB;
#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
      auto &par = particles[i];
      par.clear();
      par.buffer = buffer;
    }

    velTotalNonConRcp = Teuchos::rcp(new TV(ptclMobMapRcp, true));

    conCollectorPtr->clear();
    conSolverPtr->reset();

    forcePartNonConRcp.reset();
    velPartNonConRcp.reset();
    velNonConRcp.reset();

    velBrownRcp.reset();

    forceConURcp.reset();
    forceConBRcp.reset();
    velConURcp.reset();
    velConBRcp.reset();
  }

  void stepCalcMotion() {

    if (configPtr->KBT > 0) {
      calcVelBrown();
    }

    calcVelTotalNonCon();
  }

  /**
   * @brief write vel/force back to particle struct
   *
   */
  void stepUpdatePtcl() {
    const int nLocal = particles.size();

    auto writeBack = [&](Teuchos::RCP<TV> &vector,
                         std::array<double, 6> ParticleType::*mptr) {
      if (!vector.is_null()) {
        auto vecView = vector->getLocalView<Kokkos::HostSpace>();
#pragma omp parallel for
        for (int i = 0; i < nLocal; i++) {
          auto &data = particles[i].*mptr;
          for (int j = 0; j < 6; j++) {
            data[j] = vecView(6 * i + j, 0);
          }
        }
      }
    };

    writeBack(forcePartNonConRcp, &ParticleType::forcePartNonCon);
    writeBack(velPartNonConRcp, &ParticleType::velPartNonCon);
    writeBack(velTotalNonConRcp, &ParticleType::velTotalNonCon);
    writeBack(velBrownRcp, &ParticleType::velBrown);
    writeBack(velConBRcp, &ParticleType::velConB);
    writeBack(velConURcp, &ParticleType::velConU);
    writeBack(forceConBRcp, &ParticleType::forceConB);
    writeBack(forceConURcp, &ParticleType::forceConU);

#pragma omp parallel for
    for (int i = 0; i < nLocal; i++) {
      auto &par = particles[i];
      for (int k = 0; k < 6; k++) {
        par.vel[k] = par.velConB[k] + par.velConU[k] + par.velTotalNonCon[k];
      }
    }
  }

  void stepMovePtcl() {
    stepID++;
    const int nLocal = particles.size();
    const double dt = configPtr->dt;

    applyMonoLayer();

    if (!configPtr->particleFixed) {
#pragma omp parallel for
      for (long i = 0; i < nLocal; i++) {
        auto &sy = particles[i];
        sy.stepEuler(dt);
      }
    }
  }

  /*********************************************
   *
   *   read write
   *
   **********************************************/

  /**
   * @brief Get Data File Name
   *
   * @return auto
   */
  auto getDataFileNames() {
    std::string dataname =
        "./result/data_r" + std::to_string(commRcp->getRank()) + ".msgpack";
    std::string offsetname =
        "./result/offset_r" + std::to_string(commRcp->getRank()) + ".txt";
    return std::make_pair(dataname, offsetname);
  }

  /**
   * @brief write a simple legacy VTK file for simBox
   *
   */
  void writeBox() const {
    FILE *boxFile = fopen("./result/simBox.vtk", "w");
    fprintf(boxFile, "# vtk DataFile Version 3.0\n");
    fprintf(boxFile, "vtk file\n");
    fprintf(boxFile, "ASCII\n");
    fprintf(boxFile, "DATASET RECTILINEAR_GRID\n");
    fprintf(boxFile, "DIMENSIONS 2 2 2\n");
    fprintf(boxFile, "X_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", configPtr->simBoxLow[0],
            configPtr->simBoxHigh[0]);
    fprintf(boxFile, "Y_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", configPtr->simBoxLow[1],
            configPtr->simBoxHigh[1]);
    fprintf(boxFile, "Z_COORDINATES 2 float\n");
    fprintf(boxFile, "%g %g\n", configPtr->simBoxLow[2],
            configPtr->simBoxHigh[2]);
    fprintf(boxFile, "CELL_DATA 1\n");
    fprintf(boxFile, "POINT_DATA 8\n");
    fclose(boxFile);
  }

  /**
   * @brief read ascii dat file to rank 0
   *
   * @param datFile
   */
  void readFromDatFile(const std::string &datFile) {
    spdlog::warn("Reading file " + datFile);

    if (!commRcp->getRank()) {
      std::string line;
      particles.clear();

      std::ifstream myfile(datFile);
      while (std::getline(myfile, line)) {
        if (line[0] == '#') {
          // this is a comment line
          continue;
        } else {
          // parse a particle
          std::stringstream liness(line);
          particles.emplace_back(liness);
        }
      }
      myfile.close();
    }

    spdlog::debug("Particle number in file: {} ", particles.size());
  }

  /**
   * @brief serialize everything sequentially in msgpack format
   *
   * offset file records the beginning position of every timestep in datafile
   */
  void writeData() const {
    *offsetStreamPtr << stepID << " " << dataStreamPtr->tellp() << std::endl;
    offsetStreamPtr->flush();

    msgpack::packer<std::ofstream> opacker(*dataStreamPtr);
    opacker.pack("stepID");
    opacker.pack(stepID);
    opacker.pack("particles");
    opacker.pack(particles);
  }

  /**
   * @brief mark the end of data in current timestep
   *
   */
  void writeDataEOT() const {
    msgpack::packer<std::ofstream> opacker(*dataStreamPtr);
    opacker.pack("EOT"); // end of current timestep
    dataStreamPtr->flush();
  }

  /**
   * @brief resume from every rank's data and offset files
   *
   * data and offset files must be valid and match each other.
   *
   */
  void resume() {
    const auto &filenames = getDataFileNames();

    // step 1, read last line
    std::ifstream ioffset(filenames.second);
    long istep = -1, offset = 0;
    for (std::string line; std::getline(ioffset, line);) {
      std::stringstream liness(line);
      long a, b;
      liness >> a >> b;
      if (a <= istep || b < offset) {
        std::cout << "rank " << commRcp->getRank()
                  << " offset file error at line " << line << std::endl;
        std::exit(1);
      }
      istep = a;
      offset = b;
    }
    ioffset.close();

    // resume from last valid frame of data
    std::ifstream idata(filenames.first, std::ios::binary);
    idata.seekg(offset);

    std::stringstream buffer;
    buffer << idata.rdbuf();
    const std::string &sbuf = buffer.str();
    std::size_t len = sbuf.size();
    std::size_t off = 0;

    auto parse = [&](std::string header) {
      msgpack::object_handle result;
      // Unpack the first msgpack data.
      // off is updated when function is returned.
      result = msgpack::unpack(sbuf.data(), len, off);
      std::string msg = result.get().as<std::string>();
      if (msg != header) {
        spdlog::critical(" parsing error ", msg);
        std::exit(1);
      }
      result = msgpack::unpack(sbuf.data(), len, off);
      return result.get();
    };

    msgpack::object obj;
    obj = parse("stepID");
    this->stepID = obj.as<long>();
    obj = parse("particles");
    this->particles = obj.as<std::vector<ParticleType>>();

    assert(stepID == istep);
    assert(len == off);
    idata.close();
  }

  /*********************************************
   *
   *   statistics
   *
   **********************************************/

  /**
   * @brief calculate particle volume
   *
   * @return double total particle volume
   */
  double statPtclVol() const {
    double lclPtclVol = 0;
    const int nLocal = particles.size();
#pragma omp parallel for reduction(+ : lclPtclVol)
    for (int i = 0; i < nLocal; i++) {
      lclPtclVol += particles[i].getVolume();
    }

    double glbPtclVol = lclPtclVol;
    MPI_Allreduce(MPI_IN_PLACE, &glbPtclVol, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);

    return glbPtclVol;
  }

  std::array<double, 9> statStressConB() const {
    std::array<double, 9> stress;
    return stress;
  }

  std::array<double, 9> statStressConU() const {
    std::array<double, 9> stress;
    return stress;
  }

  std::array<double, 3> statPolarity() const {
    std::array<double, 3> polarity;
    return polarity;
  }

  /*********************************************
   *
   *   print info
   *
   **********************************************/

  void printTimingSummary(const bool zeroOut) {
    if (configPtr->timerLevel <= spdlog::level::info)
      Teuchos::TimeMonitor::summarize();
    if (zeroOut)
      Teuchos::TimeMonitor::zeroOutTimers();
  }

  /**
   * @brief display the configuration on rank 0
   *
   */
  void echo() const {
    if (commRcp->getRank() == 0) {
      for (const auto &p : particles) {
        p.echo();
      }
    }
  }
};

#endif
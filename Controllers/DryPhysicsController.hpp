/**
 * @file DryPhysicsController.hpp
 * @author wenyan4work (wenyan4work@gmail.com)
 * @brief Physics controller for dry particle physics
 * @version 1.0
 * @date 6-27-2022
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef DRYPHYSICSCONTROLLER_HPP_
#define DRYPHYSICSCONTROLLER_HPP_

#include "Sylinder/SylinderSystem.hpp"
#include "Boundary/Boundary.hpp"
#include "Constraint/NOXConstraintSolver.hpp"
#include "FDPS/particle_simulator.hpp"
#include "Trilinos/TpetraUtil.hpp"
#include "Trilinos/ZDD.hpp"
#include "Util/TRngPool.hpp"
#include "Util/Logger.hpp"

#include <unordered_map>

/**
 * @brief A master controller that aims at modeling a class of physical systems
 *
 */
class DryPhysicsController {
    bool enableTimer = false;
    int snapID;                  ///< the current id of the snapshot file to be saved. sequentially numbered from 0
    int stepCount;               ///< timestep Count. sequentially numbered from 0
    unsigned int restartRngSeed; ///< parallel seed used by restarted simulations

    // singletons
    std::shared_ptr<SylinderSystem> ptcSystemPtr;       ///< pointer to SylinderSystem
    std::shared_ptr<ConstraintSolver> conSolverPtr;       ///< pointer to ConstraintSolver
    std::shared_ptr<ConstraintCollector> conCollectorPtr; ///<  pointer to ConstraintCollector

    // MPI stuff
    std::shared_ptr<TRngPool> rngPoolPtr;      ///< TRngPool object for thread-safe random number generation
    Teuchos::RCP<const TCOMM> commRcp;         ///< TCOMM, set as a Teuchos::MpiComm object in constrctor

  public:
    SylinderConfig runConfig;   ///< sylinder system configuration. Be careful if this is modified on the fly
    // SimulationConfig runConfig; 
    ///< TODO: create two configs or one massive yaml.
    ///< TODO: give everyone sharedpointers to runConfig, no coppies!

    /**
     * @brief Construct a new DryPhysicsController object
     *
     * initialize() should be called after this constructor
     */
    DryPhysicsController() = default;

    /**
     * @brief Construct a new DryPhysicsController object
     *
     * This constructor calls initialize() internally
     * @param configFile a yaml file for SylinderConfig
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    DryPhysicsController(const std::string &configFile, const std::string &posFile);

    /**
     * @brief Construct a new DryPhysicsController object
     *
     * This constructor calls initialize() internally
     * @param config SylinderConfig object
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    DryPhysicsController(const SylinderConfig &config, const std::string &posFile);

    ~DryPhysicsController() = default;

    // forbid copy
    DryPhysicsController(const DryPhysicsController &) = delete;
    DryPhysicsController &operator=(const DryPhysicsController &) = delete;

    /**
     * @brief initialize after an empty constructor
     *
     * @param config SylinderConfig object
     * @param posFile initial configuration. use empty string ("") for no such file
     * @param argc command line argument
     * @param argv command line argument
     */
    void initialize(const SylinderConfig &config, const std::string &posFile);

    /**
     * @brief reinitialize from vtk files
     *
     * @param config SylinderConfig object
     * @param restartFile txt file containing timestep and most recent pvtp file names
     * @param argc command line argument
     * @param argv command line argument
     */
    void reinitialize(const SylinderConfig &config, const std::string &restartFile,
                      bool eulerStep = true);

    /**
     * @brief run the simulation timeloop
     *
     */
    void run();

    /**
     * @brief write a txt file containing timestep and most recent pvtp filenames info into baseFolder
     *
     * @param baseFolder
     */
    void writeTimeStepInfo(const std::string &baseFolder, const std::string &postfix);


    /**
     * @brief enable the timer in step()
     *
     * @param value
     */
    void setTimer(bool value) { enableTimer = value; }

    /**
     * one-step high level API
     */
    // get information
    /**
     * @brief Get the RngPoolPtr object
     *
     * @return std::shared_ptr<TRngPool>&
     */
    std::shared_ptr<TRngPool> &getRngPoolPtr() { return rngPoolPtr; }

    /**
     * @brief Get the CommRcp object
     *
     * @return Teuchos::RCP<const TCOMM>&
     */
    const Teuchos::RCP<const TCOMM> &getCommRcp() { return commRcp; }

    ConstraintBlockPool &getConstraintPoolNonConst() { return *(conCollectorPtr->constraintPoolPtr); };

    // write results
    std::string getCurrentResultFolder();          ///< get the current output folder path
    std::string getResultFolderWithID(int snapID); ///< get output folder path with snapID
    bool getIfWriteResultCurrentStep();            ///< check if the current step is writing (set by runConfig)
    int getSnapID() { return snapID; };            ///< get the (sequentially ordered) ID of current snapshot
    int getStepCount() { return stepCount; };      ///< get the (sequentially ordered) count of steps executed

    /**
     * @brief
     *
     * @param zeroOut zero out all timing info after printing out
     */
    void printTimingSummary(const bool zeroOut = true);
};

#endif
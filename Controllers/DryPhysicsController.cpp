#include "DryPhysicsController.hpp"

#include "MPI/CommMPI.hpp"
#include "Util/EquatnHelper.hpp"
#include "Util/GeoUtil.hpp"
#include "Util/IOHelper.hpp"
#include "Util/Logger.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <vector>
#include <chrono>

#include <vtkCellData.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkTypeInt32Array.h>
#include <vtkTypeUInt8Array.h>
#include <vtkXMLPPolyDataReader.h>
#include <vtkXMLPolyDataReader.h>

#include <mpi.h>
#include <omp.h>

DryPhysicsController::DryPhysicsController(const std::string &configFile, const std::string &posFile) {
    initialize(SylinderConfig(configFile), posFile);
}

DryPhysicsController::DryPhysicsController(const SylinderConfig &runConfig_, const std::string &posFile) {
    initialize(runConfig_, posFile);
}

void DryPhysicsController::initialize(const SylinderConfig &runConfig_, const std::string &posFile) {
    runConfig = runConfig_;
    stepCount = 0;
    snapID = 0; // the first snapshot starts from 0 in writeResult

    // store the random seed
    restartRngSeed = runConfig.rngSeed;

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);

    Logger::set_level(runConfig.logLevel);
    commRcp = getMPIWORLDTCOMM();

    // create a result folder on rank 0
    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); // prepare the output directory
    }

    // dump the settings on rank 0
    if (commRcp->getRank() == 0) {
        printf("-----------SylinderSystem Settings-----------\n");
        runConfig.dump();
    }

    // Initialize singletons

    // Notes on interactions:
    //  -All systems and interactions contain write access to conCollector for creating constraints
    //  -All systems have read only access to all other systems
    //  -Constraint solver contains write access to all systems

    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(runConfig.rngSeed);
    conCollectorPtr = std::make_shared<ConstraintCollector>();
    ptcSystemPtr = std::make_shared<SylinderSystem>(runConfig, commRcp, rngPoolPtr, conCollectorPtr);
    conSolverPtr = std::make_shared<PGDConstraintSolver>(commRcp, conCollectorPtr, ptcSystemPtr); 
    ptcSystemPtr->initialize(posFile);
    
    // prepare the output directory
    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); 
    }

    // write the initial, potentially overlapped results
    std::string baseFolder = getCurrentResultFolder();
    IOHelper::makeSubFolder(baseFolder);
    const std::string postfix = std::to_string(-1);
    ptcSystemPtr->writeResult(stepCount, baseFolder, postfix); //TODO: split this function into two: one for ptcSystem and one for ConCollector

    // Collect and resolve the initial constraints 
    // the initial configuration may be such that no non-overlapping solution exists to the nonlinear collision problem
    // we take initPreSteps initial collision resolution steps with 1 recursion each to counteract this
    spdlog::warn("Initial Constraint Resolution Begin");
    for (int i = 0; i < runConfig.initPreSteps; i++) {
        // preconstraint stuff
        conCollectorPtr->clear();
        
        ptcSystemPtr->prepareStep(stepCount);
        if (runConfig.monolayer) { 
            ptcSystemPtr->applyMonolayer();   // TODO: applyMonolayer is VERY hacky and needs updated
            ptcSystemPtr->advanceParticles(); // advance particles is always necessary after applying monbolayer
        }
        ptcSystemPtr->updateSylinderMap();
        ptcSystemPtr->calcMobOperator();
        ptcSystemPtr->buildsylinderNearDataDirectory();

        // constraint stuff
        ptcSystemPtr->collectConstraints(); 
        conSolverPtr->setup(runConfig.dt, runConfig.conResTol, runConfig.conMaxIte, 1, runConfig.conSolverChoice);

        Teuchos::RCP<Teuchos::Time> solveTimer = Teuchos::TimeMonitor::getNewCounter("ConstraintSolver::run");
        {
            Teuchos::TimeMonitor mon(*solveTimer);
            conSolverPtr->resolveConstraints(); 
        }

        // store the current configuration
        ptcSystemPtr->advanceParticles();
    }

    spdlog::warn("Initial Collision Resolution End");
}

void DryPhysicsController::reinitialize(const SylinderConfig &runConfig_, const std::string &restartFile, bool eulerStep) {
    runConfig = runConfig_;

    // Read the timestep information and pvtp filenames from restartFile
    std::string pvtpFileName;
    std::ifstream myfile(restartFile);

    myfile >> restartRngSeed;
    myfile >> stepCount;
    myfile >> snapID;
    myfile >> pvtpFileName;

    // increment the rngSeed forward by one to ensure randomness compared to previous run
    restartRngSeed++;

    // set MPI
    int mpiflag;
    MPI_Initialized(&mpiflag);
    TEUCHOS_ASSERT(mpiflag);

    Logger::set_level(runConfig.logLevel);
    commRcp = getMPIWORLDTCOMM();

    // create a result folder on rank 0
    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); // prepare the output directory
    }

    // dump the settings on rank 0
    if (commRcp->getRank() == 0) {
        printf("-----------SylinderSystem Settings-----------\n");
        runConfig.dump();
    }

    // initialize sub-classes
    // TRNG pool must be initialized after mpi is initialized
    rngPoolPtr = std::make_shared<TRngPool>(restartRngSeed);
    conCollectorPtr = std::make_shared<ConstraintCollector>();
    ptcSystemPtr = std::make_shared<SylinderSystem>(runConfig, commRcp, rngPoolPtr, conCollectorPtr);
    conSolverPtr = std::make_shared<PGDConstraintSolver>(commRcp, conCollectorPtr, ptcSystemPtr); 
    std::string baseFolder = getCurrentResultFolder();
    pvtpFileName = baseFolder + pvtpFileName;
    ptcSystemPtr->reinitialize(pvtpFileName);

    // misc
    // VTK data is wrote before the Euler step, thus we need to run one Euler step below
    if (eulerStep) {
        ptcSystemPtr->stepEuler();
        ptcSystemPtr->advanceParticles();
    }

    stepCount++;
    snapID++;
}

void DryPhysicsController::run() {
    while (getStepCount() * runConfig.dt < runConfig.timeTotal) {
        ///////////
        // setup //
        ///////////
        spdlog::warn("CurrentStep {}", stepCount);
        std::string baseFolder = getCurrentResultFolder();
        IOHelper::makeSubFolder(baseFolder);

        ////////////////////
        // pre-step stuff //
        ////////////////////
        // compute growth and devision (if necessary)
        if (runConfig.ptcGrowth.size() > 0) {
            ptcSystemPtr->calcSylinderGrowth(conSolverPtr->getParticleStress());
            ptcSystemPtr->calcSylinderDivision();
            ptcSystemPtr->advanceParticles(); // advance particles is always necessary after new particles are generated 
                                              // TODO: the current state should always be initialized upon construction
                                              //       delete this call to advance particles after this fix is done
        }
        // empty the constraint collector
        conCollectorPtr->clear();
        
        // TODO: completely desolve the prepareStep function into other, clear API calls. 
        //       I want this class to have full control of what is being called and in what order. 
        ptcSystemPtr->prepareStep(stepCount);
        if (runConfig.monolayer) { 
            ptcSystemPtr->applyMonolayer();   // TODO: applyMonolayer is VERY hacky and needs updated
            ptcSystemPtr->advanceParticles(); // advance particles is always necessary after applying monbolayer
        }
        ptcSystemPtr->updateSylinderMap();
        ptcSystemPtr->calcMobOperator();

        // compute the Brownian veloicity (if necessary)
        if (runConfig.KBT > 0) { ptcSystemPtr->calcVelocityBrown(); }

        ////////////////////////
        // unconstrained step //
        ////////////////////////
        // apply the unconstrained motion
        ptcSystemPtr->stepEuler();  // q_{r}^{k+1} = q^k + dt G^k (U_r^k + U_ext^k)
        ptcSystemPtr->applyBoxBC(); // TODO: this might not work. We'll see!
        spdlog::debug("particle position updated");

        // store the particle map and collect the constraints
        // the initial constraints are those induced by the external forces and velocities
        ptcSystemPtr->buildsylinderNearDataDirectory();
        ptcSystemPtr->collectConstraints(); 
        spdlog::debug("initial constraints collected");

        //////////////////////
        // constrained step //
        //////////////////////
        // solve for the constrained motion
        conSolverPtr->setup(runConfig.dt, runConfig.conResTol, runConfig.conMaxIte, 100, runConfig.conSolverChoice);

        Teuchos::RCP<Teuchos::Time> solveTimer = Teuchos::TimeMonitor::getNewCounter("ConstraintSolver::run");
        {
            Teuchos::TimeMonitor mon(*solveTimer);
            conSolverPtr->resolveConstraints(); 
        }

        // merge the constraint and nonconstraint vel and force
        // TODO: having a total force and velocity is unnecessary. Delete it from sylinder object but still write it out to vtk.
        ptcSystemPtr->sumForceVelocity();

        // data output stuff
        if (getIfWriteResultCurrentStep()) {
            // write result before moving. guarantee data written is consistent to geometry
            const std::string postfix = std::to_string(snapID);
            ptcSystemPtr->writeResult(stepCount, baseFolder, postfix); //TODO: split this function into two: one for ptcSystem and one for ConCollector
            writeTimeStepInfo(baseFolder, postfix);
            snapID++;
        }

        /////////////////////
        // post-step stuff //
        /////////////////////
        ptcSystemPtr->advanceParticles(); 
        ptcSystemPtr->calcOrderParameter();
        ptcSystemPtr->calcConStress();
        printTimingSummary();
        stepCount++;
    }
}

void DryPhysicsController::writeTimeStepInfo(const std::string &baseFolder, const std::string &postfix) {
    if (commRcp->getRank() == 0) {
        // write a single txt file containing timestep and most recent pvtp file names
        std::string name = baseFolder + std::string("../../TimeStepInfo.txt");
        std::string pvtpFileName = std::string("Sylinder_") + postfix + std::string(".pvtp");

        FILE *restartFile = fopen(name.c_str(), "w");
        fprintf(restartFile, "%u\n", restartRngSeed);
        fprintf(restartFile, "%u\n", stepCount);
        fprintf(restartFile, "%u\n", snapID);
        fprintf(restartFile, "%s\n", pvtpFileName.c_str());
        fclose(restartFile);
    }
}

std::string DryPhysicsController::getCurrentResultFolder() { return getResultFolderWithID(this->snapID); }

std::string DryPhysicsController::getResultFolderWithID(int snapID_) {
    TEUCHOS_ASSERT(nonnull(commRcp));
    const int num = std::max(400 / commRcp->getSize(), 1); // limit max number of files per folder
    int k = snapID_ / num;
    int low = k * num, high = k * num + num - 1;
    std::string baseFolder =
        "./result/result" + std::to_string(low) + std::string("-") + std::to_string(high) + std::string("/");
    return baseFolder;
}

bool DryPhysicsController::getIfWriteResultCurrentStep() {
    return (stepCount % static_cast<int>(runConfig.timeSnap / runConfig.dt) == 0);
}

void DryPhysicsController::printTimingSummary(const bool zeroOut) {
    if (runConfig.timerLevel <= spdlog::level::info)
        Teuchos::TimeMonitor::summarize();
    if (zeroOut)
        Teuchos::TimeMonitor::zeroOutTimers();
}

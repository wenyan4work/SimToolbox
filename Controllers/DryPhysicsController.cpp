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
    conSolverPtr = std::make_shared<ConstraintSolver>(commRcp, conCollectorPtr, ptcSystemPtr); 
    ptcSystemPtr->initialize(posFile);
    
    // prepare the output directory
    if (commRcp->getRank() == 0) {
        IOHelper::makeSubFolder("./result"); 
    }

    // Collect and resolve the initial constraints 
    spdlog::warn("Initial Constraint Resolution Begin");
    ptcSystemPtr->prepareStep(stepCount);
    ptcSystemPtr->collectConstraints(); 
    conSolverPtr->setup(runConfig.dt);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    conSolverPtr->solveConstraints(); 
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    
    conSolverPtr->writebackGamma();
    conSolverPtr->writebackForceVelocity();

    // merge the constraint and nonconstraint vel and force
    ptcSystemPtr->sumForceVelocity();

    // write the initial, potentially overlapped results
    std::string baseFolder = getCurrentResultFolder();
    IOHelper::makeSubFolder(baseFolder);
    const std::string postfix = std::to_string(-1);
    ptcSystemPtr->writeResult(stepCount, baseFolder, postfix); //TODO: split this function into two: one for ptcSystem and one for ConCollector

    // update the configuration
    // ptcSystemPtr->stepEuler(); // TODO: make sure this should be here since the collision may need to run this type of update
    ptcSystemPtr->advanceParticles();

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
    conSolverPtr = std::make_shared<ConstraintSolver>(commRcp, conCollectorPtr, ptcSystemPtr); 
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
        // setup
        spdlog::warn("CurrentStep {}", stepCount);
        std::string baseFolder = getCurrentResultFolder();
        IOHelper::makeSubFolder(baseFolder);

        // pre-constraint stuff
        conCollectorPtr->clear();
        ptcSystemPtr->prepareStep(stepCount);
        if (runConfig.KBT > 0) { ptcSystemPtr->calcVelocityBrown(); }
        ptcSystemPtr->collectConstraints();

        // constraint solve
        conSolverPtr->setup(runConfig.dt);
        conSolverPtr->solveConstraints(); 
        conSolverPtr->writebackGamma();
        conSolverPtr->writebackForceVelocity();

        // merge the constraint and nonconstraint vel and force
        ptcSystemPtr->sumForceVelocity();

        // data output stuff
        if (getIfWriteResultCurrentStep()) {
            // write result before moving. guarantee data written is consistent to geometry
            const std::string postfix = std::to_string(snapID);
            ptcSystemPtr->writeResult(stepCount, baseFolder, postfix); //TODO: split this function into two: one for ptcSystem and one for ConCollector
            writeTimeStepInfo(baseFolder, postfix);
            snapID++;
        }

        // post-step stuff
        // ptcSystemPtr->stepEuler(); // TODO: make sure this should be here since the collision may need to run this type of update
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

#include "PGDConstraintSolver.hpp"
#include "Util/Logger.hpp"

PGDConstraintSolver::PGDConstraintSolver(const Teuchos::RCP<const TCOMM> &commRcp,
                                         std::shared_ptr<ConstraintCollector> conCollectorPtr,
                                         std::shared_ptr<SylinderSystem> ptcSystemPtr)
    : commRcp_(commRcp), conCollectorPtr_(std::move(conCollectorPtr)), ptcSystemPtr_(std::move(ptcSystemPtr)) {}

void PGDConstraintSolver::reset() {
    mobMapRcp_.reset(); ///< distributed map for obj mobility. 6 dof per obj
    mobOpRcp_.reset();  ///< mobility operator, 6 dof per obj maps to 6 dof per obj

    forceConRcp_.reset(); ///< constraints force vec, 6 dof per obj
    forceExtRcp_.reset(); ///< external (non-constraint) force, 6 dof per obj
    velConRcp_.reset();   ///< constraints velocity vec, 6 dof per obj
    velExtRcp_.reset();   ///< external (non-constraint) velocity, 6 dof per obj

    DMatTransRcp_.reset(); ///< D^Trans matrix
    DMatRcp_.reset();      ///< D matrix TODO: is is better to build this explicitly or to use implicit tranpose of D^T?
    constraintDiagonalRcp_.reset(); ///< K^{-1} diagonal matrix
    biFlagRcp_.reset();

    // the linear complementarity problem 0 <= A gamma + S0 _|_ gamma >= 0
    partialSepPartialGammaOpRcp_.reset(); ///< Operator takes gamma to change in sep w.r.t gamma
                                          ///< this is the operator of BCQP problem. A = dt D^T M D + K^{-1}
    gammaRcp_.reset();                    ///< the unknown constraint Lagrange multipliers
    sep0Rcp_.reset();                     ///< initial, unconstrained constraint violation
                                          ///< this is the constant part of BCQP problem
}

void PGDConstraintSolver::setup(const double dt, const double res, const int maxIte, const int solverChoice) {
    reset();

    // store the inputs
    dt_ = dt;
    res_ = res;
    maxIte_ = maxIte;
    solverChoice_ = solverChoice;

    // initialize the constraint stuff
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();
    gammaMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
    sep0Rcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    gammaRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    biFlagRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    gammaRecRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    constraintDiagonalRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));

    // initialize the particle stuff (TODO: what about external force and velocity?)
    mobOpRcp_ = ptcSystemPtr_->getMobOperator();
    mobMapRcp_ = mobOpRcp_->getDomainMap();
    velConRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
    forceConRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));

    // build D^T, which maps from constraint Lagrange multiplier to com force vector
    mobMapRcp_ = mobOpRcp_->getDomainMap();
    DMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, gammaMapRcp_);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(DMatTransRcp_);
    DMatRcp_ = transposerDu.createTranspose();
    TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(DMatRcp_));

    // build the operator that takes gamma to change in sep w.r.t gamma
    partialSepPartialGammaOpRcp_ = Teuchos::rcp(new PartialSepPartialGammaOp(gammaMapRcp_));
    partialSepPartialGammaOpRcp_->initialize(mobOpRcp_, DMatTransRcp_, forceConRcp_, velConRcp_, dt_);

    // fill the initial gamma guess, the constraint flag, the initial unconstrained separation, and the diagonal of
    // K^{-1}
    conCollectorPtr_->fillFixedConstraintInfo(gammaRecRcp_, biFlagRcp_, sep0Rcp_, constraintDiagonalRcp_);
}

void PGDConstraintSolver::solveConstraints() {
    // TODO: Is the following reduction necessary?
    //       If it is necessary, should we call stepEuler without constraint force?

    // /////////////////////////////////////////////////
    // // Check if there are any constraints to solve //
    // /////////////////////////////////////////////////
    // const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();
    // int numGlobalConstraints;
    // MPI_Allreduce(&numLocalConstraints, &numGlobalConstraints, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // if (numGlobalConstraints == 0) {
    //     spdlog::info("No constraints to solve");
    //     return;
    // } else {
    //     spdlog::info("Global number of constraints: {:g}", numGlobalConstraints);
    // }

    //////////////////////////////
    // Create the solver object //
    //////////////////////////////
    // TODO: update the solver object to support updating A and b instead of creating an entirely new object



    ///////////////////////
    // Run the recursion //
    ///////////////////////
    // the recursion loop
    for (int r = 0; r < 2; r++) {
        // create the solver object
        BCQPSolver solver(partialSepPartialGammaOpRcp_, sep0Rcp_);
        spdlog::debug("solver constructed");

        // setup the lower bound for gamma
        // for bilaterial constraints this is -infinity
        // for unilaterial constraints this is 0
        Teuchos::RCP<TV> lbRcp = solver.getLowerBound();
        lbRcp->scale(-std::numeric_limits<double>::max() / 10, *biFlagRcp_); // 0 if biFlag=0, -infty if biFlag=1
        spdlog::debug("bound constructed");

        // solve the constraints
        IteHistory history;
        switch (solverChoice_) {
        case 0:
            solver.solveBBPGD(gammaRecRcp_, res_, maxIte_, history);
            break;
        case 1:
            solver.solveAPGD(gammaRecRcp_, res_, maxIte_, history);
            break;
        default:
            solver.solveBBPGD(gammaRecRcp_, res_, maxIte_, history);
            break;
        }
        gammaRcp_->update(1.0, *gammaRecRcp_, 1.0);

        // print the full solution history
        for (auto it = history.begin(); it != history.end() - 1; it++) {
            auto &p = *it;
            spdlog::debug("RECORD: BCQP history {:g}, {:g}, {:g}, {:g}, {:g}, {:g}", p[0], p[1], p[2], p[3], p[4],
                          p[5]);
        }
        auto &p = history.back();
        spdlog::info("RECORD: BCQP residue {:g}, {:g}, {:g}, {:g}, {:g}, {:g}", p[0], p[1], p[2], p[3], p[4], p[5]);

        // apply the recursion
        recursionStep();
    }

    // Note, each recursion step passes the results to particleSystem and taking the Euler step
    // By this point, the system will be in its final constraint-satisfying configuration
    // Print the extent to which this final configuration satisfies all constraints
    const double maxConViolation = sep0Rcp_->normInf();
    spdlog::info("RECORD: Maximum constraint violation {:g}", maxConViolation);

    // store the result in the constraint objects
    writebackGamma();
}

void PGDConstraintSolver::recursionStep() {
    // to avoid double counting forces, reset the constraint gamma and constraint sep
    conCollectorPtr_->resetConstraintVariables();

    // solve for the induced force and velocity
    DMatRcp_->apply(*gammaRcp_, *forceConRcp_);    // F_{r}^k = sum_{n=0}^{r} D_{n-1}^{k+1} gamma_{r}^k
    mobOpRcp_->apply(*forceConRcp_, *velConRcp_); // U_r^k = M^k F_r^k

    // send the constraint vel and force to the particles
    ptcSystemPtr_->saveForceVelocityConstraints(forceConRcp_, velConRcp_);

    // move the configuration from q^k to q_{r}^{k+1}
    ptcSystemPtr_->stepEuler();  // q_{r}^{k+1} = q^k + dt G^k U_r^k
    ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see

    // update all constraints
    // this must NOT generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    ptcSystemPtr_->updatePairCollision(); // TODO: this only supports systems with pure, particle-particle
                                          // collisions atm

    // add this recursion to D^T and D
    // TODO: is using an explicit transpose necessary? It doubles storage of a MASSIVE object
    conCollectorPtr_->updateConstraintMatrixVector(DMatTransRcp_);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(DMatTransRcp_);
    DMatRcp_ = transposerDu.createTranspose();
    TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(DMatRcp_));

    // update PartialSepPartialGamma
    // TODO: now much speedup do we get when we have block diagonal mobility and compute this explicitly?
    partialSepPartialGammaOpRcp_->initialize(mobOpRcp_, DMatTransRcp_, forceConRcp_, velConRcp_, dt_);

    // fill the initial gamma guess, the constraint flag, the initial unconstrained separation, and the diagonal of
    // K^{-1}
    conCollectorPtr_->fillFixedConstraintInfo(gammaRecRcp_, biFlagRcp_, sep0Rcp_, constraintDiagonalRcp_);
}

void PGDConstraintSolver::writebackGamma() { conCollectorPtr_->writeBackGamma(gammaRcp_.getConst()); }
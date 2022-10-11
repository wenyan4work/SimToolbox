#include "PGDConstraintSolver.hpp"
#include "Util/Logger.hpp"

// multiplyRAP
#include <TpetraExt_TripleMatrixMultiply.hpp>

PGDConstraintSolver::PGDConstraintSolver(const Teuchos::RCP<const TCOMM> &commRcp,
                                         std::shared_ptr<ConstraintCollector> conCollectorPtr,
                                         std::shared_ptr<SylinderSystem> ptcSystemPtr)
    : commRcp_(commRcp), conCollectorPtr_(std::move(conCollectorPtr)), ptcSystemPtr_(std::move(ptcSystemPtr)) {}

void PGDConstraintSolver::reset() {
    mobMapRcp_.reset(); ///< distributed map for obj mobility. 6 dof per obj
    mobMatRcp_.reset(); ///< mobility operator, 6 dof per obj maps to 6 dof per obj

    forceConRcp_.reset(); ///< constraints force vec, 6 dof per obj
    velConRcp_.reset();   ///< constraints velocity vec, 6 dof per obj

    DMatTransRcp_.reset(); ///< D^Trans matrix
    DMatRcp_.reset();      ///< D matrix TODO: is is better to build this explicitly or to use implicit tranpose of D^T?
    constraintDiagonalRcp_.reset(); ///< K^{-1} diagonal matrix
    biFlagRcp_.reset();

    // the linear complementarity problem 0 <= A gamma + S0 _|_ gamma >= 0
    constraintJacobianOp_.reset(); ///< operator taking gamma to change in sep w.r.t gamma
                                   ///< this is the operator of BCQP problem. A = dt D^T M D + K^{-1}

    gammaRcp_.reset(); ///< the unknown constraint Lagrange multipliers
    sep0Rcp_.reset();  ///< initial, unconstrained constraint violation
                       ///< this is the constant part of BCQP problem
}

void PGDConstraintSolver::initialize() {
    // initialize the constraint stuff
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();
    gammaMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
    sepRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    sep0Rcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    gammaRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    biFlagRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    constraintDiagonalRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));

    // initialize the particle stuff
    mobMatRcp_ = ptcSystemPtr_->getMobMatrix();
    mobMapRcp_ = mobMatRcp_->getDomainMap();
    velConRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
    forceConRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));

    // build D^T, which maps from constraint Lagrange multiplier to com force vector
    mobMapRcp_ = mobMatRcp_->getDomainMap();
    DMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, gammaMapRcp_);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(DMatTransRcp_);
    DMatRcp_ = transposerDu.createTranspose();
    TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(DMatRcp_));

    // fill the initial gamma guess, the constraint flag, the initial unconstrained separation, and the diagonal of
    // K^{-1}
    conCollectorPtr_->fillFixedConstraintInfo(gammaRcp_, biFlagRcp_, sep0Rcp_, constraintDiagonalRcp_);

    // // to reduce numerical rounding errors, we scale our LCP by 1/dt
    // sep0Rcp_->scale(1.0 / dt_);
    // constraintDiagonalRcp_->scale(1.0 / dt_);

    // build the constraint Jacobian operator
    constraintJacobianOp_ = Teuchos::rcp(new ConstraintJacobianOp(gammaMapRcp_));
    constraintJacobianOp_->initialize(mobMatRcp_, DMatRcp_, DMatTransRcp_, constraintDiagonalRcp_, dt_);
}

void PGDConstraintSolver::reinitialize() {
    // reinitialize any data structure that depends on the number of constraints
    // this is non-ideal since it involves significant alocation and dealocation

    // initialize the constraint stuff
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();
    gammaMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
    sepRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    sep0Rcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    gammaRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    biFlagRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));
    constraintDiagonalRcp_ = Teuchos::rcp(new TV(gammaMapRcp_, true));

    // build D^T, which maps from constraint Lagrange multiplier to com force vector
    DMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, gammaMapRcp_);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(DMatTransRcp_);
    DMatRcp_ = transposerDu.createTranspose();
    TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(DMatRcp_));

    // fill the initial gamma guess, the constraint flag, the initial unconstrained separation, and the diagonal of
    // K^{-1}
    conCollectorPtr_->fillFixedConstraintInfo(gammaRcp_, biFlagRcp_, sep0Rcp_, constraintDiagonalRcp_);

    // // to reduce numerical rounding errors, we scale our LCP by 1/dt
    // sep0Rcp_->scale(1.0 / dt_);
    // constraintDiagonalRcp_->scale(1.0 / dt_);

    // build the constraint Jacobian operator
    constraintJacobianOp_ = Teuchos::rcp(new ConstraintJacobianOp(gammaMapRcp_));
    constraintJacobianOp_->initialize(mobMatRcp_, DMatRcp_, DMatTransRcp_, constraintDiagonalRcp_, dt_);
}

void PGDConstraintSolver::setup(const double dt, const double res, const int maxIterations, const int maxRecursions,
                                const int solverChoice) {
    reset();

    // store the inputs
    dt_ = dt;
    res_ = res;
    maxIterations_ = maxIterations;
    maxRecursions_ = maxRecursions;
    solverChoice_ = solverChoice;
}

void PGDConstraintSolver::resolveConstraints() {
    /////////////////////////////////////////////////
    // Check if there are any constraints to solve //
    /////////////////////////////////////////////////
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();
    int numGlobalConstraints;
    MPI_Allreduce(&numLocalConstraints, &numGlobalConstraints, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (numGlobalConstraints == 0) {
        spdlog::info("No constraints to solve. System advances unconstrained");
        return;
    } else {
        spdlog::info("Global number of constraints: {}", numGlobalConstraints);
    }

    ////////////////////////////////////
    // Initialize the data structures //
    ////////////////////////////////////
    initialize();

    ///////////////////////
    // Run the recursion //
    ///////////////////////
    // the recursion loop
    double maxConViolation;
    int stepCount = 1;
    for (int r = 0; r < maxRecursions_; r++) {
        // // for debug, write out the particle positions and constraints
        // // conCollectorPtr_->writeBackGamma(xRcp.getConst());
        // const std::string postfix = std::to_string(stepCount);
        // ptcSystemPtr_->writeResult(stepCount, "./result/", postfix);
        // stepCount++;

        // create the solver object
        BCQPSolver solver(constraintJacobianOp_, sep0Rcp_);
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
            solver.solveBBPGD(sepRcp_, gammaRcp_, res_, maxIterations_, history);
            break;
        case 1:
            solver.solveAPGD(sepRcp_, gammaRcp_, res_, maxIterations_, history);
            break;
        default:
            solver.solveBBPGD(sepRcp_, gammaRcp_, res_, maxIterations_, history);
            break;
        }
        // sepRcp_->scale(dt_);

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

        // get the extent to which the new configuration satisfies all constraints
        // use sepRcp_ as temporary storage for the constraint violation values
        // sep0Rcp_->scale(dt_);
        gammaRcp_->putScalar(0.0);
        conCollectorPtr_->evalConstraintValues(gammaRcp_, sep0Rcp_, sepRcp_);
        maxConViolation = sepRcp_->normInf();
        spdlog::debug("RECORD: ReLCP constraint history {}, {:g}", r, maxConViolation);
        if (maxConViolation < res_) { // TODO: should this be a bit larger then the linear solve, say linear 1e-5 and
                                      // nonlinear 1e-4? Sometimes this helps. Sometimes it hurts ¯\_(ツ)_/¯
            break;
        }
    }

    // // for debug, write out the final particle positions and constraints
    // // conCollectorPtr_->writeBackGamma(xRcp.getConst());
    // const std::string postfix = std::to_string(stepCount);
    // ptcSystemPtr_->writeResult(stepCount, "./result/", postfix);
    // stepCount++;

    // Note, each recursion step passes the results to particleSystem and taking the Euler step
    // By this point, the system will be in its final constraint-satisfying configuration
    // the constraints will also have their final gamma and sep values

    // Print the extent to which this final configuration satisfies all constraints
    spdlog::info("RECORD: ReLCP constraint violation {:g}", maxConViolation);
}

void PGDConstraintSolver::recursionStep() {
    // store the constraint gamma and constraint sep
    conCollectorPtr_->writeBackConstraintVariables(gammaRcp_, sepRcp_);
    spdlog::debug("results stored");

    // solve for the induced force and velocity (sum the forces from previous recursions)
    DMatRcp_->apply(*gammaRcp_, *forceConRcp_, Teuchos::NO_TRANS, 1.0,
                    1.0);                          // F_{r}^k = sum_{n=0}^{r} D_{n-1}^{k+1} gamma_{r}^k
    mobMatRcp_->apply(*forceConRcp_, *velConRcp_); // U_r^k = M^k F_r^k
    spdlog::debug("induced velocity solved");

    // send the constraint vel and force to the particles
    ptcSystemPtr_->saveForceVelocityConstraints(forceConRcp_, velConRcp_);
    spdlog::debug("force and velocity stored");

    // move the configuration from q^k to q_{r}^{k+1}
    ptcSystemPtr_->stepEuler();  // q_{r}^{k+1} = q^k + dt G^k (U_r^k + U_ext^k)
    ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see!
    spdlog::debug("particle position updated");

    // collect unresolved constraints and add them to constraintCollector
    // this process can generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    spdlog::debug("near particles updated");

    ptcSystemPtr_->collectUnresolvedConstraints();
    spdlog::debug("new constraints collected");

    // reinitialize data structures that depend on the number of constraints
    reinitialize();
}

Teuchos::RCP<TV> PGDConstraintSolver::getParticleStress() const {
    // TODO: This entire operation is just a reduction operation from constraints to particles
    //       Being able to loop over each particle and fetch the constraints that impact them would be FAR better and
    //       cleaner. Instead, I'm legit building a massive sparce matrix just so I can calapse it onto an axis!!
    const Teuchos::RCP<const TMAP> ptcStressMapRcp = ptcSystemPtr_->getParticleStressMap();
    Teuchos::RCP<TV> ptcStressRcp = Teuchos::rcp(new TV(ptcStressMapRcp));

    // build S^T, which maps from constraint Lagrange multiplier to particle virial stress
    // the "true" ensures that the stresses in scaled
    // hence, muliplying by a vector of all 1s will give us the sum of stress on each particle
    const Teuchos::RCP<const TCMAT> SMatTransRcp =
        conCollectorPtr_->buildGammaToVirialStressMatrix(ptcStressMapRcp, gammaMapRcp_, true);
    Tpetra::RowMatrixTransposer<double, int, int> transposerSu(SMatTransRcp);
    const Teuchos::RCP<const TCMAT> SMatRcp = transposerSu.createTranspose();
    TEUCHOS_ASSERT(nonnull(SMatTransRcp));
    TEUCHOS_ASSERT(nonnull(SMatRcp));

    // do the computation
    const Teuchos::RCP<TV> onesRcp = Teuchos::rcp(new TV(gammaMapRcp_, true));
    onesRcp->putScalar(Teuchos::ScalarTraits<Scalar>::one());

    SMatRcp->apply(*onesRcp, *ptcStressRcp);

    return ptcStressRcp;
}

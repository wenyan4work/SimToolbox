// Our stuff
#include "NOXConstraintEvaluator.hpp"

// Teuchos support
#include <Teuchos_CommHelpers.hpp>

// Thyra support
#include <Thyra_MultiVectorStdOps.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>
#include <Thyra_VectorStdOps.hpp>

// Kokkos support
#include <Kokkos_Core.hpp>

// Constructor
EvaluatorTpetraConstraint::EvaluatorTpetraConstraint(const Teuchos::RCP<const TCOMM> &commRcp,
                                                     const Teuchos::RCP<const TOP> &mobOpRcp,
                                                     std::shared_ptr<ConstraintCollector> conCollectorPtr,
                                                     std::shared_ptr<SylinderSystem> ptcSystemPtr,
                                                     const Teuchos::RCP<TV> &forceRcp, const Teuchos::RCP<TV> &velRcp,
                                                     const double dt)
    : commRcp_(commRcp), mobOpRcp_(mobOpRcp), conCollectorPtr_(std::move(conCollectorPtr)),
      ptcSystemPtr_(std::move(ptcSystemPtr)), dt_(dt), forceRcp_(forceRcp), velRcp_(velRcp), showGetInvalidArg_(false) {
    TEUCHOS_ASSERT(nonnull(commRcp_));
    TEUCHOS_ASSERT(nonnull(mobOpRcp_));
    TEUCHOS_ASSERT(nonnull(forceRcp_));
    TEUCHOS_ASSERT(nonnull(velRcp_));

    // initialize the various objects
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();

    xMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
    sepRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    xGuessRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    statusRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    constraintDiagonalRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    partialSepPartialGammaDiagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    partialSepPartialGammaOpRcp_ = Teuchos::rcp(new PartialSepPartialGammaOp(xMapRcp_));

    // solution space
    xSpaceRcp_ = Thyra::createVectorSpace<Scalar, LO, GO, Node>(xMapRcp_);

    // residual space
    fMapRcp_ = xMapRcp_;
    fSpaceRcp_ = xSpaceRcp_;

    // build D^k, which maps from force magnatude to com force vector
    mobMapRcp_ = mobOpRcp_->getDomainMap();
    AMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, xMapRcp_);
    TEUCHOS_ASSERT(nonnull(AMatTransRcp_));

    // build the diagonal of dt (D^p)^T M^k D^p :(
    partialSepPartialGammaDiagRcp_->putScalar(1.0);

    // recursively fill the initial guess
    conCollectorPtr_->fillConstraintGuess(xGuessRcp_);
    // for (int r = 0; r < 1; r++) {
    //     recursionStep(xGuessRcp_);
    //     conCollectorPtr_->fillConstraintGuess(xGuessRcp_);
    // }
    dumpTV(xGuessRcp_, "xGuessRcp_");

    // setup in/out args
    typedef Thyra::ModelEvaluatorBase MEB;
    MEB::InArgsSetup<Scalar> inArgs;
    inArgs.setModelEvalDescription(this->description());
    inArgs.setSupports(MEB::IN_ARG_x);
    prototypeInArgs_ = inArgs;

    MEB::OutArgsSetup<Scalar> outArgs;
    outArgs.setModelEvalDescription(this->description());
    outArgs.setSupports(MEB::OUT_ARG_f);
    outArgs.setSupports(MEB::OUT_ARG_W_op);
    outArgs.setSupports(MEB::OUT_ARG_W_prec);
    prototypeOutArgs_ = outArgs;

    // set the initial condition
    x0Rcp_ = Thyra::createVector(xGuessRcp_, xSpaceRcp_);
    nominalValues_ = inArgs;
    nominalValues_.set_x(x0Rcp_);

    residTimer_ = Teuchos::TimeMonitor::getNewCounter("Model Evaluator: Residual Evaluation");
    intOpTimer_ = Teuchos::TimeMonitor::getNewCounter("Model Evaluator: Integral Operator Evaluation");
}

// Initializers/Accessors

void EvaluatorTpetraConstraint::setShowGetInvalidArgs(bool showGetInvalidArg) {
    showGetInvalidArg_ = showGetInvalidArg;
}

void EvaluatorTpetraConstraint::set_W_factory(
    const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar>> &W_factoryRcp) {
    W_factoryRcp_ = W_factoryRcp;
}

// Public functions overridden from ModelEvaulator
Teuchos::RCP<const thyra_vec_space> EvaluatorTpetraConstraint::get_x_space() const { return xSpaceRcp_; }

Teuchos::RCP<const thyra_vec_space> EvaluatorTpetraConstraint::get_f_space() const { return fSpaceRcp_; }

Thyra::ModelEvaluatorBase::InArgs<Scalar> EvaluatorTpetraConstraint::getNominalValues() const { return nominalValues_; }

Teuchos::RCP<thyra_op> EvaluatorTpetraConstraint::create_W_op() const {
    Teuchos::RCP<JacobianOperator> W_tpetra = Teuchos::rcp(new JacobianOperator(xMapRcp_));
    return Thyra::tpetraLinearOp<Scalar, LO, GO, Node>(fSpaceRcp_, xSpaceRcp_, W_tpetra);
}

// Teuchos::RCP<thyra_prec>
// EvaluatorTpetraConstraint::create_W_prec() const
// {
//   // return the precreated Thyra preconditioner
//   // this preconditioner is internally owned, I hope that is ok
//   return precThyraRcp_;
// }

Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar>> EvaluatorTpetraConstraint::get_W_factory() const {
    return W_factoryRcp_;
}

Thyra::ModelEvaluatorBase::InArgs<Scalar> EvaluatorTpetraConstraint::createInArgs() const { return prototypeInArgs_; }

// Private functions overridden from ModelEvaulatorDefaultBase

Thyra::ModelEvaluatorBase::OutArgs<Scalar> EvaluatorTpetraConstraint::createOutArgsImpl() const {
    return prototypeOutArgs_;
}

void EvaluatorTpetraConstraint::evalModelImpl(const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
                                              const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs) const {
    Teuchos::RCP<Teuchos::Time> debugTimer1 =
        Teuchos::TimeMonitor::getNewCounter("EvaluatorTpetraConstraint::update_config");
    Teuchos::RCP<Teuchos::Time> debugTimer2 = Teuchos::TimeMonitor::getNewCounter("EvaluatorTpetraConstraint::fill_f");
    Teuchos::RCP<Teuchos::Time> debugTimer3 = Teuchos::TimeMonitor::getNewCounter("EvaluatorTpetraConstraint::fill_J");

    TEUCHOS_ASSERT(nonnull(inArgs.get_x()));

    const Teuchos::RCP<thyra_vec> f_out = outArgs.get_f();
    const Teuchos::RCP<thyra_op> W_out = outArgs.get_W_op();
    const Teuchos::RCP<thyra_prec> W_prec_out = outArgs.get_W_prec();

    const bool fill_f = nonnull(f_out);
    const bool fill_W = nonnull(W_out);
    const bool fill_W_prec = nonnull(W_prec_out);

    using tpetra_extract = Thyra::TpetraOperatorVectorExtraction<Scalar, LO, GO, Node>;

    if (fill_f || fill_W || fill_W_prec) {
        // std::cout << "fill_f: " << fill_f << " | fill_W: " << fill_W << "| fill_W_prec: " << fill_W_prec <<
        // std::endl;

        ///////////////////////////////////////
        // Get the underlying tpetra objects //
        ///////////////////////////////////////
        Teuchos::RCP<const TV> xRcp = tpetra_extract::getConstTpetraVector(inArgs.get_x());

        Teuchos::RCP<TV> fRcp;
        if (fill_f) {
            fRcp = tpetra_extract::getTpetraVector(f_out);
        }

        Teuchos::RCP<JacobianOperator> JRcp;
        if (fill_W) {
            Teuchos::RCP<TOP> M_tpetra = tpetra_extract::getTpetraOperator(W_out);
            JRcp = Teuchos::rcp_dynamic_cast<JacobianOperator>(M_tpetra);
            TEUCHOS_ASSERT(nonnull(JRcp));
        }

        //////////////////////////////////////////////
        // Zero out the objects that will be filled //
        //////////////////////////////////////////////
        partialSepPartialGammaOpRcp_->unitialize();

        if (fill_f) {
            fRcp->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
        }

        if (fill_W) {
            JRcp->unitialize();
        }

        /////////////////////////////////////////////////////////////////
        // Recursively update the configuration and build the D matrix //
        /////////////////////////////////////////////////////////////////
        {
            Teuchos::TimeMonitor mon(*debugTimer1);
            // Beware, the systems will update conCollector each time evalModelImpl is called
            // we must reset the conCollector to q^k before proceeding,

            if (nonnull(xRcp)) {
                dumpTV(xRcp, "Gamma");
            }

            // TODO: the following bit is useless if there are no recursions
            // reset recursion to 0
            conCollectorPtr_->resetConstraintRecursions();  
            conCollectorPtr_->updateConstraintMatrixVector(AMatTransRcp_);

            // the recursion loop
            for (int r = 0; r < 1; r++) {
                recursionStep(xRcp);
            }
            dumpTCMAT(AMatTransRcp_, "AmatT");
            dumpTOP(mobOpRcp_, "Mobility");

            // setup PartialSepPartialGamma
            partialSepPartialGammaOpRcp_->initialize(mobOpRcp_, AMatTransRcp_, forceRcp_, velRcp_, dt_);

            // evaluate the unconstrained separation
            // sep = sep0 + dt D^T M D Xactive
            // TODO: we are currently taking two steps. One solves S0 and then when checking the solution, we step based on the new S0
            // S0 should only be updated at the beginning of each Newton step
            // 
            conCollectorPtr_->evalSepInitialValues(sepRcp_);
            dumpTV(sepRcp_, "Sep0");
            partialSepPartialGammaOpRcp_->apply(*xRcp, *sepRcp_, Teuchos::NO_TRANS, 1.0, 1.0);
            dumpTV(sepRcp_, "Sep");
        }

        //////////////////////
        // Fill the objects //
        //////////////////////

        // fill the constraint value
        if (fill_f) {
            Teuchos::TimeMonitor mon(*debugTimer2);

            // evaluate C(q^{k+1}(gamma^k), scale * gamma^k)
            conCollectorPtr_->evalConstraintValues(xRcp, partialSepPartialGammaDiagRcp_, sepRcp_, fRcp, statusRcp_);

            dumpTV(xRcp, "X");
            dumpTV(sepRcp_, "Sep");
            dumpTV(fRcp, "F");
        }

        // fill the Jacobian
        if (fill_W) {
            Teuchos::TimeMonitor mon(*debugTimer3);

            // eval the diagonal of K^{-1}(q^{k+1}, gamma^k)
            conCollectorPtr_->evalConstraintDiagonal(xRcp, constraintDiagonalRcp_);

            // setup J
            conCollectorPtr_->evalConstraintValues(xRcp, partialSepPartialGammaDiagRcp_, sepRcp_, forceMagRcp_,
                                                   statusRcp_);
            JRcp->initialize(partialSepPartialGammaOpRcp_, partialSepPartialGammaDiagRcp_, constraintDiagonalRcp_,
                             statusRcp_, dt_);

            // // for debug, dump to file
            dumpToFile(JRcp, forceMagRcp_, statusRcp_, partialSepPartialGammaDiagRcp_, xRcp, mobOpRcp_, AMatTransRcp_,
                       constraintDiagonalRcp_);

            std::cout << "" << std::endl;
        }

        ///////////
        // Debug //
        ///////////

        // for debug, advance the particles and write their positions to a vtk file
        // conCollectorPtr_->writeBackGamma(xRcp.getConst());
        const std::string postfix = std::to_string(stepCount_);
        ptcSystemPtr_->writeResult(stepCount_, "./result/", postfix);
        stepCount_++;
    }
}

void EvaluatorTpetraConstraint::recursionStep(const Teuchos::RCP<const TV> &gammaRcp) const {
    // solve for the induced force and velocity
    AMatTransRcp_->apply(*gammaRcp, *forceRcp_,
                            Teuchos::TRANS);   // F_{r}^k = sum_{n=0}^{r} D_{n-1}^{k+1} gamma_{r}^k
    mobOpRcp_->apply(*forceRcp_, *velRcp_); // U_r^k = M^k F_r^k

    // send the constraint vel and force to the particles
    ptcSystemPtr_->saveForceVelocityConstraints(forceRcp_, velRcp_);

    // move the configuration from q^k to q_{r}^{k+1}
    ptcSystemPtr_->stepEuler(0); // q_{r}^{k+1} = q^k + dt G^k U_r^k
    ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see

    // update all constraints
    // this must NOT generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    ptcSystemPtr_->updatePairCollision(); // TODO: this only supports systems with pure, particle-particle
                                            // collisions atm

    // add this recursion to D^T
    conCollectorPtr_->updateConstraintMatrixVector(AMatTransRcp_);
}

void EvaluatorTpetraConstraint::dumpToFile(const Teuchos::RCP<const TOP> &JRcp, const Teuchos::RCP<const TV> &fRcp,
                                           const Teuchos::RCP<const TV> &statusRcp,
                                           const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp,
                                           const Teuchos::RCP<const TV> &xRcp, const Teuchos::RCP<const TOP> &mobOpRcp,
                                           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
                                           const Teuchos::RCP<const TV> &constraintDiagonalRcp) const {
    // only dump nonnull objects
    if (nonnull(JRcp)) {
        dumpTOP(JRcp, "Jacobian");
    }
    if (nonnull(fRcp)) {
        dumpTV(fRcp, "f");
    }
    if (nonnull(statusRcp)) {
        dumpTV(statusRcp, "mask");
    }
    if (nonnull(partialSepPartialGammaDiagRcp)) {
        dumpTV(partialSepPartialGammaDiagRcp, "PrecInv");
    }
    if (nonnull(xRcp)) {
        dumpTV(xRcp, "Gamma");
    }
    if (nonnull(mobOpRcp)) {
        dumpTOP(mobOpRcp, "Mobility");
    }
    if (nonnull(AMatTransRcp)) {
        dumpTCMAT(AMatTransRcp, "AmatT");
    }
    if (nonnull(constraintDiagonalRcp)) {
        dumpTV(constraintDiagonalRcp, "diag");
    }
}

///////////////////////////////////////
// Jacobian operator implementations //
///////////////////////////////////////

JacobianOperator::JacobianOperator(const Teuchos::RCP<const TMAP> &xMapRcp) : xMapRcp_(xMapRcp) {}

void JacobianOperator::initialize(const Teuchos::RCP<const PartialSepPartialGammaOp> &PartialSepPartialGammaOpRcp,
                                  const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp,
                                  const Teuchos::RCP<const TV> &constraintDiagonalRcp,
                                  const Teuchos::RCP<const TV> &statusRcp, const double dt) {
    // store the objects to be used in apply
    dt_ = dt;
    statusRcp_ = statusRcp;
    constraintDiagonalRcp_ = constraintDiagonalRcp;
    partialSepPartialGammaOpRcp_ = PartialSepPartialGammaOpRcp;
    partialSepPartialGammaDiagRcp_ = partialSepPartialGammaDiagRcp;

    // check the input
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintDiagonalRcp_->getMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*partialSepPartialGammaDiagRcp_->getMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*partialSepPartialGammaOpRcp_->getRangeMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*partialSepPartialGammaOpRcp_->getDomainMap()));

    // create the internal data structures
    changeInSepRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    activeXcolRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
}

void JacobianOperator::unitialize() {
    dt_ = 0.0;
    statusRcp_.reset();
    activeXcolRcp_.reset();
    changeInSepRcp_.reset();
    constraintDiagonalRcp_.reset();
    partialSepPartialGammaOpRcp_.reset();
    partialSepPartialGammaDiagRcp_.reset();
}

Teuchos::RCP<const TMAP> JacobianOperator::getDomainMap() const { return xMapRcp_; }

Teuchos::RCP<const TMAP> JacobianOperator::getRangeMap() const { return xMapRcp_; }

bool JacobianOperator::hasTransposeApply() const { return false; }

void JacobianOperator::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode, Scalar alpha, Scalar beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                               "X and Y do not have the same numbers of vectors (columns).");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                               "X and Y do not have the same Map.\n");
    TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // The jacobian is NOT symmetric, so trans apply is NOT ok.
    TEUCHOS_ASSERT(nonnull(constraintDiagonalRcp_));

    const int numVecs = X.getNumVectors();
    for (int i = 0; i < numVecs; i++) {
        // Goal Y = alpha J^{-1}(dt D^T M A X + Diag X)  + beta Y
        auto XcolRcp = X.getVector(i);
        auto YcolRcp = Y.getVectorNonConst(i);

        // step 1. determine the active set of constraints
        dumpTV(XcolRcp, "XcolRcp");
        activeXcolRcp_->elementWiseMultiply(1.0, *XcolRcp, *statusRcp_, 0.0);
        dumpTV(statusRcp_, "statusRcp_");

        // step 2. change_in_sep = dt D^T M D Xactive
        partialSepPartialGammaOpRcp_->apply(*activeXcolRcp_, *changeInSepRcp_);
        dumpTV(changeInSepRcp_, "changeInSepRcp_");

        // step 3. solve dt D^T M D X + Diag X and store in changeInSepRcp_
        changeInSepRcp_->elementWiseMultiply(1.0, *constraintDiagonalRcp_, *activeXcolRcp_, 1.0);
        dumpTV(changeInSepRcp_, "changeInSepRcp2_");

        // step 4. apply the projection (with scaling of projected values)
        // also, account for Y = alpha Op X + beta Y
        {
            auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
            auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
            auto statusPtr = statusRcp_->getLocalView<Kokkos::HostSpace>();
            auto activeXcolPtr = activeXcolRcp_->getLocalView<Kokkos::HostSpace>();
            auto changeInSepPtr = changeInSepRcp_->getLocalView<Kokkos::HostSpace>();
            auto constraintDiagonalPtr = constraintDiagonalRcp_->getLocalView<Kokkos::HostSpace>();
            auto partialSepPartialGammaDiagPtr = partialSepPartialGammaDiagRcp_->getLocalView<Kokkos::HostSpace>();
            YcolRcp->modify<Kokkos::HostSpace>();
            const auto localSize = YcolPtr.extent(0);
#pragma omp parallel for
            for (size_t idx = 0; idx < localSize; idx++) {
                if (statusPtr(idx, 0) > 0.5) { // active
                    if (beta == Teuchos::ScalarTraits<Scalar>::zero()) {
                        YcolPtr(idx, 0) = alpha * changeInSepPtr(idx, 0);
                    } else { // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values?
                        YcolPtr(idx, 0) = alpha * changeInSepPtr(idx, 0) + beta * YcolPtr(idx, 0);
                    }
                } else { // inactive
                    if (beta == Teuchos::ScalarTraits<Scalar>::zero()) {
                        YcolPtr(idx, 0) = alpha * partialSepPartialGammaDiagPtr(idx, 0) * XcolPtr(idx, 0);
                        // YcolPtr(idx, 0) = alpha * XcolPtr(idx, 0);
                    } else { // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values?
                        YcolPtr(idx, 0) =
                            alpha * partialSepPartialGammaDiagPtr(idx, 0) * XcolPtr(idx, 0) + beta * YcolPtr(idx, 0);
                        // YcolPtr(idx, 0) = alpha * XcolPtr(idx, 0) + beta * YcolPtr(idx, 0);
                    }
                }
            }
            dumpTV(YcolRcp, "YcolRcp");
        }
    }
}

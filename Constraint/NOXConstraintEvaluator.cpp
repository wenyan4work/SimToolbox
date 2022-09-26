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
                                                     const Teuchos::RCP<const TCMAT> &mobMatRcp,
                                                     std::shared_ptr<ConstraintCollector> conCollectorPtr,
                                                     std::shared_ptr<SylinderSystem> ptcSystemPtr,
                                                     const Teuchos::RCP<TV> &forceRcp, const Teuchos::RCP<TV> &velRcp,
                                                     const double dt)
    : commRcp_(commRcp), mobMatRcp_(mobMatRcp), conCollectorPtr_(std::move(conCollectorPtr)),
      ptcSystemPtr_(std::move(ptcSystemPtr)), dt_(dt), forceRcp_(forceRcp), velRcp_(velRcp), showGetInvalidArg_(false) {
    TEUCHOS_ASSERT(nonnull(commRcp_));
    TEUCHOS_ASSERT(nonnull(mobMatRcp_));
    TEUCHOS_ASSERT(nonnull(forceRcp_));
    TEUCHOS_ASSERT(nonnull(velRcp_));

    // initialize the various objects
    const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfDOF();

    xMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
    sepRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    sep0Rcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    xGuessRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    statusRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    constraintDiagonalRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
    partialSepPartialGammaDiagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));

    // solution space
    xSpaceRcp_ = Thyra::createVectorSpace<Scalar, LO, GO, Node>(xMapRcp_);

    // residual space
    fMapRcp_ = xMapRcp_;
    fSpaceRcp_ = xSpaceRcp_;

    // build D(q^p), which maps from force magnatude to force vector
    mobMapRcp_ = mobMatRcp_->getDomainMap();
    AMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, xMapRcp_);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(AMatTransRcp_);
    AMatRcp_ = transposerDu.createTranspose();
    TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(AMatRcp_));

    // build dt (D^p)^T M^k D^p
    // this is what we will use to approximate the diagonal of dt D^{k+1} M^k D^p
    partialSepPartialGammaMatRcp_ = Teuchos::rcp(new TCMAT(xMapRcp_, 0));
    Tpetra::TripleMatrixMultiply::MultiplyRAP(*AMatRcp_, true, *mobMatRcp_, false, *AMatRcp_, false, *partialSepPartialGammaMatRcp_);
    partialSepPartialGammaMatRcp_->resumeFill();
    partialSepPartialGammaMatRcp_->scale(dt_);
    partialSepPartialGammaMatRcp_->fillComplete();

    // build the diagonal of dt (D^p)^T M^k D^p
    partialSepPartialGammaMatRcp_->getLocalDiagCopy(*partialSepPartialGammaDiagRcp_); 

    // fill the initial gamma guess, the initial unconstrained separation, and the diagonal of K^{-1}(q^{k+1}, gamma^k)
    // use forceMagRcp_ as fake storage for the unused biFlagRcp
    conCollectorPtr_->fillFixedConstraintInfo(xGuessRcp_, forceMagRcp_, sep0Rcp_, constraintDiagonalRcp_);

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
        if (fill_f) {
            fRcp->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
        }

        if (fill_W) {
            JRcp->unitialize();
        }

        ////////////////////////////////////
        // evaluate the unconstrained sep //
        ////////////////////////////////////
        sepRcp_->scale(1.0, *sep0Rcp_); // store initial separation in sep
        partialSepPartialGammaMatRcp_->apply(*xRcp, *sepRcp_, Teuchos::NO_TRANS, 1.0, 1.0);

        //////////////////////
        // Fill the objects //
        //////////////////////

        // fill the constraint value
        if (fill_f) {
            Teuchos::TimeMonitor mon(*debugTimer2);

            // evaluate C(q^{k+1}(gamma^k), scale * gamma^k)
            conCollectorPtr_->evalConstraintValues(xRcp, partialSepPartialGammaDiagRcp_, sepRcp_, fRcp, statusRcp_);
        }

        // fill the Jacobian
        if (fill_W) {
            Teuchos::TimeMonitor mon(*debugTimer3);

            // setup J
            conCollectorPtr_->evalConstraintValues(xRcp, partialSepPartialGammaDiagRcp_, sepRcp_, forceMagRcp_,
                                                   statusRcp_);
            JRcp->initialize(partialSepPartialGammaMatRcp_, partialSepPartialGammaDiagRcp_, constraintDiagonalRcp_,
                             statusRcp_, dt_);

            // // for debug, dump to file
            // dumpToFile(JRcp, forceMagRcp_, statusRcp_, partialSepPartialGammaDiagRcp_, xRcp, mobMatRcp_, AMatTransRcp_,
            //            constraintDiagonalRcp_);

            // std::cout << "" << std::endl;
        }

        ///////////
        // Debug //
        ///////////

        // // for debug, advance the particles and write their positions to a vtk file
        // // conCollectorPtr_->writeBackGamma(xRcp.getConst());
        // const std::string postfix = std::to_string(stepCount_);
        // ptcSystemPtr_->writeResult(stepCount_, "./result/", postfix);
        // stepCount_++;
    }
}

void EvaluatorTpetraConstraint::recursionStep(const Teuchos::RCP<const TV> &gammaRcp) {
    // TODO: make sure this matches the linear solver


    // store the constraint gamma and constraint sep
    conCollectorPtr_->writeBackConstraintVariables(gammaRcp, sepRcp_);

    // solve for the induced force and velocity
    AMatRcp_->apply(*gammaRcp, *forceRcp_);   // F_{r}^k = sum_{n=0}^{r} D_{n-1}^{k+1} gamma_{r}^k
    mobMatRcp_->apply(*forceRcp_, *velRcp_); // U_r^k = M^k F_r^k

    // send the constraint vel and force to the particles
    ptcSystemPtr_->saveForceVelocityConstraints(forceRcp_, velRcp_);

    // move the configuration from q^k to q_{r}^{k+1}
    ptcSystemPtr_->stepEuler(); // q_{r}^{k+1} = q^k + dt G^k U_r^k
    ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see

    // update all constraints
    // this must NOT generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    ptcSystemPtr_->updatePairCollision(); // TODO: this only supports systems with pure, particle-particle
                                            // collisions atm

    // add this recursion to D^T and D
    conCollectorPtr_->updateConstraintMatrixVector(AMatTransRcp_);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(AMatTransRcp_);
    AMatRcp_ = transposerDu.createTranspose();
    TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(AMatRcp_));

    // update PartialSepPartialGamma
    partialSepPartialGammaMatRcp_ = Teuchos::rcp(new TCMAT(xMapRcp_, 0)); // Doesn't play well with filling the old matrix :(
    Tpetra::TripleMatrixMultiply::MultiplyRAP(*AMatRcp_, true, *mobMatRcp_, false, *AMatRcp_, false, *partialSepPartialGammaMatRcp_);
    partialSepPartialGammaMatRcp_->resumeFill();
    partialSepPartialGammaMatRcp_->scale(dt_);
    partialSepPartialGammaMatRcp_->fillComplete();

    // get the diagonal of dt (D^p)^T M^k D^p
    partialSepPartialGammaMatRcp_->getLocalDiagCopy(*partialSepPartialGammaDiagRcp_); 

    // fill the initial gamma guess, the initial unconstrained separation, and the diagonal of K^{-1}(q^{k+1}, gamma^k)
    // use forceMagRcp_ as fake storage for the unused biFlagRcp
    conCollectorPtr_->fillFixedConstraintInfo(xGuessRcp_, forceMagRcp_, sep0Rcp_, constraintDiagonalRcp_);

    // set the new initial condition
    x0Rcp_ = Thyra::createVector(xGuessRcp_, xSpaceRcp_);
    nominalValues_.set_x(x0Rcp_);
}

void EvaluatorTpetraConstraint::dumpToFile(const Teuchos::RCP<const TOP> &JRcp, const Teuchos::RCP<const TV> &fRcp,
                                           const Teuchos::RCP<const TV> &statusRcp,
                                           const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp,
                                           const Teuchos::RCP<const TV> &xRcp, const Teuchos::RCP<const TCMAT> &mobMatRcp,
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
    if (nonnull(mobMatRcp)) {
        dumpTCMAT(mobMatRcp, "Mobility");
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

void JacobianOperator::initialize(const Teuchos::RCP<const TCMAT> &partialSepPartialGammaMatRcp,
                                  const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp,
                                  const Teuchos::RCP<const TV> &constraintDiagonalRcp,
                                  const Teuchos::RCP<const TV> &statusRcp, const double dt) {
    // store the objects to be used in apply
    dt_ = dt;
    statusRcp_ = statusRcp;
    constraintDiagonalRcp_ = constraintDiagonalRcp;
    partialSepPartialGammaMatRcp_ = partialSepPartialGammaMatRcp;
    partialSepPartialGammaDiagRcp_ = partialSepPartialGammaDiagRcp;

    // check the input
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintDiagonalRcp_->getMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*partialSepPartialGammaDiagRcp_->getMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*partialSepPartialGammaMatRcp_->getRangeMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*partialSepPartialGammaMatRcp_->getDomainMap()));

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
    partialSepPartialGammaMatRcp_.reset();
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
        activeXcolRcp_->elementWiseMultiply(1.0, *XcolRcp, *statusRcp_, 0.0);

        // step 2. change_in_sep = dt D^T M D Xactive
        partialSepPartialGammaMatRcp_->apply(*activeXcolRcp_, *changeInSepRcp_);

        // step 3. solve dt D^T M D X + Diag X and store in changeInSepRcp_
        changeInSepRcp_->elementWiseMultiply(1.0, *constraintDiagonalRcp_, *activeXcolRcp_, 1.0);

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
        }
    }
}

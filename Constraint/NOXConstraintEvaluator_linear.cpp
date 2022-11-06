// Our stuff
#include "NOXConstraintEvaluator.hpp"

// Teuchos support
#include <Teuchos_CommHelpers.hpp>

// Thyra support
#include <Thyra_MultiVectorStdOps.hpp>
#include <Thyra_VectorStdOps.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

// Kokkos support
#include <Kokkos_Core.hpp>

// Constructor
EvaluatorTpetraConstraint::EvaluatorTpetraConstraint(const Teuchos::RCP<const TCOMM>& commRcp,
                     const Teuchos::RCP<const TOP> &mobOpRcp, 
                     const Teuchos::RCP<const TV> &velExternalRcp, 
                     const Teuchos::RCP<const TV> &forceExternalRcp, 
                     std::shared_ptr<ConstraintCollector> conCollectorPtr,
                     std::shared_ptr<ParticleSystem> ptcSystemPtr,
                     const Teuchos::RCP<TV> &forceRcp,
                     const Teuchos::RCP<TV> &velRcp,
                     const double dt) :
  commRcp_(commRcp),
  mobOpRcp_(mobOpRcp),
  velExternalRcp_(velExternalRcp),
  forceExternalRcp_(forceExternalRcp),
  conCollectorPtr_(std::move(conCollectorPtr)), 
  ptcSystemPtr_(std::move(ptcSystemPtr)),
  dt_(dt),
  forceRcp_(forceRcp),
  velRcp_(velRcp),
  showGetInvalidArg_(false)
{
  TEUCHOS_ASSERT(nonnull(commRcp_));
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));
  TEUCHOS_ASSERT(nonnull(velExternalRcp_));
  TEUCHOS_ASSERT(nonnull(forceExternalRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));

  // initialize the various objects
  const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfConstraints();
  xMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
  xGuessRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintFlagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintScaleRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintKappaRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constrainedSepRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  unconstrainedSepRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintDiagonalRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  PartialSepPartialGammaOpRcp_ = Teuchos::rcp(new PartialSepPartialGammaOp(xMapRcp_));

  // build A(q^0), which maps from force magnatude to force vector
  mobMapRcp_ = mobOpRcp_->getDomainMap();
  AMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, xMapRcp_);
  Tpetra::RowMatrixTransposer<double, int, int> transposerDu(AMatTransRcp_);
  AMatRcp_ = transposerDu.createTranspose();
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));

  // fill the fixed constraint information from ConstraintCollector
  conCollectorPtr_->fillConstraintInformation(commRcp_, xGuessRcp_, unconstrainedSepRcp_, 
                                              constraintKappaRcp_, constraintFlagRcp_);

  // solve for the unconstrained separation, which we would like to resolve
  // compute unconstrainedSep = 1.0 * initialSep + dt (A^k)^T (U_external^k + M^k F_external^k)
  velRcp_->assign(*velExternalRcp_); // deepcopy 
  mobOpRcp_->apply(*forceExternalRcp_, *velRcp_, Teuchos::NO_TRANS, 1.0, 1.0); 
  AMatTransRcp_->apply(*velRcp_, *unconstrainedSepRcp_, Teuchos::NO_TRANS, dt, 1.0);

  // solution space
  xSpaceRcp_ = Thyra::createVectorSpace<Scalar, LO, GO, Node>(xMapRcp_);

  // residual space
  fMapRcp_ = xMapRcp_;
  fSpaceRcp_ = xSpaceRcp_;

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


void EvaluatorTpetraConstraint::
setShowGetInvalidArgs(bool showGetInvalidArg)
{
  showGetInvalidArg_ = showGetInvalidArg;
}


void EvaluatorTpetraConstraint::
set_W_factory(const Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >& W_factoryRcp)
{
  W_factoryRcp_ = W_factoryRcp;
}


// Public functions overridden from ModelEvaulator
Teuchos::RCP<const thyra_vec_space>
EvaluatorTpetraConstraint::get_x_space() const
{
  return xSpaceRcp_;
}


Teuchos::RCP<const thyra_vec_space>
EvaluatorTpetraConstraint::get_f_space() const
{
  return fSpaceRcp_;
}


Thyra::ModelEvaluatorBase::InArgs<Scalar>
EvaluatorTpetraConstraint::getNominalValues() const
{
  return nominalValues_;
}


Teuchos::RCP<thyra_op>
EvaluatorTpetraConstraint::create_W_op() const
{
  Teuchos::RCP<JacobianOperator> W_tpetra = Teuchos::rcp(new JacobianOperator(xMapRcp_));
  return Thyra::tpetraLinearOp<Scalar, LO, GO, Node>(fSpaceRcp_, xSpaceRcp_, W_tpetra);
}


Teuchos::RCP<const Thyra::LinearOpWithSolveFactoryBase<Scalar> >
EvaluatorTpetraConstraint::get_W_factory() const
{
  return W_factoryRcp_;
}


Thyra::ModelEvaluatorBase::InArgs<Scalar>
EvaluatorTpetraConstraint::createInArgs() const
{
  return prototypeInArgs_;
}


// Private functions overridden from ModelEvaulatorDefaultBase


Thyra::ModelEvaluatorBase::OutArgs<Scalar>
EvaluatorTpetraConstraint::createOutArgsImpl() const
{
  return prototypeOutArgs_;
}


void EvaluatorTpetraConstraint::
evalModelImpl(const Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
              const Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs) const
{
  TEUCHOS_ASSERT(nonnull(inArgs.get_x()));

  const Teuchos::RCP<thyra_vec> f_out = outArgs.get_f();
  const Teuchos::RCP<thyra_op> W_out = outArgs.get_W_op();
  const Teuchos::RCP<thyra_prec> W_prec_out = outArgs.get_W_prec();

  const bool fill_f = nonnull(f_out);
  const bool fill_W = nonnull(W_out);
  const bool fill_W_prec = nonnull(W_prec_out);

  using tpetra_extract = Thyra::TpetraOperatorVectorExtraction<Scalar,LO,GO,Node>;

  if (fill_f || fill_W || fill_W_prec) {
    // std::cout << "fill_f: " << fill_f << " | fill_W: " << fill_W << "| fill_W_prec: " << fill_W_prec << std::endl;

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
    PartialSepPartialGammaOpRcp_->unitialize();

    if (fill_f) {
      fRcp->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
    }

    if (fill_W) {
      JRcp->unitialize();
    }
    
    //////////////////////
    // Fill the objects //
    //////////////////////

    // setup PartialSepPartialGamma(q^k, q^k+1, gamma^k)
    PartialSepPartialGammaOpRcp_->initialize(mobOpRcp_, AMatTransRcp_, AMatRcp_, forceRcp_, velRcp_, dt_);

    // compute the new separation, sep(q^k+1)
    // sep^k+1 = sep_unconstrained + dt (A^k)^T M^k A^k gamma^k
    constrainedSepRcp_->assign(*unconstrainedSepRcp_); // deep-copy 
    PartialSepPartialGammaOpRcp_->apply(*xRcp, *constrainedSepRcp_, Teuchos::NO_TRANS, 1.0, 1.0);

    // evaluate the diagonal of the scale matrix, S(q^k+1, gamma^k)
    // fill diagonal matrix that is added to the Jacobian, K(q^k, q^k+1, gamma^k)
    {
      auto xPtr = xRcp->getLocalView<Kokkos::HostSpace>();
      auto constraintFlagPtr = constraintFlagRcp_->getLocalView<Kokkos::HostSpace>();
      auto constrainedSepPtr = constrainedSepRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintKappaPtr = constraintKappaRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintScalePtr = constraintScaleRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintDiagonalPtr = constraintDiagonalRcp_->getLocalView<Kokkos::HostSpace>();
      constraintScaleRcp_->modify<Kokkos::HostSpace>();
      const auto localSize = xPtr.extent(0);
#pragma omp parallel for
      for (size_t idx = 0; idx < localSize; idx++) {
        if (constraintFlagPtr(idx, 0)) { // spring constraint
          constraintScalePtr(idx, 0) = 1.0;
          constraintDiagonalPtr(idx, 0) = 1.0 / constraintKappaPtr(idx, 0); // TODO: this class needs access to kappa 
        } else { 
          // collision constraint using Ficher Bermiester function
          // constraintScalePtr is the partial derivative of the FB function (sep^k+1, gamma^k) w.r.t sep^k+1
          // constraintDiagonalPtr is the partial derivative of the FB function (sep^k, gamma^k) w.r.t gamma^k
          if ((std::abs(constrainedSepPtr(idx, 0)) < 1e-6) && (std::abs(xPtr(idx, 0)) < 1e-6)) {
              constraintScalePtr(idx, 0) = 0.0;
              constraintDiagonalPtr(idx, 0) = 1.0;
          } else {
              constraintScalePtr(idx, 0) = 1.0 - constrainedSepPtr(idx, 0) / std::sqrt(std::pow(constrainedSepPtr(idx, 0) , 2) + std::pow(xPtr(idx, 0) , 2));
              constraintDiagonalPtr(idx, 0) = 1.0 - xPtr(idx, 0) / std::sqrt(std::pow(constrainedSepPtr(idx, 0) , 2) + std::pow(xPtr(idx, 0) , 2));
          }
        }
        // std::cout << "constraintScalePtr(idx, 0): " << constraintScalePtr(idx, 0) << " | constraintDiagonalPtr(idx, 0): " << constraintDiagonalPtr(idx, 0) << std::endl;
      }
    }

    // update the Jacobian
    if (fill_W) {
      // setup J(q^k, q^k+1, gamma^k)
      JRcp->initialize(PartialSepPartialGammaOpRcp_, constraintScaleRcp_, constraintDiagonalRcp_, dt_);

      // for debug, dump to file
      // dumpToFile(JRcp, xRcp, mobOpRcp_, AMatTransRcp_, AMatRcp_, constraintScaleRcp_, constraintDiagonalRcp_);
    }

    if (fill_f) {
      // evaluate F(q^k+1(gamma^k), gamma^k)
      {
        auto xPtr = xRcp->getLocalView<Kokkos::HostSpace>();
        auto fPtr = fRcp->getLocalView<Kokkos::HostSpace>();
        auto constraintFlagPtr = constraintFlagRcp_->getLocalView<Kokkos::HostSpace>();
        auto constrainedSepPtr = constrainedSepRcp_->getLocalView<Kokkos::HostSpace>();
        auto constraintKappaPtr = constraintKappaRcp_->getLocalView<Kokkos::HostSpace>();
        fRcp->modify<Kokkos::HostSpace>();
        const auto localSize = xPtr.extent(0);
#pragma omp parallel for
        for (size_t idx = 0; idx < localSize; idx++) {
          if (constraintFlagPtr(idx, 0)) { // spring constraint
            // spring constraint
            // this is the value of linear spring constraint evaluated at q^k+1, gamma^k
            fPtr(idx, 0) = constrainedSepPtr(idx, 0) + 1.0 / constraintKappaPtr(idx, 0) * xPtr(idx, 0);
          } else { 
            // collision constraint using Ficher Bermiester function
            // this is the value of the FB function evaluated at gamma^k, sep^k+1
            fPtr(idx, 0) = constrainedSepPtr(idx, 0) + xPtr(idx, 0) 
                - std::sqrt(std::pow(constrainedSepPtr(idx, 0) , 2) + std::pow(xPtr(idx, 0) , 2));
          }
          std::cout << "constrainedSepPtr(idx, 0) " << constrainedSepPtr(idx, 0) << " xPtr(idx, 0) " << xPtr(idx, 0) << " fPtr(idx, 0) " << fPtr(idx, 0) << std::endl;
        }
      }
    }

    ///////////
    // Debug //
    ///////////

    // for debug, advance the particles and write their positions to a vtk file
    // conCollectorPtr_->writeBackGamma(xRcp.getConst());
    // const std::string postfix = std::to_string(stepCount_);
    // ptcSystemPtr_->writeResult(stepCount_, "./result/result0-399/", postfix);
    // stepCount_++;
  }
}


void EvaluatorTpetraConstraint::dumpToFile(const Teuchos::RCP<const TOP> &JRcp, 
                                           const Teuchos::RCP<const TV> &xRcp, 
                                           const Teuchos::RCP<const TOP> &mobOpRcp,
                                           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
                                           const Teuchos::RCP<const TCMAT> &AMatRcp,
                                           const Teuchos::RCP<const TV> &constraintScaleRcp, 
                                           const Teuchos::RCP<const TV> &constraintDiagonalRcp) const {
  // only dump nonnull objects
  if (nonnull(JRcp)) {dumpTOP(JRcp, "Jacobian");}
  if (nonnull(xRcp)) {dumpTV(xRcp, "Gamma");}
  if (nonnull(mobOpRcp)) {dumpTOP(mobOpRcp, "Mobility");}
  if (nonnull(AMatTransRcp)) {dumpTCMAT(AMatTransRcp, "AmatT");}
  if (nonnull(AMatRcp)) {dumpTCMAT(AMatRcp, "Amat");}
  if (nonnull(constraintScaleRcp)) {dumpTV(constraintScaleRcp, "scale");}
  if (nonnull(constraintDiagonalRcp)) {dumpTV(constraintDiagonalRcp, "diag");}
}

///////////////////////////////////////
// Jacobian operator implementations //
///////////////////////////////////////

JacobianOperator::JacobianOperator(const Teuchos::RCP<const TMAP> &xMapRcp) :
  xMapRcp_(xMapRcp) {}


void JacobianOperator::initialize(const Teuchos::RCP<const PartialSepPartialGammaOp> &PartialSepPartialGammaOpRcp, 
                                  const Teuchos::RCP<const TV> &constraintScaleRcp, 
                                  const Teuchos::RCP<const TV> &constraintDiagonalRcp, 
                                  const double dt)
{
  // store the objects to be used in apply
  dt_ = dt;
  constraintScaleRcp_ = constraintScaleRcp;
  constraintDiagonalRcp_ = constraintDiagonalRcp;
  PartialSepPartialGammaOpRcp_ = PartialSepPartialGammaOpRcp;

  // check the input
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintDiagonalRcp_->getMap()));
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*PartialSepPartialGammaOpRcp_->getDomainMap()));

  // create the internal data structures
  changeInSepRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
}


void JacobianOperator::unitialize()
{
  dt_ = 0.0;
  changeInSepRcp_.reset(); 
  constraintScaleRcp_.reset(); 
  constraintDiagonalRcp_.reset(); 
  PartialSepPartialGammaOpRcp_.reset(); 
}


Teuchos::RCP<const TMAP> JacobianOperator::getDomainMap() const
{
  return xMapRcp_;
}

Teuchos::RCP<const TMAP> JacobianOperator::getRangeMap() const
{
  return xMapRcp_;
}

bool JacobianOperator::hasTransposeApply() const 
{
  return true;
}

void JacobianOperator::
apply(const TMV& X, TMV& Y, Teuchos::ETransp mode,
      Scalar alpha, Scalar beta) const {
  TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                              "X and Y do not have the same numbers of vectors (columns).");
  TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                              "X and Y do not have the same Map.\n");
  TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // The jacobian is NOT symmetric, so trans apply is NOT ok.
  TEUCHOS_ASSERT(nonnull(constraintDiagonalRcp_));

  const int numVecs = X.getNumVectors();
  for (int i = 0; i < numVecs; i++) {
    auto XcolRcp = X.getVector(i);
    auto YcolRcp = Y.getVectorNonConst(i);
    // Goal Y = alpha (dt S^T A^T M A X) + alpha Diag Y + beta Y

    // step 1. change in sep = dt A^T M A X
    PartialSepPartialGammaOpRcp_->apply(*XcolRcp, *changeInSepRcp_);

    // step 2/3. solve alpha dt S^T A^T M A X and add on the diagonal alpha Diag Y + beta Y
    {
      auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
      auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
      auto changeInSepPtr = changeInSepRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintScalePtr = constraintScaleRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintDiagonalPtr = constraintDiagonalRcp_->getLocalView<Kokkos::HostSpace>();
      changeInSepRcp_->modify<Kokkos::HostSpace>();
      const auto localSize = changeInSepPtr.extent(0);
#pragma omp parallel for
      for (size_t idx = 0; idx < localSize; idx++) {
        if (beta == Teuchos::ScalarTraits<Scalar>::zero()) {
          YcolPtr(idx, 0) = alpha * constraintScalePtr(idx, 0) * changeInSepPtr(idx, 0) 
                          + alpha * constraintDiagonalPtr(idx, 0) * XcolPtr(idx, 0);        
        } else { // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values?
          YcolPtr(idx, 0) = alpha * constraintScalePtr(idx, 0) * changeInSepPtr(idx, 0) 
                          + alpha * constraintDiagonalPtr(idx, 0) * XcolPtr(idx, 0) 
                          + beta * YcolPtr(idx, 0); 
        }
      }
    }
  }
}

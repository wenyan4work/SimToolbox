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
                     std::shared_ptr<ConstraintCollector> conCollectorPtr,
                     std::shared_ptr<SylinderSystem> ptcSystemPtr,
                     const double dt) :
  commRcp_(commRcp),
  mobOpRcp_(mobOpRcp),
  conCollectorPtr_(std::move(conCollectorPtr)), 
  ptcSystemPtr_(std::move(ptcSystemPtr)),
  dt_(dt),
  showGetInvalidArg_(false)
{
  TEUCHOS_ASSERT(nonnull(commRcp_));
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));

  // initialize the various objects
  mobMapRcp_ = mobOpRcp_->getDomainMap();
  forceRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
  velRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));

  const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfConstraints();
  xMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
  xGuessRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintFlagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintScaleRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintDiagonalRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));

  // fill the fixed constraint information from ConstraintCollector
  conCollectorPtr_->fillConstraintInformation(commRcp_, xGuessRcp_, constraintFlagRcp_);

  // build A(q^0), which maps from force magnatude to force vector
  AMatTransRcp_ = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, xMapRcp_);
  Tpetra::RowMatrixTransposer<double, int, int> transposerDu(AMatTransRcp_);
  AMatRcp_ = transposerDu.createTranspose();
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));

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
    if (fill_f) {
      fRcp->putScalar(Teuchos::ScalarTraits<Scalar>::zero());
    }
    if (fill_W) {
      JRcp->unitialize();
    }
    
    /////////////////////////////
    // Reset the configuration //
    /////////////////////////////
    // Beware, the systems will update conCollector each time evalModelImpl is called
    // we must reset the conCollector to q^k before proceeding,
    // TODO: can this be replaced by storing two ConstraintCollectors?
    
    // reset the configuration to q^k
    // move the particles back to their previous positions 
    ptcSystemPtr_->resetConfiguration(); // reset configuration to q^k
    ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see

    // update all constraints to revert changes
    // this must NOT generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    ptcSystemPtr_->updatePairCollision();

    //////////////////////
    // Fill the objects //
    //////////////////////
    // evaluate the diagonal of the scale matrix, S(gamma^k)
    conCollectorPtr_->evalConstraintScale(xRcp, constraintScaleRcp_); 

    // update the Jacobian using q^k
    if (fill_W) {
      // fill diagonal of K(q^k, gamma^k)
      conCollectorPtr_->evalConstraintDiagonal(xRcp, constraintDiagonalRcp_); 

      // setup J(q^k, gamma^k)
      JRcp->initialize(mobOpRcp_, AMatTransRcp_, AMatRcp_, constraintScaleRcp_, 
                      constraintDiagonalRcp_, dt_);

      // for debug, dump to file
      // dumpToFile(JRcp, xRcp, mobOpRcp_, AMatTransRcp_, AMatRcp_, constraintScaleRcp_, constraintDiagonalRcp_);
    }

    // update the configuration to from q^k to q^k+1

    // step 0. scale gamma to get force magnatude 
    // force_mag = S(lambda^k) gamma^k
    {
      auto xPtr = xRcp->getLocalView<Kokkos::HostSpace>();
      auto forceMagPtr = forceMagRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintScalePtr = constraintScaleRcp_->getLocalView<Kokkos::HostSpace>();
      forceMagRcp_->modify<Kokkos::HostSpace>();
      const int localSize = xPtr.dimension_0();
#pragma omp parallel for
      for (int k = 0; k < localSize; k++) {
        forceMagPtr(k, 0) = constraintScalePtr(k, 0) * xPtr(k, 0);
      }
    }

    // solve for the induced force and velocity
    AMatRcp_->apply(*forceMagRcp_, *forceRcp_); // F^k = A(q^k) S(gamma^k) gamma^k
    mobOpRcp_->apply(*forceRcp_, *velRcp_); // U^k = M(q^k) F^k

    // update the particle system from the q^k to q^k+1
    // send the constraint vel and force to the particles
    ptcSystemPtr_->saveForceVelocityConstraints(forceRcp_, velRcp_);
    // merge the constraint and nonconstraint vel and force
    ptcSystemPtr_->sumForceVelocity();
    // move the particles according to the total velocity
    ptcSystemPtr_->stepEuler(); // q^k+1 = q^k + dt G(q^k) U^k
    ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see

    // update all constraints
    // this must NOT generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    ptcSystemPtr_->updatePairCollision();

    if (fill_f) {
      // fill the constraint value
      // evaluate F(q^k+1(gamma^k), gamma^k)
      conCollectorPtr_->evalConstraintValues(xRcp, fRcp); 
    }

    // if (fill_W) {
    //   // advance the reference config to the updated configuration
    //   ptcSystemPtr_->advanceParticles();
    // }

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


void JacobianOperator::initialize(const Teuchos::RCP<const TOP> &mobOpRcp, 
           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
           const Teuchos::RCP<const TCMAT> &AMatRcp,
           const Teuchos::RCP<const TV> &constraintScaleRcp, 
           const Teuchos::RCP<const TV> &constraintDiagonalRcp, 
           const double dt)
{
  // store the objects to be used in apply
  dt_ = dt;
  mobOpRcp_ = mobOpRcp;
  AMatTransRcp_ = AMatTransRcp;
  AMatRcp_ = AMatRcp;
  constraintScaleRcp_ = constraintScaleRcp;
  constraintDiagonalRcp_ = constraintDiagonalRcp;
  mobMapRcp_ = mobOpRcp_->getDomainMap(); // symmetric & domainmap = rangemap

  // check the input
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintScaleRcp_->getMap()));
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintDiagonalRcp_->getMap()));
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*AMatTransRcp_->getRangeMap()));
  TEUCHOS_ASSERT(mobMapRcp_->isSameAs(*AMatTransRcp_->getDomainMap()));

  // initialize working multivectors, zero out
  velRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
  forceRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
  forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
}


void JacobianOperator::unitialize()
{
  dt_ = 0.0;
  mobMapRcp_.reset(); 
  mobOpRcp_.reset(); 
  AMatRcp_.reset(); 
  AMatTransRcp_.reset(); 
  constraintScaleRcp_.reset();
  constraintDiagonalRcp_.reset(); 
  forceRcp_.reset(); 
  velRcp_.reset(); 
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
  // TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // The jacobian is symmetric, so trans apply is ok.
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));
  TEUCHOS_ASSERT(nonnull(forceMagRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(constraintScaleRcp_));
  TEUCHOS_ASSERT(nonnull(constraintDiagonalRcp_));

  const int numVecs = X.getNumVectors();
  for (int i = 0; i < numVecs; i++) {
    auto XcolRcp = X.getVector(i);
    auto YcolRcp = Y.getVectorNonConst(i);
    // Goal Y = alpha (dt S^T A^T M A S X) + beta Y

    // step 1. Scale times gamma to get force magnatude (f = S x)
    {
      auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
      auto forceMagPtr = forceMagRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintScalePtr = constraintScaleRcp_->getLocalView<Kokkos::HostSpace>();
      forceMagRcp_->modify<Kokkos::HostSpace>();
      const int localSize = XcolPtr.dimension_0();
#pragma omp parallel for
      for (int k = 0; k < localSize; k++) {
        forceMagPtr(k, 0) = constraintScalePtr(k, 0) * XcolPtr(k, 0);
      }
    }

    // step 2. A times force magnatude to get force/torque vector (F = A f)
    {
      AMatRcp_->apply(*forceMagRcp_, *forceRcp_);
    }

    // step 3. Mobility times force/torque vector to get velocity (U = M F)
    {
      mobOpRcp_->apply(*forceRcp_, *velRcp_);
    }

    // step 4. A^T times velocity to get adjoint (result = A^T U)
    // here, we use forceMagRcp_ as a placeholder for the result
    {
      AMatTransRcp_->apply(*velRcp_, *forceMagRcp_);
    }

    // step 5. dt times S times A^T U 
    // we merge into this step the fact that Tpetra OPs wants to solve Y = alpha * Op * X + beta * Y
    {
      auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
      auto forceMagPtr = forceMagRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintScalePtr = constraintScaleRcp_->getLocalView<Kokkos::HostSpace>();
      YcolRcp->modify<Kokkos::HostSpace>();
      const int localSize = YcolPtr.dimension_0();
#pragma omp parallel for
      for (int k = 0; k < localSize; k++) {
        YcolPtr(k, 0) = dt_ * constraintScalePtr(k, 0) * forceMagPtr(k, 0); // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values? 
        // YcolPtr(k, 0) = alpha * dt_ * constraintScalePtr(k, 0) * forceMagPtr(k, 0) + beta * YcolPtr(k, 0);
      }
    }
   
    // step 6. add diagonal. Y += alpha * Diag * X
    {
      auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
      auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
      auto constraintDiagonalPtr = constraintDiagonalRcp_->getLocalView<Kokkos::HostSpace>();
      YcolRcp->modify<Kokkos::HostSpace>();
      const int localSize = YcolPtr.dimension_0();
#pragma omp parallel for
      for (int k = 0; k < localSize; k++) {
        YcolPtr(k, 0) += alpha * constraintDiagonalPtr(k, 0) * XcolPtr(k, 0);
      }
    }
  }
}

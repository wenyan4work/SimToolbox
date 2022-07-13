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
  constraintFlagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintDiagonalRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));

  // fill the fixed constraint information from ConstraintCollector
  conCollectorPtr_->fillConstraintInformation(commRcp_, xGuessRcp_, constraintFlagRcp_);

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
    std::cout << "fill_f: " << fill_f << " | fill_W: " << fill_W << "| fill_W_prec: " << fill_W_prec << std::endl;

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
    
    //////////////////////
    // Fill the objects //
    //////////////////////

    // // setup D(q^k, lambda^k)
    Teuchos::RCP<const TCMAT> DMatTransRcp = conCollectorPtr_->buildConstraintMatrixVector(mobMapRcp_, xMapRcp_, xRcp);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(DMatTransRcp);
    Teuchos::RCP<const TCMAT> DMatRcp = transposerDu.createTranspose();

    // step 1 solve for the induced force and velocity
    DMatRcp_->apply(*xRcp, *forceRcp_); // F^k = D(q^k) lambda 
    mobOpRcp_->apply(*forceRcp_, *velRcp_); // U^k = M(q^k) F^k

    // step 2. update the particle system
    // send the constraint vel and force to the particles
    ptcSystemPtr_->saveForceVelocityConstraints(forceRcp_, velRcp_);
    // merge the constraint and nonconstraint vel and force
    ptcSystemPtr_->sumForceVelocity();
    // move the particles according to the total velocity
    ptcSystemPtr_->stepEuler(); // q^k+1 = q^k + dt G(q^k) U^k

    // step 3. update all constraints
    // this must NOT generate new constraints and MUST maintain the current constraint ordering
    ptcSystemPtr_->updatesylinderNearDataDirectory();
    ptcSystemPtr_->updatePairCollision();

    // (optional) step 4. fill the constraint values
    if (fill_f) {
      // fill the constraint based on the updated configuration
      // evaluate F(q^k+1(lambda^k), lambda^k)
      conCollectorPtr_->evalConstraintValues(xRcp, fRcp); 

      // for debug, advance the particles and write their positions to a vtk file
      // conCollectorPtr_->writeBackGamma(xRcp.getConst());
      // const std::string postfix = std::to_string(stepCount_);
      // ptcSystemPtr_->writeResult(stepCount_, "./result/result0-399/", postfix);
      // stepCount_++;
    }

    // (optional) step 5. generate the Jacobian operator using the current configuration
    if (fill_W) {

      conCollectorPtr_->evalConstraintDiagonal(xRcp, constraintDiagonalRcp_);
      // setup J(q^k, lambda^k)
      JRcp->initialize(mobOpRcp_, DMatTransRcp, DMatRcp, constraintDiagonalRcp_, dt_);

      // advance q^k to q^k+1  
    }

    // for debug, dump to file
    // dumpToFile(JRcp, xRcp, DMatTransRcp, mobOpRcp_);
  }
}


void EvaluatorTpetraConstraint::dumpToFile(const Teuchos::RCP<const TOP> JRcp, 
                                           const Teuchos::RCP<const TV> xRcp, 
                                           const Teuchos::RCP<const TCMAT> DMatTransRcp,
                                           const Teuchos::RCP<const TOP> mobOpRcp) const {
  // only dump nonnull objects
  if (nonnull(JRcp)) {dumpTOP(JRcp, "Jacobian");}
  if (nonnull(xRcp)) {dumpTV(xRcp, "Gamma");}
  if (nonnull(DMatTransRcp)) {dumpTCMAT(DMatTransRcp, "Dmat");}
  if (nonnull(mobOpRcp)) {dumpTOP(mobOpRcp, "Mobility");}
}

///////////////////////////////////////
// Jacobian operator implementations //
///////////////////////////////////////


JacobianOperator::JacobianOperator(const Teuchos::RCP<const TMAP> &xMapRcp) :
  xMapRcp_(xMapRcp) {}


void JacobianOperator::initialize(const Teuchos::RCP<const TOP> &mobOpRcp, 
           const Teuchos::RCP<const TCMAT> &DMatTransRcp,
           const Teuchos::RCP<const TCMAT> &DMatRcp,
           const Teuchos::RCP<const TV> &constraintDiagonalRcp, 
           const double dt)
{
  // store the objects to be used in apply
  dt_ = dt;
  mobOpRcp_ = mobOpRcp;
  DMatTransRcp_ = DMatTransRcp;
  DMatRcp_ = DMatRcp;
  constraintDiagonalRcp_ = constraintDiagonalRcp;
  mobMapRcp_ = mobOpRcp_->getDomainMap(); // symmetric & domainmap = rangemap

  // check the input
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintDiagonalRcp_->getMap()));
  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*DMatTransRcp_->getRangeMap()));
  TEUCHOS_ASSERT(mobMapRcp_->isSameAs(*DMatTransRcp_->getDomainMap()));

  // initialize working multivectors, zero out
  forceRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
  velRcp_ = Teuchos::rcp(new TV(mobMapRcp_, true));
}


void JacobianOperator::unitialize()
{
  dt_ = 0.0;
  mobMapRcp_.reset(); 
  mobOpRcp_.reset(); 
  DMatRcp_.reset(); 
  DMatTransRcp_.reset(); 
  constraintDiagonalRcp_.reset(); 
  forceRcp_.reset(); 
  velRcp_.reset(); 
}


Teuchos::RCP<const TMAP>JacobianOperator::getDomainMap() const
{
  return xMapRcp_;
}

Teuchos::RCP<const TMAP>JacobianOperator::getRangeMap() const
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
  TEUCHOS_ASSERT(nonnull(velRcp_));
  TEUCHOS_ASSERT(nonnull(DMatRcp_));
  TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(constraintDiagonalRcp_));


  const int numVecs = X.getNumVectors();
  for (int i = 0; i < numVecs; i++) {
    auto XcolRcp = X.getVector(i);
    auto YcolRcp = Y.getVectorNonConst(i);

    // step 1, D multiply X
    {
      DMatRcp_->apply(*XcolRcp, *forceRcp_); // Du gammac
    }

    // step 2, Vel = Mobility * FT
    {
      mobOpRcp_->apply(*forceRcp_, *velRcp_);
    }

    // step 3, scale vel by dt
    {
      velRcp_->scale(dt_);
    }

    // step 4, D^T multiply velocity
    // Y = alpha * Op * X + beta * Y
    {
      DMatTransRcp_->apply(*velRcp_, *YcolRcp, Teuchos::NO_TRANS, alpha, beta);
    }

    // step 5, add diagonal. Y += alpha * invK * X
    auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
    auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
    YcolRcp->modify<Kokkos::HostSpace>();
    auto constraintDiagonalPtr = constraintDiagonalRcp_->getLocalView<Kokkos::HostSpace>();
    const int localSize = YcolPtr.dimension_0();
#pragma omp parallel for
    for (int k = 0; k < localSize; k++) {
      YcolPtr(k, 0) += alpha * constraintDiagonalPtr(k, 0) * XcolPtr(k, 0);
    }
  }
}

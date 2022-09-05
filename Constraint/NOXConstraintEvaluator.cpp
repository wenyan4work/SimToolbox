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
                     const Teuchos::RCP<const TCMAT> &mobMatRcp, 
                     std::shared_ptr<ConstraintCollector> conCollectorPtr,
                     std::shared_ptr<SylinderSystem> ptcSystemPtr,
                     const Teuchos::RCP<TV> &forceRcp,
                     const Teuchos::RCP<TV> &velRcp,
                     const double dt) :
  commRcp_(commRcp),
  mobMatRcp_(mobMatRcp),
  conCollectorPtr_(std::move(conCollectorPtr)), 
  ptcSystemPtr_(std::move(ptcSystemPtr)),
  dt_(dt),
  forceRcp_(forceRcp),
  velRcp_(velRcp),
  showGetInvalidArg_(false)
{
  TEUCHOS_ASSERT(nonnull(commRcp_));
  TEUCHOS_ASSERT(nonnull(mobMatRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));

  // initialize the various objects
  const int numLocalConstraints = conCollectorPtr_->getLocalNumberOfConstraints();
  xMapRcp_ = getTMAPFromLocalSize(numLocalConstraints, commRcp_);
  xGuessRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  projMaskRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintFlagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
  constraintKappaRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
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
  
  // fill the fixed constraint information from ConstraintCollector
  conCollectorPtr_->fillConstraintInformation(commRcp_, xGuessRcp_, 
                                              constraintKappaRcp_, constraintFlagRcp_); 

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

// Teuchos::RCP<thyra_prec> 
// EvaluatorTpetraConstraint::create_W_prec() const 
// {
//   // return the precreated Thyra preconditioner 
//   // this preconditioner is internally owned, I hope that is ok
//   return precThyraRcp_;
// }

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
  Teuchos::RCP<Teuchos::Time> debugTimer1 = Teuchos::TimeMonitor::getNewCounter("EvaluatorTpetraConstraint::update_config");
  Teuchos::RCP<Teuchos::Time> debugTimer2 = Teuchos::TimeMonitor::getNewCounter("EvaluatorTpetraConstraint::fill_f");
  Teuchos::RCP<Teuchos::Time> debugTimer3 = Teuchos::TimeMonitor::getNewCounter("EvaluatorTpetraConstraint::fill_J");

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
    
    //////////////////////////////
    // Update the configuration //
    //////////////////////////////
    {
      Teuchos::TimeMonitor mon(*debugTimer1);
      // Beware, the systems will update conCollector each time evalModelImpl is called
      // we must reset the conCollector to q^p before proceeding,
      
      // reset the configuration to q^p, the pseudo-configuration induced by external factors
      // move the particles back to their previous positions 
      ptcSystemPtr_->resetConfiguration(); // reset configuration to q^p

      // move the configuration from q^p to q^k+1
      // solve for the induced force and velocity
      AMatRcp_->apply(*xRcp, *forceRcp_); // F^k = D(q^k) gamma^k
      mobMatRcp_->apply(*forceRcp_, *velRcp_); // U^k = M(q^k) F^k

      // update the particle system from the q^p to q^k+1
      // send the constraint vel and force to the particles
      ptcSystemPtr_->saveForceVelocityConstraints(forceRcp_, velRcp_);
      // move the particles according to the total velocity
      ptcSystemPtr_->stepEuler(0); // q^k+1 = q^p + dt G(q^k) M^k D^p gamma^k 
      ptcSystemPtr_->applyBoxBC(); // TODO: this might not work. We'll see

      // update all constraints
      // this must NOT generate new constraints and MUST maintain the current constraint ordering
      ptcSystemPtr_->updatesylinderNearDataDirectory();
      ptcSystemPtr_->updatePairCollision();
    }

    //////////////////////
    // Fill the objects //
    //////////////////////

    // fill the constraint value
    if (fill_f) {
      Teuchos::TimeMonitor mon(*debugTimer2);
      // evaluate F(q^k+1(gamma^k), gamma^k)
      conCollectorPtr_->evalConstraintValues(xRcp, fRcp, projMaskRcp_); 

      // scale the projected values by the diagonal of dt D^T M D
      // this converts their units from force to distance
      {
        auto fPtr = fRcp->getLocalView<Kokkos::HostSpace>();
        auto projMaskPtr = projMaskRcp_->getLocalView<Kokkos::HostSpace>();
        auto partialSepPartialGammaDiagPtr = partialSepPartialGammaDiagRcp_->getLocalView<Kokkos::HostSpace>();
        fRcp->modify<Kokkos::HostSpace>();
        const auto localSize = fPtr.extent(0);
  #pragma omp parallel for
        for (size_t idx = 0; idx < localSize; idx++) {
          // scale the projection for Min Map
          if (projMaskPtr(idx, 0) < 0.5) { 
            // no projection (do nothing)
          } else {
            // projection (scale by the approximnate diagonal of dt D^T M D)
            // fPtr(idx, 0) = fPtr(idx, 0) * partialSepPartialGammaDiagPtr(idx, 0);        
            fPtr(idx, 0) = fPtr(idx, 0);        
          }
        }
      }
    }
   
    // fill the Jacobian and the preconditioner
    if (fill_W) {
      Teuchos::TimeMonitor mon(*debugTimer3);
      // // compute (D^{k+1})^T with each newton iteration
      // conCollectorPtr_->updateConstraintMatrixVector(AMatTransRcp_);

      // // setup PartialSepPartialGamma(q^k+1, q^k) = (D^{k+1})^T M^k D^p
      // // update the diagonal "preconditioner"
      // {
      //   const Teuchos::RCP<TCMAT> MARcp = Teuchos::rcp(new TCMAT(mobMatRcp_->getRowMap(), 0));
      //   Tpetra::MatrixMatrix::Multiply(*mobMatRcp_, false, *AMatRcp_, false, *MARcp);

      //   partialSepPartialGammaMatRcp_->resumeFill();
      //   Tpetra::MatrixMatrix::Multiply(*AMatTransRcp_, false, *MARcp, false, *partialSepPartialGammaMatRcp_);
      //   partialSepPartialGammaMatRcp_->resumeFill();
      //   partialSepPartialGammaMatRcp_->scale(dt_);
      //   partialSepPartialGammaMatRcp_->fillComplete();

      //   // // build the diagonal of dt (D^{k+1})^T M^k D^p
      //   // partialSepPartialGammaMatRcp_->getLocalDiagCopy(*partialSepPartialGammaDiagRcp_); 
      // }

      // eval the diagonal of K^{-1}(q^k+1, gamma^k)
      conCollectorPtr_->evalConstraintDiagonal(xRcp, constraintDiagonalRcp_); 

      // setup J(q^k, q^k+1, gamma^k)
      conCollectorPtr_->evalConstraintValues(xRcp, forceMagRcp_, projMaskRcp_); 
      JRcp->initialize(partialSepPartialGammaMatRcp_, partialSepPartialGammaDiagRcp_, constraintDiagonalRcp_, projMaskRcp_, dt_);

      // // for debug, dump to file
      // dumpToFile(JRcp, forceMagRcp_, projMaskRcp_, partialSepPartialGammaDiagRcp_, xRcp, mobMatRcp_, AMatTransRcp_, AMatRcp_, constraintDiagonalRcp_);

      // std::cout << "" << std::endl;
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
                                           const Teuchos::RCP<const TV> &fRcp,
                                           const Teuchos::RCP<const TV> &projMaskRcp,
                                           const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp,
                                           const Teuchos::RCP<const TV> &xRcp, 
                                           const Teuchos::RCP<const TCMAT> &mobMatRcp,
                                           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
                                           const Teuchos::RCP<const TCMAT> &AMatRcp,
                                           const Teuchos::RCP<const TV> &constraintDiagonalRcp) const {
  // only dump nonnull objects
  if (nonnull(JRcp)) {dumpTOP(JRcp, "Jacobian");}
  if (nonnull(fRcp)) {dumpTV(fRcp, "f");}
  if (nonnull(projMaskRcp)) {dumpTV(projMaskRcp, "mask");}
  if (nonnull(partialSepPartialGammaDiagRcp)) {dumpTV(partialSepPartialGammaDiagRcp, "PrecInv");}
  if (nonnull(xRcp)) {dumpTV(xRcp, "Gamma");}
  if (nonnull(mobMatRcp)) {dumpTCMAT(mobMatRcp, "Mobility");}
  if (nonnull(AMatTransRcp)) {dumpTCMAT(AMatTransRcp, "AmatT");}
  if (nonnull(AMatRcp)) {dumpTCMAT(AMatRcp, "Amat");}
  if (nonnull(constraintDiagonalRcp)) {dumpTV(constraintDiagonalRcp, "diag");}
}

///////////////////////////////////////
// Jacobian operator implementations //
///////////////////////////////////////

JacobianOperator::JacobianOperator(const Teuchos::RCP<const TMAP> &xMapRcp) :
  xMapRcp_(xMapRcp) {}


void JacobianOperator::initialize(const Teuchos::RCP<const TCMAT> &partialSepPartialGammaMatRcp, 
                                  const Teuchos::RCP<const TV> &partialSepPartialGammaDiagRcp, 
                                  const Teuchos::RCP<const TV> &constraintDiagonalRcp, 
                                  const Teuchos::RCP<const TV> &projMaskRcp,
                                  const double dt)
{
  // store the objects to be used in apply
  dt_ = dt;
  projMaskRcp_ = projMaskRcp;
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
}


void JacobianOperator::unitialize()
{
  dt_ = 0.0;
  projMaskRcp_.reset();
  changeInSepRcp_.reset(); 
  constraintDiagonalRcp_.reset(); 
  partialSepPartialGammaMatRcp_.reset(); 
  partialSepPartialGammaDiagRcp_.reset(); 
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
  return false;
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
    // Goal Y = alpha J^{-1}(dt D^T M A X + Diag X)  + beta Y
    auto XcolRcp = X.getVector(i);
    auto YcolRcp = Y.getVectorNonConst(i);

    // step 1. change_in_sep = dt D^T M D X
    partialSepPartialGammaMatRcp_->apply(*XcolRcp, *changeInSepRcp_);
 
    // step 2/3/4. solve dt D^T M D X and add on the diagonal Diag X
    // and apply the projection (with scaling of projected values)
    // also, account for Y = alpha Op X + beta Y
    {
      auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
      auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
      auto projMaskPtr = projMaskRcp_->getLocalView<Kokkos::HostSpace>();
      auto changeInSepPtr = changeInSepRcp_->getLocalView<Kokkos::HostSpace>();
      auto constraintDiagonalPtr = constraintDiagonalRcp_->getLocalView<Kokkos::HostSpace>();
      auto partialSepPartialGammaDiagPtr = partialSepPartialGammaDiagRcp_->getLocalView<Kokkos::HostSpace>();
      YcolRcp->modify<Kokkos::HostSpace>();
      const auto localSize = YcolPtr.extent(0);
#pragma omp parallel for
      for (size_t idx = 0; idx < localSize; idx++) {
        // apply projection for Min Map
        if (projMaskPtr(idx, 0) < 0.5) { // no projection
          if (beta == Teuchos::ScalarTraits<Scalar>::zero()) {
            YcolPtr(idx, 0) = alpha * changeInSepPtr(idx, 0) + alpha * constraintDiagonalPtr(idx, 0) * XcolPtr(idx, 0);        
          } else { // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values?
            YcolPtr(idx, 0) = alpha * changeInSepPtr(idx, 0) + alpha * constraintDiagonalPtr(idx, 0) * XcolPtr(idx, 0) 
                            + beta * YcolPtr(idx, 0); 
          }
        } else { // project
          if (beta == Teuchos::ScalarTraits<Scalar>::zero()) {
            // YcolPtr(idx, 0) = alpha * partialSepPartialGammaDiagPtr(idx, 0) * XcolPtr(idx, 0);        
            YcolPtr(idx, 0) = alpha * XcolPtr(idx, 0);        
          } else { // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values?
            // YcolPtr(idx, 0) = alpha * partialSepPartialGammaDiagPtr(idx, 0) * XcolPtr(idx, 0) + beta * YcolPtr(idx, 0); 
            YcolPtr(idx, 0) = alpha * XcolPtr(idx, 0) + beta * YcolPtr(idx, 0); 
          }        
        }
      }
    }
  }
}

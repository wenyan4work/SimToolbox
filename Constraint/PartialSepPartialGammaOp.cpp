#include "PartialSepPartialGammaOp.hpp"

PartialSepPartialGammaOp::PartialSepPartialGammaOp(const Teuchos::RCP<const TMAP> &xMapRcp) :
  xMapRcp_(xMapRcp) {}


void PartialSepPartialGammaOp::initialize(const Teuchos::RCP<const TOP> &mobOpRcp, 
           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
           const Teuchos::RCP<const TCMAT> &AMatRcp,
           const Teuchos::RCP<TV> &forceRcp,
           const Teuchos::RCP<TV> &velRcp,
           const double dt)
{

  // store the objects to be used in apply
  dt_ = dt;
  mobOpRcp_ = mobOpRcp;
  AMatTransRcp_ = AMatTransRcp;
  AMatRcp_ = AMatRcp;
  velRcp_ = velRcp;
  forceRcp_ = forceRcp;

  // check the input
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));

  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*AMatTransRcp_->getRangeMap()));
  TEUCHOS_ASSERT(mobOpRcp_->getDomainMap()->isSameAs(*AMatTransRcp_->getDomainMap()));

  // initialize working multivectors, zero out
  forceMagRcp_ = Teuchos::rcp(new TV(xMapRcp_, true));
}

void PartialSepPartialGammaOp::unitialize()
{
  dt_ = 0.0;
  mobOpRcp_.reset(); 
  AMatRcp_.reset(); 
  AMatTransRcp_.reset(); 
  forceRcp_.reset(); 
  velRcp_.reset(); 
}

Teuchos::RCP<const TMAP> PartialSepPartialGammaOp::getDomainMap() const
{
  return xMapRcp_;
}

Teuchos::RCP<const TMAP> PartialSepPartialGammaOp::getRangeMap() const
{
  return xMapRcp_;
}

bool PartialSepPartialGammaOp::hasTransposeApply() const 
{
  return true;
}

void PartialSepPartialGammaOp::
apply(const TMV& X, TMV& Y, Teuchos::ETransp mode,
      Scalar alpha, Scalar beta) const {
  TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                              "X and Y do not have the same numbers of vectors (columns).");
  TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                              "X and Y do not have the same Map.\n");
  // TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // dt A^T M A is symmetric, so trans apply is ok.
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));
  TEUCHOS_ASSERT(nonnull(forceMagRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));

  const int numVecs = X.getNumVectors();
  for (int i = 0; i < numVecs; i++) {
    auto XcolRcp = X.getVector(i);
    auto YcolRcp = Y.getVectorNonConst(i);
    // Goal Y = alpha (dt A^T M A X) + beta Y

    // step 1. A times force magnatude to get force/torque vector (F = A x)
    {
      AMatRcp_->apply(*XcolRcp, *forceRcp_);
    }

    // step 2. Mobility times force/torque vector to get velocity (U = M F)
    {
      mobOpRcp_->apply(*forceRcp_, *velRcp_);
    }

    // step 3. dt A^T times velocity to get change in sep (change in sep = dt A^T U)
    // we merge into this step the fact that Tpetra OPs wants to solve Y = alpha * Op * X + beta * Y
    {
      AMatTransRcp_->apply(*velRcp_, *YcolRcp, Teuchos::NO_TRANS, alpha * dt_, beta);
    }
  }
}
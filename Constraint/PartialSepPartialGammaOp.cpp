#include "PartialSepPartialGammaOp.hpp"

PartialSepPartialGammaOp::PartialSepPartialGammaOp(const Teuchos::RCP<const TMAP> &xMapRcp) :
  xMapRcp_(xMapRcp) {}


void PartialSepPartialGammaOp::initialize(const Teuchos::RCP<const TOP> &mobOpRcp, 
           const Teuchos::RCP<const TCMAT> &AMatTransRcp,
           const Teuchos::RCP<const TCMAT> &AMatRcp,
           const Teuchos::RCP<const TV> &constraintScaleRcp, 
           const Teuchos::RCP<TV> &forceRcp,
           const Teuchos::RCP<TV> &velRcp,
           const double dt)
{

  // store the objects to be used in apply
  dt_ = dt;
  mobOpRcp_ = mobOpRcp;
  AMatTransRcp_ = AMatTransRcp;
  AMatRcp_ = AMatRcp;
  constraintScaleRcp_ = constraintScaleRcp;
  velRcp_ = velRcp;
  forceRcp_ = forceRcp;

  // check the input
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));
  TEUCHOS_ASSERT(nonnull(constraintScaleRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));

  TEUCHOS_ASSERT(xMapRcp_->isSameAs(*constraintScaleRcp_->getMap()));
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
  constraintScaleRcp_.reset();
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
  // TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // dt S^T A^T M A S is symmetric, so trans apply is ok.
  TEUCHOS_ASSERT(nonnull(mobOpRcp_));
  TEUCHOS_ASSERT(nonnull(forceRcp_));
  TEUCHOS_ASSERT(nonnull(forceMagRcp_));
  TEUCHOS_ASSERT(nonnull(velRcp_));
  TEUCHOS_ASSERT(nonnull(AMatRcp_));
  TEUCHOS_ASSERT(nonnull(AMatTransRcp_));
  TEUCHOS_ASSERT(nonnull(constraintScaleRcp_));

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
        if (beta == Teuchos::ScalarTraits<Scalar>::zero()) {
          YcolPtr(k, 0) = dt_ * constraintScalePtr(k, 0) * forceMagPtr(k, 0); // TODO: VERY IMPORTANT! Why does YcolPtr originally have -nan values?
        } else {
          YcolPtr(k, 0) = alpha * dt_ * constraintScalePtr(k, 0) * forceMagPtr(k, 0) + beta * YcolPtr(k, 0);
        } 
      }
    }
  }
}
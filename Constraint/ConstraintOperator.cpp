#include "ConstraintOperator.hpp"
#include "Util/Logger.hpp"

ConstraintOperator::ConstraintOperator(Teuchos::RCP<TOP> &mobOp_,
                                       Teuchos::RCP<TCMAT> &DMatTransRcp_,
                                       Teuchos::RCP<TV> &invKappa_)
    : commRcp(mobOp_->getDomainMap()->getComm()), mobOpRcp(mobOp_),
      DMatTransRcp(DMatTransRcp_), invKappa(invKappa_) {
  // timer
  transposeDMat =
      Teuchos::TimeMonitor::getNewCounter("ConstraintOperator::TransposeDMat");
  applyMobMat =
      Teuchos::TimeMonitor::getNewCounter("ConstraintOperator::ApplyMobility");
  applyDMat =
      Teuchos::TimeMonitor::getNewCounter("ConstraintOperator::ApplyDMat");
  applyDTransMat =
      Teuchos::TimeMonitor::getNewCounter("ConstraintOperator::ApplyDMatTrans");

  enableTimer();

  // explicit transpose
  {
    Teuchos::TimeMonitor mon(*transposeDMat);
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(DMatTransRcp);
    DMatRcp = transposerDu.createTranspose();
  }

  mobMapRcp = mobOpRcp->getDomainMap(); // symmetric & domainmap=rangemap
  gammaMapRcp = invKappa->getMap();

  // initialize working multivectors, zero out
  forceRcp = Teuchos::rcp(new TV(mobMapRcp, true));
  velRcp = Teuchos::rcp(new TV(mobMapRcp, true));
}

void ConstraintOperator::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode,
                               scalar_type alpha, scalar_type beta) const {
  TEUCHOS_TEST_FOR_EXCEPTION(
      X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
      "X and Y do not have the same numbers of vectors (columns).");
  TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()),
                             std::invalid_argument,
                             "X and Y do not have the same Map.\n");

  const int numVecs = X.getNumVectors();
  for (int i = 0; i < numVecs; i++) {
    auto XcolRcp = X.getVector(i);
    auto YcolRcp = Y.getVectorNonConst(i);

    // step 1, D multiply X
    {
      Teuchos::TimeMonitor mon(*applyDMat);
      DMatRcp->apply(*XcolRcp, *forceRcp); // Du gammac
    }

    // step 2, Vel = Mobility * FT
    {
      Teuchos::TimeMonitor mon(*applyMobMat);
      mobOpRcp->apply(*forceRcp, *velRcp);
    }

    // step 3, D^T multiply velocity
    // Y = alpha * Op * X + beta * Y
    {
      Teuchos::TimeMonitor mon(*applyDTransMat);
      DMatTransRcp->apply(*velRcp, *YcolRcp, Teuchos::NO_TRANS, alpha, beta);
    }

    // step 4, add diagonal. Y += alpha * invK * X
    auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
    auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
    YcolRcp->modify<Kokkos::HostSpace>();
    auto invKappaPtr = invKappa->getLocalView<Kokkos::HostSpace>();
    const auto localSize = YcolPtr.extent(0);
#pragma omp parallel for
    for (int k = 0; k < localSize; k++) {
      YcolPtr(k, 0) += alpha * invKappaPtr(k, 0) * XcolPtr(k, 0);
    }
  }
}

Teuchos::RCP<const TMAP> ConstraintOperator::getDomainMap() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!gammaMapRcp.is_valid_ptr(), std::invalid_argument,
                             "gammaMap must be valid");
  return gammaMapRcp;
}

Teuchos::RCP<const TMAP> ConstraintOperator::getRangeMap() const {
  TEUCHOS_TEST_FOR_EXCEPTION(!gammaMapRcp.is_valid_ptr(), std::invalid_argument,
                             "gammaMap must be valid");
  return gammaMapRcp;
}

void ConstraintOperator::enableTimer() {
  transposeDMat->enable();
  applyMobMat->enable();
  applyDMat->enable();
  applyDTransMat->enable();
}

void ConstraintOperator::disableTimer() {
  transposeDMat->disable();
  applyMobMat->disable();
  applyDMat->disable();
  applyDTransMat->disable();
}
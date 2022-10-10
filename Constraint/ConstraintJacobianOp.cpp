#include "ConstraintJacobianOp.hpp"
#include <TpetraExt_TripleMatrixMultiply.hpp>

ConstraintJacobianOp::ConstraintJacobianOp(const Teuchos::RCP<const TMAP> &xMapRcp) : xMapRcp_(xMapRcp) {}

void ConstraintJacobianOp::initialize(const Teuchos::RCP<const TOP> &mobOpRcp,
                                      const Teuchos::RCP<const TCMAT> &DMatRcp,
                                      const Teuchos::RCP<const TCMAT> &DMatTransRcp,
                                      const Teuchos::RCP<const TV> &invKappaDiagRcp, const double dt) {

    // store the objects to be used in apply
    dt_ = dt;
    DMatRcp_ = DMatRcp;
    mobOpRcp_ = mobOpRcp;
    DMatTransRcp_ = DMatTransRcp;
    invKappaDiagRcp_ = invKappaDiagRcp;

    // check the input
    TEUCHOS_ASSERT(nonnull(DMatRcp_));
    TEUCHOS_ASSERT(nonnull(mobOpRcp_));
    TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(invKappaDiagRcp_));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*DMatRcp_->getDomainMap()));
    TEUCHOS_ASSERT(xMapRcp_->isSameAs(*DMatTransRcp_->getRangeMap()));
    TEUCHOS_ASSERT(mobOpRcp_->getDomainMap()->isSameAs(*DMatTransRcp_->getDomainMap()));

    // check if mobOpRcp is a TCMAT in disguise, if so precompute D^T M D
    mobMatRcp_ = Teuchos::rcp_dynamic_cast<const TCMAT>(mobOpRcp_);
    if (nonnull(mobMatRcp_)) {
        // tell apply to use explicitly built A
        matrixFree_ = false;

        // precompute D^T M D
        partialSepPartialGammaMatRcp_ = Teuchos::rcp(new TCMAT(xMapRcp_, 0));
        Tpetra::TripleMatrixMultiply::MultiplyRAP(*DMatRcp_, true, *mobMatRcp_, false, *DMatRcp_, false,
                                                *partialSepPartialGammaMatRcp_);
    } else {
        // tell apply to use matrix free A
        matrixFree_ = true;

        // initialize working multivectors, zero out
        // we only need these for the matrix free apply
        const Teuchos::RCP<const TMAP> mobMapRcp = mobOpRcp_->getDomainMap();
        velRcp_ = Teuchos::rcp(new TV(mobMapRcp, true));
        forceRcp_ = Teuchos::rcp(new TV(mobMapRcp, true));
    }
}

void ConstraintJacobianOp::unitialize() {
    dt_ = 0.0;
    mobOpRcp_.reset();
    mobMatRcp_.reset();
    DMatTransRcp_.reset();
    forceRcp_.reset();
    velRcp_.reset();
}

Teuchos::RCP<const TMAP> ConstraintJacobianOp::getDomainMap() const { return xMapRcp_; }

Teuchos::RCP<const TMAP> ConstraintJacobianOp::getRangeMap() const { return xMapRcp_; }

bool ConstraintJacobianOp::hasTransposeApply() const { return true; }

void ConstraintJacobianOp::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode, Scalar alpha, Scalar beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                               "X and Y do not have the same numbers of vectors (columns).");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                               "X and Y do not have the same Map.\n");
    // TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // dt D^T M D + K^{-1} is symmetric, so trans apply is ok.

    if (matrixFree_) {
        applyMatrixFree(X, Y, mode, alpha, beta);
    } else {
        applyExplicitMatrix(X, Y, mode, alpha, beta);
    }
}

void ConstraintJacobianOp::applyMatrixFree(const TMV &X, TMV &Y, Teuchos::ETransp mode, Scalar alpha,
                                           Scalar beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                               "X and Y do not have the same numbers of vectors (columns).");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                               "X and Y do not have the same Map.\n");
    // TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // dt D^T M D + K^{-1} is symmetric, so trans apply is ok.
    TEUCHOS_ASSERT(nonnull(velRcp_));
    TEUCHOS_ASSERT(nonnull(forceRcp_));
    TEUCHOS_ASSERT(nonnull(mobOpRcp_));
    TEUCHOS_ASSERT(nonnull(DMatRcp_));
    TEUCHOS_ASSERT(nonnull(DMatTransRcp_));
    TEUCHOS_ASSERT(nonnull(invKappaDiagRcp_));

    const int numVecs = X.getNumVectors();
    for (int i = 0; i < numVecs; i++) {
        auto XcolRcp = X.getVector(i);
        auto YcolRcp = Y.getVectorNonConst(i);
        // Goal Y = alpha (dt D^T M D X) + beta Y

        // step 1. D times force magnatude to get force/torque vector (F = D x)
        DMatRcp_->apply(*XcolRcp, *forceRcp_);

        // step 2. Mobility times force/torque vector to get velocity (U = M F)
        mobOpRcp_->apply(*forceRcp_, *velRcp_);

        // step 3. dt D^T times velocity to get change in sep (change in sep = dt D^T U)
        // we merge into this step the fact that Tpetra OPs wants to solve Y = alpha * Op * X + beta * Y
        DMatTransRcp_->apply(*velRcp_, *YcolRcp, Teuchos::NO_TRANS, dt_ * alpha, beta);

        // step 4, add diagonal. Y += alpha * K^{-1} X
        auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
        auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
        auto invKappaDiagPtr = invKappaDiagRcp_->getLocalView<Kokkos::HostSpace>();
        YcolRcp->modify<Kokkos::HostSpace>();
        const auto localSize = YcolPtr.extent(0);
#pragma omp parallel for
        for (size_t idx = 0; idx < localSize; idx++) {
            YcolPtr(idx, 0) += dt_ * alpha * invKappaDiagPtr(idx, 0) * XcolPtr(idx, 0);
        }
    }
}

void ConstraintJacobianOp::applyExplicitMatrix(const TMV &X, TMV &Y, Teuchos::ETransp mode, Scalar alpha,
                                               Scalar beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                               "X and Y do not have the same numbers of vectors (columns).");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                               "X and Y do not have the same Map.\n");
    // TEUCHOS_ASSERT(mode == Teuchos::NO_TRANS); // dt D^T M D + K^{-1} is symmetric, so trans apply is ok.
    TEUCHOS_ASSERT(nonnull(partialSepPartialGammaMatRcp_));
    TEUCHOS_ASSERT(nonnull(invKappaDiagRcp_));

    const int numVecs = X.getNumVectors();
    for (int i = 0; i < numVecs; i++) {
        auto XcolRcp = X.getVector(i);
        auto YcolRcp = Y.getVectorNonConst(i);
        // Goal Y = alpha (dt D^T M D X + K^[-1]X) + beta Y

        // step 1, Y = alpha * D^T M D * X + beta * Y
        partialSepPartialGammaMatRcp_->apply(*XcolRcp, *YcolRcp, Teuchos::NO_TRANS, dt_ * alpha, beta);

        // step 2, add diagonal. Y += alpha * K^{-1} X
        auto XcolPtr = XcolRcp->getLocalView<Kokkos::HostSpace>();
        auto YcolPtr = YcolRcp->getLocalView<Kokkos::HostSpace>();
        auto invKappaDiagPtr = invKappaDiagRcp_->getLocalView<Kokkos::HostSpace>();
        YcolRcp->modify<Kokkos::HostSpace>();
        const auto localSize = YcolPtr.extent(0);
#pragma omp parallel for
        for (size_t idx = 0; idx < localSize; idx++) {
            YcolPtr(idx, 0) += dt_ * alpha * invKappaDiagPtr(idx, 0) * XcolPtr(idx, 0);
        }
    }
}

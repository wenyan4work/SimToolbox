#include "ConstraintOperator.hpp"

ConstraintOperator::ConstraintOperator(Teuchos::RCP<TOP> &mobOp_, Teuchos::RCP<TCMAT> &uniDuMatTrans_,
                                       Teuchos::RCP<TCMAT> &biDbMatTrans_, std::vector<double> &invKappaDiagMat_)
    : commRcp(mobOp_->getDomainMap()->getComm()), mobOpRcp(mobOp_), uniDuMatTransRcp(uniDuMatTrans_),
      biDbMatTransRcp(biDbMatTrans_), invKappaDiagMat(invKappaDiagMat_) {

    // explicit transpose
    Tpetra::RowMatrixTransposer<double, int, int> transposerDu(uniDuMatTransRcp);
    uniDuMatRcp = transposerDu.createTranspose();
    Tpetra::RowMatrixTransposer<double, int, int> transposerDb(biDbMatTransRcp);
    biDbMatRcp = transposerDb.createTranspose();

    // block maps
    mobMapRcp = mobOpRcp->getDomainMap(); // symmetric & domainmap=rangemap

    // setup global map
    // both Du and Db are globally & contiguously partitioned by rows
    // The total map should include rows from both Du and Db, where Db follows Du
    buildBlockMaps();

    // initialize working multivectors, zero out
    mobForceRcp = Teuchos::rcp(new TMV(mobMapRcp, 2, true)); // two columns: Du gammac,  Db gammab
    mobVelRcp = Teuchos::rcp(new TMV(mobMapRcp, 2, true));   // two columns: M Du gamma, M Db gammab
    deltaUniRcp = Teuchos::rcp(new TMV(uniDuMatTransRcp->getRangeMap(), 2, true)); // two columns: DuT M Du, DuT M Db
    deltaBiRcp = Teuchos::rcp(new TMV(biDbMatTransRcp->getRangeMap(), 2, true));   // two columns: DbT M Du, DbT M Db
}

void ConstraintOperator::apply(const TMV &X, TMV &Y, Teuchos::ETransp mode, scalar_type alpha, scalar_type beta) const {
    TEUCHOS_TEST_FOR_EXCEPTION(X.getNumVectors() != Y.getNumVectors(), std::invalid_argument,
                               "X and Y do not have the same numbers of vectors (columns).");
    TEUCHOS_TEST_FOR_EXCEPTION(!X.getMap()->isSameAs(*Y.getMap()), std::invalid_argument,
                               "X and Y do not have the same Map.\n");
    const int blockoffset = gammaUniBlockMapRcp->getNodeNumElements();

    const int numVecs = X.getNumVectors();
    for (int i = 0; i < numVecs; i++) {
        auto XcolRcp = X.getVector(i);
        auto YcolRcp = Y.getVectorNonConst(i);
        auto gammaUniBlock = XcolRcp->offsetView(gammaUniBlockMapRcp, 0);
        auto gammaBiBlock = XcolRcp->offsetView(gammaBiBlockMapRcp, blockoffset);
        auto deltaUniBlock = YcolRcp->offsetViewNonConst(gammaUniBlockMapRcp, 0);
        auto deltaBiBlock = YcolRcp->offsetViewNonConst(gammaBiBlockMapRcp, blockoffset);

        // step 1, Du and Db multiply X, block by block
        auto ftCol0 = mobForceRcp->getVectorNonConst(0);
        uniDuMatRcp->apply(*gammaUniBlock, *ftCol0); // Du gammac
        auto ftCol1 = mobForceRcp->getVectorNonConst(1);
        biDbMatRcp->apply(*gammaBiBlock, *ftCol1); // Db gammab

        // step 2, Vel = Mobility * FT
        mobOpRcp->apply(*mobForceRcp, *mobVelRcp);

        // step 3, Du^T and Db^T multiply velocity
        uniDuMatTransRcp->apply(*mobVelRcp, *deltaUniRcp);
        biDbMatTransRcp->apply(*mobVelRcp, *deltaBiRcp);

        // step 4, add spring constant effect: K^{-1}gammab
        // deltaUniBlock[i] += K^{-1}[i]*gammab[i]
        auto deltaBiPtr = deltaBiBlock->getLocalView<Kokkos::HostSpace>();
        deltaBiBlock->modify<Kokkos::HostSpace>();
        auto gammaBiPtr = gammaBiBlock->getLocalView<Kokkos::HostSpace>();
        const int gammaBiSize = gammaBiPtr.dimension_0();
        TEUCHOS_TEST_FOR_EXCEPTION(!gammaBiSize == invKappaDiagMat.size(), std::invalid_argument,
                                   "Kinv and gammaBi size error\n")
#pragma omp parallel for
        for (int k = 0; k < gammaBiSize; k++) {
            deltaBiPtr(k, 0) += invKappaDiagMat[k] * gammaBiPtr(k, 0);
        }

        // step 5, Y = [deltaUniBlock ; deltaBiBlock] = alpha * Op * X + beta * Y
        // Op * X = [ deltaUni(0)+deltaUni(1); deltaBi(0)+deltaBi(1)]
        // 5.1 uni block
        deltaUniBlock->update(alpha, *(deltaUniRcp->getVector(0)), alpha, *(deltaUniRcp->getVector(1)), beta);
        // 5.2 bi block
        deltaBiBlock->update(alpha, *(deltaBiRcp->getVector(0)), alpha, *(deltaBiRcp->getVector(1)), beta);
    }
}

Teuchos::RCP<const TMAP> ConstraintOperator::getDomainMap() const {
    TEUCHOS_TEST_FOR_EXCEPTION(!gammaMapRcp.is_valid_ptr(), std::invalid_argument, "gammaMap must be valid");
    return gammaMapRcp;
}

Teuchos::RCP<const TMAP> ConstraintOperator::getRangeMap() const {
    TEUCHOS_TEST_FOR_EXCEPTION(!gammaMapRcp.is_valid_ptr(), std::invalid_argument, "gammaMap must be valid");
    return gammaMapRcp;
}

void ConstraintOperator::buildBlockMaps() {
    /****
     *  For details about this sub map, see Tpetra::MultiVector::offsetView() and ::offsetViewNonConst()
     */

    auto map1 = uniDuMatRcp->getDomainMap(); // the first map, starting from 0, contiguous
    auto map2 = biDbMatRcp->getDomainMap();  // the second map, starting from 0, contiguous

    // the two sub block maps
    gammaUniBlockMapRcp = map1;
    gammaBiBlockMapRcp = map2;

    gammaMapRcp = getTMAPFromTwoBlockTMAP(map1, map2);
}